from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .restraints import add_distance_restraints, add_positional_restraints
from .state import utc_now_iso


@dataclass
class TrajectoryMinResult:
    completed_frames: int
    total_frames: int
    output_path: Path
    wall_seconds: float
    last_positions_nm: Any | None
    last_box_vectors_nm: Any | None


class TrajectoryMinimizationError(RuntimeError):
    pass


def _build_integrator(step_cfg: dict[str, Any]):
    import openmm as mm
    from openmm.unit import kelvin, picosecond

    kind = str(step_cfg.get("integrator", "langevin_middle")).lower()
    timestep = float(step_cfg.get("timestep_ps", 0.002)) * picosecond
    if kind == "verlet":
        return mm.VerletIntegrator(timestep)
    if kind == "langevin_middle":
        thermostat = step_cfg.get("thermostat", {})
        temperature = float(thermostat.get("temperature_k", 300.0)) * kelvin
        friction = float(thermostat.get("friction_per_ps", 1.0)) / picosecond
        return mm.LangevinMiddleIntegrator(temperature, friction, timestep)
    raise ValueError(f"Unsupported integrator `{kind}`")


def _read_progress(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return int(payload.get("completed_frames", 0))


def _write_progress(path: Path, completed_frames: int):
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"completed_frames": int(completed_frames), "updated_at": utc_now_iso()}, handle, indent=2)


def _count_trajectory_frames(path: Path) -> int:
    import mdtraj as md

    with md.open(str(path), "r") as handle:
        return int(len(handle))


def _frame_iter(path: Path, topology_pdb: Path, chunk_size: int = 50):
    import mdtraj as md

    for chunk in md.iterload(str(path), top=str(topology_pdb), chunk=int(chunk_size)):
        for i in range(chunk.n_frames):
            box = None
            if chunk.unitcell_vectors is not None:
                box = chunk.unitcell_vectors[i]
            yield chunk.xyz[i], box


def _infer_output_trajectory_path(step_dir: Path, input_path: Path) -> Path:
    ext = input_path.suffix.lower()
    if ext not in {".xtc", ".dcd"}:
        raise TrajectoryMinimizationError(
            f"Unsupported input trajectory format `{input_path}`. Supported: .xtc, .dcd"
        )
    return step_dir / f"trajectory{ext}"


def _create_writer(output_path: Path, topology, append: bool):
    from openmm.app import DCDFile, XTCFile

    dt_ps = 1.0
    if output_path.suffix.lower() == ".dcd":
        mode = "r+b" if append else "wb"
        handle = output_path.open(mode)
        return DCDFile(handle, topology, dt_ps, append=append), handle
    return XTCFile(str(output_path), topology, dt_ps, append=append), None


def _to_box_vectors_quantity(box_vectors_nm):
    from openmm import Vec3
    from openmm.unit import nanometer

    if box_vectors_nm is None:
        return None
    return [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in box_vectors_nm] * nanometer


def _worker_build_runtime(
    topology,
    system_xml: str,
    step_cfg: dict[str, Any],
    resolved_step_restraints: dict[str, Any],
    platform_name: str,
    platform_properties: dict[str, str],
    init_reference_positions_nm,
):
    from openmm import Platform, XmlSerializer
    from openmm.unit import nanometer
    from openmm.app import Simulation

    system = XmlSerializer.deserialize(system_xml)
    pos_force = None
    pos_atoms = None

    if "positional_atom_indices" in resolved_step_restraints:
        pos_atoms = list(resolved_step_restraints["positional_atom_indices"])
        pos_force = add_positional_restraints(
            system,
            init_reference_positions_nm * nanometer,
            pos_atoms,
            step_cfg["positional_restraints"],
        )
    if "distance_rows" in resolved_step_restraints:
        add_distance_restraints(system, resolved_step_restraints["distance_rows"])

    integrator = _build_integrator(step_cfg)
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(topology, system, integrator, platform, platform_properties)
    return simulation, pos_force, pos_atoms


def _update_positional_reference(pos_force, pos_atoms: list[int], frame_nm):
    if pos_force is None:
        return
    for force_index, atom in enumerate(pos_atoms):
        x0, y0, z0 = frame_nm[atom]
        pos_force.setParticleParameters(force_index, atom, [float(x0), float(y0), float(z0)])


def _minimize_single_frame(
    simulation,
    pos_force,
    pos_atoms: list[int] | None,
    frame_nm,
    box_vectors_nm,
    step_cfg: dict[str, Any],
):
    from openmm.unit import kilojoule_per_mole, nanometer

    if pos_force is not None and pos_atoms is not None:
        _update_positional_reference(pos_force, pos_atoms, frame_nm)
        pos_force.updateParametersInContext(simulation.context)

    simulation.context.setPositions(frame_nm * nanometer)
    periodic_box = _to_box_vectors_quantity(box_vectors_nm)
    if periodic_box is not None:
        simulation.context.setPeriodicBoxVectors(*periodic_box)

    simulation.minimizeEnergy(
        tolerance=float(step_cfg["tolerance_kj_mol_nm"]) * kilojoule_per_mole / nanometer,
        maxIterations=int(step_cfg["max_iterations"]),
    )
    state = simulation.context.getState(getPositions=True)
    out_positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    out_box = None
    box = state.getPeriodicBoxVectors()
    if box is not None:
        out_box = box.value_in_unit(nanometer)
    return out_positions, out_box


_WORKER_RUNTIME = None


def _worker_init(
    topology_pdb_path: str,
    system_xml: str,
    step_cfg: dict[str, Any],
    resolved_step_restraints: dict[str, Any],
    platform_name: str,
    platform_properties: dict[str, str],
    init_reference_positions_nm,
):
    from openmm.app import PDBFile

    global _WORKER_RUNTIME
    topology = PDBFile(str(topology_pdb_path)).topology
    simulation, pos_force, pos_atoms = _worker_build_runtime(
        topology,
        system_xml,
        step_cfg,
        resolved_step_restraints,
        platform_name,
        platform_properties,
        init_reference_positions_nm,
    )
    _WORKER_RUNTIME = (simulation, pos_force, pos_atoms, step_cfg)


def _worker_run_frame(task):
    frame_index, frame_nm, box_vectors_nm = task
    global _WORKER_RUNTIME
    simulation, pos_force, pos_atoms, step_cfg = _WORKER_RUNTIME
    out_positions, out_box = _minimize_single_frame(
        simulation, pos_force, pos_atoms, frame_nm, box_vectors_nm, step_cfg
    )
    return frame_index, out_positions, out_box


def run_trajectory_minimization_step(
    *,
    step_cfg: dict[str, Any],
    step_dir: Path,
    topology,
    base_system,
    system_pdb_path: Path,
    resolved_step_restraints: dict[str, Any],
    input_trajectory_path: Path,
    platform_name: str,
    platform_properties: dict[str, str],
) -> TrajectoryMinResult:
    from openmm import XmlSerializer
    from openmm.unit import nanometer

    output_path = _infer_output_trajectory_path(step_dir, input_trajectory_path)
    if input_trajectory_path.resolve() == output_path.resolve():
        raise TrajectoryMinimizationError("Input and output trajectory paths must be different.")

    progress_path = step_dir / "frame_progress.json"
    total_frames = _count_trajectory_frames(input_trajectory_path)
    completed_frames = _read_progress(progress_path)
    if completed_frames < 0 or completed_frames > total_frames:
        raise TrajectoryMinimizationError(
            f"Invalid frame progress {completed_frames} for trajectory with {total_frames} frames."
        )
    if completed_frames > 0 and not output_path.exists():
        raise TrajectoryMinimizationError(
            f"Progress indicates {completed_frames} completed frames but output file is missing: {output_path}"
        )

    frame_iter = _frame_iter(input_trajectory_path, system_pdb_path)
    first_frame_positions_nm = None
    first_frame_box_nm = None
    for frame_positions_nm, frame_box_nm in frame_iter:
        first_frame_positions_nm = frame_positions_nm
        first_frame_box_nm = frame_box_nm
        break
    if first_frame_positions_nm is None:
        writer, writer_handle = _create_writer(output_path, topology, append=False)
        del writer
        if writer_handle is not None:
            writer_handle.close()
        _write_progress(progress_path, 0)
        return TrajectoryMinResult(0, 0, output_path, 0.0, None, None)

    if len(first_frame_positions_nm) != topology.getNumAtoms():
        raise TrajectoryMinimizationError(
            f"Trajectory atom count {len(first_frame_positions_nm)} does not match system atom count "
            f"{topology.getNumAtoms()}."
        )

    frame_iter = _frame_iter(input_trajectory_path, system_pdb_path)
    system_xml = XmlSerializer.serialize(base_system)
    workers = int(step_cfg.get("parallel", {}).get("workers", 1))

    writer, writer_handle = _create_writer(output_path, topology, append=completed_frames > 0)
    frame_index = 0
    last_positions_nm = None
    last_box_vectors_nm = None
    wall_start = time.perf_counter()

    try:
        if workers <= 1:
            simulation, pos_force, pos_atoms = _worker_build_runtime(
                topology,
                system_xml,
                step_cfg,
                resolved_step_restraints,
                platform_name,
                platform_properties,
                first_frame_positions_nm,
            )
            for frame_positions_nm, frame_box_nm in frame_iter:
                if frame_index < completed_frames:
                    frame_index += 1
                    continue
                out_positions_nm, out_box_nm = _minimize_single_frame(
                    simulation, pos_force, pos_atoms, frame_positions_nm, frame_box_nm, step_cfg
                )
                if out_box_nm is not None:
                    writer.writeModel(out_positions_nm * nanometer, periodicBoxVectors=_to_box_vectors_quantity(out_box_nm))
                else:
                    writer.writeModel(out_positions_nm * nanometer)
                frame_index += 1
                completed_frames = frame_index
                _write_progress(progress_path, completed_frames)
                last_positions_nm = out_positions_nm
                last_box_vectors_nm = out_box_nm
        else:
            chunk_size = max(4 * workers, 8)
            pending: list[tuple[int, Any, Any]] = []
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=workers,
                    initializer=_worker_init,
                    initargs=(
                        str(system_pdb_path),
                        system_xml,
                        step_cfg,
                        resolved_step_restraints,
                        platform_name,
                        platform_properties,
                        first_frame_positions_nm,
                    ),
                ) as pool:
                    for frame_positions_nm, frame_box_nm in frame_iter:
                        if frame_index < completed_frames:
                            frame_index += 1
                            continue
                        pending.append((frame_index, frame_positions_nm, frame_box_nm))
                        frame_index += 1
                        if len(pending) >= chunk_size:
                            results = pool.map(_worker_run_frame, pending)
                            for idx, out_positions_nm, out_box_nm in results:
                                if out_box_nm is not None:
                                    writer.writeModel(
                                        out_positions_nm * nanometer,
                                        periodicBoxVectors=_to_box_vectors_quantity(out_box_nm),
                                    )
                                else:
                                    writer.writeModel(out_positions_nm * nanometer)
                                completed_frames = idx + 1
                                _write_progress(progress_path, completed_frames)
                                last_positions_nm = out_positions_nm
                                last_box_vectors_nm = out_box_nm
                            pending = []
                    if pending:
                        results = pool.map(_worker_run_frame, pending)
                        for idx, out_positions_nm, out_box_nm in results:
                            if out_box_nm is not None:
                                writer.writeModel(
                                    out_positions_nm * nanometer,
                                    periodicBoxVectors=_to_box_vectors_quantity(out_box_nm),
                                )
                            else:
                                writer.writeModel(out_positions_nm * nanometer)
                            completed_frames = idx + 1
                            _write_progress(progress_path, completed_frames)
                            last_positions_nm = out_positions_nm
                            last_box_vectors_nm = out_box_nm
            except (PermissionError, OSError):
                # Some restricted environments disallow multiprocessing semaphores.
                simulation, pos_force, pos_atoms = _worker_build_runtime(
                    topology,
                    system_xml,
                    step_cfg,
                    resolved_step_restraints,
                    platform_name,
                    platform_properties,
                    first_frame_positions_nm,
                )
                for frame_positions_nm, frame_box_nm in frame_iter:
                    if frame_index < completed_frames:
                        frame_index += 1
                        continue
                    out_positions_nm, out_box_nm = _minimize_single_frame(
                        simulation, pos_force, pos_atoms, frame_positions_nm, frame_box_nm, step_cfg
                    )
                    if out_box_nm is not None:
                        writer.writeModel(
                            out_positions_nm * nanometer,
                            periodicBoxVectors=_to_box_vectors_quantity(out_box_nm),
                        )
                    else:
                        writer.writeModel(out_positions_nm * nanometer)
                    frame_index += 1
                    completed_frames = frame_index
                    _write_progress(progress_path, completed_frames)
                    last_positions_nm = out_positions_nm
                    last_box_vectors_nm = out_box_nm
    finally:
        if writer_handle is not None:
            writer_handle.close()

    wall_seconds = time.perf_counter() - wall_start
    return TrajectoryMinResult(
        completed_frames=completed_frames,
        total_frames=total_frames,
        output_path=output_path,
        wall_seconds=wall_seconds,
        last_positions_nm=last_positions_nm,
        last_box_vectors_nm=last_box_vectors_nm,
    )
