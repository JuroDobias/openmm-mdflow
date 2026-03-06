from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .builder import build_system
from .platforms import select_platform
from .reporting import build_reporters, checkpoint_interval
from .restraints import add_distance_restraints, add_positional_restraints
from .selection import AmberMaskResolver
from .state import initialize_state, save_state, utc_now_iso


def _load_system_from_disk(system_dir: Path):
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    pdb_path = system_dir / "system.pdb"
    xml_path = system_dir / "system.xml"
    pdb = PDBFile(str(pdb_path))
    with xml_path.open("r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    return pdb.topology, pdb.positions, system, pdb_path, xml_path


def _clone_system(system):
    from openmm import XmlSerializer

    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _positions_from_state(state_path: Path):
    from openmm import XmlSerializer

    with state_path.open("r", encoding="utf-8") as handle:
        state = XmlSerializer.deserialize(handle.read())
    return state.getPositions(asNumpy=True)


def _apply_state_positions_to_context(simulation, state_path: Path):
    from openmm import XmlSerializer

    with state_path.open("r", encoding="utf-8") as handle:
        state = XmlSerializer.deserialize(handle.read())

    simulation.context.setPositions(state.getPositions(asNumpy=True))
    box_vectors = state.getPeriodicBoxVectors()
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    velocities = state.getVelocities()
    if velocities is not None:
        simulation.context.setVelocities(velocities)


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


def _resolve_step_restraints(steps: list[dict[str, Any]], topology, positions) -> dict[str, dict[str, Any]]:
    resolver = AmberMaskResolver(topology, positions)
    resolved: dict[str, dict[str, Any]] = {}

    for step_index, step_cfg in enumerate(steps):
        step_id = step_cfg["id"]
        step_resolved: dict[str, Any] = {}
        if "positional_restraints" in step_cfg:
            pos_cfg = step_cfg["positional_restraints"]
            step_resolved["positional_atom_indices"] = resolver.resolve(
                pos_cfg["mask"],
                f"steps[{step_index}].positional_restraints.mask",
            )

        if "distance_restraints" in step_cfg:
            rows: list[dict[str, Any]] = []
            for i, row in enumerate(step_cfg["distance_restraints"]):
                rows.append(
                    {
                        "group1_indices": resolver.resolve(
                            row["group1_mask"],
                            f"steps[{step_index}].distance_restraints[{i}].group1_mask",
                        ),
                        "group2_indices": resolver.resolve(
                            row["group2_mask"],
                            f"steps[{step_index}].distance_restraints[{i}].group2_mask",
                        ),
                        "r0_a": row["r0_a"],
                        "tolerance_a": row.get("tolerance_a", 0.0),
                        "k_kcal_mol_a2": row["k_kcal_mol_a2"],
                    }
                )
            step_resolved["distance_rows"] = rows
        resolved[step_id] = step_resolved
    return resolved


def _apply_step_restraints(system, step_cfg: dict[str, Any], reference_positions, resolved_step_restraints: dict[str, Any]):
    if "positional_restraints" in step_cfg:
        pos_cfg = step_cfg["positional_restraints"]
        atom_indices = resolved_step_restraints["positional_atom_indices"]
        add_positional_restraints(system, reference_positions, atom_indices, pos_cfg)

    if "distance_restraints" in step_cfg:
        add_distance_restraints(system, resolved_step_restraints["distance_rows"])


def _save_final_pdb(simulation, path: Path):
    from openmm.app import PDBFile

    state = simulation.context.getState(getPositions=True)
    box_vectors = state.getPeriodicBoxVectors()
    if box_vectors is not None:
        simulation.topology.setPeriodicBoxVectors(box_vectors)
    with path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)


def _run_md_step(simulation, step_cfg: dict[str, Any], step_dir: Path, starting_completed_steps: int) -> int:
    total_steps = int(step_cfg["n_steps"])
    completed = int(starting_completed_steps)

    for reporter in build_reporters(step_cfg, step_dir):
        simulation.reporters.append(reporter)

    ckpt_interval = checkpoint_interval(step_cfg)
    ckpt_path = step_dir / "checkpoint.xml"
    progress_path = step_dir / "checkpoint_progress.json"

    if ckpt_interval is None:
        simulation.step(total_steps - completed)
        completed = total_steps
    else:
        while completed < total_steps:
            chunk = min(int(ckpt_interval), total_steps - completed)
            simulation.step(chunk)
            completed += chunk
            simulation.saveState(str(ckpt_path))
            with progress_path.open("w", encoding="utf-8") as handle:
                json.dump({"completed_steps": completed, "updated_at": utc_now_iso()}, handle, indent=2)

    if ckpt_interval is not None:
        simulation.saveState(str(ckpt_path))
        with progress_path.open("w", encoding="utf-8") as handle:
            json.dump({"completed_steps": completed, "updated_at": utc_now_iso()}, handle, indent=2)

    return completed


def run_workflow(config: dict[str, Any]):
    from openmm import MonteCarloBarostat
    from openmm.app import Simulation
    from openmm.unit import bar, kilojoule_per_mole, nanometer

    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_root = output_dir / "steps"
    steps_root.mkdir(parents=True, exist_ok=True)

    system_dir = output_dir / "system"
    if (system_dir / "system.xml").exists() and (system_dir / "system.pdb").exists():
        topology, base_positions, base_system, pdb_path, xml_path = _load_system_from_disk(system_dir)
    else:
        artifacts = build_system(config, system_dir)
        topology, base_positions, base_system = artifacts.topology, artifacts.positions, artifacts.system
        pdb_path, xml_path = artifacts.pdb_path, artifacts.system_path

    step_restraints = _resolve_step_restraints(config["steps"], topology, base_positions)

    manifest = initialize_state(
        output_dir,
        config,
        {"system_xml": str(xml_path), "system_pdb": str(pdb_path)},
    )
    manifest.setdefault("steps", {})

    platform, platform_properties = select_platform(config["project"])
    selected_platform_name = platform.getName()
    requested_platform = str(config["project"].get("platform", "auto"))
    prev_state_path: Path | None = None

    for step_index, step_cfg in enumerate(config["steps"]):
        step_id = step_cfg["id"]
        step_dir = steps_root / f"{step_index:02d}_{step_id}"
        step_dir.mkdir(parents=True, exist_ok=True)

        done_file = step_dir / "done.ok"
        final_state_path = step_dir / "final_state.xml"
        final_pdb_path = step_dir / "final_state.pdb"
        checkpoint_path = step_dir / "checkpoint.xml"
        progress_path = step_dir / "checkpoint_progress.json"

        step_state = manifest["steps"].get(step_id, {})
        step_state.update(
            {
                "index": step_index,
                "id": step_id,
                "directory": str(step_dir),
                "final_state": str(final_state_path),
                "checkpoint": str(checkpoint_path),
                "status": "pending",
                "updated_at": utc_now_iso(),
            }
        )

        if done_file.exists() and final_state_path.exists():
            step_state["status"] = "completed"
            step_state["done_at"] = done_file.read_text(encoding="utf-8").strip()
            manifest["steps"][step_id] = step_state
            manifest["updated_at"] = utc_now_iso()
            save_state(output_dir, manifest)
            prev_state_path = final_state_path
            continue

        step_state["status"] = "running"
        manifest["steps"][step_id] = step_state
        manifest["updated_at"] = utc_now_iso()
        save_state(output_dir, manifest)

        step_system = _clone_system(base_system)

        reference_positions = base_positions
        if prev_state_path is not None and prev_state_path.exists():
            reference_positions = _positions_from_state(prev_state_path)

        _apply_step_restraints(step_system, step_cfg, reference_positions, step_restraints.get(step_id, {}))
        integrator = _build_integrator(step_cfg)

        if step_cfg["type"] == "md" and step_cfg["ensemble"] == "NPT":
            temp_k = float(step_cfg["thermostat"]["temperature_k"])
            barostat_cfg = step_cfg["barostat"]
            step_system.addForce(
                MonteCarloBarostat(
                    float(barostat_cfg["pressure_bar"]) * bar,
                    temp_k,
                    int(barostat_cfg["frequency"]),
                )
            )

        try:
            simulation = Simulation(topology, step_system, integrator, platform, platform_properties)
        except Exception as exc:
            # Common on some systems: CUDA platform detected, but runtime/driver
            # cannot compile kernels for the visible GPU architecture.
            if requested_platform == "auto" and selected_platform_name == "CUDA":
                from openmm import Platform

                cpu_platform = Platform.getPlatformByName("CPU")
                cpu_properties = {"Threads": "1"}
                print(
                    "Warning: CUDA context initialization failed in auto mode. "
                    "Falling back to CPU platform for this run."
                )
                simulation = Simulation(topology, step_system, integrator, cpu_platform, cpu_properties)
                selected_platform_name = "CPU"
                platform = cpu_platform
                platform_properties = cpu_properties
            else:
                raise exc
        simulation.context.setPositions(base_positions)

        completed_steps = 0
        if step_cfg["type"] == "md" and checkpoint_path.exists() and progress_path.exists():
            with progress_path.open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
            completed_steps = int(progress.get("completed_steps", 0))
            if completed_steps < int(step_cfg["n_steps"]):
                simulation.loadState(str(checkpoint_path))
        elif prev_state_path is not None and prev_state_path.exists():
            _apply_state_positions_to_context(simulation, prev_state_path)

        if step_cfg["type"] == "minimization":
            simulation.minimizeEnergy(
                tolerance=float(step_cfg["tolerance_kj_mol_nm"]) * kilojoule_per_mole / nanometer,
                maxIterations=int(step_cfg["max_iterations"]),
            )
            completed_steps = 0
        else:
            completed_steps = _run_md_step(simulation, step_cfg, step_dir, completed_steps)

        simulation.saveState(str(final_state_path))
        _save_final_pdb(simulation, final_pdb_path)
        done_file.write_text(utc_now_iso(), encoding="utf-8")

        step_state["status"] = "completed"
        step_state["completed_steps"] = int(completed_steps)
        step_state["done_at"] = done_file.read_text(encoding="utf-8").strip()
        step_state["updated_at"] = utc_now_iso()
        manifest["steps"][step_id] = deepcopy(step_state)
        manifest["updated_at"] = utc_now_iso()
        save_state(output_dir, manifest)

        prev_state_path = final_state_path
