from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

openmm = pytest.importorskip("openmm")
mdtraj = pytest.importorskip("mdtraj")

from openmm_mdflow.config import load_and_validate
from openmm_mdflow.workflow import run_workflow


def _write_minimal_water_pdb(path: Path):
    path.write_text(
        "\n".join(
            [
                "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O",
                "ATOM      2  H1  HOH A   1       0.095   0.000   0.000  1.00  0.00           H",
                "ATOM      3  H2  HOH A   1      -0.032   0.090   0.000  1.00  0.00           H",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def _write_two_waters_pdb(path: Path):
    path.write_text(
        "\n".join(
            [
                "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O",
                "ATOM      2  H1  HOH A   1       0.095   0.000   0.000  1.00  0.00           H",
                "ATOM      3  H2  HOH A   1      -0.032   0.090   0.000  1.00  0.00           H",
                "ATOM      4  O   HOH A   2       3.500   0.000   0.000  1.00  0.00           O",
                "ATOM      5  H1  HOH A   2       3.595   0.000   0.000  1.00  0.00           H",
                "ATOM      6  H2  HOH A   2       3.468   0.090   0.000  1.00  0.00           H",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def _write_two_atom_pdb(path: Path):
    path.write_text(
        "\n".join(
            [
                "ATOM      1  O   HOH A   1       0.300   0.000   0.000  1.00  0.00           O",
                "ATOM      2  H1  HOH A   1       0.395   0.000   0.000  1.00  0.00           H",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def _base_config(tmp_path: Path, receptor_path: Path):
    return {
        "project": {
            "name": "it",
            "output_dir": str(tmp_path / "run"),
            "platform": "CPU",
        },
        "system": {
            "receptor": {"file": str(receptor_path)},
            "ligands": [],
            "cofactors": [],
            # Keep padding large enough so PME cutoff (0.9 nm) is <= half-box length.
            "solvation": {"mode": "explicit", "ionic_strength_molar": 0.0, "padding_nm": 1.0},
        },
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
            "nonbonded_cutoff_nm": 0.9,
        },
        "steps": [
            {"id": "min1", "type": "minimization", "tolerance_kj_mol_nm": 10.0, "max_iterations": 5},
            {
                "id": "nvt1",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 2,
                "timestep_ps": 0.001,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
                "reporters": {"state": {"interval": 1}, "checkpoint": {"interval": 1}},
            },
            {
                "id": "npt1",
                "type": "md",
                "ensemble": "NPT",
                "n_steps": 2,
                "timestep_ps": 0.001,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
                "barostat": {"pressure_bar": 1.0, "frequency": 1},
                "reporters": {"state": {"interval": 1}, "checkpoint": {"interval": 1}},
            },
        ],
    }


def test_workflow_build_and_run(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    _write_minimal_water_pdb(receptor)
    cfg = _base_config(tmp_path, receptor)
    cfg_path = tmp_path / "workflow.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_workflow(load_and_validate(cfg_path))

    out = Path(cfg["project"]["output_dir"])
    assert (out / "system" / "system.xml").exists()
    assert (out / "steps" / "min1" / "done.ok").exists()
    assert (out / "steps" / "nvt1" / "done.ok").exists()
    assert (out / "steps" / "npt1" / "done.ok").exists()


def test_restart_skip_and_checkpoint(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    _write_minimal_water_pdb(receptor)
    cfg = _base_config(tmp_path, receptor)
    cfg["steps"] = [
        {
            "id": "md1",
            "type": "md",
            "ensemble": "NVT",
            "n_steps": 4,
            "timestep_ps": 0.001,
            "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
            "reporters": {"checkpoint": {"interval": 2}},
        }
    ]
    cfg_path = tmp_path / "workflow.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    validated = load_and_validate(cfg_path)
    run_workflow(validated)

    out = Path(cfg["project"]["output_dir"])
    step_dir = out / "steps" / "md1"
    done = step_dir / "done.ok"
    mtime_before = done.stat().st_mtime

    run_workflow(validated)
    assert done.stat().st_mtime == mtime_before

    done.unlink()
    (step_dir / "final_state.xml").unlink(missing_ok=True)
    progress = {"completed_steps": 2}
    with (step_dir / "checkpoint_progress.json").open("w", encoding="utf-8") as handle:
        json.dump(progress, handle)

    run_workflow(validated)
    assert done.exists()


def test_mask_restraints_workflow_run(tmp_path: Path):
    receptor = tmp_path / "receptor_two_waters.pdb"
    _write_two_waters_pdb(receptor)
    cfg = {
        "project": {
            "name": "it_masks",
            "output_dir": str(tmp_path / "run_masks"),
            "platform": "CPU",
        },
        "system": {
            "receptor": {"file": str(receptor)},
            "ligands": [],
            "cofactors": [],
            "solvation": {"mode": "vacuum"},
        },
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
            "nonbonded_cutoff_nm": 0.9,
        },
        "steps": [
            {
                "id": "min1",
                "type": "minimization",
                "tolerance_kj_mol_nm": 10.0,
                "max_iterations": 3,
                "positional_restraints": {
                    "mask": ":1@O",
                    "k_kcal_mol_a2": 10.0,
                    "tolerance_a": 0.5,
                },
            },
            {
                "id": "nvt1",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 2,
                "timestep_ps": 0.001,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
                "distance_restraints": [
                    {
                        "group1_mask": ":1@O",
                        "group2_mask": ":2@O",
                        "r0_a": 3.5,
                        "k_kcal_mol_a2": 5.0,
                    }
                ],
            },
        ],
    }
    cfg_path = tmp_path / "workflow_mask.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_workflow(load_and_validate(cfg_path))

    out = Path(cfg["project"]["output_dir"])
    assert (out / "steps" / "min1" / "done.ok").exists()
    assert (out / "steps" / "nvt1" / "done.ok").exists()


def test_step_reference_missing_file_fails(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    _write_minimal_water_pdb(receptor)
    cfg = _base_config(tmp_path, receptor)
    cfg["steps"][0]["positional_restraints"] = {"mask": ":1@O", "k_kcal_mol_a2": 10.0, "tolerance_a": 0.5}
    cfg["steps"][0]["restraint_reference"] = str(tmp_path / "missing_reference.pdb")
    cfg_path = tmp_path / "workflow_missing_ref.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    with pytest.raises(ValueError, match="positional restraint reference"):
        run_workflow(load_and_validate(cfg_path))


def test_step_reference_atom_count_mismatch_fails(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    reference_bad = tmp_path / "reference_bad.pdb"
    _write_minimal_water_pdb(receptor)
    _write_two_atom_pdb(reference_bad)
    cfg = _base_config(tmp_path, receptor)
    cfg["steps"][0]["positional_restraints"] = {"mask": ":1@O", "k_kcal_mol_a2": 10.0, "tolerance_a": 0.5}
    cfg["steps"][0]["restraint_reference"] = str(reference_bad)
    cfg_path = tmp_path / "workflow_bad_ref.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    with pytest.raises(ValueError, match="atom count mismatch"):
        run_workflow(load_and_validate(cfg_path))


def test_trajectory_minimization_from_previous_step(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    _write_minimal_water_pdb(receptor)
    out_dir = tmp_path / "run_trajmin"
    cfg = {
        "project": {"name": "trajmin", "output_dir": str(out_dir), "platform": "CPU"},
        "system": {
            "receptor": {"file": str(receptor)},
            "ligands": [],
            "cofactors": [],
            "solvation": {"mode": "vacuum"},
        },
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
            "nonbonded_cutoff_nm": 0.9,
        },
        "steps": [
            {
                "id": "md1",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 4,
                "timestep_ps": 0.001,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
                "reporters": {"traj": {"format": "dcd", "interval": 1}},
            },
            {
                "id": "trajmin1",
                "type": "trajectory_minimization",
                "tolerance_kj_mol_nm": 10.0,
                "max_iterations": 10,
                "parallel": {"workers": 2},
            },
        ],
    }
    cfg_path = tmp_path / "workflow_trajmin.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_workflow(load_and_validate(cfg_path))

    input_traj = out_dir / "steps" / "md1" / "trajectory.dcd"
    out_traj = out_dir / "steps" / "trajmin1" / "trajectory.dcd"
    assert input_traj.exists()
    assert out_traj.exists()
    assert (out_dir / "steps" / "trajmin1" / "done.ok").exists()

    top_pdb = out_dir / "system" / "system.pdb"
    in_t = mdtraj.load(str(input_traj), top=str(top_pdb))
    out_t = mdtraj.load(str(out_traj), top=str(top_pdb))
    assert in_t.n_frames == out_t.n_frames


def test_trajectory_minimization_resume_from_frame_progress(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    _write_minimal_water_pdb(receptor)
    out_dir = tmp_path / "run_traj_resume"

    setup_cfg = {
        "project": {"name": "trajsetup", "output_dir": str(out_dir), "platform": "CPU"},
        "system": {
            "receptor": {"file": str(receptor)},
            "ligands": [],
            "cofactors": [],
            "solvation": {"mode": "vacuum"},
        },
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
            "nonbonded_cutoff_nm": 0.9,
        },
        "steps": [
            {
                "id": "md1",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 4,
                "timestep_ps": 0.001,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
                "reporters": {"traj": {"format": "dcd", "interval": 1}},
            }
        ],
    }
    setup_path = tmp_path / "workflow_setup.yaml"
    with setup_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(setup_cfg, handle, sort_keys=False)
    run_workflow(load_and_validate(setup_path))

    input_traj = out_dir / "steps" / "md1" / "trajectory.dcd"
    top_pdb = out_dir / "system" / "system.pdb"
    in_t = mdtraj.load(str(input_traj), top=str(top_pdb))
    assert in_t.n_frames >= 4

    cfg = {
        "project": {"name": "trajresume", "output_dir": str(out_dir), "platform": "CPU"},
        "system": setup_cfg["system"],
        "forcefield": setup_cfg["forcefield"],
        "steps": [
            {
                "id": "trajmin_resume",
                "type": "trajectory_minimization",
                "tolerance_kj_mol_nm": 10.0,
                "max_iterations": 10,
                "input": {"trajectory": str(input_traj)},
                "parallel": {"workers": 1},
            }
        ],
    }
    cfg_path = tmp_path / "workflow_traj_resume.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    step_dir = out_dir / "steps" / "trajmin_resume"
    step_dir.mkdir(parents=True, exist_ok=True)
    partial_out = step_dir / "trajectory.dcd"
    in_t[:2].save_dcd(str(partial_out))
    with (step_dir / "frame_progress.json").open("w", encoding="utf-8") as handle:
        json.dump({"completed_frames": 2}, handle)

    run_workflow(load_and_validate(cfg_path))
    out_t = mdtraj.load(str(partial_out), top=str(top_pdb))
    assert out_t.n_frames == in_t.n_frames


def test_trajectory_minimization_fixed_reference_pdb(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    reference = tmp_path / "reference.pdb"
    _write_minimal_water_pdb(receptor)
    _write_minimal_water_pdb(reference)
    out_dir = tmp_path / "run_traj_fixed_ref"
    cfg = {
        "project": {"name": "trajfixref", "output_dir": str(out_dir), "platform": "CPU"},
        "system": {
            "receptor": {"file": str(receptor)},
            "ligands": [],
            "cofactors": [],
            "solvation": {"mode": "vacuum"},
        },
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
            "nonbonded_cutoff_nm": 0.9,
        },
        "steps": [
            {
                "id": "md1",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 4,
                "timestep_ps": 0.001,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
                "reporters": {"traj": {"format": "dcd", "interval": 1}},
            },
            {
                "id": "trajmin_fixedref",
                "type": "trajectory_minimization",
                "tolerance_kj_mol_nm": 10.0,
                "max_iterations": 10,
                "restraint_reference": str(reference),
                "positional_restraints": {"mask": ":1@O", "k_kcal_mol_a2": 10.0, "tolerance_a": 0.5},
            },
        ],
    }
    cfg_path = tmp_path / "workflow_traj_fixed_ref.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_workflow(load_and_validate(cfg_path))

    input_traj = out_dir / "steps" / "md1" / "trajectory.dcd"
    out_traj = out_dir / "steps" / "trajmin_fixedref" / "trajectory.dcd"
    assert input_traj.exists()
    assert out_traj.exists()
    top_pdb = out_dir / "system" / "system.pdb"
    in_t = mdtraj.load(str(input_traj), top=str(top_pdb))
    out_t = mdtraj.load(str(out_traj), top=str(top_pdb))
    assert in_t.n_frames == out_t.n_frames


def test_md_step_reference_input_uses_system_geometry(tmp_path: Path):
    receptor = tmp_path / "receptor.pdb"
    _write_minimal_water_pdb(receptor)
    cfg = _base_config(tmp_path, receptor)
    cfg["steps"] = [
        {
            "id": "nvt_ref_input",
            "type": "md",
            "ensemble": "NVT",
            "n_steps": 2,
            "timestep_ps": 0.001,
            "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
            "restraint_reference": "input",
            "positional_restraints": {"mask": ":1@O", "k_kcal_mol_a2": 10.0, "tolerance_a": 0.5},
        }
    ]
    cfg_path = tmp_path / "workflow_ref_input.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    run_workflow(load_and_validate(cfg_path))
    out = Path(cfg["project"]["output_dir"])
    assert (out / "steps" / "nvt_ref_input" / "done.ok").exists()
