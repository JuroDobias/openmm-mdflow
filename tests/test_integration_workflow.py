from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

openmm = pytest.importorskip("openmm")

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
    assert (out / "steps" / "00_min1" / "done.ok").exists()
    assert (out / "steps" / "01_nvt1" / "done.ok").exists()
    assert (out / "steps" / "02_npt1" / "done.ok").exists()


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
    step_dir = out / "steps" / "00_md1"
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
    assert (out / "steps" / "00_min1" / "done.ok").exists()
    assert (out / "steps" / "01_nvt1" / "done.ok").exists()
