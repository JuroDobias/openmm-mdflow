from __future__ import annotations

import pytest

from openmm_mdflow.config import ConfigError, validate_config


def _base_config():
    return {
        "project": {"name": "t1", "output_dir": "runs/t1", "platform": "auto"},
        "system": {
            "receptor": {"file": "receptor.pdb"},
            "ligands": [],
            "cofactors": [],
            "solvation": {"mode": "vacuum"},
        },
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
        },
        "steps": [
            {"id": "min1", "type": "minimization", "tolerance_kj_mol_nm": 10.0, "max_iterations": 100},
            {
                "id": "md1",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 1000,
                "timestep_ps": 0.004,
                "thermostat": {"kind": "langevin_middle", "temperature_k": 300, "friction_per_ps": 1.0},
            },
        ],
    }


def test_validate_config_success():
    cfg = validate_config(_base_config())
    assert cfg["project"]["name"] == "t1"
    assert cfg["steps"][1]["ensemble"] == "NVT"


def test_duplicate_step_id_fails():
    cfg = _base_config()
    cfg["steps"][1]["id"] = "min1"
    with pytest.raises(ConfigError, match="Duplicate step id"):
        validate_config(cfg)


def test_npt_requires_barostat():
    cfg = _base_config()
    cfg["steps"][1]["ensemble"] = "NPT"
    with pytest.raises(ConfigError, match="barostat"):
        validate_config(cfg)


def test_mask_restraints_schema_success():
    cfg = _base_config()
    cfg["steps"][0]["positional_restraints"] = {
        "mask": ":1 & !@H=",
        "k_kcal_mol_a2": 10.0,
        "tolerance_a": 1.0,
    }
    cfg["steps"][1]["distance_restraints"] = [
        {
            "group1_mask": ":1@CA",
            "group2_mask": ":1@N",
            "r0_a": 3.0,
            "tolerance_a": 0.2,
            "k_kcal_mol_a2": 2.0,
        }
    ]
    out = validate_config(cfg)
    assert out["steps"][0]["positional_restraints"]["mask"] == ":1 & !@H="
    assert out["steps"][1]["distance_restraints"][0]["group1_mask"] == ":1@CA"
    assert out["steps"][1]["distance_restraints"][0]["tolerance_a"] == 0.2


def test_distance_tolerance_defaults_to_zero():
    cfg = _base_config()
    cfg["steps"][1]["distance_restraints"] = [
        {
            "group1_mask": ":1@CA",
            "group2_mask": ":1@N",
            "r0_a": 3.0,
            "k_kcal_mol_a2": 2.0,
        }
    ]
    out = validate_config(cfg)
    assert out["steps"][1]["distance_restraints"][0]["tolerance_a"] == 0.0


def test_index_based_restraints_rejected():
    cfg = _base_config()
    cfg["steps"][0]["positional_restraints"] = {
        "atoms": [0, 1],
        "k_kcal_mol_a2": 10.0,
        "tolerance_a": 1.0,
    }
    with pytest.raises(ConfigError, match="no longer supported"):
        validate_config(cfg)

    cfg = _base_config()
    cfg["steps"][1]["distance_restraints"] = [{"atoms": [0, 1], "r0_a": 3.0, "k_kcal_mol_a2": 2.0}]
    with pytest.raises(ConfigError, match="no longer supported"):
        validate_config(cfg)


def test_trajectory_minimization_schema_success():
    cfg = _base_config()
    cfg["steps"] = [
        {
            "id": "traj_min",
            "type": "trajectory_minimization",
            "tolerance_kj_mol_nm": 8.0,
            "max_iterations": 200,
            "input": {"trajectory": "runs/x/steps/md1/trajectory.xtc"},
            "parallel": {"workers": 2},
            "distance_restraints": [
                {
                    "group1_mask": ":1@CA",
                    "group2_mask": ":1@N",
                    "r0_a": 3.0,
                    "tolerance_a": 0.2,
                    "k_kcal_mol_a2": 2.0,
                }
            ],
        }
    ]
    out = validate_config(cfg)
    step = out["steps"][0]
    assert step["type"] == "trajectory_minimization"
    assert step["parallel"]["workers"] == 2
    assert step["input"]["trajectory"].endswith("trajectory.xtc")


def test_trajectory_minimization_rejects_md_only_fields():
    cfg = _base_config()
    cfg["steps"] = [
        {
            "id": "traj_min",
            "type": "trajectory_minimization",
            "tolerance_kj_mol_nm": 8.0,
            "max_iterations": 200,
            "n_steps": 100,
        }
    ]
    with pytest.raises(ConfigError, match="not valid for `trajectory_minimization`"):
        validate_config(cfg)
