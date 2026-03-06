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
