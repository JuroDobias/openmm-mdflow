from __future__ import annotations

from pathlib import Path

import yaml

from openmm_mdflow.cli import main


def _write_yaml(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_cli_validate_success(tmp_path: Path):
    cfg = {
        "project": {"name": "x", "output_dir": str(tmp_path / "run"), "platform": "auto"},
        "system": {"receptor": {"file": "r.pdb"}, "ligands": [], "cofactors": [], "solvation": {"mode": "vacuum"}},
        "forcefield": {
            "protein": ["amber14-all.xml"],
            "water_ions": ["amber14/tip3p.xml"],
            "ligand": {"engine": "openff", "model": "openff-2.0.0", "cache": "ff.json"},
            "hydrogen_mass_amu": 1.5,
        },
        "steps": [{"id": "s1", "type": "minimization", "tolerance_kj_mol_nm": 10, "max_iterations": 10}],
    }
    config_path = tmp_path / "workflow.yaml"
    _write_yaml(config_path, cfg)
    assert main(["validate", "--config", str(config_path)]) == 0


def test_cli_validate_fail(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    _write_yaml(config_path, {"project": {"name": "x"}})
    assert main(["validate", "--config", str(config_path)]) == 2
