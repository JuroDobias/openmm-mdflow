from __future__ import annotations

from pathlib import Path

from openmm_mdflow.state import initialize_state, load_state, save_state


def test_state_roundtrip(tmp_path: Path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cfg = {"project": {"name": "x"}, "_meta": {"config_path": "/tmp/workflow.yaml"}}
    initialize_state(output_dir, cfg, {"system_xml": "a.xml", "system_pdb": "a.pdb"})
    doc = load_state(output_dir)
    doc["steps"]["s1"] = {"status": "completed"}
    save_state(output_dir, doc)
    doc2 = load_state(output_dir)
    assert doc2["steps"]["s1"]["status"] == "completed"
