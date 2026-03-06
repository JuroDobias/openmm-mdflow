from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / ".mdflow"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_file(output_dir: str | Path) -> Path:
    return state_dir(output_dir) / "state.json"


def load_state(output_dir: str | Path) -> dict[str, Any]:
    path = state_file(output_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(output_dir: str | Path, data: dict[str, Any]):
    path = state_file(output_dir)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def initialize_state(output_dir: str | Path, config: dict[str, Any], system_paths: dict[str, str]):
    state = load_state(output_dir)
    if state:
        return state
    state = {
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "project": config["project"]["name"],
        "config_path": config["_meta"]["config_path"],
        "system": system_paths,
        "steps": {},
    }
    save_state(output_dir, state)
    return state
