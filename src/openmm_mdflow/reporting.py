from __future__ import annotations

from pathlib import Path
from typing import Any


def build_reporters(step_cfg: dict[str, Any], step_dir: Path):
    from openmm.app import DCDReporter, StateDataReporter

    reporters = []
    reporters_cfg = step_cfg.get("reporters", {})
    if "state" in reporters_cfg:
        state_interval = int(reporters_cfg["state"]["interval"])
        reporters.append(
            StateDataReporter(
                str(step_dir / "state.csv"),
                state_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                density=True,
                speed=True,
                separator=",",
                append=(step_dir / "state.csv").exists(),
            )
        )
    if "traj" in reporters_cfg:
        traj_cfg = reporters_cfg["traj"]
        traj_interval = int(traj_cfg["interval"])
        traj_format = traj_cfg["format"]
        if traj_format == "dcd":
            dcd_path = step_dir / "trajectory.dcd"
            reporters.append(DCDReporter(str(dcd_path), traj_interval, append=dcd_path.exists()))
        else:
            xtc_path = step_dir / "trajectory.xtc"
            try:
                from openmm.app import XTCReporter

                reporters.append(XTCReporter(str(xtc_path), traj_interval, append=xtc_path.exists()))
            except TypeError:
                from openmm.app import XTCReporter

                reporters.append(XTCReporter(str(xtc_path), traj_interval))
    return reporters


def checkpoint_interval(step_cfg: dict[str, Any]) -> int | None:
    reporters_cfg = step_cfg.get("reporters", {})
    if "checkpoint" not in reporters_cfg:
        return None
    return int(reporters_cfg["checkpoint"]["interval"])
