from __future__ import annotations

import os
from typing import Any


def _available_platform_names() -> list[str]:
    import openmm as mm

    return [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]


def select_platform(project_cfg: dict[str, Any]):
    import openmm as mm

    requested = project_cfg.get("platform", "auto")
    names = _available_platform_names()

    if requested == "auto":
        for candidate in ("CUDA", "HIP", "OpenCL", "CPU"):
            if candidate in names:
                requested = candidate
                break
        else:
            requested = names[0]
    elif requested not in names:
        raise RuntimeError(f"Requested OpenMM platform `{requested}` is not available. Available: {names}")

    properties: dict[str, str] = {}
    if requested in {"CUDA", "HIP", "OpenCL"}:
        properties["Precision"] = "mixed"
    elif requested == "CPU":
        properties["Threads"] = os.environ.get("OMP_NUM_THREADS", "1")

    return mm.Platform.getPlatformByName(requested), properties
