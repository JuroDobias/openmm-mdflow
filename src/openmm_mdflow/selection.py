from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SelectionError(ValueError):
    """Raised when Amber mask selection cannot be resolved."""


def _preview(indices: list[int], max_items: int = 10) -> str:
    if not indices:
        return "[]"
    clipped = indices[:max_items]
    suffix = ", ..." if len(indices) > max_items else ""
    return "[" + ", ".join(str(x) for x in clipped) + suffix + "]"


@dataclass
class AmberMaskResolver:
    topology: Any
    positions: Any

    def __post_init__(self):
        try:
            import parmed as pmd
        except ImportError as exc:
            raise ImportError("Amber mask selection requires `parmed` to be installed.") from exc

        self._structure = pmd.openmm.load_topology(self.topology, xyz=self.positions)
        self._n_atoms = len(self._structure.atoms)

    def resolve(self, mask: str, label: str) -> list[int]:
        if not isinstance(mask, str) or not mask.strip():
            raise SelectionError(f"`{label}` must be a non-empty Amber mask string.")
        mask = mask.strip()

        try:
            from parmed.amber.mask import AmberMask

            selected_flags = AmberMask(self._structure, mask).Selection()
        except Exception as exc:
            raise SelectionError(f"{label}: invalid Amber mask {mask!r}: {exc}") from exc

        selected = [idx for idx, flag in enumerate(selected_flags) if bool(flag)]
        if len(selected_flags) != self._n_atoms:
            raise SelectionError(
                f"{label}: Amber mask {mask!r} returned invalid selection length "
                f"{len(selected_flags)} (expected {self._n_atoms})."
            )
        if not selected:
            raise SelectionError(
                f"{label}: Amber mask {mask!r} selected 0 atoms; indices preview {_preview(selected)}."
            )
        return selected
