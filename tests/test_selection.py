from __future__ import annotations

from pathlib import Path

import pytest

openmm = pytest.importorskip("openmm")
from openmm.app import PDBFile

from openmm_mdflow.selection import AmberMaskResolver, SelectionError


def _write_test_pdb(path: Path):
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00  0.00           C",
                "ATOM      4  O   ALA A   1       1.000   2.000   0.000  1.00  0.00           O",
                "ATOM      5  H   ALA A   1      -0.500   0.000   0.000  1.00  0.00           H",
                "ATOM      6  O   HOH B   2       3.000   0.000   0.000  1.00  0.00           O",
                "ATOM      7  H1  HOH B   2       3.100   0.000   0.000  1.00  0.00           H",
                "ATOM      8  H2  HOH B   2       2.900   0.000   0.000  1.00  0.00           H",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_amber_masks(tmp_path: Path):
    pdb_path = tmp_path / "mask_test.pdb"
    _write_test_pdb(pdb_path)
    pdb = PDBFile(str(pdb_path))
    resolver = AmberMaskResolver(pdb.topology, pdb.positions)

    assert resolver.resolve(":1", "x") == [0, 1, 2, 3, 4]
    assert resolver.resolve("@CA", "x") == [1]
    assert resolver.resolve(":ALA&!@H=", "x") == [0, 1, 2, 3]


def test_invalid_or_empty_mask_errors(tmp_path: Path):
    pdb_path = tmp_path / "mask_test.pdb"
    _write_test_pdb(pdb_path)
    pdb = PDBFile(str(pdb_path))
    resolver = AmberMaskResolver(pdb.topology, pdb.positions)

    with pytest.raises(SelectionError, match="invalid Amber mask"):
        resolver.resolve("::A", "steps[0].positional_restraints.mask")
    with pytest.raises(SelectionError, match="selected 0 atoms"):
        resolver.resolve(":LIG", "steps[0].positional_restraints.mask")
