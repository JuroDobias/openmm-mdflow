from __future__ import annotations

import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit

from openmm_mdflow.restraints import add_distance_restraints, add_positional_restraints, validate_atom_indices


def test_validate_atom_indices():
    validate_atom_indices([0, 2], 3, "x")
    with pytest.raises(ValueError):
        validate_atom_indices([3], 3, "x")


def test_add_restraints():
    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0)
    positions = unit.Quantity([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], unit.nanometer)

    add_positional_restraints(
        system,
        positions,
        {"atoms": [0, 1], "k_kcal_mol_a2": 10.0, "tolerance_a": 1.0},
    )
    add_distance_restraints(
        system,
        [{"atoms": [0, 2], "r0_a": 2.0, "k_kcal_mol_a2": 5.0}],
    )
    assert system.getNumForces() == 2
