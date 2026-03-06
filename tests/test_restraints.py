from __future__ import annotations

import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit

from openmm_mdflow.restraints import add_distance_restraints, add_positional_restraints


def test_add_positional_and_group_distance_restraints():
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)

    positions = unit.Quantity(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.2, 0.2, 0.0],
        ],
        unit.nanometer,
    )

    pos_force = add_positional_restraints(
        system,
        positions,
        [0, 1],
        {"k_kcal_mol_a2": 10.0, "tolerance_a": 1.0},
    )
    dist_force = add_distance_restraints(
        system,
        [
            {
                "group1_indices": [0, 1],
                "group2_indices": [2, 3],
                "r0_a": 2.0,
                "tolerance_a": 0.2,
                "k_kcal_mol_a2": 5.0,
            }
        ],
    )
    assert system.getNumForces() == 2
    assert pos_force.getNumParticles() == 2
    assert dist_force.getNumGroups() == 2
    assert dist_force.getNumBonds() == 1
    assert dist_force.getPerBondParameterName(1) == "tol"
    assert "step(dr)" in dist_force.getEnergyFunction()
