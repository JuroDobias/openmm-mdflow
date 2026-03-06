from __future__ import annotations

from typing import Any


KCAL_MOL_A2_TO_KJ_MOL_NM2 = 418.4
A_TO_NM = 0.1


def validate_atom_indices(atom_indices: list[int], n_atoms: int, label: str):
    for atom in atom_indices:
        if atom < 0 or atom >= n_atoms:
            raise ValueError(f"{label}: atom index {atom} out of range [0, {n_atoms - 1}]")


def add_positional_restraints(system, reference_positions, cfg: dict[str, Any]):
    import openmm as mm
    from openmm.unit import nanometer

    atoms = cfg["atoms"]
    k = cfg["k_kcal_mol_a2"] * KCAL_MOL_A2_TO_KJ_MOL_NM2
    tol = cfg["tolerance_a"] * A_TO_NM

    # Use periodicdistance() so restrained atoms in periodic systems are compared
    # with minimum-image convention instead of raw Cartesian deltas.
    expr = "0.5*k*step(dr-tol)*(dr-tol)^2; dr=periodicdistance(x,y,z,x0,y0,z0)"
    force = mm.CustomExternalForce(expr)
    force.setName("PositionalRestraints")
    force.addGlobalParameter("k", float(k))
    force.addGlobalParameter("tol", float(tol))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    positions_nm = reference_positions.value_in_unit(nanometer)
    for atom in atoms:
        x0, y0, z0 = positions_nm[atom]
        force.addParticle(atom, [float(x0), float(y0), float(z0)])
    system.addForce(force)
    return force


def add_distance_restraints(system, rows: list[dict[str, Any]]):
    import openmm as mm

    expr = "0.5*k*(r-r0)^2"
    force = mm.CustomBondForce(expr)
    force.setName("DistanceRestraints")
    force.addPerBondParameter("r0")
    force.addPerBondParameter("k")
    for row in rows:
        a1, a2 = row["atoms"]
        r0_nm = float(row["r0_a"] * A_TO_NM)
        k_kj_mol_nm2 = float(row["k_kcal_mol_a2"] * KCAL_MOL_A2_TO_KJ_MOL_NM2)
        force.addBond(a1, a2, [r0_nm, k_kj_mol_nm2])
    system.addForce(force)
    return force
