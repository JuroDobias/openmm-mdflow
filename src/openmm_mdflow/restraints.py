from __future__ import annotations

from typing import Any


KCAL_MOL_A2_TO_KJ_MOL_NM2 = 418.4
A_TO_NM = 0.1


def add_positional_restraints(system, reference_positions, atom_indices: list[int], cfg: dict[str, Any]):
    import openmm as mm
    from openmm.unit import nanometer

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
    for atom in atom_indices:
        x0, y0, z0 = positions_nm[atom]
        force.addParticle(atom, [float(x0), float(y0), float(z0)])
    system.addForce(force)
    return force


def _group_mass_weights(system, atom_indices: list[int]) -> list[float]:
    from openmm.unit import dalton

    masses_da = [float(system.getParticleMass(atom).value_in_unit(dalton)) for atom in atom_indices]
    if sum(masses_da) <= 0.0:
        raise ValueError(
            "Distance restraint group has zero total mass. "
            f"Selected indices: {atom_indices[:10]}"
        )
    return masses_da


def add_distance_restraints(system, rows: list[dict[str, Any]]):
    import openmm as mm

    # Flat-bottom harmonic potential: no force inside [r0-tol, r0+tol].
    expr = "0.5*k*step(dr)*dr^2; dr=abs(distance(g1,g2)-r0)-tol"
    force = mm.CustomCentroidBondForce(2, expr)
    force.setName("DistanceRestraints")
    force.setUsesPeriodicBoundaryConditions(system.usesPeriodicBoundaryConditions())
    force.addPerBondParameter("r0")
    force.addPerBondParameter("tol")
    force.addPerBondParameter("k")
    for row in rows:
        group1 = row["group1_indices"]
        group2 = row["group2_indices"]
        g1 = force.addGroup(group1, _group_mass_weights(system, group1))
        g2 = force.addGroup(group2, _group_mass_weights(system, group2))
        r0_nm = float(row["r0_a"] * A_TO_NM)
        tol_nm = float(row.get("tolerance_a", 0.0) * A_TO_NM)
        k_kj_mol_nm2 = float(row["k_kcal_mol_a2"] * KCAL_MOL_A2_TO_KJ_MOL_NM2)
        force.addBond([g1, g2], [r0_nm, tol_nm, k_kj_mol_nm2])
    system.addForce(force)
    return force
