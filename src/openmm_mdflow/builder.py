from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BuildArtifacts:
    system_path: Path
    pdb_path: Path
    topology: Any
    positions: Any
    system: Any


def _set_residue_and_chain(topology, residue_name: str | None, chain_id: str):
    if residue_name:
        for residue in topology.residues():
            residue.name = residue_name
    for chain in topology.chains():
        chain.id = chain_id


def _to_openmm_component(file_path: Path):
    from openmm import Vec3
    from openmm.app import PDBFile
    from openmm.unit import angstrom

    ext = file_path.suffix.lower()
    if ext == ".pdb":
        pdb = PDBFile(str(file_path))
        return pdb.topology, pdb.positions, None
    if ext == ".sdf":
        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_file(str(file_path), file_format="SDF", allow_undefined_stereo=True)
        topology = molecule.to_topology().to_openmm(ensure_unique_atom_names=True)
        positions_np = molecule.conformers[0].to("angstrom").magnitude
        positions = [Vec3(float(row[0]), float(row[1]), float(row[2])) for row in positions_np] * angstrom
        return topology, positions, molecule
    raise ValueError(f"Unsupported file type for `{file_path}`. Use .pdb or .sdf.")


def _build_template_generator(engine: str, model: str, molecules: list[Any], cache_file: str):
    if engine == "gaff":
        from openmmforcefields.generators import GAFFTemplateGenerator

        return GAFFTemplateGenerator(molecules=molecules, forcefield=model, cache=cache_file)
    if engine == "openff":
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        return SMIRNOFFTemplateGenerator(molecules=molecules, forcefield=model, cache=cache_file)
    if engine == "espaloma":
        from openmmforcefields.generators import EspalomaTemplateGenerator

        return EspalomaTemplateGenerator(molecules=molecules, forcefield=model, cache=cache_file)
    raise ValueError(f"Unsupported ligand engine `{engine}`")


def _infer_water_model(water_ion_files: list[str]) -> str | None:
    joined = " ".join(x.lower() for x in water_ion_files)
    mapping = (
        # addSolvent() does not have an `opc` keyword; tip4pew geometry is the
        # closest 4-site model for placing waters before applying OPC templates.
        ("opc3", "tip3p"),
        ("opc", "tip4pew"),
        ("tip5p", "tip5p"),
        ("tip4pew", "tip4pew"),
        ("tip4p", "tip4pew"),
        ("spce", "spce"),
        ("tip3p", "tip3p"),
    )
    for key, model in mapping:
        if key in joined:
            return model
    return None


def build_system(config: dict[str, Any], output_dir: str | Path) -> BuildArtifacts:
    from openmm import XmlSerializer
    from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, PME, HBonds
    from openmm.unit import amu, molar, nanometer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    system_path = output_path / "system.xml"
    pdb_path = output_path / "system.pdb"

    system_cfg = config["system"]
    ff_cfg = config["forcefield"]
    solvation_cfg = system_cfg["solvation"]

    receptor_file = Path(system_cfg["receptor"]["file"])
    receptor_topology, receptor_positions, receptor_molecule = _to_openmm_component(receptor_file)
    _set_residue_and_chain(receptor_topology, None, "A")
    modeller = Modeller(receptor_topology, receptor_positions)

    small_molecules: list[Any] = []
    if receptor_molecule is not None:
        small_molecules.append(receptor_molecule)

    ligand_chain_ord = ord("L")
    for ligand in system_cfg["ligands"]:
        topology, positions, molecule = _to_openmm_component(Path(ligand["file"]))
        _set_residue_and_chain(topology, ligand.get("residue_name") or "LIG", chr(ligand_chain_ord))
        ligand_chain_ord += 1
        modeller.add(topology, positions)
        if molecule is not None:
            small_molecules.append(molecule)

    cofactor_chain_ord = ord("C")
    for cofactor in system_cfg["cofactors"]:
        topology, positions, molecule = _to_openmm_component(Path(cofactor["file"]))
        _set_residue_and_chain(topology, cofactor.get("residue_name") or "COF", chr(cofactor_chain_ord))
        cofactor_chain_ord += 1
        modeller.add(topology, positions)
        if molecule is not None:
            small_molecules.append(molecule)

    forcefield = ForceField(*ff_cfg["protein"], *ff_cfg["water_ions"])

    if small_molecules:
        ligand_cfg = ff_cfg["ligand"]
        try:
            generator = _build_template_generator(
                ligand_cfg["engine"],
                ligand_cfg["model"],
                small_molecules,
                ligand_cfg["cache"],
            )
        except ImportError as exc:
            raise ImportError(
                "Small-molecule parameterization requires openmmforcefields and OpenFF components. "
                "Install with `pip install -e '.[setup]'`."
            ) from exc
        forcefield.registerTemplateGenerator(generator.generator)

    mode = solvation_cfg["mode"]
    hydrogen_mass_amu = ff_cfg["hydrogen_mass_amu"]
    cutoff_nm = ff_cfg["nonbonded_cutoff_nm"]
    if mode == "explicit":
        water_model = _infer_water_model(ff_cfg["water_ions"])
        # Required for models with virtual sites (for example OPC) when the input
        # structure already contains crystallographic waters.
        modeller.addExtraParticles(forcefield)
        kwargs: dict[str, Any] = {
            "padding": solvation_cfg["padding_nm"] * nanometer,
            "ionicStrength": solvation_cfg["ionic_strength_molar"] * molar,
        }
        if water_model is not None:
            kwargs["model"] = water_model
        modeller.addSolvent(forcefield, **kwargs)
        # Keep water topology consistent with the selected model after solvation.
        modeller.addExtraParticles(forcefield)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff_nm * nanometer,
            constraints=HBonds,
            rigidWater=True,
            removeCMMotion=False,
            hydrogenMass=hydrogen_mass_amu * amu,
        )
    elif mode == "implicit":
        model = solvation_cfg["implicit_model"]
        model_files = {"OBC2": "implicit/obc2.xml", "GBN2": "implicit/gbn2.xml", "HCT": "implicit/hct.xml"}
        forcefield.loadFile(model_files[model])
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
            rigidWater=True,
            removeCMMotion=False,
            hydrogenMass=hydrogen_mass_amu * amu,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
            rigidWater=True,
            removeCMMotion=False,
            hydrogenMass=hydrogen_mass_amu * amu,
        )

    with system_path.open("w", encoding="utf-8") as handle:
        handle.write(XmlSerializer.serialize(system))
    with pdb_path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    return BuildArtifacts(
        system_path=system_path,
        pdb_path=pdb_path,
        topology=modeller.topology,
        positions=modeller.positions,
        system=system,
    )
