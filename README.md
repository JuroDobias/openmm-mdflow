# openmm-mdflow

`openmm-mdflow` is a standalone MD workflow tool built on OpenMM 8.4+.

It is designed for YAML-driven workflows with:
- Protein/ligand/cofactor system setup
- Ligand parameterization with `openff`, `gaff`, or `espaloma`
- Step-based minimization and MD
- Positional and distance restraints (atom index based)
- Checkpointing and automatic restart/skip behavior

This tool does not import or depend on `atom_openmm` at runtime.

## Installation

```bash
source ~/Software/pymol/etc/profile.d/conda.sh
conda activate atm8.4.0
cd /home/juro/Software/AToM-OpenMM/tool/openmm-mdflow
pip install -e ".[test]"
```

To use ligand/cofactor parameterization from SDF files:

```bash
pip install -e ".[setup]"
```

## CLI

```bash
openmm-mdflow validate --config workflow.yaml
openmm-mdflow run --config workflow.yaml
```

## Output Layout

`openmm-mdflow` writes deterministic output paths:

- `output_dir/system/system.xml`
- `output_dir/system/system.pdb`
- `output_dir/steps/<NN>_<step_id>/...`
- `output_dir/.mdflow/state.json`

Each completed step writes `done.ok`. Re-running the same workflow:
- Skips steps with `done.ok` + `final_state.xml`
- Resumes interrupted MD steps from `checkpoint.xml` when available

## YAML Spec (v1)

```yaml
project:
  name: my_run
  output_dir: runs/my_run
  platform: CUDA  # CUDA|HIP|OpenCL|CPU|auto

system:
  receptor:
    file: receptor.pdb
  ligands:
    - file: lig1.sdf
      residue_name: L1
  cofactors:
    - file: cof1.sdf
      residue_name: COF
  solvation:
    mode: explicit  # explicit|implicit|vacuum
    ionic_strength_molar: 0.15
    padding_nm: 1.0

forcefield:
  protein: ["amber14-all.xml"]
  water_ions: ["amber14/tip3p.xml"]
  ligand:
    engine: espaloma  # espaloma|openff|gaff
    model: espaloma-0.3.2
    cache: ff.json
  hydrogen_mass_amu: 1.5

steps:
  - id: min1
    type: minimization
    tolerance_kj_mol_nm: 10.0
    max_iterations: 1000
    positional_restraints:
      atoms: [1, 2, 3]
      k_kcal_mol_a2: 25.0
      tolerance_a: 1.5

  - id: npt_eq
    type: md
    ensemble: NPT
    n_steps: 250000
    timestep_ps: 0.004
    thermostat:
      kind: langevin_middle
      temperature_k: 300
      friction_per_ps: 1.0
    barostat:
      pressure_bar: 1.0
      frequency: 25
    distance_restraints:
      - atoms: [10, 220]
        r0_a: 3.0
        k_kcal_mol_a2: 5.0
    reporters:
      traj:
        format: xtc
        interval: 5000
      state:
        interval: 1000
      checkpoint:
        interval: 5000
```

## Notes

- Receptor currently supports `.pdb` and `.sdf`.
- Ligands/cofactors support `.sdf` or `.pdb`.
- SDF-based ligand/cofactor parameterization requires `openmmforcefields` and OpenFF components.
