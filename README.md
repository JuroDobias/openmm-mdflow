# openmm-mdflow

`openmm-mdflow` is a standalone MD workflow tool built on OpenMM 8.4+.

It is designed for YAML-driven workflows with:
- Protein/ligand/cofactor system setup
- Ligand parameterization with `openff`, `gaff`, or `espaloma`
- Step-based minimization and MD
- Positional and distance restraints via Amber/ParmEd mask expressions
- Checkpointing and automatic restart/skip behavior

This tool does not import or depend on `atom_openmm` at runtime.

## Installation

Assuming conda is already initialized and available in your shell:

```bash
cd tool/openmm-mdflow
conda env create -f environment.yml
# if the env already exists:
conda env update -f environment.yml --prune
conda activate openmm-mdflow
pip install -e .
openmm-mdflow --help
```

`environment.yml` pins Python 3.12 and includes `espaloma`, so `forcefield.ligand.engine: espaloma` works in a fresh environment.

Optional editable install with extras (alternative to `environment.yml` dependency set):

```bash
pip install -e ".[setup,test]"
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
- `output_dir/steps/<step_id>/...`
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
      mask: ":1-3 & !@H="
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
      - group1_mask: ":L1 & !@H="
        group2_mask: ":45,67,89@CA,C,N,O"
        r0_a: 3.0
        tolerance_a: 0.2
        k_kcal_mol_a2: 5.0
    reporters:
      traj:
        format: xtc
        interval: 5000
      state:
        interval: 1000
      checkpoint:
        interval: 5000

  - id: trajmin_post
    type: trajectory_minimization
    tolerance_kj_mol_nm: 10.0
    max_iterations: 500
    # optional, defaults to previous step trajectory
    input:
      trajectory: runs/my_run/steps/npt_eq/trajectory.xtc
    parallel:
      workers: 2
```

## Notes

- Receptor currently supports `.pdb` and `.sdf`.
- Ligands/cofactors support `.sdf` or `.pdb`.
- SDF-based ligand/cofactor parameterization requires `openmmforcefields` and OpenFF components.
- Restraints now require Amber mask fields (`mask`, `group1_mask`, `group2_mask`); old index-based `atoms` keys are not supported.
- Distance restraints are flat-bottom harmonic around `r0_a` with optional `tolerance_a` (default `0.0` A).
- `trajectory_minimization` minimizes each input trajectory frame and writes a new trajectory in the current step directory.
