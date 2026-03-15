from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when workflow configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Configuration must be a mapping at top level.")
    return data


def load_and_validate(path: str | Path) -> dict[str, Any]:
    raw = load_config(path)
    validated = validate_config(raw)
    validated["_meta"] = {"config_path": str(Path(path).resolve())}
    return validated


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"`{label}` must be a mapping.")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"`{label}` must be a list.")
    return value


def _as_non_negative_int(value: Any, label: str, allow_zero: bool = True) -> int:
    if not isinstance(value, int):
        raise ConfigError(f"`{label}` must be an integer.")
    if value < 0 or (value == 0 and not allow_zero):
        msg = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"`{label}` must be {msg}.")
    return value


def _as_positive_float(value: Any, label: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"`{label}` must be a number.")
    value = float(value)
    if value < 0 or (value == 0 and not allow_zero):
        msg = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"`{label}` must be {msg}.")
    return value


def _as_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"`{label}` must be a non-empty string.")
    return value.strip()


def _validate_positional_restraints(value: Any, label: str) -> dict[str, Any]:
    data = _require_mapping(value, label)
    if "atoms" in data:
        raise ConfigError(
            f"`{label}.atoms` is no longer supported. "
            f"Use `{label}.mask` with an Amber/ParmEd mask expression."
        )
    return {
        "mask": _as_non_empty_string(data.get("mask"), f"{label}.mask"),
        "k_kcal_mol_a2": _as_positive_float(data.get("k_kcal_mol_a2"), f"{label}.k_kcal_mol_a2"),
        "tolerance_a": _as_positive_float(data.get("tolerance_a", 0.0), f"{label}.tolerance_a", allow_zero=True),
    }


def _validate_distance_restraints(value: Any, label: str) -> list[dict[str, Any]]:
    rows = _require_list(value, label)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = _require_mapping(row, f"{label}[{i}]")
        if "atoms" in item:
            raise ConfigError(
                f"`{label}[{i}].atoms` is no longer supported. "
                f"Use `{label}[{i}].group1_mask` and `{label}[{i}].group2_mask`."
            )
        out.append(
            {
                "group1_mask": _as_non_empty_string(item.get("group1_mask"), f"{label}[{i}].group1_mask"),
                "group2_mask": _as_non_empty_string(item.get("group2_mask"), f"{label}[{i}].group2_mask"),
                "r0_a": _as_positive_float(item.get("r0_a"), f"{label}[{i}].r0_a"),
                "tolerance_a": _as_positive_float(
                    item.get("tolerance_a", 0.0), f"{label}[{i}].tolerance_a", allow_zero=True
                ),
                "k_kcal_mol_a2": _as_positive_float(item.get("k_kcal_mol_a2"), f"{label}[{i}].k_kcal_mol_a2"),
            }
        )
    return out


def _validate_step(step: Any, index: int) -> dict[str, Any]:
    label = f"steps[{index}]"
    data = _require_mapping(step, label)
    step_id = data.get("id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise ConfigError(f"`{label}.id` must be a non-empty string.")
    step_type = data.get("type")
    if step_type not in {"minimization", "md", "trajectory_minimization"}:
        raise ConfigError(f"`{label}.type` must be `minimization`, `md`, or `trajectory_minimization`.")

    cleaned: dict[str, Any] = {"id": step_id.strip(), "type": step_type}
    has_new_reference = "restraint_reference" in data
    has_legacy_reference = "restraint_reference_pdb" in data
    if has_new_reference and has_legacy_reference:
        raise ConfigError(
            f"`{label}.restraint_reference` and `{label}.restraint_reference_pdb` cannot be used together. "
            "Use only `restraint_reference`."
        )
    if has_new_reference:
        cleaned["restraint_reference"] = _as_non_empty_string(
            data.get("restraint_reference"), f"{label}.restraint_reference"
        )
    elif has_legacy_reference:
        cleaned["restraint_reference"] = _as_non_empty_string(
            data.get("restraint_reference_pdb"), f"{label}.restraint_reference_pdb"
        )

    if "positional_restraints" in data:
        cleaned["positional_restraints"] = _validate_positional_restraints(
            data["positional_restraints"], f"{label}.positional_restraints"
        )
    if "distance_restraints" in data:
        cleaned["distance_restraints"] = _validate_distance_restraints(
            data["distance_restraints"], f"{label}.distance_restraints"
        )

    if step_type in {"minimization", "trajectory_minimization"}:
        cleaned["tolerance_kj_mol_nm"] = _as_positive_float(
            data.get("tolerance_kj_mol_nm", 10.0), f"{label}.tolerance_kj_mol_nm"
        )
        cleaned["max_iterations"] = _as_non_negative_int(
            data.get("max_iterations", 0), f"{label}.max_iterations", allow_zero=True
        )
        cleaned["integrator"] = data.get("integrator", "langevin_middle")
        cleaned["timestep_ps"] = _as_positive_float(data.get("timestep_ps", 0.002), f"{label}.timestep_ps")

        if step_type == "trajectory_minimization":
            forbidden_keys = ("ensemble", "n_steps", "thermostat", "barostat", "reporters")
            for key in forbidden_keys:
                if key in data:
                    raise ConfigError(f"`{label}.{key}` is not valid for `trajectory_minimization`.")

            input_cfg = _require_mapping(data.get("input", {}), f"{label}.input")
            cleaned_input: dict[str, Any] = {}
            if "trajectory" in input_cfg:
                cleaned_input["trajectory"] = _as_non_empty_string(input_cfg["trajectory"], f"{label}.input.trajectory")
            cleaned["input"] = cleaned_input

            parallel_cfg = _require_mapping(data.get("parallel", {}), f"{label}.parallel")
            cleaned["parallel"] = {
                "workers": _as_non_negative_int(
                    parallel_cfg.get("workers", 1), f"{label}.parallel.workers", allow_zero=False
                )
            }
    else:
        ensemble = data.get("ensemble")
        if ensemble not in {"NVT", "NPT"}:
            raise ConfigError(f"`{label}.ensemble` must be `NVT` or `NPT`.")
        cleaned["ensemble"] = ensemble
        cleaned["n_steps"] = _as_non_negative_int(data.get("n_steps"), f"{label}.n_steps", allow_zero=False)
        cleaned["timestep_ps"] = _as_positive_float(data.get("timestep_ps"), f"{label}.timestep_ps")
        cleaned["integrator"] = data.get("integrator", "langevin_middle")

        thermostat = _require_mapping(data.get("thermostat"), f"{label}.thermostat")
        kind = thermostat.get("kind")
        if kind != "langevin_middle":
            raise ConfigError(f"`{label}.thermostat.kind` must be `langevin_middle` in v1.")
        cleaned["thermostat"] = {
            "kind": kind,
            "temperature_k": _as_positive_float(thermostat.get("temperature_k"), f"{label}.thermostat.temperature_k"),
            "friction_per_ps": _as_positive_float(
                thermostat.get("friction_per_ps"), f"{label}.thermostat.friction_per_ps"
            ),
        }

        if ensemble == "NPT":
            barostat = _require_mapping(data.get("barostat"), f"{label}.barostat")
            cleaned["barostat"] = {
                "pressure_bar": _as_positive_float(barostat.get("pressure_bar"), f"{label}.barostat.pressure_bar"),
                "frequency": _as_non_negative_int(barostat.get("frequency"), f"{label}.barostat.frequency", allow_zero=False),
            }
        elif "barostat" in data:
            raise ConfigError(f"`{label}.barostat` is only valid for NPT.")

        reporters = _require_mapping(data.get("reporters", {}), f"{label}.reporters")
        cleaned_reporters: dict[str, Any] = {}
        if "traj" in reporters:
            traj = _require_mapping(reporters["traj"], f"{label}.reporters.traj")
            traj_format = traj.get("format", "xtc")
            if traj_format not in {"xtc", "dcd"}:
                raise ConfigError(f"`{label}.reporters.traj.format` must be `xtc` or `dcd`.")
            cleaned_reporters["traj"] = {
                "format": traj_format,
                "interval": _as_non_negative_int(
                    traj.get("interval"), f"{label}.reporters.traj.interval", allow_zero=False
                ),
            }
        if "state" in reporters:
            state = _require_mapping(reporters["state"], f"{label}.reporters.state")
            cleaned_reporters["state"] = {
                "interval": _as_non_negative_int(
                    state.get("interval"), f"{label}.reporters.state.interval", allow_zero=False
                )
            }
        if "checkpoint" in reporters:
            checkpoint = _require_mapping(reporters["checkpoint"], f"{label}.reporters.checkpoint")
            cleaned_reporters["checkpoint"] = {
                "interval": _as_non_negative_int(
                    checkpoint.get("interval"), f"{label}.reporters.checkpoint.interval", allow_zero=False
                )
            }
        cleaned["reporters"] = cleaned_reporters

    return cleaned


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(config)

    project = _require_mapping(data.get("project"), "project")
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("`project.name` must be a non-empty string.")
    platform = project.get("platform", "auto")
    if platform not in {"CUDA", "HIP", "OpenCL", "CPU", "auto"}:
        raise ConfigError("`project.platform` must be CUDA|HIP|OpenCL|CPU|auto.")
    output_dir = project.get("output_dir", f"runs/{name.strip()}")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("`project.output_dir` must be a non-empty string.")
    cleaned_project = {"name": name.strip(), "output_dir": output_dir.strip(), "platform": platform}

    system = _require_mapping(data.get("system"), "system")
    receptor = _require_mapping(system.get("receptor"), "system.receptor")
    receptor_file = receptor.get("file")
    if not isinstance(receptor_file, str) or not receptor_file.strip():
        raise ConfigError("`system.receptor.file` must be a non-empty string.")

    ligands = _require_list(system.get("ligands", []), "system.ligands")
    cofactors = _require_list(system.get("cofactors", []), "system.cofactors")

    def _validate_components(rows: list[Any], label: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for i, row in enumerate(rows):
            item = _require_mapping(row, f"{label}[{i}]")
            file_path = item.get("file")
            if not isinstance(file_path, str) or not file_path.strip():
                raise ConfigError(f"`{label}[{i}].file` must be a non-empty string.")
            residue_name = item.get("residue_name")
            if residue_name is not None and (not isinstance(residue_name, str) or not residue_name.strip()):
                raise ConfigError(f"`{label}[{i}].residue_name` must be a non-empty string when provided.")
            out.append({"file": file_path.strip(), "residue_name": residue_name.strip() if residue_name else ""})
        return out

    solvation = _require_mapping(system.get("solvation", {}), "system.solvation")
    mode = solvation.get("mode", "explicit")
    if mode not in {"explicit", "implicit", "vacuum"}:
        raise ConfigError("`system.solvation.mode` must be explicit|implicit|vacuum.")
    cleaned_solvation: dict[str, Any] = {"mode": mode}
    if mode == "explicit":
        cleaned_solvation["ionic_strength_molar"] = _as_positive_float(
            solvation.get("ionic_strength_molar", 0.15), "system.solvation.ionic_strength_molar", allow_zero=True
        )
        cleaned_solvation["padding_nm"] = _as_positive_float(
            solvation.get("padding_nm", 1.0), "system.solvation.padding_nm"
        )
    elif mode == "implicit":
        implicit_model = solvation.get("implicit_model", "OBC2")
        if implicit_model not in {"OBC2", "GBN2", "HCT"}:
            raise ConfigError("`system.solvation.implicit_model` must be OBC2|GBN2|HCT.")
        cleaned_solvation["implicit_model"] = implicit_model

    cleaned_system = {
        "receptor": {"file": receptor_file.strip()},
        "ligands": _validate_components(ligands, "system.ligands"),
        "cofactors": _validate_components(cofactors, "system.cofactors"),
        "solvation": cleaned_solvation,
    }

    forcefield = _require_mapping(data.get("forcefield"), "forcefield")
    protein = _require_list(forcefield.get("protein", ["amber14-all.xml"]), "forcefield.protein")
    water_ions = _require_list(forcefield.get("water_ions", ["amber14/tip3p.xml"]), "forcefield.water_ions")
    ligand = _require_mapping(forcefield.get("ligand", {}), "forcefield.ligand")
    engine = ligand.get("engine", "openff")
    if engine not in {"openff", "gaff", "espaloma"}:
        raise ConfigError("`forcefield.ligand.engine` must be openff|gaff|espaloma.")
    model_default = {"openff": "openff-2.0.0", "gaff": "gaff-2.11", "espaloma": "espaloma-0.3.2"}[engine]
    cleaned_forcefield = {
        "protein": [str(x) for x in protein],
        "water_ions": [str(x) for x in water_ions],
        "ligand": {
            "engine": engine,
            "model": str(ligand.get("model", model_default)),
            "cache": str(ligand.get("cache", "ff.json")),
        },
        "hydrogen_mass_amu": _as_positive_float(
            forcefield.get("hydrogen_mass_amu", 1.5), "forcefield.hydrogen_mass_amu"
        ),
        "nonbonded_cutoff_nm": _as_positive_float(
            forcefield.get("nonbonded_cutoff_nm", 0.9), "forcefield.nonbonded_cutoff_nm"
        ),
    }

    steps_raw = _require_list(data.get("steps"), "steps")
    if not steps_raw:
        raise ConfigError("`steps` must not be empty.")
    steps: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, step_raw in enumerate(steps_raw):
        step = _validate_step(step_raw, i)
        if step["id"] in seen_ids:
            raise ConfigError(f"Duplicate step id `{step['id']}`.")
        seen_ids.add(step["id"])
        steps.append(step)

    return {
        "project": cleaned_project,
        "system": cleaned_system,
        "forcefield": cleaned_forcefield,
        "steps": steps,
    }
