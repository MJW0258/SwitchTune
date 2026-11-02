#!/usr/bin/env python3
"""Run locally deployed Boltz-2 predictions from a SwitchTune manifest.

The local Boltz-2 CLI writes several output files per prediction rather than
returning one API response. This runner validates those files and normalizes
them into the response/summary shape consumed by the existing SwitchTune
collectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True, allow_nan=False)


def read_log_tail(path: Path, max_lines: int = 80, max_chars: int = 12000) -> str:
    """Return a bounded tail so CLI failures remain visible in the summary CSV."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Unable to read Boltz log: {exc}"
    return "\n".join(lines[-max_lines:])[-max_chars:]


def _write_local_fallback_yaml(source: Path, destination: Path, mode: str) -> None:
    """Write a narrow diagnostic fallback input for a failed Boltz invocation.

    A ligand/template parse error used to make every candidate look like a
    generic ``failed_missing_structure``.  Keeping the fallback inputs in the
    run directory lets us distinguish a bad template from a bad ligand while
    still allowing a run to continue when the model can predict the protein.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PyYAML is a runtime dependency
        raise RuntimeError("PyYAML is required to create Boltz fallback inputs.") from exc
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Boltz input is not a mapping: {source}")
    sequences = data.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError(f"Boltz input has no sequences list: {source}")
    if mode == "no_template":
        data.pop("templates", None)
    elif mode == "no_ligands_keep_template":
        data["sequences"] = [
            item for item in sequences
            if isinstance(item, dict) and "protein" in item
        ]
        data.pop("constraints", None)
    elif mode == "no_ligands":
        data["sequences"] = [
            item for item in sequences
            if isinstance(item, dict) and "protein" in item
        ]
        data.pop("constraints", None)
        data.pop("templates", None)
    else:
        raise ValueError(f"Unsupported Boltz fallback mode: {mode}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _is_oom_log(log_tail: str) -> bool:
    """Return True for CUDA/PyTorch OOM messages emitted by Boltz-2.

    Boltz-2 commonly reports an OOM as ``ran out of memory, skipping batch``
    and still exits with code zero.  Matching that text is therefore required;
    checking only the process return code misses the failure.
    """
    text = str(log_tail or "").lower()
    return any(
        token in text
        for token in (
            "outofmemoryerror",
            "cuda out of memory",
            "ran out of memory",
            "cudnn_status_alloc_failed",
            "cuda error: out of memory",
        )
    )


def _failure_class(log_tail: str, return_code: int, status: str) -> str:
    """Classify common CLI failures without hiding the original log.

    Check GPU OOM before ligand/template keywords.  A fallback filename such as
    ``fallback_no_ligands.yaml`` contains the word ``ligands`` and previously
    caused an OOM to be mislabeled as a ligand parse error.
    """
    text = str(log_tail or "").lower()
    if _is_oom_log(text):
        return "gpu_oom"
    # A successful Boltz-2 process commonly logs informational CUDA/GPU
    # lines (for example, "GPU available: True").  Those lines are not a
    # runtime failure.  Classify the remaining error patterns only when the
    # process or output status is actually unsuccessful.
    if int(return_code or 0) == 0 and str(status or "").startswith("completed"):
        return ""
    if any(token in text for token in ("unknown option", "no such option", "invalid value")):
        return "cli_argument_error"
    if any(token in text for token in ("parse_pdb", "parse_mmcif", "template parse", "template.*error")):
        return "template_parse_error"
    if any(token in text for token in ("ccd", "smiles", "rdkit", "molecule", "ligand parse")):
        return "ligand_parse_error"
    if any(token in text for token in ("cuda", "cudnn", "gpu")):
        return "gpu_runtime_error"
    if any(token in text for token in ("permission denied", "no space left", "disk")):
        return "filesystem_error"
    if return_code:
        return "cli_failed"
    if status != "completed":
        return "missing_local_output"
    return ""


def nested_shape(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value)] + nested_shape(value[0])
    return []


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_scalar(value: Any) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        return _as_scalar(value[0])
    return _finite_number(value)


def _as_json_number(value: Any) -> int | float | None:
    number = _finite_number(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return number


def _iter_npz_keys(data: Any) -> Iterable[str]:
    try:
        return list(data.files)
    except Exception:
        return []


def load_npz_array(path: Path, preferred_key: str) -> Any:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment prerequisite
        raise RuntimeError("The local Boltz-2 runner requires numpy.") from exc

    with np.load(path, allow_pickle=False) as data:
        keys = list(_iter_npz_keys(data))
        if not keys:
            raise ValueError(f"NPZ archive contains no arrays: {path}")
        key = preferred_key if preferred_key in keys else keys[0]
        array = data[key]
        while getattr(array, "ndim", 0) > 2 and array.shape[0] == 1:
            array = array[0]
        if not getattr(array, "size", 0):
            raise ValueError(f"NPZ array is empty: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"NPZ array contains NaN/Inf values: {path}")
        return array.tolist()


def pae_matrix_status(value: Any) -> tuple[bool, list[int]]:
    shape = nested_shape(value)
    matrix = value
    if (
        isinstance(matrix, list)
        and len(matrix) == 1
        and isinstance(matrix[0], list)
        and matrix[0]
        and isinstance(matrix[0][0], list)
    ):
        matrix = matrix[0]
    if not isinstance(matrix, list) or not matrix:
        return False, shape
    size = len(matrix)
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            return False, shape
        for item in row:
            number = _finite_number(item)
            if number is None:
                return False, shape
    return True, shape


def _find_first(root: Path, patterns: list[str], variant: str = "") -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.rglob(pattern) if path.is_file())
    if not matches:
        return None
    variant_lower = variant.lower()
    matches.sort(
        key=lambda path: (
            0 if variant_lower and variant_lower in path.name.lower() else 1,
            len(path.parts),
            str(path),
        )
    )
    return matches[0]


def find_local_prediction_files(root: Path, variant: str) -> dict[str, Path | None]:
    return {
        "structure": _find_first(root, [f"{variant}_model_*.cif", f"{variant}_model_*.mmcif", f"{variant}_model_*.pdb", "*.cif", "*.mmcif", "*.pdb"], variant),
        "confidence": _find_first(root, [f"confidence_{variant}_model_*.json", "confidence_*_model_*.json", "confidence.json"], variant),
        "pae": _find_first(root, [f"pae_{variant}_model_*.npz", "pae_*_model_*.npz", "pae.npz"], variant),
        "pde": _find_first(root, [f"pde_{variant}_model_*.npz", "pde_*_model_*.npz", "pde.npz"], variant),
        "plddt": _find_first(root, [f"plddt_{variant}_model_*.npz", "plddt_*_model_*.npz", "plddt.npz"], variant),
    }


def _confidence_value(confidence: dict[str, Any], key: str) -> float | None:
    return _as_scalar(confidence.get(key))


def normalize_local_prediction(
    *,
    variant: str,
    output_root: Path,
    response_path: Path,
    elapsed_seconds: float,
    require_pae: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    files = find_local_prediction_files(output_root, variant)
    structure_path = files["structure"]
    confidence_path = files["confidence"]
    pae_path = files["pae"]
    pde_path = files["pde"]
    plddt_path = files["plddt"]

    pae: Any = None
    pde: Any = None
    plddt: Any = None
    errors: list[str] = []
    warnings: list[str] = []

    if pae_path is not None:
        try:
            pae = load_npz_array(pae_path, "pae")
        except Exception as exc:
            errors.append(f"PAE load failed: {exc}")
    else:
        errors.append("PAE NPZ output was not found.")

    pae_usable, pae_shape = pae_matrix_status(pae)
    if structure_path is None:
        errors.append("Boltz-2 structure output was not found.")
    if pde_path is not None:
        try:
            pde = load_npz_array(pde_path, "pde")
        except Exception as exc:
            warnings.append(f"PDE load failed: {exc}")
    else:
        warnings.append("PDE NPZ output was not found.")
    if plddt_path is not None:
        try:
            plddt = load_npz_array(plddt_path, "plddt")
        except Exception as exc:
            warnings.append(f"pLDDT load failed: {exc}")
    else:
        warnings.append("pLDDT NPZ output was not found; CIF B-factors may still provide pLDDT.")

    confidence: dict[str, Any] = {}
    if confidence_path is not None:
        try:
            confidence = read_json(confidence_path)
        except Exception as exc:
            warnings.append(f"Confidence JSON load failed: {exc}")
    else:
        warnings.append("Confidence JSON output was not found.")

    status = "completed"
    if structure_path is None:
        status = "failed_missing_structure"
    elif require_pae and not pae_usable:
        status = "failed_missing_required_pae"
    elif errors:
        status = "failed_local_output"

    response: dict[str, Any] = {
        "backend": "boltz2_local",
        "variant": variant,
        "confidence_scores": [_confidence_value(confidence, "confidence_score")],
        "ptm_scores": [_confidence_value(confidence, "ptm")],
        "iptm_scores": [_confidence_value(confidence, "iptm")],
        "ligand_iptm_scores": [_confidence_value(confidence, "ligand_iptm")],
        "protein_iptm_scores": [_confidence_value(confidence, "protein_iptm")],
        "complex_plddt_scores": [_confidence_value(confidence, "complex_plddt")],
        "complex_iplddt_scores": [_confidence_value(confidence, "complex_iplddt")],
        "complex_pde_scores": [_confidence_value(confidence, "complex_pde")],
        "complex_ipde_scores": [_confidence_value(confidence, "complex_ipde")],
        "chains_ptm_scores": confidence.get("chains_ptm"),
        "pair_chains_iptm_scores": confidence.get("pair_chains_iptm"),
        "pae": pae,
        "pde": pde,
        "plddt": plddt,
        "confidence": confidence,
        "metrics": {
            "total_time_seconds": elapsed_seconds,
            "model_inference_time_seconds": elapsed_seconds,
        },
        "structures": (
            [{"path": str(structure_path), "format": structure_path.suffix.lower().lstrip(".")}]
            if structure_path is not None
            else []
        ),
        "local_files": {
            "output_root": str(output_root),
            "structure_path": str(structure_path) if structure_path else "",
            "confidence_json": str(confidence_path) if confidence_path else "",
            "pae_npz": str(pae_path) if pae_path else "",
            "pde_npz": str(pde_path) if pde_path else "",
            "plddt_npz": str(plddt_path) if plddt_path else "",
        },
        "required_pae": bool(require_pae),
        "pae_present": pae is not None,
        "pae_matrix_usable": pae_usable,
        "pae_shape": pae_shape,
        "pde_present": pde is not None,
        "pde_shape": nested_shape(pde) if pde is not None else [],
        "plddt_present": plddt is not None,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(response_path, response)

    row: dict[str, Any] = {
        "variant": variant,
        "status": status,
        "http_status": "",
        "structure_path": str(structure_path) if structure_path else "",
        "response_json": str(response_path),
        "error": "; ".join(errors),
        "required_pae": bool(require_pae),
        "pae_present": pae is not None,
        "pae_matrix_usable": pae_usable,
        "pae_shape": json.dumps(pae_shape, ensure_ascii=True) if pae_shape else "",
        "confidence_score": _confidence_value(confidence, "confidence_score"),
        "ptm_score": _confidence_value(confidence, "ptm"),
        "iptm_score": _confidence_value(confidence, "iptm"),
        "ligand_iptm_score": _confidence_value(confidence, "ligand_iptm"),
        "protein_iptm_score": _confidence_value(confidence, "protein_iptm"),
        "complex_plddt_score": _confidence_value(confidence, "complex_plddt"),
        "complex_iplddt_score": _confidence_value(confidence, "complex_iplddt"),
        "complex_pde_score": _confidence_value(confidence, "complex_pde"),
        "complex_ipde_score": _confidence_value(confidence, "complex_ipde"),
        "pde_present": pde is not None,
        "pde_shape": json.dumps(nested_shape(pde), ensure_ascii=True) if pde is not None else "",
        "plddt_present": plddt is not None,
        "warnings": "; ".join(warnings),
        "metric_total_time_seconds": elapsed_seconds,
        "metric_model_inference_time_seconds": elapsed_seconds,
        "structure_prediction_write_full_pae": True,
        "structure_prediction_write_full_pde": True,
        "backend": "boltz2_local",
    }
    return row, response


def local_boltz_command(
    *,
    executable: str,
    yaml_path: Path,
    output_root: Path,
    cache_dir: str,
    devices: int,
    accelerator: str,
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    max_parallel_samples: int,
    num_workers: int,
    preprocessing_threads: int,
    no_kernels: bool,
    write_full_pae: bool,
    write_full_pde: bool,
    output_format: str,
    override: bool,
) -> list[str]:
    command = [
        executable,
        "predict",
        str(yaml_path),
        "--out_dir",
        str(output_root),
        "--cache",
        str(cache_dir),
        "--devices",
        str(int(devices)),
        "--accelerator",
        str(accelerator),
        "--recycling_steps",
        str(int(recycling_steps)),
        "--sampling_steps",
        str(int(sampling_steps)),
        "--diffusion_samples",
        str(int(diffusion_samples)),
        "--max_parallel_samples",
        str(int(max_parallel_samples)),
        "--num_workers",
        str(int(num_workers)),
        "--preprocessing-threads",
        str(int(preprocessing_threads)),
    ]
    if no_kernels:
        command.append("--no_kernels")
    if write_full_pae:
        command.append("--write_full_pae")
    if write_full_pde:
        command.append("--write_full_pde")
    if output_format:
        command.extend(["--output_format", output_format])
    if override:
        command.append("--override")
    return command


def _attempt_profile(
    *,
    profile: str,
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    max_parallel_samples: int,
    num_workers: int,
    preprocessing_threads: int,
    accelerator: str,
) -> dict[str, int | str]:
    """Resolve a named retry profile while keeping all settings auditable."""
    profile = str(profile or "standard").strip().lower()
    if profile in {"standard", "default", ""}:
        return {
            "name": "standard",
            "recycling_steps": int(recycling_steps),
            "sampling_steps": int(sampling_steps),
            "diffusion_samples": int(diffusion_samples),
            "max_parallel_samples": int(max_parallel_samples),
            "num_workers": int(num_workers),
            "preprocessing_threads": int(preprocessing_threads),
            "accelerator": str(accelerator),
        }
    if profile in {"oom", "oom_reduced", "low_memory"}:
        # Sequence length, template features and ligand tokens dominate the
        # activation footprint.  One diffusion sample and one worker are
        # already the lowest safe settings; reducing recycles/steps is the
        # remaining useful retry on 8-GB cards.
        return {
            "name": "oom_reduced",
            "recycling_steps": 0,
            "sampling_steps": 5,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "num_workers": 1,
            "preprocessing_threads": 1,
            "accelerator": str(accelerator),
        }
    if profile in {"oom_cpu", "cpu"}:
        return {
            "name": "oom_cpu",
            # CPU fallback changes the memory device, not the scientific
            # sampling protocol. Preserve the manifest's standard settings.
            "recycling_steps": int(recycling_steps),
            "sampling_steps": int(sampling_steps),
            "diffusion_samples": int(diffusion_samples),
            "max_parallel_samples": int(max_parallel_samples),
            "num_workers": int(num_workers),
            "preprocessing_threads": int(preprocessing_threads),
            "accelerator": "cpu",
        }
    raise ValueError(f"Unsupported Boltz-2 attempt profile: {profile}")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for preferred in (
        "variant",
        "status",
        "http_status",
        "structure_path",
        "response_json",
        "error",
    ):
        fields.append(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Boltz-2 CLI predictions from a SwitchTune manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    boltz = manifest.get("boltz2")
    if not isinstance(boltz, dict):
        raise ValueError("Manifest does not contain a `boltz2` section.")
    if str(boltz.get("backend") or "boltz2_local") != "boltz2_local":
        raise ValueError("This runner requires manifest boltz2.backend=boltz2_local.")

    prediction_dir = Path(str(manifest.get("prediction_dir"))).expanduser()
    response_dir = Path(str(boltz.get("response_dir") or prediction_dir.parent / "boltz2_responses")).expanduser()
    summary_csv = Path(
        str(boltz.get("summary_csv") or prediction_dir.parent / "boltz2_local_prediction_summary.csv")
    ).expanduser()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    executable = str(boltz.get("executable") or "boltz")
    cache_dir = str(boltz.get("cache_dir") or "./models/boltz2")
    devices = int(boltz.get("devices") or 1)
    accelerator = str(boltz.get("accelerator") or "gpu")
    recycling_steps = int(boltz.get("recycling_steps") or 1)
    sampling_steps = int(boltz.get("sampling_steps") or 20)
    diffusion_samples = int(boltz.get("diffusion_samples") or 1)
    max_parallel_samples = int(boltz.get("max_parallel_samples") or 1)
    num_workers = int(boltz.get("num_workers") or 1)
    preprocessing_threads = int(boltz.get("preprocessing_threads") or 1)
    no_kernels = bool(boltz.get("no_kernels", True))
    oom_cpu_fallback = bool(boltz.get("oom_cpu_fallback", False))
    write_full_pae = bool(boltz.get("write_full_pae", True))
    write_full_pde = bool(boltz.get("write_full_pde", True))
    output_format = str(boltz.get("output_format") or "mmcif")
    require_pae = True
    if not write_full_pae:
        raise ValueError("require_pae=true is mandatory; local Boltz-2 must use --write_full_pae.")

    records = boltz.get("input_yaml_records") or []
    if not isinstance(records, list):
        raise ValueError("boltz2.input_yaml_records must be a list.")
    if args.max_records > 0:
        records = records[: args.max_records]

    rows: list[dict[str, Any]] = []
    failures = 0
    fallback_modes = boltz.get(
        "fallback_modes",
        ["no_template", "no_ligands_keep_template", "no_ligands"],
    )
    if not isinstance(fallback_modes, list):
        fallback_modes = ["no_template", "no_ligands_keep_template", "no_ligands"]
    fallback_modes = [str(mode).strip().lower() for mode in fallback_modes if str(mode).strip()]
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        variant = str(record.get("variant") or f"variant_{index}")
        yaml_path = Path(str(record.get("yaml_path") or record.get("input_yaml_path") or "")).expanduser()
        output_root = prediction_dir / variant
        response_path = response_dir / f"{variant}.response.json"
        existing_files = find_local_prediction_files(output_root, variant)
        existing_valid = (
            existing_files["structure"] is not None
            and existing_files["pae"] is not None
        )
        if existing_valid and not args.force:
            try:
                row, _ = normalize_local_prediction(
                    variant=variant,
                    output_root=output_root,
                    response_path=response_path,
                    elapsed_seconds=0.0,
                    require_pae=require_pae,
                )
            except Exception as exc:
                row = {
                    "variant": variant,
                    "status": "failed_exception",
                    "structure_path": "",
                    "response_json": str(response_path),
                    "error": str(exc),
                }
            if row.get("status") == "completed":
                row["status"] = "skipped_existing"
            else:
                failures += 1
            rows.append(row)
            print(f"[{index}/{len(records)}] {variant} {row.get('status')}", flush=True)
            continue

        if not yaml_path.exists():
            failures += 1
            row = {
                "variant": variant,
                "status": "failed_missing_input",
                "structure_path": "",
                "response_json": str(response_path),
                "error": f"Input YAML not found: {yaml_path}",
                "required_pae": True,
                "pae_matrix_usable": False,
                "backend": "boltz2_local",
            }
            rows.append(row)
            print(f"{variant} failed_missing_input: {yaml_path}", flush=True)
            continue

        output_root.mkdir(parents=True, exist_ok=True)
        # A failed process may leave a partially populated output directory.
        # The normalizer below only accepts a structure together with a valid
        # PAE matrix, so it is safe to reuse those files on a later invocation.
        # Attempts are tuples of (label, yaml, profile).  The low-memory retry
        # is inserted only after an actual OOM; parse failures still use the
        # ligand/template fallbacks below.
        attempts: list[tuple[str, Path, str]] = [("primary", yaml_path, "standard")]
        for mode in fallback_modes:
            fallback_yaml = response_dir / f"{variant}.fallback_{mode}.yaml"
            try:
                _write_local_fallback_yaml(yaml_path, fallback_yaml, mode)
            except Exception as exc:
                # Keep the primary attempt usable even if a fallback cannot be generated.
                print(f"{variant} fallback_{mode} input generation failed: {exc}", flush=True)
                continue
            attempts.append((f"fallback_{mode}", fallback_yaml, "standard"))

        final_row: dict[str, Any] | None = None
        final_response: dict[str, Any] = {}
        final_command: list[str] = []
        final_log_path = response_dir / f"{variant}.boltz_local.log"
        final_return_code = 1
        final_log_tail = ""
        final_elapsed = 0.0
        attempted_oom_inputs: set[str] = set()
        attempted_cpu_inputs: set[str] = set()
        attempt_index = 0
        while attempt_index < len(attempts):
            attempt_mode, attempt_yaml, attempt_profile_name = attempts[attempt_index]
            attempt_index += 1
            profile = _attempt_profile(
                profile=attempt_profile_name,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                diffusion_samples=diffusion_samples,
                max_parallel_samples=max_parallel_samples,
                num_workers=num_workers,
                preprocessing_threads=preprocessing_threads,
                accelerator=accelerator,
            )
            command = local_boltz_command(
                executable=executable,
                yaml_path=attempt_yaml,
                output_root=output_root,
                cache_dir=cache_dir,
                devices=devices,
                accelerator=str(profile["accelerator"]),
                recycling_steps=int(profile["recycling_steps"]),
                sampling_steps=int(profile["sampling_steps"]),
                diffusion_samples=int(profile["diffusion_samples"]),
                max_parallel_samples=int(profile["max_parallel_samples"]),
                num_workers=int(profile["num_workers"]),
                preprocessing_threads=int(profile["preprocessing_threads"]),
                no_kernels=no_kernels,
                write_full_pae=write_full_pae,
                write_full_pde=write_full_pde,
                output_format=output_format,
                override=True,
            )
            log_path = response_dir / (
                f"{variant}.boltz_local.log" if attempt_index == 0
                else f"{variant}.boltz_local.{attempt_mode}.log"
            )
            print(
                f"[{index}/{len(records)}] running local Boltz-2 for {variant} "
                f"({attempt_mode}, profile={profile['name']})",
                flush=True,
            )
            started = time.monotonic()
            return_code = 1
            command_error = ""
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write("$ " + " ".join(command) + "\n\n")
                log_handle.flush()
                try:
                    child_env = os.environ.copy()
                    # Reduce allocator fragmentation across the many
                    # one-variant Boltz invocations in a long batch.
                    child_env.setdefault(
                        "PYTORCH_CUDA_ALLOC_CONF",
                        "expandable_segments:True",
                    )
                    completed = subprocess.run(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        env=child_env,
                    )
                    return_code = int(completed.returncode)
                except Exception as exc:
                    command_error = str(exc)
                    log_handle.write(f"\nRUNNER_EXCEPTION: {command_error}\n")
            elapsed = time.monotonic() - started
            try:
                row, response = normalize_local_prediction(
                    variant=variant,
                    output_root=output_root,
                    response_path=response_path,
                    elapsed_seconds=elapsed,
                    require_pae=require_pae,
                )
            except Exception as exc:
                row = {
                    "variant": variant,
                    "status": "failed_exception",
                    "structure_path": "",
                    "response_json": str(response_path),
                    "error": str(exc),
                    "required_pae": True,
                    "pae_matrix_usable": False,
                    "backend": "boltz2_local",
                }
                response = {}
            log_tail = read_log_tail(log_path)
            if return_code != 0:
                existing_error = str(row.get("error") or "")
                command_error = command_error or f"Boltz-2 CLI exited with code {return_code}; see {log_path}"
                row["error"] = "; ".join(item for item in (existing_error, command_error) if item)
                if row.get("status") == "completed":
                    row["status"] = "failed_cli"
            row.update(
                {
                    "return_code": return_code,
                    "log_tail": log_tail,
                    "cli_error_visible": bool(return_code != 0 or row.get("status") != "completed"),
                    "attempt_mode": attempt_mode,
                    "attempt_profile": str(profile["name"]),
                    "attempt_recycling_steps": int(profile["recycling_steps"]),
                    "attempt_sampling_steps": int(profile["sampling_steps"]),
                    "attempted_input_yaml": str(attempt_yaml),
                    "failure_class": _failure_class(log_tail, return_code, str(row.get("status") or "")),
                }
            )
            if response_path.exists():
                try:
                    response_data = read_json(response_path)
                    response_data.update(
                        {
                            "cli_return_code": return_code,
                            "cli_log_path": str(log_path),
                            "cli_log_tail": log_tail,
                            "attempt_mode": attempt_mode,
                        }
                    )
                    write_json(response_path, response_data)
                except Exception:
                    pass
            final_row = row
            final_response = response
            final_command = command
            final_log_path = log_path
            final_return_code = return_code
            final_log_tail = log_tail
            final_elapsed = elapsed
            # Boltz-2 may return code 0 while Lightning skips an OOM batch.
            # Retry the same ligand/template-aware input with a reduced
            # activation footprint before trying less faithful fallbacks.
            if (
                _is_oom_log(log_tail)
                and attempt_profile_name == "standard"
                and str(attempt_yaml) not in attempted_oom_inputs
            ):
                attempted_oom_inputs.add(str(attempt_yaml))
                attempts.insert(
                    attempt_index,
                    (f"{attempt_mode}_oom_reduced", attempt_yaml, "oom_reduced"),
                )
            if (
                oom_cpu_fallback
                and _is_oom_log(log_tail)
                and attempt_profile_name == "oom_reduced"
                and str(attempt_yaml) not in attempted_cpu_inputs
            ):
                attempted_cpu_inputs.add(str(attempt_yaml))
                attempts.insert(
                    attempt_index,
                    (f"{attempt_mode}_cpu", attempt_yaml, "oom_cpu"),
                )
            if return_code == 0 and row.get("status") == "completed":
                if attempt_mode != "primary":
                    row["status"] = f"completed_{attempt_mode}"
                    row["fallback_used"] = True
                    row["fallback_warning"] = (
                        "Primary ligand/template-aware Boltz prediction failed; "
                        f"prediction completed using {attempt_mode}. Inspect primary log before interpreting ligand metrics."
                    )
                break

        assert final_row is not None
        final_row["command"] = " ".join(final_command)
        final_row["log_path"] = str(final_log_path)
        final_row["metric_total_time_seconds"] = final_elapsed
        final_row["metric_model_inference_time_seconds"] = final_elapsed
        if final_row.get("status", "").startswith("completed"):
            # A fallback is a usable structure, even though it is scientifically
            # qualified by fallback_warning above.
            pass
        else:
            failures += 1
        rows.append(final_row)
        print(
            f"{variant} {final_row.get('status')}: PAE usable={final_row.get('pae_matrix_usable')} "
            f"structure={bool(final_row.get('structure_path'))} "
            f"failure_class={final_row.get('failure_class', '')}",
            flush=True,
        )

    _write_summary(summary_csv, rows)
    summary = {
        "status": "completed" if failures == 0 else "completed_with_failures",
        "backend": "boltz2_local",
        "manifest": str(manifest_path),
        "candidate_count": len(records),
        "failure_count": failures,
        "require_pae": True,
        "prediction_dir": str(prediction_dir),
        "response_dir": str(response_dir),
        "summary_csv": str(summary_csv),
        "cache_dir": cache_dir,
        "oom_cpu_fallback": oom_cpu_fallback,
    }
    write_json(summary_csv.with_suffix(".summary.json"), summary)
    print("Boltz-2 local batch summary:")
    print(json.dumps(summary, indent=2))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
