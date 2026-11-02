#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
        json.dump(data, handle, indent=2, ensure_ascii=True)


def iter_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield item
            yield from iter_values(item)
    elif isinstance(value, list):
        for item in value:
            yield item
            yield from iter_values(item)


def nested_shape(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value)] + nested_shape(value[0])
    return []


def pae_matrix_status(value: Any) -> tuple[bool, list[int]]:
    """Validate Boltz PAE as a finite square matrix, accepting [1, n, n]."""
    original_shape = nested_shape(value)
    matrix_like = value
    if (
        isinstance(matrix_like, list)
        and len(matrix_like) == 1
        and isinstance(matrix_like[0], list)
        and matrix_like[0]
        and isinstance(matrix_like[0][0], list)
    ):
        matrix_like = matrix_like[0]
    if not isinstance(matrix_like, list) or not matrix_like:
        return False, original_shape
    matrix_size = len(matrix_like)
    for row in matrix_like:
        if not isinstance(row, list) or len(row) != matrix_size:
            return False, original_shape
        for item in row:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                return False, original_shape
            if not math.isfinite(numeric):
                return False, original_shape
    return True, original_shape


def required_pae_summary(response: dict[str, Any]) -> dict[str, Any]:
    pae = response.get("pae")
    usable, shape = pae_matrix_status(pae)
    return {
        "required_pae": True,
        "pae_present": pae is not None,
        "pae_matrix_usable": usable,
        "pae_shape": json.dumps(shape, ensure_ascii=True) if shape else "",
    }


def structure_suffix(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("data_") or "_atom_site." in stripped or "_chem_comp." in stripped:
        return ".cif"
    if stripped.startswith("ATOM") or stripped.startswith("HETATM") or "\nATOM" in text or "\nHETATM" in text:
        return ".pdb"
    return ".cif"


def extract_structure_text(response: dict[str, Any]) -> tuple[str, str]:
    preferred_keys = (
        "structure",
        "structure_cif",
        "cif",
        "mmcif",
        "pdb",
        "structure_pdb",
    )
    stack: list[Any] = [response]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key in preferred_keys:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value, structure_suffix(value)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)

    for value in iter_values(response):
        if not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) < 100:
            continue
        if text.startswith("data_") or "_atom_site." in text:
            return value, ".cif"
        if text.startswith("ATOM") or "\nATOM" in text or "\nHETATM" in text:
            return value, ".pdb"
    return "", ""


def extract_confidence_summary(response: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "confidence",
        "confidence_score",
        "confidence_scores",
        "ptm",
        "iptm",
        "complex_plddt",
        "plddt",
        "predicted_aligned_error",
    }
    summary: dict[str, Any] = {}

    def add_value(key: str, value: Any) -> None:
        if isinstance(value, (int, float, str, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float, str, bool)):
            singular = key[:-1] if key.endswith("s") else key
            summary[singular] = value[0]

    for key, value in response.items():
        if key in keys:
            add_value(key, value)
    for value in iter_values(response):
        if isinstance(value, dict):
            for key in keys:
                if key in value and key not in summary:
                    add_value(key, value[key])
    metrics = response.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                summary[f"metric_{key}"] = value
    return summary


def post_with_retries(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[int, dict[str, Any], str]:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("The Boltz-2 NIM runner requires `requests`. Install with: pip install requests") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    retryable = {408, 409, 425, 429, 500, 502, 503, 504}
    last_text = ""
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_text = str(exc)
            if attempt >= max_retries:
                return 599, {"error": last_text}, last_text
            wait = sleep_seconds * (2 ** attempt)
            print(
                f"Retryable request error {type(exc).__name__}; "
                f"sleeping {wait:.1f}s before retry {attempt + 1}/{max_retries}."
            )
            time.sleep(wait)
            continue
        last_text = response.text
        if response.status_code < 400:
            try:
                return response.status_code, response.json(), response.text
            except Exception:
                return response.status_code, {"raw_text": response.text}, response.text
        if response.status_code not in retryable or attempt >= max_retries:
            try:
                data = response.json()
            except Exception:
                data = {"error_text": response.text}
            return response.status_code, data, response.text
        wait = sleep_seconds * (2 ** attempt)
        print(f"Retryable HTTP {response.status_code}; sleeping {wait:.1f}s before retry {attempt + 1}/{max_retries}.")
        time.sleep(wait)
    return 599, {"error_text": last_text}, last_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hosted Boltz-2 predictions from a SwitchTune manifest.")
    parser.add_argument("--manifest", required=True, help="structure_prediction_job_manifest.json")
    parser.add_argument("--force", action="store_true", help="Re-run even if output structure exists.")
    parser.add_argument(
        "--drop-structural-templates",
        action="store_true",
        help="Debug mode: remove polymers[*].structural_templates from each payload before submission.",
    )
    parser.add_argument(
        "--drop-ligands",
        action="store_true",
        help="Debug mode: remove top-level ligands from each payload before submission.",
    )
    parser.add_argument(
        "--save-submitted-payloads",
        action="store_true",
        help="Write the exact payload submitted after debug transformations.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Debug mode: submit at most this many records.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    boltz = manifest.get("boltz2")
    if not isinstance(boltz, dict):
        raise ValueError("Manifest does not contain a `boltz2` section.")

    endpoint = str(boltz.get("endpoint") or "").strip()
    api_key_env = str(boltz.get("api_key_env") or "BOLTZ2_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip()
    if not endpoint:
        raise ValueError("Boltz-2 endpoint is empty in manifest.")
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")

    prediction_dir = Path(str(manifest.get("prediction_dir"))).expanduser()
    response_dir = Path(str(boltz.get("response_dir") or prediction_dir.parent / "boltz2_responses")).expanduser()
    summary_csv = Path(str(boltz.get("summary_csv") or prediction_dir.parent / "boltz2_nim_prediction_summary.csv")).expanduser()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    timeout = int(boltz.get("request_timeout_seconds") or 600)
    max_retries = int(boltz.get("max_retries") or 5)
    sleep_seconds = float(boltz.get("rate_limit_sleep_seconds") or 3.0)
    pae_missing_retries = max(0, int(boltz.get("pae_missing_retries") or 2))
    # SwitchTune requires usable full PAE for every Boltz-2 structure job.
    require_pae = True
    records = boltz.get("payload_records") or []
    if not isinstance(records, list):
        raise ValueError("boltz2.payload_records must be a list.")
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]

    rows: list[dict[str, Any]] = []
    failures = 0
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        variant = str(record.get("variant") or f"variant_{idx}")
        payload_path = Path(str(record.get("payload_path"))).expanduser()
        raw_response_path = response_dir / f"{variant}.response.json"
        existing = sorted(prediction_dir.glob(f"{variant}.*"))
        existing = [p for p in existing if p.suffix.lower() in {".cif", ".pdb"}]
        if existing and not args.force:
            confidence: dict[str, Any] = {}
            pae_info = {
                "required_pae": require_pae,
                "pae_present": False,
                "pae_matrix_usable": False,
                "pae_shape": "",
            }
            if raw_response_path.exists():
                try:
                    response = read_json(raw_response_path)
                    confidence = extract_confidence_summary(response)
                    pae_info = required_pae_summary(response)
                except Exception:
                    confidence = {}
            if require_pae and not pae_info["pae_matrix_usable"]:
                failures += 1
                rows.append(
                    {
                        "variant": variant,
                        "status": "failed_missing_required_pae",
                        "http_status": "",
                        "structure_path": str(existing[0]),
                        "response_json": str(raw_response_path) if raw_response_path.exists() else "",
                        "error": (
                            "Existing structure was not reused because the response lacks a usable PAE matrix; "
                            "rerun with --force after obtaining a response with write_full_pae=true."
                        ),
                        **pae_info,
                        **confidence,
                    }
                )
                print(
                    f"{variant} failed_missing_required_pae: existing structure has no usable PAE response",
                    flush=True,
                )
                continue
            rows.append(
                {
                    "variant": variant,
                    "status": "skipped_existing",
                    "http_status": "",
                    "structure_path": str(existing[0]),
                    "response_json": str(raw_response_path) if raw_response_path.exists() else "",
                    "error": "",
                    **pae_info,
                    **confidence,
                }
            )
            continue

        payload = read_json(payload_path)
        if args.drop_structural_templates:
            for polymer in payload.get("polymers") or []:
                if isinstance(polymer, dict):
                    polymer.pop("structural_templates", None)
        if args.drop_ligands:
            payload["ligands"] = []
        submitted_payload_path = ""
        if args.save_submitted_payloads:
            submitted_payload_path = str(response_dir / f"{variant}.submitted_payload.json")
            write_json(Path(submitted_payload_path), payload)
        print(f"[{idx}/{len(records)}] submitting {variant} to Boltz-2 NIM")
        try:
            pae_retry_count = 0
            http_status = 599
            response_json: dict[str, Any] = {}
            raw_text = ""
            pae_info: dict[str, Any] = {
                "required_pae": require_pae,
                "pae_present": False,
                "pae_matrix_usable": False,
                "pae_shape": "",
            }
            for pae_attempt in range(pae_missing_retries + 1):
                http_status, response_json, raw_text = post_with_retries(
                    endpoint=endpoint,
                    api_key=api_key,
                    payload=payload,
                    timeout=timeout,
                    max_retries=max_retries,
                    sleep_seconds=sleep_seconds,
                )
                pae_info = required_pae_summary(response_json)
                if (
                    http_status < 400
                    and require_pae
                    and not pae_info["pae_matrix_usable"]
                    and pae_attempt < pae_missing_retries
                ):
                    pae_retry_count += 1
                    print(
                        f"{variant} returned HTTP {http_status} without a usable PAE matrix; "
                        f"retrying PAE-aware request {pae_retry_count}/{pae_missing_retries}.",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    continue
                break
            write_json(raw_response_path, response_json)
            structure_text, suffix = extract_structure_text(response_json)
            structure_path = ""
            status = "completed" if http_status < 400 else "failed"
            error = "" if http_status < 400 else raw_text[-1000:]
            if (
                structure_text
                and http_status < 400
                and (not require_pae or pae_info["pae_matrix_usable"])
            ):
                out_path = prediction_dir / f"{variant}{suffix}"
                out_path.write_text(structure_text, encoding="utf-8")
                structure_path = str(out_path)
            if status == "completed" and require_pae and not pae_info["pae_matrix_usable"]:
                status = "failed_missing_required_pae"
                error = (
                    "Boltz-2 returned HTTP success but no usable PAE matrix while "
                    "write_full_pae=true was requested after "
                    f"{pae_retry_count} PAE-aware retries."
                )
                failures += 1
            elif not structure_text and status == "completed":
                status = "completed_no_structure_detected"
                error = "No CIF/PDB structure text was detected in the response."
                failures += 1
            else:
                if status != "completed":
                    failures += 1
            confidence = extract_confidence_summary(response_json)
            rows.append(
                {
                    "variant": variant,
                    "status": status,
                    "http_status": http_status,
                    "structure_path": structure_path,
                    "response_json": str(raw_response_path),
                    "submitted_payload_json": submitted_payload_path,
                    "error": error,
                    "pae_retry_count": pae_retry_count,
                    **pae_info,
                    **confidence,
                }
            )
            if status != "completed":
                compact_error = str(error or raw_text or "").replace("\n", " ")[:800]
                print(f"{variant} {status}: HTTP {http_status}; {compact_error}", flush=True)
        except Exception as exc:
            failures += 1
            rows.append(
                {
                    "variant": variant,
                    "status": "failed_exception",
                    "http_status": "",
                    "structure_path": "",
                    "response_json": str(raw_response_path),
                    "submitted_payload_json": submitted_payload_path,
                    "error": str(exc),
                    "pae_retry_count": pae_retry_count if "pae_retry_count" in locals() else 0,
                }
            )
            print(f"{variant} failed_exception: {exc}", flush=True)
        if idx < len(records):
            time.sleep(sleep_seconds)

    fields: list[str] = []
    for preferred in ("variant", "status", "http_status", "structure_path", "response_json", "error"):
        fields.append(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "completed" if failures == 0 else "completed_with_failures",
        "manifest": str(manifest_path),
        "endpoint": endpoint,
        "api_key_env": api_key_env,
        "candidate_count": len(records),
        "failure_count": failures,
        "require_pae": require_pae,
        "pae_missing_retries": pae_missing_retries,
        "prediction_dir": str(prediction_dir),
        "response_dir": str(response_dir),
        "summary_csv": str(summary_csv),
    }
    write_json(summary_csv.with_suffix(".summary.json"), summary)
    print("Boltz-2 NIM batch summary:")
    print(json.dumps(summary, indent=2))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
