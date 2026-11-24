#!/usr/bin/env python3
"""Run/resume official AlphaFold 3 jobs and normalize their outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from switchtune.alphafold3 import collect_candidate, write_json


def build_command(record: dict, config: dict) -> list[str]:
    model_dir = Path(config["model_dir"]).expanduser()
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        raise ValueError(
            "AF3 model_dir is missing/empty. Obtain official model parameters and set AF3_MODEL_DIR."
        )
    if config["msa_mode"] == "search" and not Path(config["db_dir"]).is_dir():
        raise ValueError("AF3 search requires downloaded databases at AF3_DB_DIR")
    output_base = Path(record["output_base"])
    output_base.mkdir(parents=True, exist_ok=True)
    input_path = Path(record["input_path"]).resolve()
    if config["deployment"] == "docker":
        command = ["docker", "run", "--rm", "--gpus", "all"]
        if hasattr(os, "getuid"):
            command += ["--user", f"{os.getuid()}:{os.getgid()}"]
        command += [
            "--env",
            "HOME=/tmp",
            "--env",
            "XDG_CACHE_HOME=/tmp/.cache",
            "--volume",
            f"{input_path.parent}:/mnt/af_input:ro",
            "--volume",
            f"{output_base}:/mnt/af_output",
            "--volume",
            f"{model_dir.resolve()}:/mnt/models:ro",
        ]
        if config["msa_mode"] == "search":
            command += [
                "--volume",
                f"{Path(config['db_dir']).resolve()}:/mnt/public_databases:ro",
            ]
        command += [
            config["docker_image"],
            "python",
            "run_alphafold.py",
            f"--json_path=/mnt/af_input/{input_path.name}",
            "--output_dir=/mnt/af_output",
            "--model_dir=/mnt/models",
        ]
        if config["msa_mode"] == "search":
            command += ["--db_dir=/mnt/public_databases"]
    else:
        script = Path(config["repo"]) / "run_alphafold.py"
        if not script.is_file():
            raise ValueError(
                "Set AF3_REPO to the official AlphaFold3 checkout containing run_alphafold.py"
            )
        command = [
            config["python"],
            str(script),
            f"--json_path={input_path}",
            f"--output_dir={output_base}",
            f"--model_dir={model_dir.resolve()}",
        ]
        if config["msa_mode"] == "search":
            command += [f"--db_dir={config['db_dir']}"]
    command += [
        f"--run_data_pipeline={'true' if config['msa_mode'] == 'search' else 'false'}",
        f"--num_diffusion_samples={config['num_samples']}",
        "--force_output_dir=true",
    ]
    reserved = {
        "--json_path",
        "--input_dir",
        "--output_dir",
        "--model_dir",
        "--db_dir",
        "--run_data_pipeline",
        "--run_inference",
        "--num_diffusion_samples",
        "--num_seeds",
        "--force_output_dir",
        "--compress_large_output_files",
    }
    if any(
        str(arg).split("=", 1)[0] in reserved for arg in config.get("extra_args", [])
    ):
        raise ValueError(
            "AF3 extra_args must not override managed input/output, sample, or pipeline flags"
        )
    return command + list(config.get("extra_args", []))


def save_summary(config: dict, rows: list[dict]) -> None:
    path = Path(config["summary_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or [
        "variant",
        "status",
        "error",
    ]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    failures = sum(row["status"].startswith("failed") for row in rows)
    write_json(
        path.with_suffix(".summary.json"),
        {
            "status": "failed"
            if failures == len(rows) and rows
            else "completed_with_failures"
            if failures
            else "completed",
            "candidate_count": len(rows),
            "failure_count": failures,
            "completed_count": len(rows) - failures,
        },
    )


def run_batch(
    manifest: dict, *, collect_only: bool = False, timeout: int = 0
) -> list[dict]:
    config = manifest["alphafold3"]
    rows = []
    for record in config["input_records"]:
        row = collect_candidate(record, config, Path(config["response_dir"]))
        if row["status"] == "completed":
            row["status"] = "skipped_existing" if not collect_only else "completed"
        elif not collect_only:
            log = Path(record["output_base"]) / "alphafold3.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            return_code, error = None, ""
            try:
                command = build_command(record, config)
                print(f"AlphaFold3: {record['variant']}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=timeout or None,
                        check=False,
                    )
                return_code = result.returncode
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
            # Artifact validation is authoritative, including after process errors.
            row = collect_candidate(record, config, Path(config["response_dir"]))
            row.update(
                return_code=return_code,
                log_path=str(log),
                elapsed_seconds=round(time.monotonic() - start, 3),
            )
            if row["status"].startswith("failed"):
                row["error"] = "; ".join(
                    x
                    for x in [error, f"CLI return code: {return_code}", row["error"]]
                    if x
                )
        rows.append(row)
        save_summary(config, rows)
    if not rows:
        save_summary(config, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Rescan existing outputs without loading AF3 or running inference",
    )
    parser.add_argument("--timeout-seconds", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = run_batch(
        manifest, collect_only=args.collect_only, timeout=args.timeout_seconds
    )
    return 1 if any(row["status"].startswith("failed") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
