#!/usr/bin/env python3
# Choose and configure the structure predictor.
"""Choose a predictor before installing heavyweight model dependencies (stdlib only)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def hardware_report(storage_root: Path) -> dict:
    storage_root = storage_root.expanduser().resolve()
    existing = storage_root
    while not existing.exists():
        existing = existing.parent
    gpu = "NVIDIA GPU information unavailable"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        gpu = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "storage_root": str(storage_root),
        "free_disk_gib": round(shutil.disk_usage(existing).free / 2**30, 1),
        "gpu": gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["boltz2_local", "alphafold3"])
    parser.add_argument("--storage-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "config/structure_backend.local.json",
    )
    parser.add_argument("--deployment", choices=["docker", "native"], default="docker")
    parser.add_argument(
        "--msa-mode",
        choices=["search", "single_sequence", "provided"],
        default="search",
    )
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--db-dir", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--python", default="python")
    parser.add_argument("--docker-image", default="alphafold3")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    report = hardware_report(args.storage_root)
    print(json.dumps(report, indent=2))
    print(
        "Boltz-2: smaller deployment; GPU memory needs depend on sequence/complex length."
    )
    print(
        "AlphaFold3: upstream recommends >=64 GB host RAM; validated on A100/H100 80 GB."
    )
    print(
        "AF3 full databases: ~630 GB extracted; allow up to 1 TB for full installation."
    )
    print(
        "AF3 inference with prepared MSAs/templates can omit search databases, but still needs weights and sufficient GPU memory."
    )
    print(
        "These are planning guidelines, not a guarantee that a particular protein fits."
    )
    if args.inspect_only:
        return
    engine = args.engine
    if not engine:
        choice = input("Select predictor: [1] Boltz-2, [2] AlphaFold3: ").strip()
        if choice not in {"1", "2"}:
            parser.error("Choose 1 or 2, or supply --engine")
        engine = {"1": "boltz2_local", "2": "alphafold3"}[choice]
    root = args.storage_root.expanduser().resolve()
    config = {"engine": engine, "hardware_at_configuration": report}
    if engine == "alphafold3":
        config["alphafold3"] = {
            "deployment": args.deployment,
            "docker_image": args.docker_image,
            "repo": str(Path(args.repo).expanduser().resolve()) if args.repo else "",
            "python": args.python,
            "msa_mode": args.msa_mode,
            "model_dir": str(Path(args.model_dir).expanduser().resolve())
            if args.model_dir
            else str(root / "af3/models"),
            "db_dir": str(Path(args.db_dir).expanduser().resolve())
            if args.db_dir
            else str(root / "af3/databases"),
            "seeds": [1],
            "num_samples": 5,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {engine} selection to {args.output}")
    print(
        "Next: follow docs/installation.md for this predictor; the launcher and MCP server read this selection automatically."
    )


if __name__ == "__main__":
    main()
