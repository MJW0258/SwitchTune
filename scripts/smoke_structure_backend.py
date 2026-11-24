#!/usr/bin/env python3
"""Prepare or execute one WT control with the deployment-selected predictor."""
# ruff: noqa: E402 -- Support running this file without installing the package.

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from switchtune.mcp_server import (
    collect_structure_prediction_results,
    prepare_structure_prediction_job,
    standardize_structure_prediction_results,
)
from switchtune.structure_backend import configured_engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure", type=Path, required=True, help="Target-state PDB or mmCIF"
    )
    parser.add_argument("--chain", default="A")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the installed predictor on the server",
    )
    args = parser.parse_args()
    os.environ.setdefault("SWITCHTUNE_HOME", str(ROOT))
    os.environ.setdefault(
        "SWITCHTUNE_TOOL_REGISTRY", str(ROOT / "config/tool_registry.yaml")
    )
    os.environ.setdefault("SWITCHTUNE_CONFIG", str(ROOT / "config/switchtune.yaml"))
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.SeqUtils import seq1

    structure_parser = (
        MMCIFParser(QUIET=True)
        if args.structure.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = structure_parser.get_structure("target", str(args.structure))
    sequence = "".join(
        seq1(r.resname) for r in structure[0][args.chain] if r.id[0] == " "
    )
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        parser.error("Choose a protein chain containing standard amino acids")
    name = "structure_smoke_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(os.environ.get("SWITCHTUNE_RUNS", ROOT / "SwitchTune_runs")) / name
    root.mkdir(parents=True, exist_ok=True)
    candidates = root / "wt_control.csv"
    with candidates.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "sequence"])
        writer.writeheader()
        writer.writerow({"variant": "WT", "sequence": sequence})
    job = prepare_structure_prediction_job(
        name,
        str(candidates),
        engine="auto",
        target_state_structure=str(args.structure.resolve()),
        template_chain_id=args.chain,
        execute=args.execute,
    )
    report = {"configured_engine": configured_engine(), "job": job}
    if args.execute:
        collected = collect_structure_prediction_results(
            name, job["prediction_dir"], job_manifest_path=job["job_manifest_path"]
        )
        standardized = standardize_structure_prediction_results(
            name, collected["output_csv"], require_pae=True
        )
        report.update(collection=collected, standardization=standardized)
        passed = (
            collected["valid_structure_count"] == 1
            and standardized["standardized_count"] == 1
        )
        report["passed"] = passed
    else:
        print(
            "Prepared only. Add --execute on a configured Linux GPU server to test real inference."
        )
        passed = True
    output = root / "smoke_report.json"
    output.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Backend: {job['engine']}; manifest: {job['job_manifest_path']}")
    print(f"Smoke report: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
