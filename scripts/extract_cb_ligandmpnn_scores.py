# Extract ProteinMPNN and LigandMPNN state scores.
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["SWITCHTUNE_CB_ALLOW_EXTERNAL_EXTRACT"] = "0"

from switchtune.mcp_server import extract_conformation_bias_scores  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SwitchTune CB scores from LigandMPNN .pt outputs."
    )
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--job_manifest_path", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--score_min", type=float, default=None)
    parser.add_argument("--score_max", type=float, default=None)
    parser.add_argument("--score_mode", default="single_mutant_delta")
    parser.add_argument("--chain_id", default=None)
    parser.add_argument("--position_column", default="position")
    parser.add_argument("--sequence_index_column", default="sequence_index_1based")
    parser.add_argument("--wt_column", default="wt_aa")
    parser.add_argument("--mut_column", default="mut_aa")
    parser.add_argument("--sequence_column", default="sequence")
    args = parser.parse_args()

    result = extract_conformation_bias_scores(
        project_name=args.project_name,
        job_manifest_path=args.job_manifest_path,
        output_csv=args.output_csv,
        score_min=args.score_min,
        score_max=args.score_max,
        score_mode=args.score_mode,
        chain_id=args.chain_id,
        position_column=args.position_column,
        sequence_index_column=args.sequence_index_column,
        wt_column=args.wt_column,
        mut_column=args.mut_column,
        sequence_column=args.sequence_column,
    )
    print("SWITCHTUNE_CB_EXTRACT_JSON:" + json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
