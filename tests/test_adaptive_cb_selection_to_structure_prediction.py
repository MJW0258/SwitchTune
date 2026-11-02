from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from switchtune.mcp_server import (
    generate_saturation_library,
    prepare_structure_prediction_job,
    select_state_specific_candidates_adaptive,
)


def _make_toy_scored_library(project_name: str) -> str:
    library = generate_saturation_library(project_name, "ACDEFGHIKLMNPQRSTVWY", "2-8")
    source = Path(library["csv_path"])
    output = source.with_name("toy_cb_scored_library.csv")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    for idx, row in enumerate(rows):
        # A deterministic broad score distribution with central and tail variants.
        row["bias_score"] = f"{2.8 * math.sin(idx * 0.37):.6f}"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + ["bias_score"])
        writer.writeheader()
        writer.writerows(rows)
    return str(output)


def main() -> None:
    os.environ.setdefault("SWITCHTUNE_RUNS", str(PROJECT_ROOT / "test_runs"))
    project_name = os.environ.get("SWITCHTUNE_ADAPTIVE_TEST_PROJECT", "chey_two_state_cb_test")
    scored_csv = os.environ.get(
        "SWITCHTUNE_ADAPTIVE_TEST_SCORED_CSV",
        str(PROJECT_ROOT / "test_runs" / project_name / "02_conformation_bias_scoring" / "cb_scored_library.csv"),
    )

    if not Path(scored_csv).expanduser().exists():
        project_name = "toy_adaptive_cb_selection_test"
        scored_csv = _make_toy_scored_library(project_name)
        print("Using toy scored library because CheY scored CSV was not found.")

    adaptive = select_state_specific_candidates_adaptive(
        project_name=project_name,
        scored_library_csv=scored_csv,
        score_column="bias_score",
        strategy=os.environ.get("SWITCHTUNE_ADAPTIVE_STRATEGY", "auto"),
        target_retained_fraction=(
            float(os.environ["SWITCHTUNE_ADAPTIVE_FRACTION"])
            if os.environ.get("SWITCHTUNE_ADAPTIVE_FRACTION")
            else None
        ),
        target_count=(
            int(os.environ["SWITCHTUNE_ADAPTIVE_TARGET_COUNT"])
            if os.environ.get("SWITCHTUNE_ADAPTIVE_TARGET_COUNT")
            else None
        ),
        min_candidates=int(os.environ.get("SWITCHTUNE_ADAPTIVE_MIN", "30")),
        max_candidates=int(os.environ.get("SWITCHTUNE_ADAPTIVE_MAX", "250")),
        center=os.environ.get("SWITCHTUNE_ADAPTIVE_CENTER", "zero"),
    )

    structure_job = prepare_structure_prediction_job(
        project_name=project_name,
        candidates_csv=adaptive["output_csv"],
        engine=os.environ.get("SWITCHTUNE_STRUCTURE_ENGINE", "boltz2_nim"),
        top_n=int(os.environ.get("SWITCHTUNE_STRUCTURE_TOP_N", str(adaptive["retained_count"]))),
    )

    print("ADAPTIVE_CB_SELECTION_TO_STRUCTURE_PREDICTION_TEST: PASSED")
    print("Project:", project_name)
    print("Input scored CSV:", scored_csv)
    print("Adaptive selected CSV:", adaptive["output_csv"])
    print("Adaptive retained:", adaptive["retained_count"])
    print("Adaptive fraction:", adaptive["retained_fraction"])
    print("Adaptive score window:", adaptive["score_min"], adaptive["score_max"])
    print("Fixed [-1, 1] retained:", adaptive["fixed_minus1_plus1_count"])
    print("Structure input FASTA:", structure_job["fasta_path"])
    print("Structure run script:", structure_job["run_script"])


if __name__ == "__main__":
    main()
