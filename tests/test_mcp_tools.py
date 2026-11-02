from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from switchtune.mcp_server import (
    create_project,
    filter_library_by_constraints,
    generate_saturation_library,
    inspect_bias_score_distribution,
    parse_design_prompt,
    prepare_bias_scoring_inputs,
    recommend_bias_threshold,
    run_conformation_bias_scoring,
    select_state_specific_candidates,
    setup_saturation_screening_project,
    summarize_library,
    validate_inputs,
    write_decision_log,
)

TEST_STRUCTURE = ROOT / "toy_target_state.pdb"
TEST_REFERENCE_STRUCTURE = ROOT / "toy_reference_state.pdb"
TEST_STRUCTURE.write_text(
    "HEADER    TOY TARGET STRUCTURE FOR SWITCHTUNE MCP TEST\nEND\n",
    encoding="utf-8",
)
TEST_REFERENCE_STRUCTURE.write_text(
    "HEADER    TOY REFERENCE STRUCTURE FOR SWITCHTUNE MCP TEST\nEND\n",
    encoding="utf-8",
)

prompt = (
    "Design state-specific mutations for ToyProtein around residues 2-3 "
    "to preserve switchability in the toy target state."
)

print("parse_design_prompt:")
parsed = parse_design_prompt(prompt)
print(json.dumps(parsed, indent=2, ensure_ascii=False))

print("\ncreate_project:")
created = create_project(
    project_name="toy_switchtune",
    protein_name="ToyProtein",
    target_state_name="toy_state",
    design_objective="Test SwitchTune MCP connection.",
    metadata={"source": "test_mcp_tools.py"},
)
print(json.dumps(created, indent=2, ensure_ascii=False))

print("\nwrite_decision_log:")
decision = write_decision_log(
    project_name="toy_switchtune",
    step_name="parse_design_prompt",
    decision="proceed",
    rationale="Toy prompt contains residue range and protein name.",
    input_summary={"prompt": prompt},
    output_summary={"parsed_fields": parsed},
)
print(json.dumps(decision, indent=2, ensure_ascii=False))

print("\nvalidate_inputs:")
result = validate_inputs(
    project_name="toy_switchtune",
    protein_name="ToyProtein",
    sequence="ACDEFGHIK",
    target_state_structure=str(TEST_STRUCTURE),
    target_state_name="toy_state",
    mutation_region="2-3",
    design_objective="Test state-specific screening MCP connection.",
    first_residue_number=1,
)
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\ngenerate_saturation_library:")
library = generate_saturation_library(
    project_name="toy_switchtune",
    sequence="ACDEFGHIK",
    mutation_region="2-3",
    first_residue_number=1,
)
print(json.dumps(library, indent=2, ensure_ascii=False))

print("\nsummarize_library:")
lib_summary = summarize_library(
    project_name="toy_switchtune",
    library_csv=library["csv_path"],
)
print(json.dumps(lib_summary, indent=2, ensure_ascii=False))

print("\nfilter_library_by_constraints:")
filtered = filter_library_by_constraints(
    project_name="toy_switchtune",
    library_csv=library["csv_path"],
    exclude_positions="3",
)
print(json.dumps(filtered, indent=2, ensure_ascii=False))

print("\nprepare_bias_scoring_inputs:")
bias_manifest = prepare_bias_scoring_inputs(
    project_name="toy_switchtune",
    library_csv=library["csv_path"],
    target_state_structure=str(TEST_STRUCTURE),
    target_state_name="toy_state",
    first_residue_number=1,
)
print(json.dumps(bias_manifest, indent=2, ensure_ascii=False))

toy_scored_csv = Path(bias_manifest["output_csv"]).with_name("toy_bias_scores.csv")
with open(bias_manifest["output_csv"], "r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
for idx, row in enumerate(rows):
    # Deterministic toy scores spanning negative to positive values.
    row["bias_score"] = f"{-1.5 + 0.08 * idx:.3f}"
with toy_scored_csv.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\ninspect_bias_score_distribution:")
dist = inspect_bias_score_distribution(
    project_name="toy_switchtune",
    scored_csv=str(toy_scored_csv),
    score_column="bias_score",
    score_min=-1.0,
    score_max=1.0,
)
print(json.dumps(dist, indent=2, ensure_ascii=False))

print("\nrecommend_bias_threshold:")
threshold = recommend_bias_threshold(
    project_name="toy_switchtune",
    scored_csv=str(toy_scored_csv),
    score_column="bias_score",
    target_retained_fraction=0.5,
)
print(json.dumps(threshold, indent=2, ensure_ascii=False))

print("\nrun_conformation_bias_scoring:")
cb_job = run_conformation_bias_scoring(
    project_name="toy_switchtune",
    library_csv=library["csv_path"],
    target_state_structure=str(TEST_STRUCTURE),
    reference_state_structure=str(TEST_REFERENCE_STRUCTURE),
    target_state_name="toy_target_state",
    reference_state_name="toy_reference_state",
    model_type="protein_mpnn",
    parse_chains="A",
    first_residue_number=1,
    execute=False,
    collect_scores=False,
)
print(json.dumps(cb_job, indent=2, ensure_ascii=False))

print("\nselect_state_specific_candidates:")
selected = select_state_specific_candidates(
    project_name="toy_switchtune",
    scored_library_csv=str(toy_scored_csv),
    score_column="bias_score",
    score_min=-1.0,
    score_max=1.0,
)
print(json.dumps(selected, indent=2, ensure_ascii=False))

print("\nsetup_saturation_screening_project:")
combined = setup_saturation_screening_project(
    project_name="toy_switchtune_combined",
    protein_name="ToyProtein",
    sequence="ACDEFGHIK",
    target_state_structure=str(TEST_STRUCTURE),
    target_state_name="toy_state",
    mutation_region="2-3",
    design_objective="Test state-specific screening MCP composite tool.",
    first_residue_number=1,
)
print(json.dumps(combined, indent=2, ensure_ascii=False))

print("\nSmoke test complete.")
