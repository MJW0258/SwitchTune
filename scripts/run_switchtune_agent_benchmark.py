#!/usr/bin/env python3
"""Run a formal SwitchTune benchmark through Biomni agent.go().

This is the public-release style benchmark entrypoint. It deliberately starts
from a natural-language prompt, registers the SwitchTune MCP server with
Biomni A1, uses the guarded agent.go() handoff, and invokes a separate Biomni
reviewer at every checkpoint plus the final round.

Do not confuse this with run_switchtune_formal_benchmark.py, which is a
deprecated direct-MCP runner kept only for old timing/debug experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(os.environ.get("SWITCHTUNE_HOME", SCRIPT_DIR.parent)).expanduser().resolve()
RUNS_ROOT = Path(os.environ.get("SWITCHTUNE_RUNS", "./SwitchTune_runs")).expanduser()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


BIOEMU_METRIC_FIELDS = (
    "bioemu_sample_count",
    "bioemu_complete_sample_count",
    "bioemu_complete_sample_fraction",
    "bioemu_rmsd_to_target_mean_A",
    "bioemu_rmsd_to_reference_mean_A",
    "bioemu_target_like_fraction",
    "bioemu_target_preference_margin_A",
    "bioemu_pairwise_rmsd_mean_A",
    "bioemu_rmsf_global_mean_A",
    "bioemu_mutation_window_rmsf_mean_A",
    "bioemu_radius_gyration_mean_A",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_agent_failure(error: str) -> str:
    """Classify external LLM billing failures separately from workflow failures."""
    text = str(error or "").lower()
    billing_markers = (
        "insufficient balance",
        "billing_error",
        "payment required",
        "quota exceeded",
        "credits exhausted",
    )
    if "403" in text and any(marker in text for marker in billing_markers):
        return "blocked_external_llm_billing"
    if any(marker in text for marker in billing_markers):
        return "blocked_external_llm_billing"
    return ""


def print_header(title: str) -> None:
    print("\n" + "=" * 92, flush=True)
    print(title, flush=True)
    print("=" * 92, flush=True)


def read_sequence(sequence: str, sequence_file: str) -> str:
    if sequence:
        return "".join(sequence.split()).upper()
    if not sequence_file:
        return ""
    lines = Path(sequence_file).expanduser().read_text(encoding="utf-8").splitlines()
    return "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith(">")
    ).upper()


def read_design_prompt(prompt: str, prompt_file: str) -> str:
    """Read the user's natural-language design request from text or a file."""
    if str(prompt or "").strip():
        return str(prompt).strip()
    if str(prompt_file or "").strip():
        return Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
    return ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


def recover_front_half_after_llm_failure(
    *,
    project_name: str,
    protein_name: str,
    target_state_structure: str,
    reference_state_structure: str,
    target_state_name: str,
    reference_state_name: str,
    parse_chains: str,
    sequence: str,
    first_residue_number: int,
    model_type: str,
    ligand_codes: str,
    design_objective: str,
    error_text: str,
    mutation_region: str = "",
) -> dict[str, Any] | None:
    """Recover the setup checkpoint without another upstream LLM request.

    The normal formal path starts with ``agent.go``. If the provider rejects a
    continuation request after the first PDB observation (upstream LLM 502), the
    SwitchTune workflow can still make a blind, auditable setup decision from
    the two supplied WT structures. This recovery is limited to provider
    request failures; ordinary Python or tool errors remain fatal.
    """
    lowered = str(error_text or "").lower()
    if not any(marker in lowered for marker in ("502", "internalservererror", "improperly formed request")):
        return None
    try:
        from switchtune.mcp_server import (
            _autonomous_structural_mutation_region,
            inspect_two_state_pdb_pair,
            setup_two_state_cb_scoring_project_once,
        )

        inspection = inspect_two_state_pdb_pair(
            pdb_a=target_state_structure,
            pdb_b=reference_state_structure,
            protein_name=protein_name,
            design_goal=design_objective,
        )
        region = str(mutation_region or "").strip()
        if region:
            rationale = "Use the explicit user-specified mutation region unchanged during LLM recovery."
        else:
            region = str(inspection.get("suggested_mutation_region") or "").strip()
            rationale = str(inspection.get("mutation_region_rationale") or "").strip()
        if not region:
            chain_id = next(
                (item.strip() for item in re.split(r"[,;\s]+", parse_chains) if item.strip()),
                "A",
            )
            region, rationale = _autonomous_structural_mutation_region(
                target_state_structure,
                reference_state_structure,
                chain_id,
            )
        if not region:
            return None

        result = setup_two_state_cb_scoring_project_once(
            project_name=project_name,
            protein_name=protein_name,
            sequence=sequence,
            target_state_structure=target_state_structure,
            reference_state_structure=reference_state_structure,
            mutation_region=region,
            target_state_name=target_state_name,
            reference_state_name=reference_state_name,
            design_objective=design_objective,
            model_type=model_type or "protein_mpnn",
            parse_chains=parse_chains,
            first_residue_number=int(first_residue_number),
            force_refresh=False,
            stop_after_setup=False,
        )
        if str(result.get("status") or "").lower() not in {"completed", "prepared"}:
            return None
        return {
            "status": "recovered_without_llm_continuation",
            "mutation_region": region,
            "mutation_region_rationale": rationale,
            "inspection": inspection,
            "setup": result,
        }
    except Exception as exc:
        return {"status": "recovery_failed", "error": repr(exc)}


def continue_front_half_after_llm_recovery(
    *,
    recovery: dict[str, Any],
    project_name: str,
    target_state_structure: str,
    reference_state_structure: str,
    target_state_name: str,
    reference_state_name: str,
    cb_selection_mode: str,
    cb_score_min: float,
    cb_score_max: float,
    structure_engine: str,
    ligand_codes: str,
    structure_top_n: int,
    cb_timeout_seconds: int,
    workflow_timeout_seconds: int,
) -> dict[str, Any]:
    """Finish the deterministic front half after an LLM continuation failure.

    ``agent.go()`` normally performs these calls after setup.  When the
    provider rejects the continuation (for example with an upstream LLM 502), the
    recovery path must perform the same audited MCP calls itself; otherwise the
    checkpoint runner starts with a project that has a library but no retained
    candidate CSV.
    """
    setup = recovery.get("setup") if isinstance(recovery, dict) else None
    if not isinstance(setup, dict):
        return {"status": "recovery_front_half_failed", "error": "recovery did not contain setup result"}

    try:
        from switchtune.mcp_server import (
            execute_conformation_bias_scoring_job,
            prepare_structure_prediction_job,
            select_state_specific_candidates,
            select_state_specific_candidates_adaptive,
            select_structure_prediction_engine,
        )

        manifest_path = str(setup.get("cb_job_manifest_path") or "")
        if not manifest_path:
            raise ValueError("setup result did not contain cb_job_manifest_path")

        cb_result = execute_conformation_bias_scoring_job(
            project_name=project_name,
            job_manifest_path=manifest_path,
            timeout_seconds=int(cb_timeout_seconds),
            skip_if_outputs_exist=True,
            collect_scores=True,
        )
        if str(cb_result.get("status") or "").lower() != "completed":
            return {
                "status": "recovery_front_half_failed",
                "error": f"CB scoring did not complete: {cb_result.get('status')}",
                "cb_execution": cb_result,
            }

        scored_csv = str(cb_result.get("scored_library_csv") or "")
        if not scored_csv:
            raise ValueError("CB execution completed without scored_library_csv")
        mode = str(cb_selection_mode or "adaptive").strip().lower()
        if mode == "fixed_window":
            selection = select_state_specific_candidates(
                project_name=project_name,
                scored_library_csv=scored_csv,
                score_column="bias_score",
                score_min=float(cb_score_min),
                score_max=float(cb_score_max),
            )
        else:
            selection = select_state_specific_candidates_adaptive(
                project_name=project_name,
                scored_library_csv=scored_csv,
                score_column="bias_score",
                strategy="auto",
                center="zero",
            )
        if str(selection.get("status") or "").lower() != "completed":
            return {
                "status": "recovery_front_half_failed",
                "error": f"CB candidate selection did not complete: {selection.get('status')}",
                "cb_execution": cb_result,
                "cb_selection": selection,
            }

        candidates_csv = str(selection.get("output_csv") or "")
        if not candidates_csv:
            raise ValueError("CB candidate selection completed without output_csv")
        engine_decision = select_structure_prediction_engine(
            project_name=project_name,
            target_state_structure=target_state_structure,
            reference_state_structure=reference_state_structure,
            ligand_codes=ligand_codes,
            requested_engine=structure_engine,
        )
        structure = prepare_structure_prediction_job(
            project_name=project_name,
            candidates_csv=candidates_csv,
            engine=str(engine_decision.get("recommended_engine") or structure_engine or "auto"),
            top_n=int(structure_top_n) if int(structure_top_n) > 0 else 0,
            target_state_structure=target_state_structure,
            reference_state_structure=reference_state_structure,
            template_mode="target_state",
            boltz_use_template=True,
            boltz_write_full_pae=True,
            ligand_codes=ligand_codes,
            cofactor_required=bool(engine_decision.get("ligand_required", False)),
            execute=False,
            timeout_seconds=int(workflow_timeout_seconds),
        )
        if str(structure.get("status") or "").lower() not in {"completed", "prepared"}:
            return {
                "status": "recovery_front_half_failed",
                "error": f"structure job preparation did not complete: {structure.get('status')}",
                "cb_execution": cb_result,
                "cb_selection": selection,
                "engine_decision": engine_decision,
                "structure_prediction": structure,
            }
        return {
            "status": "recovered_front_half_completed",
            "cb_execution": cb_result,
            "cb_selection": selection,
            "engine_decision": engine_decision,
            "structure_prediction": structure,
        }
    except Exception as exc:
        return {
            "status": "recovery_front_half_failed",
            "error": repr(exc),
        }


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists() or not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def mutation_site_from_variant(value: Any) -> str:
    """Extract a residue-site key from a single- or multi-mutant ID."""
    text = str(value or "").strip().upper()
    positions = re.findall(
        r"(?<![A-Z0-9])[A-Z*](\d+)[A-Z*](?=$|[^A-Z0-9])",
        text,
    )
    if not positions:
        positions = re.findall(r"[A-Z*](\d+)[A-Z*]", text)
    unique = list(dict.fromkeys(positions))
    return ";".join(unique) or "unknown"


def csv_row_count(path: Path | str) -> int:
    return len(read_csv_rows(Path(path))[0]) if str(path) else 0


def has_workflow_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    artifact_names = {
        "project_manifest.json",
        "decision_log.jsonl",
        "jobs",
        "01_saturation_library",
        "02_conformation_bias_scoring",
        "03_state_specific_candidates",
        "04_structure_prediction",
        "04b_structure_confidence_filter",
        "05_inverse_folding_scoring",
        "06_bioemu_sampling",
        "07_rosetta_energy",
        "08_final_recommendation",
    }
    return any((run_dir / name).exists() for name in artifact_names)


def configure_environment() -> None:
    os.environ["SWITCHTUNE_HOME"] = str(REPO)
    os.environ.setdefault("SWITCHTUNE_RUNS", str(RUNS_ROOT))
    os.environ.setdefault("SWITCHTUNE_CONFIG", str(REPO / "config" / "switchtune.yaml"))
    os.environ.setdefault("SWITCHTUNE_TOOL_REGISTRY", str(REPO / "config" / "tool_registry.yaml"))
    os.environ.setdefault("SWITCHTUNE_DECISION_POLICY", str(REPO / "config" / "decision_policy.yaml"))
    os.environ.setdefault("SWITCHTUNE_STRUCTURE_ENGINE", "boltz2_local")
    # Short proteins, single-chain oligomeric proxies, and ligand-free state
    # surrogates can yield complete Boltz-2/PAE outputs below the generic
    # absolute confidence thresholds.  Keep the strict gate unchanged, but
    # allow the MCP server to create an explicitly labelled relative-quality
    # review pool so a completed prediction stage is not mistaken for a hard
    # execution failure.  Set this to 0 to restore strict-only behaviour.
    os.environ.setdefault("SWITCHTUNE_ALLOW_RELATIVE_STRUCTURE_FALLBACK", "1")
    # The MCP server is launched as a child process. Keep autonomous region
    # selection enabled in that child so dynamic-region runs do not require a
    # fragile second LLM continuation after the initial PDB inspection.
    os.environ.setdefault("SWITCHTUNE_AUTONOMOUS_REGION_FALLBACK", "0")
    os.environ.setdefault("BOLTZ_CACHE", "./models/boltz2")
    os.environ.setdefault("BOLTZ_LOCAL_EXECUTABLE", "boltz")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    # ESM-IF1 must stay in the active biomni_e1 environment unless explicitly
    # overridden by the user.
    os.environ["ESM_IF1_ENV"] = os.environ.get("ESM_IF1_ENV", "current")
    os.environ.setdefault("ESM_IF1_REPO", "./external/esm")
    os.environ.setdefault(
        "ESM_IF1_SCORE_SCRIPT",
        "./external/esm/examples/inverse_folding/score_log_likelihoods.py",
    )

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO)]
    if current_pythonpath:
        parts.append(current_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def build_prompt(
    *,
    project_name: str,
    protein_name: str,
    target_state_structure: str,
    reference_state_structure: str,
    target_state_name: str,
    reference_state_name: str,
    parse_chains: str,
    mutation_region: str,
    first_residue_number: int,
    model_type: str,
    user_prompt: str,
    design_objective: str,
    sequence: str,
    ligand_codes: str,
    structure_engine: str,
    cb_selection_mode: str,
    cb_score_min: float,
    cb_score_max: float,
    bioemu_top_n: int,
    bioemu_max_per_site: int,
) -> str:
    sequence_block = f"\nsequence:\n{sequence}\n" if sequence else ""
    region_line = (
        f"mutation_region: {mutation_region}\n"
        if mutation_region
        else (
            "mutation_region: agent-selected after inspection\n"
            "mutation_region_format: one or more exact ranges or positions, comma-separated\n"
        )
    )
    model_line = f"model_type: {model_type}\n" if model_type else ""
    ligand_line = f"ligand_codes: {ligand_codes}\n" if ligand_codes else ""
    return f"""You are connected to the SwitchTune MCP server through Biomni A1.

Run a formal SwitchTune benchmark workflow from this natural-language request.

project_name: {project_name}
protein_name: {protein_name}
target_state_structure: {target_state_structure}
reference_state_structure: {reference_state_structure}
target_state_name: {target_state_name}
reference_state_name: {reference_state_name}
parse_chains: {parse_chains}
first_residue_number: {first_residue_number}
{region_line}{model_line}{ligand_line}structure_engine: {structure_engine}
cb_selection_mode: {cb_selection_mode}
user_design_prompt: {user_prompt or "(not supplied; use the compatibility objective below)"}
design_objective: {design_objective}
{sequence_block}
Formal benchmark mode:
- Do not use smoke-test budgets.
- Do not limit structure prediction to five variants.
- Use empirical adaptive CB selection. Choose a central retention fraction from the
  observed score distribution and record the rationale. Do not apply a universal score
  threshold or a fixed retained count.
- Do not use BioEmu or Rosetta budgets to set the CB retention count.
- The checkpoint workflow will send all candidates retained by the selected CB policy to local Boltz-2 structure prediction.
- Let the agent choose the BioEmu and final candidate budget from the valid evidence pool.
  The adaptive budget is bounded to 20-30 candidates when that many valid variants exist.
  Use {bioemu_max_per_site} variants maximum per mutation site and the configured sample count.
  Prefer broad residue-site coverage before filling additional substitutions at
  the same site. The guard will audit this constraint and backfill from the
  same valid checkpoint pool when necessary.
- After BioEmu, retain the complete adaptive 20-30 candidate set for Rosetta.
- Rosetta must rank/evaluate that full set; do not use the final recommendation top-N
  as a Rosetta input filter.
- The final recommendation top-N is applied only after Rosetta evidence is merged.
- Keep require_pae=true, request full PAE/PDE from Boltz-2, and use the local Boltz-2 CLI with the target-state structure as template.

Mandatory blind-test policy:
- This is a blind benchmark. During candidate generation, mutation-region selection,
  checkpoint review, ranking, and final recommendation, use only the supplied
  wild-type sequence and wild-type structures plus outputs generated by this run.
- Do not search, browse, retrieve, or consult PubMed, PMC, Google Scholar,
  Crossref, DOI pages, journal pages, mutation databases, disease databases,
  benchmark answer keys, previous native Biomni/SwitchTune outputs, remembered
  engineering variants, or mutant PDB/mmCIF entries.
- Do not use literature-known mutation identities or expected benchmark hits to
  choose sites, substitutions, scores, or rationales. If an output field would
  require literature evidence, write `not assessed in blind test`.
- WT input validation is allowed only for the explicitly supplied WT structures
  and WT sequence. Post-hoc literature comparison must occur after this run in a
  separate analysis and must not be fed back into the workflow.

Required agent behavior:
1. First execute block must call inspect_two_state_pdb_pair exactly once.
   Use the MCP tool's exact argument names and call shape:
   `inspect_two_state_pdb_pair(pdb_a="{target_state_structure}",
   pdb_b="{reference_state_structure}", protein_name="{protein_name}",
   design_goal="{design_objective}")`. Pass these four keywords only:
   do not add state-name, chain, or duplicate structure-path keywords.
2. Use the same project_name for every SwitchTune tool call.
3. Treat the explicit target_state_structure, reference_state_structure,
   target_state_name, and reference_state_name above as authoritative for this
   run. Do not swap the state direction based only on ligand clues from PDB
   inspection; use those clues to validate the assignment and report conflicts.
4. Choose one or more precise contiguous intervals, non-contiguous intervals, or
   discrete positions from the two inspected structures and the supplied sequence.
   Report the structural rationale, excluded regions, and candidate-budget allocation
   before setup. Encode the final choice as a parseable comma-separated string such as
   `220-230,240-250,236`. Never pass an empty value, `infer`, or `1-1` to setup.
5. After inspection and, when needed, mutation-region selection, follow the
   SwitchTune guard handoff to set up the project, run CB scoring, apply the requested CB selection policy, and prepare the structure-prediction job.
   The next SwitchTune tool after inspection must be
   setup_two_state_cb_scoring_project_once. Do not call
   generate_saturation_library, setup_two_state_cb_scoring_project, or
   run_conformation_bias_scoring directly in formal benchmark mode.
6. In checkpoint mode, stop after the front-half handoff; the benchmark harness will run the expensive downstream stages and invoke Biomni checkpoint reviewers.
7. Do not rerun completed stages and do not repeat PDB inspection after receiving a valid observation.
8. After `setup_two_state_cb_scoring_project_once` returns `status=completed`
   or `reused_existing=true`, treat the setup checkpoint as locked. Do not
   inspect the PDB pair again, do not select a new region, and do not call
   either setup tool again. Return `<solution>` or continue only with the
   downstream handoff requested by the guard.
"""


def count_tool_mentions(text: str) -> dict[str, int]:
    tools = (
        "inspect_two_state_pdb_pair",
        "setup_two_state_cb_scoring_project_once",
        "execute_conformation_bias_scoring_job",
        "select_state_specific_candidates",
        "select_state_specific_candidates_adaptive",
        "prepare_structure_prediction_job",
        "filter_structure_confidence",
        "prepare_inverse_folding_scoring_job",
        "prepare_bioemu_sampling_job",
        "select_rosetta_analysis_strategy",
        "prepare_rosetta_energy_job",
        "prepare_final_candidate_recommendation",
        "record_agentic_candidate_decision",
    )
    return {name: text.count(name) for name in tools}


def load_decision_records(decision_log: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not decision_log.exists():
        return records
    for line in decision_log.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except Exception:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def decision_count(
    records: list[dict[str, Any]],
    *,
    reviewer: str = "",
    scope: str = "",
) -> int:
    total = 0
    for record in records:
        if record.get("step_name") != "agentic_candidate_decision":
            continue
        if reviewer and str(record.get("reviewer") or "") != reviewer:
            continue
        if scope and str(record.get("decision_scope") or "") != scope:
            continue
        total += 1
    return total


def bioemu_metric_summary(bioemu_csv: Path) -> dict[str, Any]:
    rows, fields = read_csv_rows(bioemu_csv)
    metric_fields = [field for field in BIOEMU_METRIC_FIELDS if field in fields]
    completed = [
        row for row in rows
        if str(row.get("bioemu_metrics_status") or "").strip().lower() == "completed"
    ]
    nonempty_by_variant: dict[str, int] = {}
    for row in completed:
        variant = str(row.get("variant") or f"row_{len(nonempty_by_variant) + 1}")
        nonempty_by_variant[variant] = sum(
            1
            for field in metric_fields
            if str(row.get(field) or "").strip().lower() not in {"", "nan", "none", "null", "na"}
        )
    return {
        "path": str(bioemu_csv),
        "row_count": len(rows),
        "field_count": len(fields),
        "metric_fields": metric_fields,
        "completed_metric_rows": len(completed),
        "nonempty_metric_count_by_variant": nonempty_by_variant,
    }


def build_validation(
    *,
    project_name: str,
    run_dir: Path,
    checkpoint_report: dict[str, Any],
    final_review_passed: bool,
    expected_bioemu_top_n: int,
    expected_bioemu_num_samples: int,
    expected_rosetta_top_n: int,
    min_structure_candidates: int,
    expected_bioemu_max_per_site: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    report_dir = run_dir / "00_agent_reports"
    decision_log = run_dir / "decision_log.jsonl"
    records = load_decision_records(decision_log)
    manifest = read_json(run_dir / "project_manifest.json")
    final_csv = run_dir / "08_final_recommendation" / "final_candidate_recommendation.csv"
    structure_evidence = run_dir / "04_structure_prediction" / "structure_prediction_evidence.csv"
    bioemu_manifest = run_dir / "06_bioemu_sampling" / "bioemu_sampling_job_manifest.json"
    bioemu_jobs = run_dir / "06_bioemu_sampling" / "bioemu_sampling_jobs.csv"
    bioemu_evidence = run_dir / "06_bioemu_sampling" / "bioemu_evidence.csv"
    rosetta_manifest = run_dir / "07_rosetta_energy" / "rosetta_energy_job_manifest.json"
    invocation = report_dir / "agent_go_invocation.json"

    expected_checkpoint_reviewers = {
        "structure_confidence": "biomni_structure_confidence_checkpoint",
        "inverse_folding": "biomni_inverse_folding_checkpoint",
        "bioemu": "biomni_bioemu_checkpoint",
        "rosetta": "biomni_rosetta_checkpoint",
    }

    checkpoint_reviews = checkpoint_report.get("checkpoint_reviews") or []
    checkpoint_stages = {
        str(item.get("stage_name") or ""): item
        for item in checkpoint_reviews
        if isinstance(item, dict)
    }
    for stage, reviewer in expected_checkpoint_reviewers.items():
        item = checkpoint_stages.get(stage) or {}
        check(
            f"{stage}_checkpoint_decision_recorded",
            bool(item.get("decision_recorded"))
            and str(item.get("decision_scope") or "") == "checkpoint"
            and str(item.get("decision_reviewer") or "") == reviewer,
            {
                "reviewer": item.get("decision_reviewer"),
                "scope": item.get("decision_scope"),
                "selected_variants": item.get("selected_variants"),
                "log_path": item.get("log_path"),
            },
        )

    check(
        "final_round_biomni_review_recorded",
        final_review_passed and decision_count(records, reviewer="biomni_final_round", scope="final_round") >= 1,
        {"decision_log": str(decision_log), "final_review_passed": final_review_passed},
    )

    candidate_counts = checkpoint_report.get("candidate_counts") or {}
    structure_count = int(
        candidate_counts.get("cb_selected_candidates")
        or candidate_counts.get("adaptive_cb_candidates")
        or 0
    )
    check(
        "formal_structure_candidate_set_not_smoke_test",
        structure_count >= int(min_structure_candidates),
        {
            "cb_selected_candidates": structure_count,
            "min_structure_candidates": min_structure_candidates,
            "policy": checkpoint_report.get("structure_candidate_policy"),
        },
    )
    check(
        "structure_policy_all_cb_retained",
        checkpoint_report.get("structure_candidate_policy")
        in {
            "all_cb_retained_candidates",
            "reused_existing_structure_prediction",
        },
        {
            "policy": checkpoint_report.get("structure_candidate_policy"),
            "structure_top_n": checkpoint_report.get("structure_top_n"),
        },
    )

    structure_rows, structure_fields = read_csv_rows(structure_evidence)
    pae_field = next(
        (field for field in ("structure_prediction_pae_matrix_usable", "pae_matrix_usable") if field in structure_fields),
        "",
    )
    full_pae_field = next(
        (field for field in ("structure_prediction_write_full_pae", "write_full_pae") if field in structure_fields),
        "",
    )
    check(
        "structure_predictions_have_usable_pae",
        bool(structure_rows)
        and bool(pae_field)
        and all(str(row.get(pae_field) or "").strip().lower() in {"true", "1", "yes", "passed", "pass"} for row in structure_rows),
        {
            "path": str(structure_evidence),
            "row_count": len(structure_rows),
            "pae_field": pae_field,
        },
    )
    check(
        "structure_predictions_requested_full_pae",
        bool(structure_rows)
        and bool(full_pae_field)
        and all(str(row.get(full_pae_field) or "").strip().lower() in {"true", "1", "yes", "passed", "pass"} for row in structure_rows),
        {
            "path": str(structure_evidence),
            "row_count": len(structure_rows),
            "full_pae_field": full_pae_field,
        },
    )

    bioemu_job = read_json(bioemu_manifest)
    bioemu_candidate_count = int(bioemu_job.get("candidate_count") or 0)
    attainable_candidate_floor = max(
        1,
        min(
            20,
            int(
                candidate_counts.get("inverse_folding_checkpoint_candidates")
                or candidate_counts.get("structure_checkpoint_candidates")
                or 20
            ),
        ),
    )
    check(
        "bioemu_candidate_budget_applied",
        attainable_candidate_floor <= bioemu_candidate_count <= int(expected_bioemu_top_n) <= 30,
        {
            "candidate_count": bioemu_candidate_count,
            "expected_upper_budget": expected_bioemu_top_n,
            "attainable_candidate_floor": attainable_candidate_floor,
            "accepted_range": "attainable_floor..bioemu_top_n, with bioemu_top_n<=30",
            "manifest": str(bioemu_manifest),
        },
    )
    check(
        "bioemu_sample_budget_applied",
        int(bioemu_job.get("num_samples") or 0) == int(expected_bioemu_num_samples),
        {
            "num_samples": bioemu_job.get("num_samples"),
            "expected": expected_bioemu_num_samples,
            "manifest": str(bioemu_manifest),
        },
    )
    bioemu_job_rows, bioemu_job_fields = read_csv_rows(bioemu_jobs)
    bioemu_site_counts: dict[str, int] = {}
    for row in bioemu_job_rows:
        site = mutation_site_from_variant(row.get("variant"))
        bioemu_site_counts[site] = bioemu_site_counts.get(site, 0) + 1
    check(
        "bioemu_site_diversity_cap_applied",
        bool(bioemu_job_rows)
        and max(bioemu_site_counts.values(), default=0) <= int(expected_bioemu_max_per_site),
        {
            "candidate_count": len(bioemu_job_rows),
            "site_counts": dict(sorted(bioemu_site_counts.items())),
            "maximum_site_count": max(bioemu_site_counts.values(), default=0),
            "expected_max_per_site": expected_bioemu_max_per_site,
            "jobs_csv": str(bioemu_jobs),
            "fields": bioemu_job_fields,
        },
    )
    metric_summary = bioemu_metric_summary(bioemu_evidence)
    check(
        "bioemu_multi_metric_evidence_present",
        metric_summary["completed_metric_rows"] > 0 and len(metric_summary["metric_fields"]) >= 8,
        metric_summary,
    )

    rosetta_job = read_json(rosetta_manifest)
    check(
        "rosetta_strategy_selected_and_propagated",
        bool(rosetta_job.get("strategy_name"))
        and bool(rosetta_job.get("strategy_focus"))
        and bool(rosetta_job.get("mode")),
        {
            "strategy_name": rosetta_job.get("strategy_name"),
            "strategy_focus": rosetta_job.get("strategy_focus"),
            "mode": rosetta_job.get("mode"),
            "manifest": str(rosetta_manifest),
        },
    )
    bioemu_checkpoint_count = int(candidate_counts.get("bioemu_checkpoint_candidates") or 0)
    rosetta_candidate_count = int(candidate_counts.get("rosetta_candidates") or 0)
    # The nominal formal budget is 20-30, but a strict upstream structure gate
    # can produce fewer than 20 valid WT-derived candidates.  In that case the
    # attainable floor is the number available after inverse-folding review;
    # requiring 20 would make an otherwise complete run fail mechanically.
    check(
        "bioemu_checkpoint_retains_formal_candidate_set",
        attainable_candidate_floor <= bioemu_checkpoint_count <= int(expected_bioemu_top_n),
        {
            "bioemu_checkpoint_candidates": bioemu_checkpoint_count,
            "expected_upper_budget": expected_bioemu_top_n,
            "attainable_candidate_floor": attainable_candidate_floor,
            "policy": checkpoint_report.get("bioemu_candidate_policy"),
        },
    )
    check(
        "rosetta_receives_full_bioemu_checkpoint_set",
        rosetta_candidate_count == bioemu_checkpoint_count
        and attainable_candidate_floor <= rosetta_candidate_count <= int(expected_bioemu_top_n),
        {
            "rosetta_candidates": rosetta_candidate_count,
            "bioemu_checkpoint_candidates": bioemu_checkpoint_count,
            "expected_range": f"{attainable_candidate_floor}..{expected_bioemu_top_n}",
            "attainable_candidate_floor": attainable_candidate_floor,
            "policy": checkpoint_report.get("rosetta_candidate_policy"),
        },
    )
    check(
        "rosetta_ranking_only_policy",
        checkpoint_report.get("rosetta_candidate_policy")
        == "all_bioemu_checkpoint_candidates_ranking_only",
        {
            "policy": checkpoint_report.get("rosetta_candidate_policy"),
            "final_recommendation_top_n": expected_rosetta_top_n,
        },
    )
    rosetta_checkpoint_count = int(candidate_counts.get("rosetta_checkpoint_candidates") or 0)
    check(
        "rosetta_checkpoint_retains_full_ranking_set",
        rosetta_checkpoint_count == rosetta_candidate_count,
        {
            "rosetta_candidates": rosetta_candidate_count,
            "rosetta_checkpoint_candidates": rosetta_checkpoint_count,
            "policy": "ranking_only",
        },
    )

    check(
        "final_recommendation_nonempty",
        csv_row_count(final_csv) > 0,
        {"path": str(final_csv), "row_count": csv_row_count(final_csv)},
    )
    check(
        "final_recommendation_budget_is_applied_after_rosetta",
        0 < csv_row_count(final_csv) <= int(expected_rosetta_top_n),
        {
            "path": str(final_csv),
            "row_count": csv_row_count(final_csv),
            "final_top_n": expected_rosetta_top_n,
        },
    )
    check(
        "agent_go_invocation_recorded",
        read_json(invocation).get("agent_go_called") is True
        or read_json(invocation).get("resumed_existing_project") is True,
        {"path": str(invocation), "invocation": read_json(invocation)},
    )

    if manifest.get("stage") != "agentic_candidate_decision_recorded":
        warnings.append(
            "project_manifest.json stage is not agentic_candidate_decision_recorded; "
            "this can happen after an idempotent final-review reuse."
        )

    return {
        "status": "passed" if not errors else "failed",
        "project_name": project_name,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "candidate_counts": candidate_counts,
        "bioemu_metric_summary": metric_summary,
        "decision_counts": {
            "checkpoint": decision_count(records, scope="checkpoint"),
            "final_round": decision_count(records, reviewer="biomni_final_round", scope="final_round"),
        },
        "updated_at": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a formal SwitchTune benchmark via Biomni agent.go() and checkpoint reviews."
    )
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--protein_name", required=True)
    parser.add_argument("--target_state_structure", required=True)
    parser.add_argument("--reference_state_structure", required=True)
    parser.add_argument("--target_state_name", default="")
    parser.add_argument("--reference_state_name", default="")
    parser.add_argument("--parse_chains", default="A")
    parser.add_argument("--first_residue_number", type=int, default=1)
    parser.add_argument("--model_type", default="")
    parser.add_argument(
        "--prompt",
        default="",
        help="Natural-language description of the desired mutation-design task.",
    )
    parser.add_argument(
        "--prompt_file",
        default="",
        help="UTF-8 file containing the natural-language design request.",
    )
    parser.add_argument("--design_objective", default="")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--sequence_file", default="")
    parser.add_argument("--ligand_codes", default="")
    parser.add_argument("--structure_engine", default=os.environ.get("SWITCHTUNE_STRUCTURE_ENGINE", "boltz2_local"))
    parser.add_argument(
        "--cb_selection_mode",
        choices=("adaptive",),
        default="adaptive",
        help="CB candidate retention policy. Default: empirical adaptive selection.",
    )
    parser.add_argument("--structure_top_n", type=int, default=0)
    parser.add_argument("--review_top_n", type=int, default=50)
    parser.add_argument("--min_structure_candidates", type=int, default=6)
    parser.add_argument(
        "--bioemu_top_n",
        type=int,
        default=0,
        help="Candidate budget. 0 delegates selection to the agent's adaptive 20-30 policy.",
    )
    parser.add_argument(
        "--bioemu_max_per_site",
        type=int,
        default=6,
        help="Maximum variants from one mutation site entering BioEmu.",
    )
    parser.add_argument("--bioemu_num_samples", type=int, default=500)
    parser.add_argument("--bioemu_batch_size", type=int, default=10)
    parser.add_argument(
        "--rosetta_top_n",
        type=int,
        default=0,
        help="Final recommendation budget. 0 delegates selection to the adaptive 20-30 policy.",
    )
    parser.add_argument("--agent_timeout_seconds", type=int, default=120)
    parser.add_argument("--workflow_timeout_seconds", type=int, default=86400)
    parser.add_argument("--cb_timeout_seconds", type=int, default=7200)
    parser.add_argument("--bioemu_timeout_seconds", type=int, default=86400)
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--max_execute_calls", type=int, default=56)
    parser.add_argument("--recursion_limit", type=int, default=260)
    parser.add_argument("--biomni_data_path", default="./data")
    parser.add_argument("--llm", default=os.environ.get("SWITCHTUNE_LLM_MODEL", "your-model-id"))
    parser.add_argument("--llm_source", default=os.environ.get("SWITCHTUNE_LLM_SOURCE", "Custom"))
    parser.add_argument("--base_url", default=os.environ.get("SWITCHTUNE_LLM_BASE_URL", ""))
    parser.add_argument("--api_key_env", default="SWITCHTUNE_LLM_API_KEY")
    parser.add_argument("--mcp_config", default="")
    parser.add_argument("--prompt_path", default="")
    parser.add_argument("--force_existing_project", action="store_true")
    parser.add_argument(
        "--resume_existing_project",
        action="store_true",
        help="Resume a project with a completed setup checkpoint without calling agent.go again.",
    )
    parser.add_argument(
        "--reuse_existing_structure_prediction",
        action="store_true",
        help=(
            "When resuming, skip Boltz-2 submission and rescan the existing "
            "structure_prediction output directory. Valid structure+PAE files "
            "are reused; missing candidates remain failed."
        ),
    )
    parser.add_argument("--debug_raw_agent", action="store_true")
    parser.add_argument("--no_run_external", action="store_true")
    args = parser.parse_args()
    # Retained for compatibility with the checkpoint/recovery function
    # signatures; public runs always use adaptive empirical selection.
    args.cb_score_min = -1.0
    args.cb_score_max = 1.0

    if int(args.bioemu_top_n) != 0 and not 20 <= int(args.bioemu_top_n) <= 30:
        parser.error("--bioemu_top_n must be 0 (adaptive) or between 20 and 30.")
    if int(args.bioemu_max_per_site) <= 0:
        parser.error("--bioemu_max_per_site must be positive.")
    if int(args.bioemu_num_samples) <= 0:
        parser.error("--bioemu_num_samples must be positive.")
    if int(args.structure_top_n) < 0:
        parser.error("--structure_top_n must be >= 0. Use 0 for all adaptive CB-retained candidates.")
    if int(args.rosetta_top_n) != 0 and not 20 <= int(args.rosetta_top_n) <= 30:
        parser.error("--rosetta_top_n must be 0 (adaptive) or between 20 and 30.")

    configure_environment()
    project_name = args.project_name.strip()
    run_dir = RUNS_ROOT / project_name
    report_dir = run_dir / "00_agent_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(args.prompt_path).expanduser() if args.prompt_path else report_dir / f"{project_name}_agent_benchmark_prompt.md"
    agent_log_path = report_dir / f"{project_name}_agent_go_log.txt"
    final_review_log_path = report_dir / f"{project_name}_final_round_review_log.txt"
    invocation_path = report_dir / "agent_go_invocation.json"
    summary_path = report_dir / f"{project_name}_agent_benchmark_summary.json"
    validation_path = report_dir / f"{project_name}_agent_benchmark_validation.json"

    print_header("SwitchTune Formal Agent Benchmark")
    print(f"Python: {sys.executable}", flush=True)
    print(f"Repository: {REPO}", flush=True)
    print(f"Project: {project_name}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"LLM: {args.llm}", flush=True)
    print(f"Structure engine: {args.structure_engine}", flush=True)
    print("CB selection: empirical adaptive central fraction", flush=True)
    print(f"Structure candidate policy: {'all adaptive CB-retained' if args.structure_top_n == 0 else args.structure_top_n}", flush=True)
    print(
        f"BioEmu candidates/samples/max-per-site: "
        f"{'adaptive 20-30' if args.bioemu_top_n == 0 else args.bioemu_top_n}/"
        f"{args.bioemu_num_samples}/{args.bioemu_max_per_site}",
        flush=True,
    )
    print(
        f"Rosetta candidates: all BioEmu checkpoint candidates; "
        f"final recommendation budget={'adaptive 20-30' if args.rosetta_top_n == 0 else args.rosetta_top_n}",
        flush=True,
    )

    if args.api_key_env not in os.environ:
        print(f"ERROR: missing API key environment variable: {args.api_key_env}", flush=True)
        return 2
    if args.resume_existing_project and not (run_dir / "project_manifest.json").exists():
        print(
            "ERROR: --resume_existing_project requires an existing project_manifest.json.",
            flush=True,
        )
        return 2
    if args.reuse_existing_structure_prediction and not args.resume_existing_project:
        print(
            "ERROR: --reuse_existing_structure_prediction requires "
            "--resume_existing_project.",
            flush=True,
        )
        return 2
    if has_workflow_artifacts(run_dir) and not args.force_existing_project and not args.resume_existing_project:
        print(
            "ERROR: project already contains workflow artifacts. Use a new --project_name "
            "for a true benchmark run, or pass --force_existing_project intentionally.",
            flush=True,
        )
        return 2

    sequence = read_sequence(args.sequence, args.sequence_file)
    user_prompt = read_design_prompt(args.prompt, args.prompt_file)
    design_objective = args.design_objective or user_prompt or (
        f"Design state-compatible mutations for {args.protein_name} while preserving the intended allosteric switch behavior."
    )
    target_state_name = args.target_state_name or f"{args.protein_name}_target"
    reference_state_name = args.reference_state_name or f"{args.protein_name}_reference"
    mcp_config = args.mcp_config or os.environ.get("SWITCHTUNE_MCP_CONFIG", str(REPO / "mcp_config.example.yaml"))
    run_external = not args.no_run_external

    prompt = build_prompt(
        project_name=project_name,
        protein_name=args.protein_name,
        target_state_structure=args.target_state_structure,
        reference_state_structure=args.reference_state_structure,
        target_state_name=target_state_name,
        reference_state_name=reference_state_name,
        parse_chains=args.parse_chains,
        mutation_region="",
        first_residue_number=args.first_residue_number,
        model_type=args.model_type,
        user_prompt=user_prompt,
        design_objective=design_objective,
        sequence=sequence,
        ligand_codes=args.ligand_codes,
        structure_engine=args.structure_engine,
        cb_selection_mode=args.cb_selection_mode,
        cb_score_min=args.cb_score_min,
        cb_score_max=args.cb_score_max,
        bioemu_top_n=args.bioemu_top_n,
        bioemu_max_per_site=args.bioemu_max_per_site,
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    invocation = {
        "status": "prepared",
        "agent_go_called": False,
        "resumed_existing_project": bool(args.resume_existing_project),
        "project_name": project_name,
        "prompt_path": str(prompt_path),
        "user_prompt": user_prompt,
        "prompt_source": (
            "argument" if user_prompt and args.prompt
            else "file" if user_prompt and args.prompt_file
            else "compatibility_default"
        ),
        "design_objective": design_objective,
        "prepared_at": utc_now(),
    }
    write_json(invocation_path, invocation)

    from checkpoint_workflow_runner import run_checkpoint_workflow  # noqa: E402
    from biomni_runtime import (  # noqa: E402
        build_agentic_review_prompt,
        make_biomni_agent,
        read_latest_final_round_decision,
        read_top_candidates,
        run_agent_session,
    )

    print_header("Initialize Biomni A1")
    agent = make_biomni_agent(
        biomni_data_path=args.biomni_data_path,
        llm=args.llm,
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
        source=args.llm_source,
        mcp_config=mcp_config,
        timeout_seconds=args.agent_timeout_seconds,
        max_steps=args.max_steps,
        max_execute_calls=args.max_execute_calls,
        recursion_limit=args.recursion_limit,
        thread_id=42,
        debug_raw_agent=args.debug_raw_agent,
        checkpoint_mode=True,
        review_mode=False,
        cb_timeout_seconds=args.cb_timeout_seconds,
        workflow_timeout_seconds=args.workflow_timeout_seconds,
    )
    review_agent = make_biomni_agent(
        biomni_data_path=args.biomni_data_path,
        llm=args.llm,
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
        source=args.llm_source,
        mcp_config=mcp_config,
        timeout_seconds=args.agent_timeout_seconds,
        max_steps=min(args.max_steps, 96),
        max_execute_calls=min(args.max_execute_calls, 24),
        recursion_limit=max(args.recursion_limit, 120),
        thread_id=43,
        debug_raw_agent=args.debug_raw_agent,
        checkpoint_mode=False,
        review_mode=True,
        cb_timeout_seconds=args.cb_timeout_seconds,
        workflow_timeout_seconds=args.workflow_timeout_seconds,
    )

    timings: dict[str, float] = {}
    started_total = time.perf_counter()
    print_header("Natural-Language Prompt Sent to agent.go()")
    print(prompt, flush=True)

    write_json(
        invocation_path,
        {
            **invocation,
            "status": "started",
            "agent_go_called": not args.resume_existing_project,
            "agent_go_callable": callable(getattr(agent, "go", None)),
            "resume_mode": bool(args.resume_existing_project),
            "started_at": utc_now(),
        },
    )
    if args.resume_existing_project:
        print(
            "Resume mode: skipping agent.go, inspection, and setup; using the existing project manifest.",
            flush=True,
        )
        log = []
        answer = "<solution>Resumed from the existing setup checkpoint.</solution>"
        agent_exception = ""
        agent_elapsed = 0.0
        timings["initial_agent_go_seconds"] = 0.0
    else:
        started = time.perf_counter()
        log, answer, agent_exception, agent_elapsed = run_agent_session(
            agent=agent,
            prompt=prompt,
            log_path=agent_log_path,
            title="Run agent.go (formal benchmark front half)",
        )
        timings["initial_agent_go_seconds"] = round(time.perf_counter() - started, 3)
    agent_exception_before_recovery = agent_exception
    agent_recovery: dict[str, Any] = {}
    if agent_exception:
        recovery = recover_front_half_after_llm_failure(
            project_name=project_name,
            protein_name=args.protein_name,
            target_state_structure=args.target_state_structure,
            reference_state_structure=args.reference_state_structure,
            target_state_name=target_state_name,
            reference_state_name=reference_state_name,
            parse_chains=args.parse_chains,
            sequence=sequence,
            first_residue_number=args.first_residue_number,
            model_type=args.model_type,
            ligand_codes=args.ligand_codes,
            design_objective=design_objective,
            error_text=agent_exception,
            mutation_region="",
        )
        if recovery and recovery.get("status") == "recovered_without_llm_continuation":
            agent_recovery = recovery
            front_half = continue_front_half_after_llm_recovery(
                recovery=recovery,
                project_name=project_name,
                target_state_structure=args.target_state_structure,
                reference_state_structure=args.reference_state_structure,
                target_state_name=target_state_name,
                reference_state_name=reference_state_name,
                cb_selection_mode=args.cb_selection_mode,
                cb_score_min=args.cb_score_min,
                cb_score_max=args.cb_score_max,
                structure_engine=args.structure_engine,
                ligand_codes=args.ligand_codes,
                structure_top_n=args.structure_top_n,
                cb_timeout_seconds=args.cb_timeout_seconds,
                workflow_timeout_seconds=args.workflow_timeout_seconds,
            )
            agent_recovery["front_half"] = front_half
            if front_half.get("status") == "recovered_front_half_completed":
                agent_exception = ""
                print(
                    "LLM continuation failed; recovered and completed the SwitchTune front half "
                    "directly from WT structure inspection. "
                    f"mutation_region={recovery.get('mutation_region')}",
                    flush=True,
                )
            else:
                agent_exception = str(
                    front_half.get("error")
                    or "SwitchTune front-half recovery did not complete."
                )
                print(f"RECOVERY_FRONT_HALF_EXCEPTION: {agent_exception}", flush=True)
        elif recovery:
            agent_recovery = recovery
    write_json(
        invocation_path,
        {
            **read_json(invocation_path),
            "status": (
                "recovered_without_llm_continuation"
                if agent_recovery.get("status") == "recovered_without_llm_continuation"
                and not agent_exception
                else "completed" if not agent_exception else "failed"
            ),
            "agent_go_exception": agent_exception,
            "agent_go_exception_before_recovery": agent_exception_before_recovery,
            "agent_recovery": agent_recovery,
            "failure_category": classify_agent_failure(agent_exception),
            "elapsed_seconds": round(agent_elapsed, 1),
            "completed_at": utc_now(),
        },
    )

    # A successful HTTP/LLM call is not sufficient evidence that the formal
    # SwitchTune workflow started. Detect a normal return after a legacy
    # direct-tool path or an early setup error before invoking checkpoint
    # stages with an empty project.
    manifest_after_agent = read_json(run_dir / "project_manifest.json")
    if manifest_after_agent and (user_prompt or args.design_objective):
        manifest_after_agent.update(
            {
                "user_prompt": user_prompt,
                "prompt_source": (
                    "argument" if user_prompt and args.prompt
                    else "file" if user_prompt and args.prompt_file
                    else "compatibility_objective"
                ),
                "design_objective": design_objective,
            }
        )
        write_json(run_dir / "project_manifest.json", manifest_after_agent)
    guard_event_log_after_agent = report_dir / "switchtune_go_guard_full_workflow_events.jsonl"
    guard_event_text_after_agent = (
        guard_event_log_after_agent.read_text(encoding="utf-8", errors="replace")
        if guard_event_log_after_agent.exists()
        else ""
    )
    formal_handoff_detected = (
        bool(manifest_after_agent.get("cb_job_manifest_path"))
        or bool(manifest_after_agent.get("cb_run_script"))
        or '"stage": "setup_two_state_cb_scoring_project_once"' in guard_event_text_after_agent
        or '"stage": "execute_conformation_bias_scoring_job"' in guard_event_text_after_agent
        or (
            (run_dir / "01_saturation_library").exists()
            and (run_dir / "02_conformation_bias_scoring").exists()
        )
    )
    if not agent_exception and not formal_handoff_detected:
        agent_exception = (
            "Biomni agent.go() returned without entering the guarded SwitchTune "
            "workflow: no setup_two_state_cb_scoring_project_once handoff was recorded."
        )
        print(f"AGENT_HANDOFF_EXCEPTION: {agent_exception}", flush=True)
        write_json(
            invocation_path,
            {
                **read_json(invocation_path),
                "status": "failed",
                "agent_go_exception": agent_exception,
                "handoff_detected": False,
                "completed_at": utc_now(),
            },
        )

    # Some otherwise valid agent sessions return immediately after
    # setup_two_state_cb_scoring_project_once.  That leaves the CB manifest in
    # place but no state-specific candidate CSV, so the checkpoint runner cannot
    # start.  Resume the deterministic front half from the existing setup
    # manifest without repeating inspection or setup.  This is also idempotent:
    # execute_conformation_bias_scoring_job skips completed outputs.
    if not agent_exception and formal_handoff_detected:
        candidate_csv_paths = (
            run_dir / "03_state_specific_candidates" / "state_specific_retained_candidates.csv",
            run_dir / "03_state_specific_candidates" / "state_specific_retained_candidates_adaptive.csv",
        )
        if not any(path.exists() for path in candidate_csv_paths):
            setup_manifest_path = str(manifest_after_agent.get("cb_job_manifest_path") or "")
            if setup_manifest_path:
                print(
                    "Setup checkpoint exists but CB-selected candidates are missing; "
                    "resuming deterministic front half from the existing manifest.",
                    flush=True,
                )
                recovery = {
                    "status": "setup_completed_without_front_half",
                    "setup": {
                        "project_name": project_name,
                        "cb_job_manifest_path": setup_manifest_path,
                    },
                }
                front_half_started = time.perf_counter()
                front_half = continue_front_half_after_llm_recovery(
                    recovery=recovery,
                    project_name=project_name,
                    target_state_structure=args.target_state_structure,
                    reference_state_structure=args.reference_state_structure,
                    target_state_name=target_state_name,
                    reference_state_name=reference_state_name,
                    cb_selection_mode=args.cb_selection_mode,
                    cb_score_min=args.cb_score_min,
                    cb_score_max=args.cb_score_max,
                    structure_engine=args.structure_engine,
                    ligand_codes=args.ligand_codes,
                    structure_top_n=args.structure_top_n,
                    cb_timeout_seconds=args.cb_timeout_seconds,
                    workflow_timeout_seconds=args.workflow_timeout_seconds,
                )
                timings["deterministic_front_half_resume_seconds"] = round(
                    time.perf_counter() - front_half_started, 3
                )
                agent_recovery = {
                    **agent_recovery,
                    "status": "resumed_from_completed_setup",
                    "setup_manifest_path": setup_manifest_path,
                    "front_half": front_half,
                }
                if front_half.get("status") != "recovered_front_half_completed":
                    agent_exception = str(
                        front_half.get("error")
                        or "Deterministic front-half resume did not complete."
                    )
                    print(f"FRONT_HALF_RESUME_EXCEPTION: {agent_exception}", flush=True)
                else:
                    manifest_after_agent = read_json(run_dir / "project_manifest.json")
                    print(
                        "Deterministic front-half resume completed; proceeding to checkpoint workflow.",
                        flush=True,
                    )
                write_json(
                    invocation_path,
                    {
                        **read_json(invocation_path),
                        "status": "completed" if not agent_exception else "failed",
                        "agent_go_exception": agent_exception,
                        "agent_recovery": agent_recovery,
                        "handoff_detected": formal_handoff_detected,
                        "completed_at": utc_now(),
                    },
                )

    checkpoint_report: dict[str, Any] = {}
    if not agent_exception:
        started = time.perf_counter()
        try:
            existing_manifest = read_json(run_dir / "project_manifest.json")
            reuse_structure = bool(
                args.resume_existing_project
                and args.reuse_existing_structure_prediction
            )
            existing_prediction_dir = str(
                existing_manifest.get("structure_prediction_output_dir") or ""
            )
            existing_job_manifest = str(
                existing_manifest.get("structure_prediction_job_manifest_path") or ""
            )
            checkpoint_report = run_checkpoint_workflow(
                project_name=project_name,
                review_agent=review_agent,
                target_state_structure=args.target_state_structure,
                reference_state_structure=args.reference_state_structure,
                target_state_name=target_state_name,
                reference_state_name=reference_state_name,
                parse_chains=args.parse_chains,
                # Legacy target_count is only a fallback for structure
                # selection. Rosetta's final top-N must not leak into it.
                target_count=min(args.bioemu_top_n, 30),
                bioemu_top_n=args.bioemu_top_n,
                bioemu_num_samples=args.bioemu_num_samples,
                bioemu_batch_size=args.bioemu_batch_size,
                bioemu_timeout_seconds=args.bioemu_timeout_seconds,
                rosetta_top_n=args.rosetta_top_n,
                run_external=run_external,
                timeout_seconds=args.workflow_timeout_seconds,
                cb_timeout_seconds=args.cb_timeout_seconds,
                ligand_codes=args.ligand_codes,
                structure_top_n=args.structure_top_n,
                review_top_n=args.review_top_n,
                structure_engine=args.structure_engine,
                max_per_mutation_site=args.bioemu_max_per_site,
                reuse_structure_prediction=reuse_structure,
                existing_structure_prediction_dir=(existing_prediction_dir or None),
                existing_structure_job_manifest=(existing_job_manifest or None),
            )
        except Exception as exc:
            checkpoint_report = {"status": "failed", "error": repr(exc)}
            print(f"CHECKPOINT_WORKFLOW_EXCEPTION: {repr(exc)}", flush=True)
        timings["checkpoint_workflow_seconds"] = round(time.perf_counter() - started, 3)
    else:
        checkpoint_report = {"status": "failed", "error": agent_exception}
        timings["checkpoint_workflow_seconds"] = 0.0

    final_csv = run_dir / "08_final_recommendation" / "final_candidate_recommendation.csv"
    merged_csv = run_dir / "08_final_recommendation" / "switchtune_evidence_merged.csv"
    decision_log = run_dir / "decision_log.jsonl"
    final_review_passed = False
    final_decision: dict[str, Any] = {}
    final_review_exception = ""
    final_review_answer = ""
    final_review_prompt_path = report_dir / f"{project_name}_final_round_review_prompt.md"

    if checkpoint_report.get("status") == "completed" and final_csv.exists():
        from switchtune.mcp_server import prepare_agentic_candidate_review  # noqa: E402

        started = time.perf_counter()
        review_packet = prepare_agentic_candidate_review(
            project_name=project_name,
            evidence_csv=str(merged_csv),
            top_n=max(1, csv_row_count(final_csv)),
            include_stage_reviews=True,
        )
        final_review_prompt = build_agentic_review_prompt(
            project_name=project_name,
            merged_csv=str(merged_csv),
            final_csv=str(final_csv),
            review_packet_json=str(review_packet.get("packet_json") or ""),
            review_packet=review_packet,
            top_candidates=read_top_candidates(final_csv, limit=max(1, csv_row_count(final_csv))),
        )
        final_review_prompt_path.write_text(final_review_prompt, encoding="utf-8")
        review_agent._switchtune_thread_id = f"switchtune-{project_name}-final-round"
        final_log, final_review_answer, final_review_exception, _elapsed = run_agent_session(
            agent=review_agent,
            prompt=final_review_prompt,
            log_path=final_review_log_path,
            title="Run agent.go (final-round Biomni review)",
        )
        final_log_text = "\n".join(map(str, final_log))
        final_decision = read_latest_final_round_decision(decision_log)
        final_review_passed = (
            not final_review_exception
            and "Traceback" not in final_log_text
            and "record_agentic_candidate_decision" in final_log_text
            and bool(final_decision)
        )
        timings["final_round_review_seconds"] = round(time.perf_counter() - started, 3)
    else:
        timings["final_round_review_seconds"] = 0.0

    resolved_expected_bioemu = int(
        checkpoint_report.get("bioemu_top_n")
        or args.bioemu_top_n
        or (checkpoint_report.get("candidate_counts") or {}).get("bioemu_candidates")
        or 20
    )
    resolved_expected_rosetta = int(
        checkpoint_report.get("rosetta_top_n")
        or args.rosetta_top_n
        or (checkpoint_report.get("candidate_counts") or {}).get("final_recommendation_candidates")
        or 20
    )
    validation = build_validation(
        project_name=project_name,
        run_dir=run_dir,
        checkpoint_report=checkpoint_report,
        final_review_passed=final_review_passed,
        expected_bioemu_top_n=resolved_expected_bioemu,
        expected_bioemu_num_samples=args.bioemu_num_samples,
        expected_rosetta_top_n=resolved_expected_rosetta,
        min_structure_candidates=args.min_structure_candidates,
        expected_bioemu_max_per_site=args.bioemu_max_per_site,
    )
    write_json(validation_path, validation)

    timings["total_seconds"] = round(time.perf_counter() - started_total, 3)
    guard_event_log = report_dir / "switchtune_go_guard_full_workflow_events.jsonl"
    guard_event_text = guard_event_log.read_text(encoding="utf-8", errors="replace") if guard_event_log.exists() else ""
    failure_category = classify_agent_failure(agent_exception or final_review_exception)
    summary_status = (
        failure_category
        if failure_category
        else (
            "passed"
            if validation["status"] == "passed" and not agent_exception and final_review_passed
            else "failed"
        )
    )
    summary = {
        "status": summary_status,
        "project_name": project_name,
        "workflow": "SwitchTune formal Biomni agent benchmark",
        "parameters": {
            "protein_name": args.protein_name,
            "user_prompt": user_prompt,
            "design_objective": design_objective,
            "structure_engine": args.structure_engine,
            "cb_selection_mode": args.cb_selection_mode,
            "cb_score_min": args.cb_score_min,
            "cb_score_max": args.cb_score_max,
            "structure_top_n": args.structure_top_n,
            "bioemu_top_n": args.bioemu_top_n,
            "bioemu_num_samples": args.bioemu_num_samples,
            "bioemu_batch_size": args.bioemu_batch_size,
            "bioemu_max_per_site": args.bioemu_max_per_site,
            "rosetta_top_n": args.rosetta_top_n,
            "rosetta_input_policy": "all_bioemu_checkpoint_candidates_ranking_only",
            "final_recommendation_policy": "rosetta_top_n_after_evidence_merge",
            "require_pae": True,
            "reuse_existing_structure_prediction": bool(args.reuse_existing_structure_prediction),
            "run_external": run_external,
        },
        "timings_seconds": timings,
        "candidate_counts": checkpoint_report.get("candidate_counts") or {},
        "checkpoint_status": checkpoint_report.get("status"),
        "checkpoint_report_path": str(report_dir / "checkpoint_workflow_summary.json"),
        "validation_path": str(validation_path),
        "prompt_path": str(prompt_path),
        "agent_log_path": str(agent_log_path),
        "final_review_prompt_path": str(final_review_prompt_path),
        "final_review_log_path": str(final_review_log_path),
        "agent_go_invocation_path": str(invocation_path),
        "decision_log_path": str(decision_log),
        "guard_event_log_path": str(guard_event_log),
        "guard_tool_mentions": count_tool_mentions(guard_event_text),
        "initial_agent_exception": agent_exception,
        "initial_agent_exception_before_recovery": agent_exception_before_recovery,
        "agent_recovery": agent_recovery,
        "failure_category": failure_category,
        "resume_guidance": (
            "Project setup may already be complete. Restore external LLM billing, "
            "then resume from project_manifest.json; do not repeat inspection or setup."
            if failure_category == "blocked_external_llm_billing"
            else ""
        ),
        "final_review_exception": final_review_exception,
        "latest_final_round_decision": final_decision,
        "final_review_passed": final_review_passed,
        "final_recommendation_csv": str(final_csv),
        "final_recommendation_count": csv_row_count(final_csv),
        "answer": answer,
        "final_review_answer": final_review_answer,
        "updated_at": utc_now(),
    }
    write_json(summary_path, summary)

    print_header("SwitchTune Formal Agent Benchmark Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    print_header("Validation")
    print(json.dumps(validation, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)
    print(f"Validation JSON: {validation_path}", flush=True)
    print("SWITCHTUNE_AGENT_BENCHMARK: " + ("PASSED" if summary["status"] == "passed" else "FAILED"), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
