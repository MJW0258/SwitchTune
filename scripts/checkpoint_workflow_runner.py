from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import time
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMPTY_DECISION_TOKENS = {
    "NONE",
    "EMPTY",
    "NO_CANDIDATES",
    "REJECT_ALL",
    "REJECT-ALL",
    "[]",
    "[ ]",
}


def _safe_float(value: Any) -> Optional[float]:
    """Return a finite float for optional CSV/JSON metrics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


from switchtune.mcp_server import (
    collect_bioemu_outputs,
    collect_inverse_folding_scores,
    collect_rosetta_scores,
    collect_structure_prediction_results,
    adaptive_candidate_budget,
    compare_structure_predictions_to_states,
    filter_structure_confidence,
    merge_switchtune_evidence_tables,
    prepare_bioemu_sampling_job,
    prepare_final_candidate_recommendation,
    prepare_inverse_folding_scoring_job,
    prepare_rosetta_energy_job,
    prepare_structure_prediction_job,
    recommend_next_workflow_step,
    review_switchtune_stage_evidence,
    select_rows_with_site_diversity,
    select_candidates_from_evidence,
    select_rosetta_analysis_strategy,
    select_structure_prediction_engine,
    standardize_structure_prediction_results,
)


def _runs_root() -> Path:
    return Path(os.environ.get("SWITCHTUNE_RUNS", "./SwitchTune_runs")).expanduser().resolve()


def _project_dir(project_name: str) -> Path:
    return _runs_root() / project_name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _csv_data_row_count(path_text: str) -> int:
    path = Path(path_text).expanduser()
    if not path.exists() or not path.is_file():
        return 0
    return len(_read_csv_rows(path)[0])


def _result_variant_count(result: dict[str, Any], *, evidence_csv: str = "") -> int:
    """Read variant counts across collectors with different legacy field names."""
    for key in (
        "evidence_variant_count",
        "variant_count",
        "score_row_count",
        "scored_variant_count",
        "completed_count",
        "valid_structure_count",
    ):
        value = result.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
    return _csv_data_row_count(evidence_csv) if evidence_csv else 0


def _variant_from_row(row: dict[str, Any]) -> str:
    value = row.get("variant") or row.get("name") or row.get("id")
    if value:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    pos = row.get("position")
    wt = row.get("wt_aa")
    mut = row.get("mut_aa")
    if pos and wt and mut:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{wt}{pos}{mut}").strip("_")
    return ""


def _preview_lines(path: str, limit: int = 5, *, field_whitelist: Optional[list[str]] = None) -> list[str]:
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        return ["- No candidate table was available."]
    rows, _fields = _read_csv_rows(csv_path)
    lines: list[str] = []
    for row in rows[:limit]:
        visible_items = row.items() if not field_whitelist else ((key, row.get(key, "")) for key in field_whitelist)
        line_text = ", ".join(
            f"{key}={value}"
            for key, value in visible_items
            if value not in {"", None}
        )
        lines.append(f"- {line_text}" if line_text else "- (empty row)")
    if not lines:
        lines.append("- No readable candidates.")
    return lines


def _filter_csv_by_variants(input_csv: str, selected_variants: Iterable[str], *, suffix: str) -> tuple[str, dict[str, Any]]:
    csv_path = Path(input_csv).expanduser()
    if not csv_path.exists():
        return input_csv, {
            "status": "missing_input",
            "input_csv": input_csv,
            "selected_variants": [],
            "reason": "candidate table not found",
        }

    selected = [
        str(item).strip()
        for item in selected_variants
        if str(item).strip() and str(item).strip().upper() not in EMPTY_DECISION_TOKENS
    ]
    rows, fieldnames = _read_csv_rows(csv_path)
    if not selected:
        locked_path = csv_path.with_name(f"{csv_path.stem}_{suffix}.csv")
        fieldnames = fieldnames or ["variant", "sequence"]
        _write_csv_rows(locked_path, [], fieldnames)
        return str(locked_path), {
            "status": "blocked_no_candidates",
            "input_csv": input_csv,
            "locked_csv": str(locked_path),
            "selected_variants": [],
            "retained_count": 0,
            "reason": "No shortlist was recorded by the checkpoint reviewer.",
        }

    selected_set = set(selected)
    retained = [row for row in rows if _variant_from_row(row) in selected_set]
    if not retained:
        locked_path = csv_path.with_name(f"{csv_path.stem}_{suffix}.csv")
        fieldnames = fieldnames or ["variant", "sequence"]
        _write_csv_rows(locked_path, [], fieldnames)
        return str(locked_path), {
            "status": "blocked_no_candidates",
            "input_csv": input_csv,
            "locked_csv": str(locked_path),
            "selected_variants": selected,
            "retained_count": 0,
            "reason": "The recorded shortlist did not match any current candidate row.",
        }

    locked_path = csv_path.with_name(f"{csv_path.stem}_{suffix}.csv")
    fieldnames = fieldnames or list(retained[0].keys())
    _write_csv_rows(locked_path, retained, fieldnames)
    return str(locked_path), {
        "status": "locked",
        "input_csv": input_csv,
        "locked_csv": str(locked_path),
        "selected_variants": selected,
        "retained_count": len(retained),
    }


def _repair_checkpoint_lock_for_site_diversity(
    *,
    current_candidates_csv: str,
    locked_csv: str,
    preferred_variants: Iterable[str],
    target_count: int,
    max_per_mutation_site: int,
    stage_name: str,
    allow_review_pool_backfill: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Repair a valid Biomni lock without changing the scientific evidence.

    Biomni remains responsible for the evidence-based shortlist. This guard
    only replaces excess variants from an overrepresented residue with
    candidates from the same valid checkpoint pool, prioritizing one variant
    per available site before filling the remaining budget.
    """
    current_path = Path(current_candidates_csv).expanduser()
    locked_path = Path(locked_csv).expanduser()
    rows, fieldnames = _read_csv_rows(current_path)
    preferred = [str(item).strip() for item in preferred_variants if str(item).strip()]
    if not rows:
        return str(locked_path), {
            "status": "blocked_no_candidates",
            "stage_name": stage_name,
            "policy": "site_round_robin_then_rank",
            "target_count": 0,
            "max_per_mutation_site": int(max_per_mutation_site),
            "reason": "The checkpoint candidate table is empty.",
        }

    preferred_set = set(preferred)
    selection_rows = rows
    if not allow_review_pool_backfill:
        selection_rows = [
            row
            for row in rows
            if str(row.get("structure_confidence_gate") or "").strip().lower()
            not in {"agent_review", "relative_fallback_review"}
            or _variant_from_row(row) in preferred_set
        ]
    effective_target = min(max(1, int(target_count)), len(selection_rows))
    repaired_rows, diversity_info = select_rows_with_site_diversity(
        selection_rows,
        target_count=effective_target,
        max_per_mutation_site=int(max_per_mutation_site),
        preferred_variants=preferred,
    )
    output_fields = list(fieldnames)
    for row in repaired_rows:
        for key in row:
            if key not in output_fields:
                output_fields.append(key)
    _write_csv_rows(locked_path, repaired_rows, output_fields)
    return str(locked_path), {
        "status": "repaired",
        "stage_name": stage_name,
        "policy": diversity_info.get("policy"),
        "target_count": effective_target,
        "max_per_mutation_site": int(max_per_mutation_site),
        "allow_review_pool_backfill": bool(allow_review_pool_backfill),
        "selection_pool_count": len(selection_rows),
        "original_selected_variant_count": len(preferred),
        "original_selected_variants": preferred,
        "retained_count": len(repaired_rows),
        "retained_variants": [
            _variant_from_row(row)
            for row in repaired_rows
        ],
        "site_diversity": diversity_info,
    }


def _extract_latest_selected_variants(decision_log_path: Path) -> tuple[list[str], dict[str, Any]]:
    records = _read_agentic_decision_records(decision_log_path)
    latest = records[-1] if records else {}
    if not latest:
        return [], {}
    return _selected_variants_from_decision_record(latest), latest


def _read_agentic_decision_records(decision_log_path: Path) -> list[dict[str, Any]]:
    if not decision_log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in decision_log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except Exception:
            continue
        if isinstance(record, dict) and record.get("step_name") == "agentic_candidate_decision":
            records.append(record)
    return records


def _selected_variants_from_decision_record(record: dict[str, Any]) -> list[str]:
    selected: list[str] = []
    output_summary = record.get("output_summary") or {}
    decision_json = output_summary.get("decision_json")
    if decision_json:
        decision_path = Path(str(decision_json)).expanduser()
        if decision_path.exists():
            decision = _read_json(decision_path)
            raw_selected = decision.get("selected_variants") or []
            selected = _normalize_selected_variants(raw_selected)
    if not selected:
        raw_decision = str(record.get("decision") or "")
        selected = _normalize_selected_variants(
            item.strip() for item in re.split(r"[,;\s]+", raw_decision) if item.strip()
        )
    return selected


def _current_candidate_variants(path: str) -> set[str]:
    csv_path = Path(path).expanduser()
    if not csv_path.exists() or not csv_path.is_file():
        return set()
    try:
        rows, _fields = _read_csv_rows(csv_path)
    except Exception:
        return set()
    return {
        variant
        for row in rows
        if (variant := _variant_from_row(row))
    }


def _find_valid_checkpoint_decision(
    records: list[dict[str, Any]],
    *,
    stage_name: str,
    current_candidates_csv: str,
    evidence_packet_json: str = "",
    minimum_selected_count: int = 1,
    allow_near_complete: bool = False,
) -> tuple[dict[str, Any], list[str], str]:
    """Return the newest decision that is valid for this exact checkpoint."""
    expected_reviewer = f"biomni_{stage_name}_checkpoint".lower()
    current_variants = _current_candidate_variants(current_candidates_csv)
    if not current_variants:
        return {}, [], "The current checkpoint candidate table is empty or unreadable."
    required_selected_count = max(
        1,
        min(int(minimum_selected_count or 1), len(current_variants)),
    )

    saw_expected_record = False
    for record in reversed(records):
        reviewer = str(record.get("reviewer") or "").strip().lower()
        scope = str(record.get("decision_scope") or "").strip().lower()
        if reviewer != expected_reviewer or scope != "checkpoint":
            continue
        if evidence_packet_json:
            record_evidence = str(
                (record.get("input_summary") or {}).get("evidence_packet_json") or ""
            )
            if record_evidence and record_evidence != str(evidence_packet_json):
                continue
        saw_expected_record = True
        selected = _selected_variants_from_decision_record(record)
        if any(
            token.upper() in {"ALL", "ALL_CURRENT_CANDIDATES", "CURRENT_CANDIDATES"}
            for token in selected
        ):
            # Ranking-only checkpoints may use a compact, auditable shorthand
            # instead of forcing the LLM to echo a 20-30 item variant list.
            selected = sorted(current_variants)
        if not selected:
            return {}, [], "The checkpoint decision explicitly selected no candidates."
        if len(selected) < required_selected_count:
            # A strict structure-confidence gate can legitimately leave fewer
            # than the nominal 20-candidate formal budget.  If the reviewer
            # already recorded a shortlist that is short by only one row, keep
            # the decision auditable and let the deterministic site-diversity
            # repair choose the missing row from the same current pool.  This
            # avoids retrying an idempotent decision tool forever.
            if (
                allow_near_complete
                and len(current_variants) <= 20
                and len(selected) >= max(1, required_selected_count - 1)
            ):
                return record, selected, (
                    "Accepted a near-complete shortlist; deterministic guard "
                    "may backfill from the same current candidate pool."
                )
            return {}, [], (
                "The checkpoint decision selected too few candidates for the "
                f"downstream budget: selected {len(selected)}, required at least "
                f"{required_selected_count} from {len(current_variants)} available."
            )
        if not set(selected).issubset(current_variants):
            unknown = sorted(set(selected) - current_variants)
            return {}, [], (
                "The checkpoint decision selected variants that are not present in "
                f"the current candidate table: {', '.join(unknown)}."
            )
        return record, selected, ""

    if saw_expected_record:
        return {}, [], "The checkpoint decision was not valid for the current candidates."
    return {}, [], (
        "No decision record matched the required reviewer/scope: "
        f"reviewer={expected_reviewer}, decision_scope=checkpoint."
    )


def _normalize_selected_variants(values: Iterable[Any]) -> list[str]:
    if isinstance(values, str):
        text = values.strip()
        if text in {"[]", "[ ]", "[\n]"}:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            values = parsed
        else:
            cleaned = text.strip("[]").replace('"', " ").replace("'", " ")
            values = re.split(r"[,;\s]+", cleaned) if cleaned else []

    tokens = []
    for item in values:
        token = str(item).strip().strip("[]").strip("\"'")
        if token:
            tokens.append(token)
    return [item for item in tokens if item.upper() not in EMPTY_DECISION_TOKENS]


def _run_agent_session(
    *,
    agent: Any,
    prompt: str,
    log_path: Path,
    title: str,
) -> tuple[list[str], str, str, float]:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    start = time.time()
    try:
        log, answer = agent.go(prompt)
        agent_exception = ""
    except Exception as exc:
        log = getattr(agent, "log", []) or []
        answer = ""
        agent_exception = repr(exc)
        print(f"AGENT_EXCEPTION: {agent_exception}", flush=True)
    elapsed = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(("\n\n" + "=" * 100 + "\n\n").join(map(str, log)), encoding="utf-8")
    return log, answer, agent_exception, elapsed


def _build_checkpoint_review_prompt(
    *,
    project_name: str,
    stage_name: str,
    evidence_csv: str,
    current_candidates_csv: str,
    top_candidate_lines: list[str],
    stage_review: Optional[dict[str, Any]] = None,
    minimum_selected_count: int = 1,
    diversity_target_count: Optional[int] = None,
    max_per_mutation_site: Optional[int] = None,
    rescue_candidate_lines: Optional[list[str]] = None,
) -> str:
    stage_focus = {
        "structure_confidence": "structure prediction quality, target-likeness, pLDDT, and PAE",
        "inverse_folding": "inverse-folding compatibility and sequence-structure agreement",
        "bioemu": "dynamic support from BioEmu sampling outputs",
        "rosetta": "Rosetta energy support and any energy-based cautions",
        "final": "the full evidence stack across all completed stages",
        "all": "the full evidence stack across all completed stages",
    }.get(stage_name, "the available evidence")

    top_lines = top_candidate_lines or ["- No ranked candidates were available."]
    minimum_selected_count = max(1, int(minimum_selected_count or 1))
    available_candidate_count = len(
        [line for line in top_candidate_lines if str(line).strip()]
    )
    effective_required_count = min(
        minimum_selected_count,
        available_candidate_count,
    ) if available_candidate_count else minimum_selected_count
    if stage_name in {"bioemu", "rosetta"}:
        budget_rule = (
            f"- Retain all {minimum_selected_count} current candidates unless a row is a hard execution failure; "
            "this checkpoint is ranking/evidence review, not a population-size bottleneck.\n"
            "- Do not prune candidates merely because one BioEmu/Rosetta metric is missing; record the missing "
            "evidence and keep the candidate for downstream ranking.\n"
            if minimum_selected_count > 1
            else
            "- This checkpoint is ranking/evidence review, not a population-size bottleneck. "
            "Do not prune candidates merely because one metric is missing.\n"
        )
    else:
        budget_rule = (
            f"- Select at least {effective_required_count} current candidates. The current table contains "
            f"{available_candidate_count or 'an unknown number of'} candidates; if it contains fewer than "
            f"{minimum_selected_count}, select every available candidate rather than inventing variants. "
            "this preserves the downstream formal benchmark budget while still allowing evidence-based pruning.\n"
            if minimum_selected_count > 1
            else ""
        )
    if (
        diversity_target_count is not None
        and max_per_mutation_site is not None
        and stage_name in {"structure_confidence", "inverse_folding", "bioemu"}
    ):
        budget_rule += (
            f"- The downstream target is {int(diversity_target_count)} candidates with at most "
            f"{int(max_per_mutation_site)} variants per mutation site. Prefer site diversity when "
            "the evidence is otherwise comparable; the workflow guard will audit and repair "
            "over-concentrated shortlists from the same valid candidate pool.\n"
        )

    review_lines: list[str] = []
    review_json = ""
    if isinstance(stage_review, dict):
        review_json = str(stage_review.get("review_json") or "")
        for item in (stage_review.get("stage_reviews") or [])[:3]:
            if not isinstance(item, dict):
                continue
            review_lines.append(
                "- "
                + str(item.get("stage_name") or stage_name)
                + f": rows={item.get('row_count', 'unknown')}, metrics={', '.join(item.get('key_metric_columns') or [])}"
            )
            for finding in (item.get("findings") or [])[:3]:
                review_lines.append(f"  finding: {finding}")
            for warning in (item.get("warnings") or [])[:3]:
                review_lines.append(f"  caution: {warning}")
            top_rows = item.get("top_rows_by_metric") or {}
            if isinstance(top_rows, dict):
                for metric, rows in list(top_rows.items())[:2]:
                    variants = [
                        str(row.get("variant") or "")
                        for row in (rows or [])[:3]
                        if isinstance(row, dict) and row.get("variant")
                    ]
                    if variants:
                        review_lines.append(f"  top by {metric}: {', '.join(variants)}")
    if not review_lines:
        review_lines.append("- Prepared stage review was unavailable; use current candidate rows and evidence CSV path.")

    rescue_lines = rescue_candidate_lines or []
    rescue_section = ""
    if rescue_lines:
        rescue_section = (
            "\n\nAgent-review rescue pool (strict gate failures; explicit selection required):\n"
            + "\n".join(rescue_lines)
        )
    rescue_rule = (
        (
            "- Rescue-pool candidates are not automatically passed. Select a rescue-pool "
            "variant only when its global confidence, local pLDDT, PAE decomposition, "
            "target-like state, and upstream switchability evidence justify the exception.\n"
        )
        if rescue_lines
        else ""
    )

    return (
        f"You are the Biomni checkpoint reviewer for SwitchTune project `{project_name}`.\n\n"
        f"This checkpoint is about `{stage_name}`. Focus on {stage_focus}.\n"
        "The workflow harness has already prepared the evidence summary.\n"
        "Do not open the raw CSV or start a new evidence review.\n"
        "Decide from the prepared summary below and write the decision immediately.\n\n"
        "Required tool:\n"
        "- `record_agentic_candidate_decision` exactly once\n\n"
        f"Decision metadata: reviewer=\"biomni_{stage_name}_checkpoint\", decision_scope=\"checkpoint\".\n\n"
        "Hard rules:\n"
        "- Emit one `<execute>` block containing only the required decision-tool call; "
        "do not combine `<execute>` and `<solution>` in the same assistant message.\n"
        "- Wait for the tool observation before returning `<solution>`.\n"
        "- Do not call any stage-review or evidence-packet-preparation tool.\n"
        "- Do not restart from PDB inspection, CB setup, structure prediction, or inverse-folding setup.\n"
        "- This remains a blind test: use only the prepared WT-derived evidence and outputs from this run. Do not search or consult literature, mutation databases, benchmark answer keys, previous reports, remembered variants, or mutant structures. If literature evidence is requested, write `not assessed in blind test`.\n"
        + (
            "- Keep the shortlist conservative if the evidence is ambiguous.\n"
            "- If the evidence is strong, prune aggressively; if it is mixed, retain the safer candidates.\n"
            if stage_name not in {"bioemu", "rosetta"}
            else
            "- Preserve the current candidate population; use the evidence to rank and annotate candidates, "
            "not to create a new hard bottleneck.\n"
        )
        + rescue_rule
        + budget_rule
        + "- Record your decision before returning.\n\n"
        f"Evidence CSV: `{evidence_csv}`\n"
        f"Prepared review JSON: `{review_json}`\n"
        f"Current candidate CSV: `{current_candidates_csv}`\n\n"
        "Prepared stage-review summary:\n"
        + "\n".join(review_lines)
        + "\n\n"
        "Current candidates:\n"
        + "\n".join(top_lines)
        + rescue_section
        + "\n\n"
        "Suggested action sequence:\n"
        + (
            "1. For this ranking-only checkpoint, call "
            "`record_agentic_candidate_decision` with "
            "selected_variants=\"ALL_CURRENT_CANDIDATES\"; this explicitly preserves "
            "the complete current candidate set.\n"
            if stage_name in {"bioemu", "rosetta"}
            else
            f"1. Call `record_agentic_candidate_decision(project_name=\"{project_name}\", selected_variants=\"...\", decision_text=\"...\", evidence_packet_json=\"{review_json or evidence_csv}\", reviewer=\"biomni_{stage_name}_checkpoint\", decision_scope=\"checkpoint\")` once.\n"
        )
        + "2. Return a brief `<solution>` that states the chosen shortlist and why.\n"
    )


def _checkpoint_review(
    *,
    review_agent: Any,
    project_name: str,
    stage_name: str,
    evidence_csv: str,
    current_candidates_csv: str,
    review_log_dir: Path,
    top_n: int = 5,
    minimum_selected_count: int = 1,
    diversity_target_count: Optional[int] = None,
    max_per_mutation_site: Optional[int] = None,
) -> dict[str, Any]:
    prompt_path = review_log_dir / f"{stage_name}_checkpoint_prompt.md"
    log_path = review_log_dir / f"{stage_name}_checkpoint_review_log.txt"
    decision_log_path = _project_dir(project_name) / "decision_log.jsonl"
    if review_agent is not None:
        review_agent._switchtune_thread_id = (
            f"switchtune-{project_name}-{stage_name}-checkpoint"
        )
    decision_count_before = len(_read_agentic_decision_records(decision_log_path))
    stage_review = review_switchtune_stage_evidence(
        project_name=project_name,
        stage_name=stage_name,
        evidence_csv=evidence_csv,
        top_n=top_n,
    )
    if stage_name == "inverse_folding":
        field_whitelist = [
            "variant",
            "inverse_folding_compatibility_score",
            "inverse_folding_model_count",
            "structure_confidence_score",
            "structure_target_similarity_score",
            "bias_score_abs",
        ]
    elif stage_name == "bioemu":
        field_whitelist = [
            "variant",
            "bioemu_sampling_complete",
            "bioemu_metrics_status",
            "bioemu_error_summary",
            "bioemu_sample_count",
            "bioemu_complete_sample_fraction",
            "bioemu_target_like_fraction",
            "bioemu_target_preference_margin_A",
            "bioemu_rmsd_to_target_mean_A",
            "bioemu_mutation_window_rmsf_mean_A",
            "bioemu_pairwise_rmsd_mean_A",
            "bioemu_radius_gyration_mean_A",
            "structure_target_similarity_score",
            "inverse_folding_compatibility_score",
            "bias_score_abs",
        ]
    elif stage_name == "rosetta":
        field_whitelist = [
            "variant",
            "rosetta_primary_score_mean",
            "inverse_folding_compatibility_score",
            "structure_target_similarity_score",
            "bias_score_abs",
        ]
    else:
        field_whitelist = [
            "variant",
            "structure_confidence_gate",
            "structure_confidence_score",
            "structure_confidence_confidence_score",
            "structure_confidence_mutation_window_plddt_mean",
            "structure_confidence_pae_primary_metric",
        ]
    top_candidate_lines = _preview_lines(
        current_candidates_csv,
        limit=top_n,
        field_whitelist=field_whitelist,
    )
    rescue_candidate_lines: list[str] = []
    if stage_name == "structure_confidence":
        current_rows, _current_fields = _read_csv_rows(
            Path(current_candidates_csv).expanduser()
        )
        rescue_rows = [
            row
            for row in current_rows
            if str(row.get("structure_confidence_gate") or "").strip().lower()
            in {"agent_review", "relative_fallback_review"}
        ]
        rescue_rows.sort(
            key=lambda row: (
                _safe_float(
                    row.get("structure_confidence_pae_primary_metric")
                )
                if _safe_float(
                    row.get("structure_confidence_pae_primary_metric")
                )
                is not None
                else float("inf"),
                str(row.get("variant") or ""),
            )
        )
        for row in rescue_rows:
            visible_items = (
                ("variant", row.get("variant", "")),
                ("gate", row.get("structure_confidence_gate", "")),
                (
                    "confidence",
                    row.get("structure_confidence_confidence_score", ""),
                ),
                (
                    "local_plddt",
                    row.get("structure_confidence_mutation_window_plddt_mean", ""),
                ),
                (
                    "mutation_pae",
                    row.get("structure_confidence_mutation_mean_pae_to_protein", ""),
                ),
                (
                    "window_pae",
                    row.get("structure_confidence_pae_primary_metric", ""),
                ),
                ("state_qc", row.get("structure_confidence_state_qc", "")),
                ("bias_abs", row.get("bias_score_abs", "")),
            )
            rescue_candidate_lines.append(
                "- "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in visible_items
                    if value not in {"", None}
                )
            )
    prompt = _build_checkpoint_review_prompt(
        project_name=project_name,
        stage_name=stage_name,
        evidence_csv=evidence_csv,
        current_candidates_csv=current_candidates_csv,
        top_candidate_lines=top_candidate_lines,
        stage_review=stage_review,
        minimum_selected_count=minimum_selected_count,
        diversity_target_count=diversity_target_count,
        max_per_mutation_site=max_per_mutation_site,
        rescue_candidate_lines=rescue_candidate_lines,
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    log, answer, agent_exception, elapsed = _run_agent_session(
        agent=review_agent,
        prompt=prompt,
        log_path=log_path,
        title=f"Checkpoint review: {stage_name}",
    )
    retry_used = False
    retry_log_path = ""
    reused_existing_decision = False

    def evaluate_decision() -> tuple[
        list[dict[str, Any]], dict[str, Any], list[str], str
    ]:
        records = _read_agentic_decision_records(decision_log_path)
        new_records = records[decision_count_before:]
        valid_record, valid_selected, validation_reason = _find_valid_checkpoint_decision(
            new_records,
            stage_name=stage_name,
            current_candidates_csv=current_candidates_csv,
            evidence_packet_json=str(stage_review.get("review_json") or evidence_csv),
            minimum_selected_count=minimum_selected_count,
            allow_near_complete=stage_name in {"structure_confidence", "inverse_folding"},
        )
        return new_records, valid_record, valid_selected, validation_reason

    new_decisions, latest_decision, selected_variants, validation_reason = evaluate_decision()
    decision_recorded = bool(latest_decision)

    # Resumable runs should reuse a valid decision already recorded for this
    # exact stage/evidence packet instead of invoking the reviewer again. The
    # decision tool is intentionally idempotent, so a duplicate call would not
    # create a new record and would otherwise look like a missing decision.
    if not decision_recorded:
        existing_records = _read_agentic_decision_records(decision_log_path)
        (
            existing_decision,
            existing_selected,
            existing_reason,
        ) = _find_valid_checkpoint_decision(
            existing_records,
            stage_name=stage_name,
            current_candidates_csv=current_candidates_csv,
            evidence_packet_json=str(stage_review.get("review_json") or evidence_csv),
            minimum_selected_count=minimum_selected_count,
            allow_near_complete=stage_name in {"structure_confidence", "inverse_folding"},
        )
        if existing_decision:
            latest_decision = existing_decision
            selected_variants = existing_selected
            validation_reason = ""
            decision_recorded = True
            reused_existing_decision = True
            new_decisions = []
            log = [
                f"Reused existing valid checkpoint decision for {stage_name}: "
                f"{', '.join(selected_variants)}"
            ]
            answer = (
                f"Reused the existing Biomni {stage_name} checkpoint decision "
                "for this resumable workflow."
            )
            log_path.write_text("\n".join(log), encoding="utf-8")
        elif existing_reason and not validation_reason:
            validation_reason = existing_reason

    # Retry once on a fresh graph thread when the model returned an
    # exploratory response, malformed tool call, or text-only decision.
    # The retry explicitly forbids file inspection and requires a tool
    # observation before any solution text.
    if not decision_recorded and review_agent is not None:
        retry_used = True
        retry_log_path_obj = review_log_dir / f"{stage_name}_checkpoint_retry_1_log.txt"
        retry_prompt = (
            f"The previous `{stage_name}` checkpoint response did not create a valid "
            "candidate-decision record. This is a controlled retry.\n\n"
            "Do not inspect, open, or load any file. Do not use pandas, shell, or "
            "any exploratory Python code. Use only the prepared summary and current "
            "candidate list below.\n"
            "Your next response must contain exactly one `<execute>` block whose only "
            "tool action is `record_agentic_candidate_decision`. Do not include "
            "`<solution>` in the same message as the execute block. Wait for the "
            "tool observation, then return a brief solution.\n\n"
            f"Use reviewer=\"biomni_{stage_name}_checkpoint\" and "
            "decision_scope=\"checkpoint\". "
            + (
                "Select all current candidate variants. This is a ranking-only checkpoint; "
                "you may use selected_variants=\"ALL_CURRENT_CANDIDATES\"; do not prune "
                "the downstream candidate population.\n\n"
                if stage_name in {"bioemu", "rosetta"}
                else
                "Select a non-empty subset of the current candidate variants; retaining all current candidates "
                "is acceptable when the evidence does not justify pruning.\n\n"
            )
            + (
                f"The current candidate list contains {len(top_candidate_lines)} rows. Select at least "
                f"{min(max(1, int(minimum_selected_count or 1)), len(top_candidate_lines))} candidates; "
                "if fewer than the nominal budget are available, select all available candidates.\n\n"
                if max(1, int(minimum_selected_count or 1)) > 1
                else ""
            )
            + f"Project: `{project_name}`\n"
            f"Prepared review JSON: `{stage_review.get('review_json', evidence_csv)}`\n"
            "Current candidates:\n"
            + "\n".join(top_candidate_lines)
            + "\n\n"
            "Required call shape:\n"
            f"`record_agentic_candidate_decision(project_name=\"{project_name}\", "
            "selected_variants=\"...\", decision_text=\"...\", "
            f"evidence_packet_json=\"{stage_review.get('review_json', evidence_csv)}\", "
            f"reviewer=\"biomni_{stage_name}_checkpoint\", "
            "decision_scope=\"checkpoint\")`\n"
        )
        retry_prompt_path = review_log_dir / f"{stage_name}_checkpoint_retry_1_prompt.md"
        retry_prompt_path.write_text(retry_prompt, encoding="utf-8")
        retry_log_path = str(retry_log_path_obj)
        review_agent._switchtune_thread_id = (
            f"switchtune-{project_name}-{stage_name}-checkpoint-retry-1"
        )
        retry_log, retry_answer, retry_exception, retry_elapsed = _run_agent_session(
            agent=review_agent,
            prompt=retry_prompt,
            log_path=retry_log_path_obj,
            title=f"Checkpoint review retry: {stage_name}",
        )
        log = list(log) + [
            "",
            "=" * 100,
            f"CHECKPOINT REVIEW RETRY: {stage_name}",
            "=" * 100,
        ] + list(retry_log)
        answer = retry_answer or answer
        if retry_exception:
            agent_exception = retry_exception
        elapsed += retry_elapsed
        log_path.write_text(
            ("\n\n" + "=" * 100 + "\n\n").join(map(str, log)),
            encoding="utf-8",
        )
        new_decisions, latest_decision, selected_variants, validation_reason = evaluate_decision()
        decision_recorded = bool(latest_decision)

    review_log_text = "\n".join(map(str, log))
    decision_scope = str(latest_decision.get("decision_scope") or "").strip().lower()
    decision_reviewer = str(latest_decision.get("reviewer") or "").strip().lower()
    decision_selected_variants = list(selected_variants)
    locked_csv, lock_info = _filter_csv_by_variants(
        current_candidates_csv,
        decision_selected_variants,
        suffix=f"{stage_name}_checkpoint_locked",
    )
    effective_selected_variants = list(decision_selected_variants)
    if (
        decision_recorded
        and _csv_data_row_count(locked_csv) > 0
        and diversity_target_count is not None
        and max_per_mutation_site is not None
    ):
        locked_csv, diversity_lock_info = _repair_checkpoint_lock_for_site_diversity(
            current_candidates_csv=current_candidates_csv,
            locked_csv=locked_csv,
            preferred_variants=decision_selected_variants,
            target_count=int(diversity_target_count),
            max_per_mutation_site=int(max_per_mutation_site),
            stage_name=stage_name,
            allow_review_pool_backfill=stage_name != "structure_confidence",
        )
        effective_selected_variants = list(
            diversity_lock_info.get("retained_variants") or decision_selected_variants
        )
        lock_info["site_diversity_repair"] = diversity_lock_info
        lock_info["status"] = diversity_lock_info.get("status", lock_info.get("status"))
        lock_info["retained_count"] = diversity_lock_info.get(
            "retained_count",
            lock_info.get("retained_count", 0),
        )
    lock_info["decision_selected_variants"] = decision_selected_variants
    lock_info["decision_recorded"] = decision_recorded
    lock_info["new_agentic_decision_count"] = len(new_decisions)
    lock_info["selected_variant_count"] = len(effective_selected_variants)
    lock_info["minimum_selected_count"] = max(1, int(minimum_selected_count or 1))
    lock_info["diversity_target_count"] = (
        int(diversity_target_count)
        if diversity_target_count is not None
        else None
    )
    lock_info["max_per_mutation_site"] = (
        int(max_per_mutation_site)
        if max_per_mutation_site is not None
        else None
    )
    lock_info["expected_reviewer"] = f"biomni_{stage_name}_checkpoint"
    lock_info["expected_decision_scope"] = "checkpoint"
    lock_info["decision_validation_reason"] = validation_reason
    lock_info["retry_used"] = retry_used
    lock_info["reused_existing_decision"] = reused_existing_decision
    if not decision_recorded:
        lock_info["status"] = "blocked_no_candidates"
        lock_info["reason"] = validation_reason or "Checkpoint reviewer did not record a candidate decision."
    return {
        "stage_name": stage_name,
        "prompt_path": str(prompt_path),
        "log_path": str(log_path),
        "retry_log_path": retry_log_path,
        "reused_existing_decision": reused_existing_decision,
        "answer": answer,
        "agent_exception": agent_exception,
        "elapsed_seconds": round(elapsed, 1),
        "decision_recorded": decision_recorded,
        "new_agentic_decision_count": len(new_decisions),
        "review_tool_mentions": {
            "record_agentic_candidate_decision": review_log_text.count("record_agentic_candidate_decision"),
            "review_switchtune_stage_evidence": review_log_text.count("review_switchtune_stage_evidence"),
            "prepare_agentic_candidate_review": review_log_text.count("prepare_agentic_candidate_review"),
        },
        "decision_scope": decision_scope,
        "decision_reviewer": decision_reviewer,
        "selected_variants": effective_selected_variants,
        "minimum_selected_count": max(1, int(minimum_selected_count or 1)),
        "diversity_target_count": (
            int(diversity_target_count)
            if diversity_target_count is not None
            else None
        ),
        "max_per_mutation_site": (
            int(max_per_mutation_site)
            if max_per_mutation_site is not None
            else None
        ),
        "decision_log_record": latest_decision,
        "prepared_stage_review": stage_review,
        "locked_csv": locked_csv,
        "lock_info": lock_info,
        "status": "completed" if _csv_data_row_count(locked_csv) > 0 else "blocked_no_candidates",
        "log_lines": len(log),
    }


def run_checkpoint_workflow(
    *,
    project_name: str,
    review_agent: Any,
    target_state_structure: str,
    reference_state_structure: str,
    target_state_name: str,
    reference_state_name: str,
    parse_chains: str,
    target_count: int,
    bioemu_top_n: int,
    bioemu_num_samples: int,
    bioemu_batch_size: int,
    rosetta_top_n: int,
    run_external: bool,
    timeout_seconds: int,
    cb_timeout_seconds: int,
    bioemu_timeout_seconds: int = 1800,
    ligand_codes: str = "BEF,MN",
    structure_top_n: Optional[int] = None,
    review_top_n: int = 50,
    structure_engine: str = "auto",
    max_per_mutation_site: int = 6,
    reuse_structure_prediction: bool = False,
    existing_structure_prediction_dir: Optional[str] = None,
    existing_structure_job_manifest: Optional[str] = None,
) -> dict[str, Any]:
    if int(max_per_mutation_site) <= 0:
        raise ValueError("max_per_mutation_site must be positive.")
    run_dir = _project_dir(project_name)
    report_dir = run_dir / "00_agent_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # WT inspection and predicted-polymer chains are separate namespaces.
    # SwitchTune writes Boltz-2 inputs as one polymer on chain A by default,
    # even when the WT template was inspected on chain C (Dronpa) or with a
    # selector such as A,B (HbI). Read the generated manifest when available;
    # old manifests did not record this field and safely fall back to A.
    state_chain = next(
        (
            token.strip()
            for token in re.split(r"[,;\s]+", str(parse_chains or ""))
            if token.strip()
        ),
        "A",
    )

    project_manifest = _read_json(run_dir / "project_manifest.json")
    # Older Boltz manifests did not record ``polymer_chain_id``.  Leave this
    # empty while inspecting the manifest so the legacy input-record fallback
    # below is actually reachable; Boltz's documented single-polymer default
    # is A when no chain metadata exists.
    prediction_chain = ""
    structure_manifest_hint = str(
        existing_structure_job_manifest
        or project_manifest.get("structure_prediction_job_manifest_path")
        or run_dir / "04_structure_prediction" / "structure_prediction_job_manifest.json"
    ).strip()
    structure_manifest_data = (
        _read_json(Path(structure_manifest_hint).expanduser())
        if structure_manifest_hint
        else {}
    )
    boltz_manifest = structure_manifest_data.get("boltz2") if isinstance(structure_manifest_data, dict) else {}
    if isinstance(boltz_manifest, dict):
        prediction_chain = str(
            boltz_manifest.get("polymer_chain_id")
            or boltz_manifest.get("boltz_chain_id")
            or ""
        ).strip()[:1] or prediction_chain
        if not prediction_chain:
            records = boltz_manifest.get("input_yaml_records") or boltz_manifest.get("payload_records") or []
            if isinstance(records, list) and records and isinstance(records[0], dict):
                prediction_chain = str(records[0].get("chain_id") or "A").strip()[:1] or "A"
    prediction_chain = prediction_chain or "A"
    selection_mode = str(project_manifest.get("state_specific_selection_mode") or "").strip().lower()
    candidate_paths = {
        "fixed_window": run_dir / "03_state_specific_candidates" / "state_specific_retained_candidates.csv",
        "adaptive": run_dir / "03_state_specific_candidates" / "state_specific_retained_candidates_adaptive.csv",
    }
    if selection_mode == "fixed_window":
        candidate_order = ("fixed_window", "adaptive")
    else:
        candidate_order = ("adaptive", "fixed_window")
    candidate_csv = next(
        (candidate_paths[mode] for mode in candidate_order if candidate_paths[mode].exists()),
        candidate_paths[candidate_order[0]],
    )
    if not candidate_csv.exists():
        raise FileNotFoundError(
            "Checkpoint mode expected a CB-selected candidate CSV at one of: "
            + ", ".join(str(path) for path in candidate_paths.values())
        )

    # A zero budget delegates the final 20--30 candidate count to the
    # evidence-aware controller. The budget is resolved from the current
    # valid pool and recorded in the checkpoint report.
    available_candidate_count = _csv_data_row_count(str(candidate_csv)) or 0
    resolved_bioemu_top_n = (
        adaptive_candidate_budget(available_candidate_count)
        if int(bioemu_top_n) <= 0
        else int(bioemu_top_n)
    )

    # Keep ``target_count`` for legacy smoke tests. Formal benchmark runs pass
    # ``structure_top_n=0`` to send every candidate retained after CB scoring
    # into structure prediction.
    if structure_top_n is None:
        resolved_structure_top_n = max(1, int(target_count))
        structure_selection_policy = "legacy_target_count"
    elif int(structure_top_n) <= 0:
        resolved_structure_top_n = _csv_data_row_count(str(candidate_csv))
        structure_selection_policy = "all_cb_retained_candidates"
    else:
        resolved_structure_top_n = min(
            int(structure_top_n),
            _csv_data_row_count(str(candidate_csv)),
        )
        structure_selection_policy = "explicit_structure_top_n"
    if resolved_structure_top_n <= 0:
        raise ValueError(
            "Checkpoint mode cannot resolve a positive structure candidate count "
            f"from {candidate_csv}."
        )
    review_top_n = max(5, int(review_top_n))

    structure_manifest = run_dir / "04_structure_prediction" / "structure_prediction_job_manifest.json"
    if not reuse_structure_prediction and not structure_manifest.exists():
        raise FileNotFoundError(
            f"Checkpoint mode expected a prepared structure manifest at {structure_manifest}. "
            "Run the initial prompt-driven front half first."
        )
    if reuse_structure_prediction:
        reused_prediction_dir = Path(str(existing_structure_prediction_dir or "")).expanduser()
        reused_job_manifest = Path(str(existing_structure_job_manifest or "")).expanduser()
        if not reused_prediction_dir.exists():
            raise FileNotFoundError(
                "reuse_structure_prediction=True requires an existing prediction directory: "
                f"{reused_prediction_dir}"
            )
        if not reused_job_manifest.exists():
            raise FileNotFoundError(
                "reuse_structure_prediction=True requires the original structure job manifest: "
                f"{reused_job_manifest}"
            )

    report: dict[str, Any] = {
        "status": "completed",
        "workflow": "SwitchTune checkpoint mode",
        "project_name": project_name,
        "target_state_name": target_state_name,
        "reference_state_name": reference_state_name,
        "checkpoint_candidate_csv": str(candidate_csv),
        "cb_selection_mode": selection_mode or (
            "adaptive" if candidate_csv.name.endswith("_adaptive.csv") else "fixed_window"
        ),
        "candidate_counts": {
            "cb_selected_candidates": _csv_data_row_count(str(candidate_csv)),
        },
        "structure_candidate_policy": (
            "reused_existing_structure_prediction"
            if reuse_structure_prediction
            else structure_selection_policy
        ),
        "structure_top_n": resolved_structure_top_n,
        "review_top_n": review_top_n,
        "bioemu_top_n": int(resolved_bioemu_top_n),
        "bioemu_candidate_budget_policy": (
            "agent_adaptive_20_to_30"
            if int(bioemu_top_n) <= 0
            else "user_configured"
        ),
        "bioemu_max_per_mutation_site": int(max_per_mutation_site),
        "bioemu_site_diversity_policy": "site_round_robin_then_rank",
        "checkpoint_reviews": [],
        "candidate_locks": [],
    }

    def stop_workflow(status: str, stage_name: str, reason: str, candidate_csv_path: str = "") -> dict[str, Any]:
        report["status"] = status
        report["blocked_stage"] = stage_name
        report["failure_reason"] = reason
        if candidate_csv_path:
            report["blocked_candidate_csv"] = candidate_csv_path
            report["blocked_candidate_count"] = _csv_data_row_count(candidate_csv_path)
        report_path = _write_json(report_dir / "checkpoint_workflow_summary.json", report)
        report["report_path"] = report_path
        _write_json(report_dir / "checkpoint_workflow_summary.json", report)
        return report

    if reuse_structure_prediction:
        structure = {
            "status": "reused_existing_structure_prediction",
            "prediction_dir": str(reused_prediction_dir),
            "job_manifest_path": str(reused_job_manifest),
            "candidate_count": _csv_data_row_count(str(candidate_csv)),
            "run_external": False,
            "require_pae": True,
        }
        report["structure_reuse"] = {
            "enabled": True,
            "prediction_dir": str(reused_prediction_dir),
            "source_job_manifest": str(reused_job_manifest),
            "source_project": str(reused_job_manifest.parents[2]),
            "note": "Boltz-2 submission was skipped; existing prediction outputs were rescanned.",
        }
    else:
        structure_engine_decision = select_structure_prediction_engine(
            project_name=project_name,
            target_state_structure=target_state_structure,
            reference_state_structure=reference_state_structure,
            ligand_codes=ligand_codes,
            requested_engine=structure_engine,
        )
        report["structure_engine_decision"] = structure_engine_decision

        structure = prepare_structure_prediction_job(
            project_name=project_name,
            candidates_csv=str(candidate_csv),
            engine=str(structure_engine_decision.get("recommended_engine") or "auto"),
            top_n=resolved_structure_top_n,
            target_state_structure=target_state_structure,
            reference_state_structure=reference_state_structure,
            template_mode="target_state",
            boltz_use_template=True,
            boltz_write_full_pae=True,
            ligand_codes=ligand_codes,
            cofactor_required=bool(structure_engine_decision.get("ligand_required", False)),
            execute=run_external,
            timeout_seconds=timeout_seconds,
        )
        report["structure_preparation"] = structure

    collect_structure = collect_structure_prediction_results(
        project_name=project_name,
        prediction_dir=str(structure.get("prediction_dir") or ""),
        job_manifest_path=str(structure.get("job_manifest_path") or ""),
    )
    report["structure_collection"] = collect_structure
    report["candidate_counts"]["structure_predictions_collected"] = _result_variant_count(
        collect_structure,
        evidence_csv=str(collect_structure.get("evidence_csv") or ""),
    )

    if "valid_structure_count" in collect_structure:
        valid_structure_count = int(collect_structure.get("valid_structure_count") or 0)
    elif "completed_count" in collect_structure:
        valid_structure_count = int(collect_structure.get("completed_count") or 0)
    else:
        valid_structure_count = int(collect_structure.get("structure_count") or 0)
    if valid_structure_count == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "structure_prediction",
            "Structure prediction produced no valid structures after the strict PAE gate. "
            "Inspect structure_prediction_evidence.csv and the Boltz-2 response summary.",
            str(collect_structure.get("evidence_csv") or ""),
        )

    standardized = standardize_structure_prediction_results(
        project_name=project_name,
        structures_csv=str(collect_structure.get("output_csv") or ""),
        chain_id=prediction_chain,
        require_pae=True,
    )
    report["structure_standardization"] = standardized
    report["candidate_counts"]["standardized_structures"] = int(
        standardized.get("standardized_count") or 0
    )
    if int(standardized.get("standardized_count") or 0) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "structure_standardization",
            "Structure standardization produced no valid PDBs after the strict PAE gate. "
            "Do not continue to inverse folding, BioEmu, or Rosetta.",
            str(standardized.get("error_csv") or ""),
        )

    standardized_csv = str(standardized.get("output_csv") or collect_structure.get("output_csv") or "")
    project_manifest = _read_json(run_dir / "project_manifest.json")
    mutation_region = str(project_manifest.get("mutation_region") or "").strip()
    comparison_regions = "global"
    if mutation_region:
        comparison_regions += f",switch_region:{mutation_region}"
    compare = compare_structure_predictions_to_states(
        project_name=project_name,
        structures_csv=standardized_csv,
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        prediction_chain=prediction_chain,
        target_chain=state_chain,
        reference_chain=state_chain,
        regions=comparison_regions,
        mapping="sequence_alignment",
    )
    report["structure_comparison"] = compare

    confidence = filter_structure_confidence(
        project_name=project_name,
        candidates_csv=str(candidate_csv),
        structures_csv=standardized_csv,
        structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
        structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
        chain_id=prediction_chain,
        mutation_window_radius=3,
        min_confidence_score=0.7,
        min_complex_plddt=0.7,
        min_local_plddt=70.0,
        require_pae=True,
        require_target_like=True,
        require_complete=True,
        enable_agent_rescue_review=True,
        rescue_max_mutation_mean_pae=4.0,
        rescue_min_confidence_score=0.90,
        rescue_min_complex_plddt=0.85,
        rescue_min_local_plddt=80.0,
        rescue_max_mutation_mean_pae_to_protein=2.0,
        rescue_max_mutation_window_internal_pae=2.0,
        rescue_require_target_like=True,
    )
    report["structure_confidence"] = confidence
    confidence_csv = str(confidence.get("output_csv") or "")
    confidence_review_csv = str(
        confidence.get("review_csv") or confidence_csv
    )
    report["candidate_counts"]["structure_confidence_strict_candidates"] = (
        _csv_data_row_count(confidence_csv)
    )
    report["candidate_counts"]["structure_confidence_review_candidates"] = (
        _csv_data_row_count(confidence_review_csv)
    )
    if _csv_data_row_count(confidence_review_csv) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "structure_confidence",
            "The structure-confidence gate produced no strict or rescue-review candidates. "
            "Inspect structure_confidence_evidence.csv and structure_confidence_rejected_candidates.csv.",
            confidence_review_csv,
        )

    structure_review = _checkpoint_review(
        review_agent=review_agent,
        project_name=project_name,
        stage_name="structure_confidence",
        evidence_csv=str(confidence.get("evidence_csv") or ""),
        current_candidates_csv=confidence_review_csv,
        review_log_dir=report_dir,
        top_n=review_top_n,
        minimum_selected_count=min(20, int(resolved_bioemu_top_n)),
        diversity_target_count=min(
            int(resolved_bioemu_top_n),
            _csv_data_row_count(confidence_review_csv),
        ),
        max_per_mutation_site=int(max_per_mutation_site),
    )
    report["checkpoint_reviews"].append(structure_review)
    report["candidate_locks"].append(structure_review["lock_info"])
    report["candidate_counts"]["structure_checkpoint_candidates"] = _csv_data_row_count(
        structure_review["locked_csv"]
    )

    structure_shortlist_csv = structure_review["locked_csv"]
    if _csv_data_row_count(structure_shortlist_csv) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "structure_confidence",
            "The structure-confidence checkpoint retained zero candidates. "
            "Inspect the strict pLDDT/PAE and target-like rejection evidence before continuing.",
            structure_shortlist_csv,
        )

    inverse_folding = prepare_inverse_folding_scoring_job(
        project_name=project_name,
        structures_csv=standardized_csv,
        candidates_csv=structure_shortlist_csv,
        model_set="esm_if1,protein_mpnn",
        chain_id=prediction_chain,
        top_n=resolved_structure_top_n,
        execute=run_external,
        timeout_seconds=timeout_seconds,
    )
    report["inverse_folding_preparation"] = inverse_folding
    if inverse_folding.get("status") == "blocked_no_candidates":
        return stop_workflow(
            "blocked_no_candidates",
            "inverse_folding",
            "Inverse-folding preparation received an empty candidate table.",
            structure_shortlist_csv,
        )

    inverse_collect = collect_inverse_folding_scores(
        project_name=project_name,
        score_dir=str(inverse_folding.get("score_dir") or ""),
        candidates_csv=structure_shortlist_csv,
    )
    report["inverse_folding_collection"] = inverse_collect
    report["candidate_counts"]["inverse_folding_scored"] = _result_variant_count(
        inverse_collect,
        evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
    )
    if int(inverse_collect.get("score_file_count") or 0) == 0:
        return stop_workflow(
            "failed",
            "inverse_folding",
            "Inverse-folding scoring produced no score files. Inspect inverse_folding_command_status.csv and per-variant logs.",
            structure_shortlist_csv,
        )

    inverse_review = _checkpoint_review(
        review_agent=review_agent,
        project_name=project_name,
        stage_name="inverse_folding",
        evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
        current_candidates_csv=structure_shortlist_csv,
        review_log_dir=report_dir,
        top_n=review_top_n,
        minimum_selected_count=min(20, int(resolved_bioemu_top_n)),
        diversity_target_count=min(
            int(resolved_bioemu_top_n),
            _csv_data_row_count(structure_shortlist_csv),
        ),
        max_per_mutation_site=int(max_per_mutation_site),
    )
    report["checkpoint_reviews"].append(inverse_review)
    report["candidate_locks"].append(inverse_review["lock_info"])
    report["candidate_counts"]["inverse_folding_checkpoint_candidates"] = _csv_data_row_count(
        inverse_review["locked_csv"]
    )

    inverse_shortlist_csv = inverse_review["locked_csv"]
    if _csv_data_row_count(inverse_shortlist_csv) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "inverse_folding",
            "The inverse-folding checkpoint retained zero candidates. Do not prepare BioEmu or Rosetta jobs.",
            inverse_shortlist_csv,
        )

    bioemu_selection = select_candidates_from_evidence(
        project_name=project_name,
        stage_name="bioemu",
        candidate_csv=inverse_shortlist_csv,
        cb_scored_csv=str((run_dir / "02_conformation_bias_scoring" / "cb_scored_library.csv")),
        structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
        structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
        structure_confidence_evidence_csv=str(confidence.get("evidence_csv") or ""),
        inverse_folding_evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
        top_n=resolved_bioemu_top_n,
        max_per_mutation_site=int(max_per_mutation_site),
        site_diversity_policy="site_round_robin_then_rank",
    )
    report["bioemu_selection"] = bioemu_selection
    report["bioemu_site_diversity"] = bioemu_selection.get("site_diversity") or {}
    report["candidate_counts"]["bioemu_candidates"] = _csv_data_row_count(
        str(bioemu_selection.get("output_csv") or "")
    )
    if _csv_data_row_count(str(bioemu_selection.get("output_csv") or "")) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "bioemu_selection",
            "Evidence selection produced zero candidates for BioEmu.",
            str(bioemu_selection.get("output_csv") or ""),
        )

    bioemu = prepare_bioemu_sampling_job(
        project_name=project_name,
        candidates_csv=str(bioemu_selection.get("output_csv") or ""),
        top_n=resolved_bioemu_top_n,
        num_samples=bioemu_num_samples,
        batch_size=bioemu_batch_size,
        execute=run_external,
        timeout_seconds=bioemu_timeout_seconds,
    )
    report["bioemu_preparation"] = bioemu
    if bioemu.get("status") == "blocked_no_candidates":
        return stop_workflow(
            "blocked_no_candidates",
            "bioemu",
            "BioEmu preparation received an empty candidate table.",
            str(bioemu_selection.get("output_csv") or ""),
        )

    bioemu_collect = collect_bioemu_outputs(
        project_name=project_name,
        sample_root=str(bioemu.get("sample_root") or ""),
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        mutation_region=str(_read_json(run_dir / "project_manifest.json").get("mutation_region") or ""),
        first_residue_number=int(
            _read_json(run_dir / "project_manifest.json").get("first_residue_number") or 1
        ),
        chain_id=prediction_chain,
        compute_metrics=True,
    )
    report["bioemu_collection"] = bioemu_collect
    report["candidate_counts"]["bioemu_evidence"] = _result_variant_count(
        bioemu_collect,
        evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
    )
    if _result_variant_count(
        bioemu_collect,
        evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
    ) == 0:
        return stop_workflow(
            "failed",
            "bioemu",
            "BioEmu collection produced no variant-level evidence. Inspect BioEmu command status and per-variant logs.",
            str(bioemu_selection.get("output_csv") or ""),
        )

    bioemu_candidate_count = _csv_data_row_count(str(bioemu_selection.get("output_csv") or ""))
    bioemu_review = _checkpoint_review(
        review_agent=review_agent,
        project_name=project_name,
        stage_name="bioemu",
        evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
        current_candidates_csv=str(bioemu_selection.get("output_csv") or ""),
        review_log_dir=report_dir,
        top_n=max(resolved_bioemu_top_n, bioemu_candidate_count, 5),
        # Preserve the full formal BioEmu candidate set for Rosetta. The
        # final recommendation budget must not be reused as a checkpoint
        # pruning budget.
        minimum_selected_count=max(
            1,
            bioemu_candidate_count,
        ),
        diversity_target_count=bioemu_candidate_count,
        max_per_mutation_site=int(max_per_mutation_site),
    )
    report["checkpoint_reviews"].append(bioemu_review)
    report["candidate_locks"].append(bioemu_review["lock_info"])
    report["candidate_counts"]["bioemu_checkpoint_candidates"] = _csv_data_row_count(
        bioemu_review["locked_csv"]
    )
    report["bioemu_candidate_policy"] = "retain_all_bioemu_candidates_for_rosetta"

    bioemu_shortlist_csv = bioemu_review["locked_csv"]
    if _csv_data_row_count(bioemu_shortlist_csv) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "bioemu",
            "The BioEmu checkpoint retained zero candidates. Do not prepare Rosetta.",
            bioemu_shortlist_csv,
        )

    rosetta_selection = select_candidates_from_evidence(
        project_name=project_name,
        stage_name="rosetta",
        candidate_csv=bioemu_shortlist_csv,
        cb_scored_csv=str((run_dir / "02_conformation_bias_scoring" / "cb_scored_library.csv")),
        structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
        structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
        structure_confidence_evidence_csv=str(confidence.get("evidence_csv") or ""),
        inverse_folding_evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
        bioemu_evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
        top_n=0,
        retain_all_candidates=True,
    )
    report["rosetta_selection"] = rosetta_selection
    report["candidate_counts"]["rosetta_candidates"] = _csv_data_row_count(
        str(rosetta_selection.get("output_csv") or "")
    )
    report["rosetta_candidate_policy"] = "all_bioemu_checkpoint_candidates_ranking_only"
    if _csv_data_row_count(str(rosetta_selection.get("output_csv") or "")) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "rosetta_selection",
            "Evidence selection produced zero candidates for Rosetta.",
            str(rosetta_selection.get("output_csv") or ""),
        )

    standardized_path = Path(standardized_csv).expanduser()
    standardized_pdb_dir = standardized_path.parent / "standardized_pdbs"
    input_pdb_dir = str(standardized_pdb_dir if standardized_pdb_dir.exists() else standardized_path.parent)
    manifest = _read_json(run_dir / "project_manifest.json")
    rosetta_strategy = select_rosetta_analysis_strategy(
        project_name=project_name,
        design_objective=str(manifest.get("design_objective") or ""),
        mutation_region=str(manifest.get("mutation_region") or ""),
        candidate_csv=str(rosetta_selection.get("output_csv") or ""),
        structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
        bioemu_evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        requested_focus="auto",
        requested_mode="auto",
    )
    report["rosetta_strategy"] = rosetta_strategy
    rosetta = prepare_rosetta_energy_job(
        project_name=project_name,
        mode=str(rosetta_strategy.get("mode") or "score"),
        input_pdb_dir=input_pdb_dir,
        structures_csv=standardized_csv,
        candidates_csv=str(rosetta_selection.get("output_csv") or ""),
        protocol_xml=str(rosetta_strategy.get("protocol_xml") or "") or None,
        interface=str(rosetta_strategy.get("interface") or "") or None,
        nstruct=int(rosetta_strategy.get("nstruct") or 1),
        strategy_name=str(rosetta_strategy.get("strategy_name") or ""),
        strategy_focus=str(rosetta_strategy.get("focus") or ""),
        strategy_reason=str(rosetta_strategy.get("reason") or ""),
        requested_metrics=",".join(rosetta_strategy.get("metrics_to_collect") or []),
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        fallback_strategy=str(rosetta_strategy.get("fallback_strategy") or ""),
        strategy_confidence=rosetta_strategy.get("confidence"),
        execute=run_external,
        timeout_seconds=timeout_seconds,
    )
    report["rosetta_preparation"] = rosetta
    if rosetta.get("status") == "blocked_no_candidates":
        return stop_workflow(
            "blocked_no_candidates",
            "rosetta",
            "Rosetta preparation received an empty candidate table.",
            str(rosetta_selection.get("output_csv") or ""),
        )

    rosetta_collect = collect_rosetta_scores(
        project_name=project_name,
        score_file=str(rosetta.get("score_file") or rosetta.get("scorefile") or ""),
        job_manifest_path=str(rosetta.get("job_manifest_path") or ""),
    )
    report["rosetta_collection"] = rosetta_collect
    report["candidate_counts"]["rosetta_evidence"] = _result_variant_count(
        rosetta_collect,
        evidence_csv=str(rosetta_collect.get("evidence_csv") or ""),
    )
    if _result_variant_count(
        rosetta_collect,
        evidence_csv=str(rosetta_collect.get("evidence_csv") or ""),
    ) == 0:
        return stop_workflow(
            "failed",
            "rosetta",
            "Rosetta collection produced no variant-level evidence. Inspect Rosetta status and logs.",
            str(rosetta_selection.get("output_csv") or ""),
        )

    rosetta_candidate_count = _csv_data_row_count(str(rosetta_selection.get("output_csv") or ""))
    rosetta_review = _checkpoint_review(
        review_agent=review_agent,
        project_name=project_name,
        stage_name="rosetta",
        evidence_csv=str(rosetta_collect.get("evidence_csv") or ""),
        current_candidates_csv=str(rosetta_selection.get("output_csv") or ""),
        review_log_dir=report_dir,
        top_n=max(int(rosetta_top_n), rosetta_candidate_count, 5),
        # Rosetta ranks the complete BioEmu checkpoint set. It must not
        # silently reduce 20-30 candidates to the final recommendation budget.
        minimum_selected_count=max(
            1,
            rosetta_candidate_count,
        ),
    )
    report["checkpoint_reviews"].append(rosetta_review)
    report["candidate_locks"].append(rosetta_review["lock_info"])
    report["candidate_counts"]["rosetta_checkpoint_candidates"] = _csv_data_row_count(
        rosetta_review["locked_csv"]
    )

    final_shortlist_csv = rosetta_review["locked_csv"]
    if _csv_data_row_count(final_shortlist_csv) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "rosetta",
            "The Rosetta checkpoint retained zero candidates; no final recommendation was generated.",
            final_shortlist_csv,
        )

    merged = merge_switchtune_evidence_tables(
        project_name=project_name,
        candidate_csv=final_shortlist_csv,
        cb_scored_csv=str((run_dir / "02_conformation_bias_scoring" / "cb_scored_library.csv")),
        structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
        structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
        structure_confidence_evidence_csv=str(confidence.get("evidence_csv") or ""),
        inverse_folding_evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
        bioemu_evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
        rosetta_evidence_csv=str(rosetta_collect.get("evidence_csv") or ""),
        # Keep the complete Rosetta-ranked set available to final review.
        top_n=_csv_data_row_count(final_shortlist_csv),
    )
    report["evidence_merge"] = merged
    if int(merged.get("ranked_variant_count") or 0) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "evidence_merge",
            "Evidence merge produced no numerically ranked candidates. "
            "Do not create a final recommendation from empty or non-numeric evidence.",
            str(merged.get("ranked_csv") or merged.get("merged_csv") or ""),
        )

    resolved_rosetta_top_n = (
        adaptive_candidate_budget(int(merged.get("ranked_variant_count") or 0))
        if int(rosetta_top_n) <= 0
        else int(rosetta_top_n)
    )
    report["rosetta_top_n"] = int(resolved_rosetta_top_n)
    report["rosetta_candidate_budget_policy"] = (
        "agent_adaptive_20_to_30" if int(rosetta_top_n) <= 0 else "user_configured"
    )
    final = prepare_final_candidate_recommendation(
        project_name=project_name,
        evidence_csv=str(merged.get("merged_csv") or merged.get("ranked_csv") or ""),
        rank_column="consensus_score",
        ascending=False,
        # ``rosetta_top_n`` is the final recommendation budget only.
        top_n=resolved_rosetta_top_n,
    )
    report["final_recommendation"] = final
    if int(final.get("candidate_count") or 0) == 0:
        return stop_workflow(
            "blocked_no_candidates",
            "final_recommendation",
            "Final recommendation contains zero candidates after numeric-evidence filtering.",
            str(final.get("output_csv") or ""),
        )
    report["final_candidate_csv"] = final_shortlist_csv

    report_path = _write_json(report_dir / "checkpoint_workflow_summary.json", report)
    report["report_path"] = report_path
    report["recommendation"] = recommend_next_workflow_step(
        project_name=project_name,
        desired_structure_candidate_count=resolved_structure_top_n,
        desired_bioemu_candidate_count=resolved_bioemu_top_n,
        desired_rosetta_candidate_count=resolved_rosetta_top_n,
    )
    missing_decision_stages = [
        str(item.get("stage_name"))
        for item in report.get("checkpoint_reviews", [])
        if not item.get("decision_recorded")
    ]
    report["missing_agentic_decision_stages"] = missing_decision_stages
    report["status"] = "completed" if not missing_decision_stages else "completed_with_missing_agentic_decisions"
    _write_json(report_dir / "checkpoint_workflow_summary.json", report)
    return report
