from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_workflow_runner import run_checkpoint_workflow


REPO = Path(os.environ.get("SWITCHTUNE_HOME", REPO_ROOT)).expanduser().resolve()
DEFAULT_PROJECT_NAME = "switchtune_workflow"
RUNS_ROOT = Path(os.environ.get("SWITCHTUNE_RUNS", "./SwitchTune_runs"))
DEFAULT_PROMPT_PATH = REPO / "prompts" / "switchtune_agent_prompt.md"

PROMPT_TEMPLATE = """Run the SwitchTune workflow for a two-state conformational-switching protein.

project_name: {project_name}
{workflow_mode_note}

protein_name: {protein_name}
target_state_structure: {target_state_structure}
reference_state_structure: {reference_state_structure}
target_state_name: {target_state_name}
reference_state_name: {reference_state_name}
parse_chains: {parse_chains}
mutation_region: agent-selected after inspection
structure_engine: {structure_engine}

Execution policy:
- Use one SwitchTune MCP step per execute block.
- First execute block must call inspect_two_state_pdb_pair exactly once.
- After inspection, choose and record a concrete mutation region from the supplied structures and sequence.
- Follow each returned next_tool and agent_instruction directly.
- Use empirical adaptive CB selection; never apply a fixed score window or fixed retained count.
- Send all candidates retained by CB selection to structure prediction.
- Choose downstream budgets adaptively from the valid evidence pool, retaining 20-30 candidates when available.
- Keep using the same project_name for every tool call.
- Do not search literature or external mutation databases during the run.
"""

EXPECTED_STAGE_FILES = {
    "library": "01_saturation_library/saturation_single_mutant_library.csv",
    "cb_scores": "02_conformation_bias_scoring/cb_scored_library.csv",
    "adaptive_candidates": "03_state_specific_candidates/state_specific_retained_candidates_adaptive.csv",
    "structure_results": "04_structure_prediction/structure_prediction_results.csv",
    "structure_pdb_results": "04_structure_prediction/structure_prediction_results_pdb.csv",
    "structure_evidence": "04_structure_prediction/structure_prediction_evidence.csv",
    "structure_confidence": "04b_structure_confidence_filter/structure_confidence_evidence.csv",
    "inverse_folding": "05_inverse_folding_scoring/inverse_folding_evidence.csv",
    "bioemu": "06_bioemu_sampling/bioemu_evidence.csv",
    "rosetta": "07_rosetta_energy/rosetta_evidence.csv",
    "merged": "08_final_recommendation/switchtune_evidence_merged.csv",
    "final": "08_final_recommendation/final_candidate_recommendation.csv",
    "agentic_review": "08_final_recommendation/agentic_candidate_review_packet.json",
}

AGENTIC_REVIEW_PROMPT_PATH = REPO / "prompts" / "switchtune_agent_review_prompt.md"


def print_header(title: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(title, flush=True)
    print("=" * 88, flush=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_row_count(path: Path | str) -> int:
    candidate_path = Path(path)
    if not candidate_path.exists():
        return 0
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_top_candidates(path: Path, limit: int = 5) -> list[dict[str, str]]:
    if not path.exists():
        return []
    fields = (
        "final_rank",
        "variant",
        "consensus_score",
        "bias_score",
        "structure_confidence_pass",
        "structure_target_similarity_score",
        "inverse_folding_compatibility_score",
        "bioemu_sampling_complete",
        "rosetta_primary_score_mean",
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({field: row.get(field, "") for field in fields})
            if len(rows) >= limit:
                break
    return rows


def has_existing_workflow_artifacts(run_dir: Path) -> bool:
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


def count_tool_mentions(log_text: str) -> dict[str, int]:
    tool_names = (
        "inspect_two_state_pdb_pair",
        "setup_two_state_cb_scoring_project_once",
        "execute_conformation_bias_scoring_job",
        "select_state_specific_candidates",
        "select_state_specific_candidates_adaptive",
        "prepare_structure_prediction_job",
        "collect_structure_prediction_results",
        "standardize_structure_prediction_results",
        "compare_structure_predictions_to_states",
        "filter_structure_confidence",
        "prepare_inverse_folding_scoring_job",
        "collect_inverse_folding_scores",
        "prepare_bioemu_sampling_job",
        "collect_bioemu_outputs",
        "select_rosetta_analysis_strategy",
        "prepare_rosetta_energy_job",
        "collect_rosetta_scores",
        "merge_switchtune_evidence_tables",
        "prepare_final_candidate_recommendation",
        "prepare_agentic_candidate_review",
        "review_switchtune_stage_evidence",
        "record_agentic_candidate_decision",
        "recommend_next_workflow_step",
    )
    return {name: len(re.findall(rf"\b{re.escape(name)}\b", log_text)) for name in tool_names}


def read_latest_jsonl_record(path: Path, *, step_name: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        if step_name is None or record.get("step_name") == step_name:
            latest = record
    return latest


def is_final_round_decision(record: dict[str, Any]) -> bool:
    """Return whether a decision-log record belongs to the final review.

    Checkpoint decisions must not satisfy the final-review completion check.
    Older records did not have ``decision_scope`` and used the default
    reviewer name, so retain a backwards-compatible reviewer fallback.
    """
    scope = str(record.get("decision_scope") or "").strip().lower()
    if scope:
        return scope in {"final", "final_round", "final-round"}
    reviewer = str(record.get("reviewer") or "").strip().lower()
    if "checkpoint" in reviewer:
        return False
    return reviewer in {"", "biomni", "biomni_final", "biomni_final_round"} or "final" in reviewer


def read_latest_final_round_decision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except Exception:
            continue
        if isinstance(record, dict) and record.get("step_name") == "agentic_candidate_decision":
            if is_final_round_decision(record):
                latest = record
    return latest


def build_agentic_review_prompt(
    *,
    project_name: str,
    merged_csv: str,
    final_csv: str,
    review_packet_json: str,
    review_packet: dict[str, Any] | None,
    top_candidates: list[dict[str, str]],
) -> str:
    def _compact_text(value: Any, limit: int = 180) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    candidate_lines: list[str] = []
    packet_candidates = (review_packet or {}).get("candidate_reviews") or []
    for item in packet_candidates:
        support = "; ".join(_compact_text(v, 90) for v in (item.get("supporting_evidence") or [])[:2])
        cautions = "; ".join(_compact_text(v, 90) for v in (item.get("caution_flags") or [])[:2])
        line = (
            f"- rank {item.get('rank')}: {item.get('variant')} "
            f"(consensus={item.get('consensus_score')}, selection={item.get('selection_score')})"
        )
        if support:
            line += f" | support: {support}"
        if cautions:
            line += f" | cautions: {cautions}"
        candidate_lines.append(line)
    if not candidate_lines:
        for row in top_candidates:
            fields = ", ".join(
                f"{key}={value}"
                for key, value in row.items()
                if value not in {"", None}
            )
            if fields:
                candidate_lines.append(f"- {fields}")
    if not candidate_lines:
        candidate_lines.append("- No ranked candidates were available.")

    stage_lines: list[str] = []
    for stage in (review_packet or {}).get("stage_reviews", [])[:5]:
        findings = stage.get("findings") or []
        finding_text = "; ".join(_compact_text(item, 110) for item in findings[:2])
        if not finding_text:
            finding_text = "no compact findings available"
        stage_lines.append(f"- {stage.get('stage_name', 'stage')}: {finding_text}")
    if not stage_lines:
        stage_lines.append("- No compact stage summaries were available.")

    return (
        f"You are now the final Biomni reviewer for SwitchTune project `{project_name}`.\n\n"
        "The evidence packet has already been prepared by the harness.\n"
        "Do not call `review_switchtune_stage_evidence` or `prepare_agentic_candidate_review` again.\n"
        "Do not rerun any earlier design, scoring, or prediction stages.\n"
        "Your only tool action is `record_agentic_candidate_decision`, exactly once.\n"
        "The decision must be marked as a final-round review with reviewer=\"biomni_final_round\" "
        "and decision_scope=\"final_round\".\n\n"
        "Hard rules:\n"
        "- Emit one `<execute>` block containing only the decision-tool call; do not "
        "combine `<execute>` and `<solution>` in the same assistant message.\n"
        "- Wait for the tool observation before returning `<solution>`.\n"
        "- Do not call inspect_two_state_pdb_pair, setup, CB scoring, structure prediction, inverse folding, BioEmu, or Rosetta.\n"
        "- Do not rerun any completed workflow stage.\n"
        "- This remains a blind test: use only the prepared WT-derived evidence and outputs from this run. Do not search or consult literature, mutation databases, benchmark answer keys, previous reports, remembered variants, or mutant structures. If literature evidence is requested, write `not assessed in blind test`.\n"
        "- Base your judgement on the prepared packet, the ranked candidates, and the compact stage summaries below.\n"
        "- Do not run exploratory Python or pandas code and do not reopen the raw CSV files; the harness has already "
        "loaded and summarized the evidence for you.\n"
        "- In the prepared packet, the visible candidate display number is `rank`. The final recommendation CSV uses "
        "`final_rank`; do not assume that the final CSV contains a `rank` column.\n"
        "- If the evidence supports the current ranking, keep it; if not, explain the tradeoff clearly.\n"
        "- Record the decision once, then stop.\n\n"
        f"Merged evidence CSV: `{merged_csv}`\n"
        f"Final recommendation CSV: `{final_csv}`\n"
        f"Review packet JSON: `{review_packet_json}`\n\n"
        "Current ranked candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\n"
        "Compact stage summaries:\n"
        + "\n".join(stage_lines)
        + "\n\n"
        "Suggested action sequence:\n"
        "1. Read the prepared packet summary above.\n"
        f"2. Call `record_agentic_candidate_decision(project_name=\"{project_name}\", selected_variants=\"...\", decision_text=\"...\", evidence_packet_json=\"{review_packet_json}\", reviewer=\"biomni_final_round\", decision_scope=\"final_round\")` once.\n"
        "3. Return a concise `<solution>` summarizing the final judgement and the decision record path.\n"
    )


def make_biomni_agent(
    *,
    biomni_data_path: str,
    llm: str,
    base_url: str,
    api_key: str,
    mcp_config: str,
    timeout_seconds: int,
    max_steps: int,
    max_execute_calls: int,
    recursion_limit: int,
    thread_id: int,
    debug_raw_agent: bool,
    checkpoint_mode: bool = False,
    review_mode: bool = False,
    cb_timeout_seconds: int = 3600,
    workflow_timeout_seconds: int = 3600,
    source: str = "Custom",
):
    from biomni.agent import A1
    from switchtune.biomni_go_guard import install_switchtune_go_guard
    from switchtune.biomni_mcp_adapter import patch_switchtune_mcp_results

    agent = A1(
        path=biomni_data_path,
        llm=llm,
        source=source,
        base_url=base_url or None,
        api_key=api_key,
        use_tool_retriever=True,
        timeout_seconds=timeout_seconds,
        expected_data_lake_files=[],
    )
    agent.add_mcp(config_path=mcp_config)
    patch_switchtune_mcp_results(agent)
    install_switchtune_go_guard(
        agent,
        replace_go=True,
        max_steps=max_steps if not review_mode else min(max_steps, 96),
        max_execute_calls=max_execute_calls if not review_mode else min(max_execute_calls, 24),
        max_same_execute_calls=1,
        recursion_limit=recursion_limit if not review_mode else max(recursion_limit, 120),
        thread_id=thread_id,
        auto_advance_after_inspection=not debug_raw_agent and not review_mode,
        auto_execute_cb_after_setup=checkpoint_mode and not debug_raw_agent and not review_mode,
        auto_run_full_workflow_after_inspection=(
            not checkpoint_mode and not debug_raw_agent and not review_mode
        ),
        auto_advance_on_repeated_tool=not debug_raw_agent,
        stop_on_prepared_tool=False,
        cb_timeout_seconds=cb_timeout_seconds,
        workflow_timeout_seconds=workflow_timeout_seconds,
        stop_after_agentic_decision=review_mode,
    )
    return agent


def run_agent_session(
    *,
    agent: Any,
    prompt: str,
    log_path: Path,
    title: str,
) -> tuple[list[str], str, str, float]:
    print_header(title)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short natural-language SwitchTune workflow test.")
    parser.add_argument("--project_name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--protein_name", default="protein")
    parser.add_argument("--target_state_structure", required=True)
    parser.add_argument("--reference_state_structure", required=True)
    parser.add_argument("--target_state_name", default="target_state")
    parser.add_argument("--reference_state_name", default="reference_state")
    parser.add_argument("--parse_chains", default="A")
    parser.add_argument("--structure_engine", default=os.environ.get("SWITCHTUNE_STRUCTURE_ENGINE", "boltz2_local"))
    parser.add_argument("--ligand_codes", default="")
    parser.add_argument("--prompt_path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--mcp_config", default=os.environ.get("SWITCHTUNE_MCP_CONFIG", str(REPO / "mcp_config.example.yaml")))
    parser.add_argument("--biomni_data_path", default="./data")
    parser.add_argument("--llm", default=os.environ.get("SWITCHTUNE_LLM_MODEL", "your-model-id"))
    parser.add_argument("--llm_source", default=os.environ.get("SWITCHTUNE_LLM_SOURCE", "Custom"))
    parser.add_argument("--base_url", default=os.environ.get("SWITCHTUNE_LLM_BASE_URL", ""))
    parser.add_argument("--api_key_env", default="SWITCHTUNE_LLM_API_KEY")
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=120,
        help="Biomni/LLM request timeout. Keep separate from long-running external jobs.",
    )
    parser.add_argument(
        "--workflow_timeout_seconds",
        type=int,
        default=1800,
        help="Timeout for each long-running external SwitchTune job, including local Boltz-2.",
    )
    parser.add_argument("--max_steps", type=int, default=160)
    parser.add_argument("--max_execute_calls", type=int, default=48)
    parser.add_argument("--recursion_limit", type=int, default=220)
    parser.add_argument("--target_count", type=int, default=0)
    parser.add_argument("--bioemu_top_n", type=int, default=0)
    parser.add_argument("--bioemu_num_samples", type=int, default=20)
    parser.add_argument("--bioemu_batch_size", type=int, default=10)
    parser.add_argument("--bioemu_timeout_seconds", type=int, default=1800)
    parser.add_argument("--rosetta_top_n", type=int, default=0)
    parser.add_argument("--cb_timeout_seconds", type=int, default=3600)
    parser.add_argument("--force_existing_project", action="store_true")
    parser.add_argument(
        "--checkpoint_mode",
        action="store_true",
        help="Run a checkpointed workflow with Biomni reviews after structure confidence, IF, BioEmu, and Rosetta.",
    )
    parser.add_argument(
        "--debug_raw_agent",
        action="store_true",
        help="Disable SwitchTune guard auto-advance so repeated LLM behavior can be inspected.",
    )
    args = parser.parse_args()

    project_name = args.project_name
    run_dir = RUNS_ROOT / project_name
    report_dir = run_dir / "00_agent_reports"
    agent_log_path = report_dir / f"{project_name}_agent_go_log.txt"
    review_log_path = report_dir / f"{project_name}_agentic_review_log.txt"
    console_log_path = report_dir / f"{project_name}_console.log"
    summary_path = report_dir / f"{project_name}_agent_go_summary.json"
    final_csv = run_dir / EXPECTED_STAGE_FILES["final"]
    merged_csv = run_dir / EXPECTED_STAGE_FILES["merged"]
    agentic_review_packet_json = run_dir / EXPECTED_STAGE_FILES["agentic_review"]
    stage_review_json = run_dir / "08_final_recommendation" / "switchtune_stage_evidence_review_all.json"
    decision_log_path = run_dir / "decision_log.jsonl"

    print_header("SwitchTune Biomni workflow test")
    print(f"Python: {sys.executable}", flush=True)
    print(f"Repo: {REPO}", flush=True)
    print(f"Project: {project_name}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Prompt path: {args.prompt_path}", flush=True)
    print(f"Final CSV: {final_csv}", flush=True)
    print(f"Biomni timeout: {args.timeout_seconds}s", flush=True)
    print(f"External workflow timeout: {args.workflow_timeout_seconds}s", flush=True)

    if not REPO.exists():
        print(f"ERROR: repo not found: {REPO}", flush=True)
        return 2
    if args.api_key_env not in os.environ:
        print(f"ERROR: missing API key environment variable: {args.api_key_env}", flush=True)
        return 2
    if has_existing_workflow_artifacts(run_dir) and not args.force_existing_project:
        print(
            "ERROR: project directory already contains workflow artifacts. Use a new --project_name or pass "
            "--force_existing_project if this is an intentional resume/overwrite-style test.",
            flush=True,
        )
        return 2

    # Keep direct in-process SwitchTune tool calls and the MCP subprocess on the
    # same workspace/config roots.
    os.environ.setdefault("SWITCHTUNE_HOME", str(REPO))
    os.environ.setdefault("SWITCHTUNE_RUNS", str(RUNS_ROOT))
    os.environ.setdefault("SWITCHTUNE_CONFIG", str(REPO / "config" / "switchtune.yaml"))
    os.environ.setdefault("SWITCHTUNE_TOOL_REGISTRY", str(REPO / "config" / "tool_registry.yaml"))
    os.environ.setdefault("SWITCHTUNE_DECISION_POLICY", str(REPO / "config" / "decision_policy.yaml"))

    # Keep ESM-IF1 in the active agent environment.
    os.environ["ESM_IF1_ENV"] = "current"
    os.environ.setdefault("ESM_IF1_REPO", "./external/esm")
    os.environ.setdefault(
        "ESM_IF1_SCORE_SCRIPT",
        "./external/esm/examples/inverse_folding/score_log_likelihoods.py",
    )
    os.environ.setdefault("SWITCHTUNE_STRUCTURE_ENGINE", "boltz2_local")
    os.environ.setdefault("BOLTZ_CACHE", "./models/boltz2")

    prompt = PROMPT_TEMPLATE.format(
        project_name=project_name,
        protein_name=args.protein_name,
        target_state_structure=args.target_state_structure,
        reference_state_structure=args.reference_state_structure,
        target_state_name=args.target_state_name,
        reference_state_name=args.reference_state_name,
        parse_chains=args.parse_chains,
        structure_engine=args.structure_engine,
        workflow_mode_note=(
            "Checkpoint mode: the initial agent call should stop after front-half preparation; "
            "the harness will continue with staged reviews and downstream execution."
            if args.checkpoint_mode
            else "Full workflow mode: the agent may continue through the full pipeline when the guard allows it."
        ),
        target_count=args.target_count,
        bioemu_top_n=args.bioemu_top_n,
        bioemu_num_samples=args.bioemu_num_samples,
        rosetta_top_n=args.rosetta_top_n,
    )
    prompt_path = Path(args.prompt_path).expanduser()
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    print_header("Initialize Biomni")
    sys.path.insert(0, str(REPO))
    agent = make_biomni_agent(
        biomni_data_path=args.biomni_data_path,
        llm=args.llm,
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
        source=args.llm_source,
        mcp_config=args.mcp_config,
        timeout_seconds=args.timeout_seconds,
        max_steps=args.max_steps,
        max_execute_calls=args.max_execute_calls,
        recursion_limit=args.recursion_limit,
        thread_id=42,
        debug_raw_agent=args.debug_raw_agent,
        checkpoint_mode=args.checkpoint_mode,
        review_mode=False,
    )
    review_agent = make_biomni_agent(
        biomni_data_path=args.biomni_data_path,
        llm=args.llm,
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
        source=args.llm_source,
        mcp_config=args.mcp_config,
        timeout_seconds=args.timeout_seconds,
        max_steps=args.max_steps,
        max_execute_calls=args.max_execute_calls,
        recursion_limit=args.recursion_limit,
        thread_id=43,
        debug_raw_agent=args.debug_raw_agent,
        checkpoint_mode=False,
        review_mode=True,
    )

    print_header("Short Natural-Language Prompt")
    print(prompt, flush=True)

    log, answer, agent_exception, elapsed = run_agent_session(
        agent=agent,
        prompt=prompt,
        log_path=agent_log_path,
        title="Run agent.go",
    )

    checkpoint_report: dict[str, Any] = {}
    if args.checkpoint_mode:
        try:
            checkpoint_report = run_checkpoint_workflow(
                project_name=project_name,
                review_agent=review_agent,
                target_state_structure=args.target_state_structure,
                reference_state_structure=args.reference_state_structure,
                target_state_name=args.target_state_name,
                reference_state_name=args.reference_state_name,
                parse_chains=args.parse_chains,
                ligand_codes=args.ligand_codes,
                structure_engine=args.structure_engine,
                target_count=args.target_count,
                structure_top_n=0,
                bioemu_top_n=args.bioemu_top_n,
                bioemu_num_samples=args.bioemu_num_samples,
                bioemu_batch_size=args.bioemu_batch_size,
                bioemu_timeout_seconds=args.bioemu_timeout_seconds,
                rosetta_top_n=args.rosetta_top_n,
                run_external=True,
                timeout_seconds=args.workflow_timeout_seconds,
                cb_timeout_seconds=args.cb_timeout_seconds,
            )
        except Exception as exc:
            checkpoint_report = {"status": "failed", "error": repr(exc)}
            print(f"CHECKPOINT_WORKFLOW_EXCEPTION: {repr(exc)}", flush=True)

    stage_files = {name: str(run_dir / rel) for name, rel in EXPECTED_STAGE_FILES.items()}
    stage_exists = {name: Path(path).exists() for name, path in stage_files.items()}
    top_candidates = read_top_candidates(final_csv)
    log_text = "\n".join(map(str, log))
    guard_event_log_path = report_dir / "switchtune_go_guard_full_workflow_events.jsonl"
    guard_event_log_text = guard_event_log_path.read_text(encoding="utf-8") if guard_event_log_path.exists() else ""
    tracebacks = len(re.findall(r"\bTraceback\b|AGENT_EXCEPTION|RESUME_TEST_FORBIDDEN_TOOL_CALLED", log_text + agent_exception))
    tool_mentions = count_tool_mentions(guard_event_log_text)
    repeated_tool_mentions = {
        name: count
        for name, count in tool_mentions.items()
        if count > 8 and name not in {"recommend_next_workflow_step"}
    }
    initial_agent_passed = not agent_exception and tracebacks == 0
    checkpoint_workflow_passed = (
        bool(args.checkpoint_mode)
        and checkpoint_report.get("status") == "completed"
        and final_csv.exists()
        and bool(top_candidates)
    )
    workflow_passed = (
        initial_agent_passed
        and (
            checkpoint_workflow_passed
            if args.checkpoint_mode
            else final_csv.exists() and bool(top_candidates)
        )
    )

    review_prompt_path = AGENTIC_REVIEW_PROMPT_PATH.expanduser()
    review_prompt = ""
    review_log = []
    review_answer = ""
    review_exception = ""
    review_elapsed = 0.0
    review_log_text = ""
    review_guard_event_text = ""
    review_guard_event_text_before = ""
    review_tracebacks = 0
    review_tool_mentions: dict[str, int] = {}
    latest_decision = {}
    final_decision_before_review = {}
    review_passed = False
    review_skipped_reason = ""
    final_decision_before_review = read_latest_final_round_decision(decision_log_path)
    if workflow_passed and final_decision_before_review:
        review_passed = True
        latest_decision = final_decision_before_review
        review_answer = "Skipped final agentic review because a final-round decision was already recorded."
        review_skipped_reason = "existing_final_round_decision"
    elif workflow_passed:
        from switchtune.mcp_server import prepare_agentic_candidate_review

        decision_count_before = len(
            [
                line
                for line in decision_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        ) if decision_log_path.exists() else 0
        review_guard_event_text_before = (
            guard_event_log_path.read_text(encoding="utf-8") if guard_event_log_path.exists() else ""
        )
        review_packet = prepare_agentic_candidate_review(
            project_name=project_name,
            evidence_csv=str(merged_csv),
            top_n=max(1, csv_row_count(final_csv)),
            include_stage_reviews=True,
        )
        review_prompt = build_agentic_review_prompt(
            project_name=project_name,
            merged_csv=str(merged_csv),
            final_csv=str(final_csv),
            review_packet_json=str(review_packet.get("packet_json") or agentic_review_packet_json),
            review_packet=review_packet,
            top_candidates=read_top_candidates(final_csv, limit=max(1, csv_row_count(final_csv))),
        )
        agentic_review_packet_json = Path(str(review_packet.get("packet_json") or agentic_review_packet_json))
        review_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        review_prompt_path.write_text(review_prompt, encoding="utf-8")
        review_agent._switchtune_thread_id = f"switchtune-{project_name}-final-round"
        review_log, review_answer, review_exception, review_elapsed = run_agent_session(
            agent=review_agent,
            prompt=review_prompt,
            log_path=review_log_path,
            title="Run agent.go (agentic review)",
        )
        review_log_text = "\n".join(map(str, review_log))
        review_guard_event_text = guard_event_log_path.read_text(encoding="utf-8") if guard_event_log_path.exists() else ""
        if review_guard_event_text_before and review_guard_event_text.startswith(review_guard_event_text_before):
            review_guard_event_text = review_guard_event_text[len(review_guard_event_text_before) :]
        review_tracebacks = len(
            re.findall(r"\bTraceback\b|AGENT_EXCEPTION|RESUME_TEST_FORBIDDEN_TOOL_CALLED", review_log_text + review_exception)
        )
        review_tool_mentions = count_tool_mentions(review_guard_event_text + "\n" + review_log_text)
        latest_decision = read_latest_final_round_decision(decision_log_path)
        decision_count_after = len(
            [
                line
                for line in decision_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        ) if decision_log_path.exists() else 0
        review_passed = (
            not review_exception
            and review_tracebacks == 0
            and review_tool_mentions.get("record_agentic_candidate_decision", 0) > 0
            and review_tool_mentions.get("review_switchtune_stage_evidence", 0) == 0
            and review_tool_mentions.get("prepare_agentic_candidate_review", 0) == 0
            and decision_count_after > decision_count_before
            and bool(latest_decision)
        )
    else:
        review_skipped_reason = "workflow_failed"

    summary = {
        "status": "passed" if (workflow_passed and review_passed) else "failed",
        "project_name": project_name,
        "checkpoint_mode": args.checkpoint_mode,
        "initial_agent_passed": initial_agent_passed,
        "checkpoint_workflow_passed": checkpoint_workflow_passed,
        "workflow_passed": workflow_passed,
        "review_passed": review_passed,
        "elapsed_seconds": round(elapsed, 1),
        "review_elapsed_seconds": round(review_elapsed, 1),
        "checkpoint_report": checkpoint_report,
        "agent_exception": agent_exception,
        "review_exception": review_exception,
        "traceback_like_count": tracebacks,
        "review_traceback_like_count": review_tracebacks,
        "final_csv": str(final_csv),
        "final_csv_exists": final_csv.exists(),
        "top_candidates": top_candidates,
        "stage_exists": stage_exists,
        "stage_files": stage_files,
        "tool_mentions": tool_mentions,
        "review_tool_mentions": review_tool_mentions,
        "possible_repeated_tool_mentions": repeated_tool_mentions,
        "agent_log_path": str(agent_log_path),
        "review_log_path": str(review_log_path),
        "guard_event_log_path": str(guard_event_log_path),
        "guard_event_log_exists": guard_event_log_path.exists(),
        "review_guard_event_log_path": str(guard_event_log_path),
        "console_log_path": str(console_log_path),
        "prompt_path": str(prompt_path),
        "review_prompt_path": str(review_prompt_path),
        "merged_csv": str(merged_csv),
        "agentic_review_packet_json": str(agentic_review_packet_json),
        "stage_review_json": str(stage_review_json),
        "decision_log_path": str(decision_log_path),
        "latest_agentic_decision": latest_decision,
        "review_skipped_reason": review_skipped_reason,
        "final_decision_before_review": final_decision_before_review,
    }
    write_json(summary_path, summary)

    print_header("Final Answer")
    print(answer, flush=True)
    if review_prompt:
        print_header("Agentic Review Answer")
        print(review_answer, flush=True)
    print_header("Short Prompt Workflow Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Agent log saved to: {agent_log_path}", flush=True)
    if review_prompt:
        print(f"Agentic review log saved to: {review_log_path}", flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)
    print(f"SHORT_PROMPT_FULL_WORKFLOW_TEST: {'PASSED' if workflow_passed and review_passed else 'FAILED'}", flush=True)
    return 0 if workflow_passed and review_passed else 1


if __name__ == "__main__":
    rc = main()
    print(f"\nSCRIPT_FINISHED_WITH_EXIT_CODE={rc}", flush=True)
    raise SystemExit(rc)
