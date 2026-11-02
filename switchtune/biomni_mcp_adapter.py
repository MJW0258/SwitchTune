from __future__ import annotations

import builtins
import json
import sys
from functools import wraps
from typing import Any, Iterable, Optional


DEFAULT_TOOL_NAMES = (
    "create_project",
    "parse_design_prompt",
    "write_decision_log",
    "validate_inputs",
    "generate_saturation_library",
    "summarize_library",
    "filter_library_by_constraints",
    "setup_saturation_screening_project",
    "prepare_bias_scoring_inputs",
    "inspect_bias_score_distribution",
    "recommend_bias_threshold",
    "run_conformation_bias_scoring",
    "setup_two_state_cb_scoring_project",
    "setup_two_state_cb_scoring_project_once",
    "inspect_two_state_pdb_pair",
    "execute_conformation_bias_scoring_job",
    "extract_conformation_bias_scores",
    "select_state_specific_candidates",
    "select_state_specific_candidates_adaptive",
    "select_structure_prediction_engine",
    "prepare_structure_prediction_job",
    "collect_structure_prediction_results",
    "standardize_structure_prediction_results",
    "compare_structure_predictions_to_states",
    "filter_structure_confidence",
    "prepare_inverse_folding_scoring_job",
    "collect_inverse_folding_scores",
    "select_candidates_from_evidence",
    "prepare_bioemu_sampling_job",
    "collect_bioemu_outputs",
    "select_rosetta_analysis_strategy",
    "prepare_rosetta_energy_job",
    "collect_rosetta_scores",
    "merge_switchtune_evidence_tables",
    "prepare_final_candidate_recommendation",
    "review_switchtune_stage_evidence",
    "prepare_agentic_candidate_review",
    "record_agentic_candidate_decision",
    "recommend_next_workflow_step",
    "get_job_status",
    "get_project_summary",
)


def _parse_json_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return text
    try:
        return json.loads(stripped)
    except Exception:
        return text


def parse_biomni_mcp_result(value: Any) -> Any:
    """Convert Biomni MCP text-content wrappers back into Python objects.

    Biomni's current MCP wrapper commonly returns a JSON string like:
    {"type":"text", "text":"{...actual tool result...}"}

    This adapter unwraps that object and parses the inner `text` JSON so agent-generated
    code can safely call `.get()` on tool results.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return parse_biomni_mcp_result(value[0])
        return [parse_biomni_mcp_result(item) for item in value]

    if isinstance(value, dict):
        if "text" in value:
            return parse_biomni_mcp_result(value.get("text"))
        return value

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return parse_biomni_mcp_result(text_attr)

    if not isinstance(value, str):
        return value

    outer = _parse_json_text(value)
    if outer is value:
        return value

    if isinstance(outer, dict) and "text" in outer:
        return parse_biomni_mcp_result(outer.get("text"))

    return outer


def _wrap_function(func):
    if getattr(func, "_switchtune_result_patched", False):
        return func

    @wraps(func)
    def wrapper(*args, __func=func, **kwargs):
        return parse_biomni_mcp_result(__func(*args, **kwargs))

    wrapper._switchtune_result_patched = True  # type: ignore[attr-defined]
    return wrapper


def patch_switchtune_mcp_results(
    agent: Any = None,
    module_name: str = "mcp_servers.switchtune",
    tool_names: Optional[Iterable[str]] = None,
) -> Any:
    """Patch Biomni's dynamic MCP module so SwitchTune tools return dicts.

    Call this after `agent.add_mcp(...)`:

        from switchtune.biomni_mcp_adapter import patch_switchtune_mcp_results
        patch_switchtune_mcp_results(agent)
    """
    if module_name not in sys.modules:
        raise RuntimeError(
            f"{module_name!r} is not loaded. Call agent.add_mcp(...) before patching."
        )

    module = sys.modules[module_name]
    names = tuple(tool_names or DEFAULT_TOOL_NAMES)

    for name in names:
        func = getattr(module, name, None)
        if func is None:
            continue
        wrapper = _wrap_function(func)
        setattr(module, name, wrapper)

        if agent is not None and hasattr(agent, "_custom_functions"):
            if name in agent._custom_functions:
                agent._custom_functions[name] = wrapper

        if hasattr(builtins, "_biomni_custom_functions"):
            if name in builtins._biomni_custom_functions:
                builtins._biomni_custom_functions[name] = wrapper

        if agent is not None and hasattr(agent, "module2api"):
            for tool_list in agent.module2api.values():
                for tool_schema in tool_list:
                    if isinstance(tool_schema, dict) and tool_schema.get("name") == name:
                        tool_schema["fn"] = wrapper

    return module
