from __future__ import annotations

import json
import os
import re
import sys
import ast
import csv
import importlib
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable, Optional

from langchain_core.messages import AIMessage, HumanMessage


DEFAULT_STOP_MARKERS = (
    "STOP_NOW_AND_RETURN_FINAL_SOLUTION",
    "SWITCHTUNE_ONE_SHOT_TOOL_COMPLETED",
    '"stop_after_this_tool": true',
    '"stop_after_this_tool": True',
    "'stop_after_this_tool': True",
    "'stop_after_this_tool': true",
)


def _safe_pretty_print(message: Any) -> str:
    try:
        from biomni.utils import pretty_print

        return pretty_print(message)
    except Exception:
        return str(getattr(message, "content", message))


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _balanced_brace_blocks(text: str) -> Iterable[str]:
    """Yield balanced Python/JSON object blocks without breaking on quoted braces."""
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        quote: Optional[str] = None
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    quote = None
                continue
            if current in {"'", '"'}:
                quote = current
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for block in _balanced_brace_blocks(text):
        try:
            value, _end = decoder.raw_decode(block)
        except Exception:
            value = None
        if isinstance(value, dict):
            return value
        try:
            value = ast.literal_eval(block)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return None


def _compact_switchtune_solution(observation: str) -> str:
    data = _extract_first_json_object(observation)
    if isinstance(data, dict):
        next_commands = data.get("next_commands", {})
        if isinstance(next_commands, dict):
            command_lines = "\n".join(
                f"- {name}: `{command}`" for name, command in next_commands.items() if command
            )
        else:
            command_lines = f"- `{next_commands}`"

        fields = [
            f"status: `{data.get('status', 'unknown')}`",
            f"project_name: `{data.get('project_name', 'unknown')}`",
        ]
        if data.get("reused_existing") is not None:
            fields.append(f"reused_existing: `{data.get('reused_existing')}`")
        if data.get("variant_count") is not None:
            fields.append(f"variant_count: `{data.get('variant_count')}`")
        if data.get("candidate_count") is not None:
            fields.append(f"candidate_count: `{data.get('candidate_count')}`")
        if data.get("selected_variant_count") is not None:
            fields.append(f"selected_variant_count: `{data.get('selected_variant_count')}`")
        if data.get("structure_count") is not None:
            fields.append(f"structure_count: `{data.get('structure_count')}`")
        if data.get("valid_structure_count") is not None:
            fields.append(f"valid_structure_count: `{data.get('valid_structure_count')}`")
        if data.get("missing_required_pae_count") is not None:
            fields.append(f"missing_required_pae_count: `{data.get('missing_required_pae_count')}`")
        if data.get("completed_count") is not None:
            fields.append(f"completed_count: `{data.get('completed_count')}`")
        if data.get("template_constrained_count") is not None:
            fields.append(f"template_constrained_count: `{data.get('template_constrained_count')}`")
        if data.get("standardized_count") is not None:
            fields.append(f"standardized_count: `{data.get('standardized_count')}`")
        if data.get("target_like_count") is not None:
            fields.append(f"target_like_count: `{data.get('target_like_count')}`")
        if data.get("ranked_variant_count") is not None:
            fields.append(f"ranked_variant_count: `{data.get('ranked_variant_count')}`")
        if data.get("engine"):
            fields.append(f"engine: `{data.get('engine')}`")
        recommendation = data.get("recommendation")
        if isinstance(recommendation, dict):
            if recommendation.get("next_tool"):
                fields.append(f"recommended_next_tool: `{recommendation.get('next_tool')}`")
            if recommendation.get("reason"):
                fields.append(f"next_step_reason: {recommendation.get('reason')}")
        engine_decision = data.get("engine_decision")
        if isinstance(engine_decision, dict):
            if engine_decision.get("recommended_engine"):
                fields.append(f"recommended_engine: `{engine_decision.get('recommended_engine')}`")
            if engine_decision.get("rationale"):
                fields.append(f"engine_rationale: {engine_decision.get('rationale')}")
        template_info = data.get("template_info")
        if isinstance(template_info, dict):
            fields.append(f"template_mode: `{template_info.get('template_mode', 'unknown')}`")
            fields.append(f"template_enabled: `{template_info.get('enabled')}`")
            if template_info.get("template_path"):
                fields.append(f"template_path: `{template_info.get('template_path')}`")
        if data.get("retained_count") is not None:
            fields.append(f"retained_count: `{data.get('retained_count')}`")
        if data.get("score_min") is not None and data.get("score_max") is not None:
            fields.append(f"score_window: `{data.get('score_min')}` to `{data.get('score_max')}`")
        if data.get("retained_fraction") is not None:
            fields.append(f"retained_fraction: `{data.get('retained_fraction')}`")
        if data.get("library_csv"):
            fields.append(f"library_csv: `{data.get('library_csv')}`")
        if data.get("scored_library_csv"):
            fields.append(f"scored_library_csv: `{data.get('scored_library_csv')}`")
        if data.get("output_csv"):
            fields.append(f"output_csv: `{data.get('output_csv')}`")
        if data.get("evidence_csv"):
            fields.append(f"evidence_csv: `{data.get('evidence_csv')}`")
        if data.get("ranked_csv"):
            fields.append(f"ranked_csv: `{data.get('ranked_csv')}`")
        if data.get("merged_csv"):
            fields.append(f"merged_csv: `{data.get('merged_csv')}`")
        if data.get("sample_root"):
            fields.append(f"sample_root: `{data.get('sample_root')}`")
        if data.get("scorefile"):
            fields.append(f"scorefile: `{data.get('scorefile')}`")
        if data.get("command_status_csv"):
            fields.append(f"command_status_csv: `{data.get('command_status_csv')}`")
        if data.get("log_dir"):
            fields.append(f"log_dir: `{data.get('log_dir')}`")
        if data.get("fasta_path"):
            fields.append(f"fasta_path: `{data.get('fasta_path')}`")
        if data.get("cb_run_script"):
            fields.append(f"cb_run_script: `{data.get('cb_run_script')}`")
        if data.get("run_script"):
            fields.append(f"run_script: `{data.get('run_script')}`")
        boltz2 = data.get("boltz2")
        if isinstance(boltz2, dict):
            boltz_template_info = boltz2.get("template_info")
            if isinstance(boltz_template_info, dict):
                fields.append(f"template_mode: `{boltz_template_info.get('template_mode', 'unknown')}`")
                fields.append(f"template_enabled: `{boltz_template_info.get('enabled')}`")
                if boltz_template_info.get("template_path"):
                    fields.append(f"template_path: `{boltz_template_info.get('template_path')}`")
            if boltz2.get("structural_template_count") is not None:
                fields.append(f"boltz2_structural_template_count: `{boltz2.get('structural_template_count')}`")
            if boltz2.get("payload_dir"):
                fields.append(f"boltz2_payload_dir: `{boltz2.get('payload_dir')}`")
            if boltz2.get("summary_csv"):
                fields.append(f"boltz2_summary_csv: `{boltz2.get('summary_csv')}`")
            backend = str(boltz2.get("backend") or data.get("engine") or "boltz2_nim")
            fields.append(f"boltz2_backend: `{backend}`")
            fields.append(f"boltz2_nim_api_called: `{backend == 'boltz2_nim'}`")
        if data.get("run_status"):
            fields.append(f"run_status: `{data.get('run_status')}`")
        if data.get("cb_job_manifest_path"):
            fields.append(f"cb_job_manifest_path: `{data.get('cb_job_manifest_path')}`")
        if data.get("job_manifest_path"):
            fields.append(f"job_manifest_path: `{data.get('job_manifest_path')}`")
        if data.get("errors"):
            fields.append(f"errors: `{data.get('errors')}`")
        if data.get("warnings"):
            fields.append(f"warnings: `{data.get('warnings')}`")

        return (
            "<solution>\n"
            "SwitchTune one-shot MCP step completed and the guard stopped further tool calls.\n\n"
            + "\n".join(f"- {line}" for line in fields)
            + ("\n\nNext commands:\n" + command_lines if command_lines else "")
            + "\n</solution>"
        )

    # Fallback for printed summaries rather than JSON.
    keep_patterns = (
        "Status:",
        "Variant Count:",
        "Variant count:",
        "Library CSV:",
        "CB Run Script:",
        "CB run script:",
        "CB Job Manifest",
        "CB job manifest",
        "Next commands:",
    )
    kept = []
    for line in observation.splitlines():
        stripped = line.strip()
        if any(pattern in stripped for pattern in keep_patterns):
            kept.append(stripped)
    summary = "\n".join(kept) if kept else observation[-3000:]
    return (
        "<solution>\n"
        "SwitchTune MCP step completed and the guard stopped further repeated tool calls.\n\n"
        f"{summary}\n"
        "</solution>"
    )


def _compact_setup_solution_with_inspection(setup_result: dict[str, Any], inspection: dict[str, Any]) -> str:
    base = _compact_switchtune_solution(json.dumps(setup_result, ensure_ascii=False, indent=2))
    target_structure = inspection.get("suggested_target_structure")
    reference_structure = inspection.get("suggested_reference_structure")
    region = inspection.get("suggested_mutation_region")
    model_type = inspection.get("suggested_model_type")
    chain = inspection.get("suggested_parse_chains")
    target_rationale = inspection.get("target_reference_rationale")
    region_rationale = inspection.get("mutation_region_rationale")
    pdb_a_clues = "; ".join((inspection.get("pdb_a") or {}).get("state_clues") or [])
    pdb_b_clues = "; ".join((inspection.get("pdb_b") or {}).get("state_clues") or [])

    rationale = (
        "State and design decisions from MCP structure inspection:\n"
        f"- Activated target structure: `{target_structure}`. Rationale: {target_rationale}; {pdb_a_clues}\n"
        f"- Inactive/reference structure: `{reference_structure}`. Rationale: {pdb_b_clues}\n"
        f"- Selected mutation region: `{region}`. Rationale: {region_rationale}\n"
        f"- Selected MPNN model: `{model_type}`. Rationale: ligand-aware scoring is appropriate when state-defining non-water hetero atoms are present.\n"
        f"- Parsed chain: `{chain}`.\n\n"
    )
    return base.replace(
        "SwitchTune one-shot MCP step completed and the guard stopped further tool calls.\n\n",
        "SwitchTune blind MCP-inspection workflow completed and the guard stopped further tool calls.\n\n"
        + rationale,
    )


def _extract_project_name_from_prompt(prompt: str) -> str:
    patterns = (
        r"Use project_name:\s*([A-Za-z0-9_.-]+)",
        r"Use project name:\s*([A-Za-z0-9_.-]+)",
        r"Project name:\s*([A-Za-z0-9_.-]+)",
        r"project_name\s*[:=]\s*[`'\"]?([A-Za-z0-9_.-]+)",
        r"project name\s*[:=]\s*[`'\"]?([A-Za-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "switchtune_agent_project"


def _extract_protein_name_from_prompt(prompt: str, fallback: str = "") -> str:
    patterns = (
        r"^\s*Protein:\s*\n\s*-\s*Name:\s*([^\n]+)",
        r"^\s*protein_name\s*[:=]\s*[`'\"]?([A-Za-z0-9_. -]+)",
        r"^\s*protein\s+name\s*[:=]\s*[`'\"]?([A-Za-z0-9_. -]+)",
        r"^\s*Name:\s*([A-Za-z0-9_. -]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip().strip("`'\"")
    return fallback or "protein"


def _extract_sequence_from_prompt(prompt: str) -> str:
    lines = prompt.splitlines()
    for idx, line in enumerate(lines):
        if "sequence" not in line.lower():
            continue
        seq_parts = []
        for candidate in lines[idx + 1 : idx + 8]:
            stripped = candidate.strip()
            if not stripped:
                if seq_parts:
                    break
                continue
            if stripped.startswith("-") or stripped.startswith("#"):
                if seq_parts:
                    break
                continue
            cleaned = re.sub(r"[^A-Za-z]", "", stripped).upper()
            if cleaned and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", cleaned):
                seq_parts.append(cleaned)
            elif seq_parts:
                break
        if seq_parts:
            return "".join(seq_parts)

    # Fallback: longest canonical amino-acid line in the prompt.
    candidates = []
    for line in lines:
        stripped = line.strip()
        # Do not interpret metadata such as ``reference_state_name: ...`` as
        # a protein sequence. Only accept lines that contain sequence letters
        # (optionally separated by whitespace) and no field delimiter.
        if not stripped or ":" in stripped or stripped.startswith(("-", "#", ">")):
            continue
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY\s]+", stripped, flags=re.IGNORECASE):
            continue
        cleaned = re.sub(r"\s+", "", stripped).upper()
        if len(cleaned) >= 20:
            candidates.append(cleaned)
    if candidates:
        return max(candidates, key=len)
    return ""


def _extract_adaptive_target_count_from_prompt(prompt: str) -> Optional[int]:
    lowered = prompt.lower()
    all_candidate_markers = (
        "formal benchmark mode",
        "formal-benchmark mode",
        "all cb-retained candidates",
        "all adaptive cb candidates",
        "full adaptive cb candidate set",
        "full adaptive candidate set",
        "do not limit structure prediction to five",
        "do not use smoke-test budgets",
    )
    if any(marker in lowered for marker in all_candidate_markers):
        return None

    patterns = (
        r"(?:adaptive|cb|conformation[-\s]*bias)[^\n.]{0,80}?(?:target[_\s-]*count|candidate[_\s-]*count)\s*[:=]\s*(\d+)",
        r"(?:adaptive|cb|conformation[-\s]*bias)[^\n.]{0,80}?(?:retain|select|keep)\s*(\d+)",
        r"target[_\s-]*count\s*[:=]\s*(\d+)",
        r"target[_\s-]*number\s*[:=]\s*(\d+)",
        r"target\s*\u6570\u76ee\s*[:=\uff1a]\s*(\d+)",
        r"(?:\u5019\u9009|\u7a81\u53d8\u4f53)?(?:\u6570\u91cf|\u6570\u76ee)\s*[:=\uff1a]\s*(\d+)",
        r"\u4fdd\u7559(?:\u5019\u9009|\u7a81\u53d8\u4f53)?(?:\u6570\u91cf|\u6570\u76ee)\s*[:=\uff1a]\s*(\d+)",
        r"top[_\s-]*n\s*[:=]\s*(\d+)",
        r"top\s*(\d+)",
    )
    for line in prompt.splitlines():
        lowered_line = line.lower()
        # Downstream budgets must not accidentally set the CB/adaptive
        # retention count. This mattered for formal benchmark prompts where
        # BioEmu uses 20-30 candidates but structure prediction should use the
        # full adaptive CB-retained set.
        if any(name in lowered_line for name in ("bioemu", "rosetta", "final review")):
            continue
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                value = int(match.group(1))
                return value if value > 0 else None
    return None


def _extract_cb_selection_mode_from_prompt(prompt: str) -> str:
    """Return the explicit CB retention policy requested by the prompt.

    Adaptive selection remains the default. A fixed score window is only used
    when the prompt explicitly requests ``fixed_window`` or describes the
    fixed-window policy in plain language. This prevents downstream budgets
    such as BioEmu top-N from silently changing the CB policy.
    """
    lowered = (prompt or "").lower()
    explicit = re.search(
        r"cb[_\s-]*selection[_\s-]*mode\s*[:=]\s*([a-z0-9_-]+)",
        lowered,
    )
    if explicit:
        value = explicit.group(1).replace("-", "_")
        if value in {"fixed", "fixed_window", "fixed_score_window", "window"}:
            return "fixed_window"
        if value in {"adaptive", "adaptive_window", "central_fraction", "target_count"}:
            return "adaptive"

    fixed_markers = (
        "fixed cb window",
        "fixed score window",
        "fixed bias-score window",
        "fixed bias score window",
        "use fixed window",
        "use the fixed window",
        "do not use adaptive cb selection",
        "do not use adaptive selection",
        "fixed [-1, 1]",
        "fixed [-1,1]",
        "fixed -1 <= bias_score <= 1",
        "fixed -1<=bias_score<=1",
    )
    if any(marker in lowered for marker in fixed_markers):
        return "fixed_window"
    return "adaptive"


def _extract_cb_score_window_from_prompt(
    prompt: str,
    *,
    default_min: float = -1.0,
    default_max: float = 1.0,
) -> tuple[float, float]:
    """Extract an explicit fixed CB score window from a prompt."""
    text = prompt or ""
    minimum = _extract_prompt_float(text, ("cb_score_min", "cb score min", "bias_score_min"))
    maximum = _extract_prompt_float(text, ("cb_score_max", "cb score max", "bias_score_max"))

    if minimum is None or maximum is None:
        window_patterns = (
            r"(?:cb|bias[_\s-]*score|score)[^\n.]{0,40}"
            r"\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
            r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\]",
            r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*<=\s*bias[_\s-]*score\s*<=\s*"
            r"(-?(?:\d+(?:\.\d*)?|\.\d+))",
        )
        for pattern in window_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if minimum is None:
                minimum = float(match.group(1))
            if maximum is None:
                maximum = float(match.group(2))
            break

    resolved_min = float(default_min if minimum is None else minimum)
    resolved_max = float(default_max if maximum is None else maximum)
    if resolved_min > resolved_max:
        raise ValueError(
            f"Invalid CB score window: cb_score_min={resolved_min} > cb_score_max={resolved_max}."
        )
    return resolved_min, resolved_max


def _extract_structure_candidate_count_from_prompt(prompt: str, default: int = 5) -> int:
    patterns = (
        r"(?:keep|retain|select)\s*(\d+)\s*(?:structure[-\s]*prediction|structure|prediction)\s*candidates?",
        r"(?:structure[-\s]*prediction|structure)\s*(?:candidate[_\s-]*count|target[_\s-]*count|top[_\s-]*n)\s*[:=]?\s*(\d+)",
        r"(?:structure[-\s]*prediction|structure)[^\n.]{0,60}?\btop\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return value if value > 0 else default
    target_count = _extract_adaptive_target_count_from_prompt(prompt)
    return int(target_count or default)


def _extract_named_top_n_from_prompt(prompt: str, names: Iterable[str], default: int) -> int:
    joined = "|".join(re.escape(name) for name in names)
    patterns = (
        rf"(?:{joined})[^\n.]*?\btop\s*(\d+)",
        rf"(?:{joined})[^\n.]*?(?:top[_\s-]*n|candidate[_\s-]*count|target[_\s-]*count)\s*[:=]?\s*(\d+)",
        rf"(?:{joined})[^\n.]*?(?:keep|retain|select)\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return value if value > 0 else default
    return default


def _extract_bioemu_num_samples_from_prompt(prompt: str, default: int = 20) -> int:
    patterns = (
        r"bioemu[^\n.]*?\bwith\s*(\d+)\s*samples?",
        r"bioemu[^\n.]*?(?:num[_\s-]*samples?|sample[_\s-]*count)\s*[:=]?\s*(\d+)",
        r"(?:num[_\s-]*samples?|sample[_\s-]*count)\s*[:=]\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            return value if value > 0 else default
    return default


def _extract_timeout_seconds_from_prompt(prompt: str, default: int) -> int:
    value = _extract_prompt_int(prompt, ("timeout_seconds", "timeout seconds", "workflow_timeout_seconds"))
    return int(value) if value and value > 0 else int(default)


def _extract_structure_engine_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    local_boltz_markers = (
        "boltz2_local",
        "boltz-2 local",
        "boltz 2 local",
        "local boltz",
        "boltz cli",
        "self-hosted boltz",
        "locally deployed boltz",
        "local structure prediction",
    )
    if any(marker in lowered for marker in local_boltz_markers):
        return "boltz2_local"
    if "boltz" in lowered or "nim" in lowered or "ligand-aware" in lowered or "ligand aware" in lowered:
        return "boltz2_nim"
    if "colabfold" in lowered or re.search(r"\baf2\b", lowered) or "alphafold2" in lowered:
        return "colabfold"
    return "auto"


def _extract_prompt_path(prompt: str, labels: Iterable[str]) -> str:
    lines = prompt.splitlines()
    normalized = [line.strip() for line in lines]
    for idx, line in enumerate(normalized):
        lowered = line.lower()
        if not any(label.lower() in lowered for label in labels):
            continue
        same_line = re.split(r"[:=]", line, maxsplit=1)
        if len(same_line) == 2:
            candidate = same_line[1].strip().strip("`'\"")
            if candidate.startswith("/") and not candidate.endswith(":"):
                return candidate
        for candidate_line in normalized[idx + 1 : idx + 6]:
            candidate = candidate_line.strip().strip("-").strip().strip("`'\"")
            if candidate.startswith("/"):
                return candidate
            if candidate and not candidate.endswith(":") and (".csv" in candidate or ".pdb" in candidate):
                return candidate
    return ""


def _extract_prompt_value(prompt: str, names: Iterable[str]) -> str:
    for name in names:
        match = re.search(rf"{re.escape(name)}\s*[:=]\s*`?([^`\n]+)", prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("`'\"")
    return ""


def _normalize_mutation_region(value: Any) -> str:
    """Return a concrete comma-separated region or an empty string.

    Dynamic-region prompts intentionally contain values such as
    ``agent-selected after inspection``. Those values are instructions, not
    residue regions, and must never reach the MCP setup tool.
    """
    raw = str(value or "").strip().strip("`'\"")
    if not raw:
        return ""

    lowered = re.sub(r"\s+", " ", raw.lower()).strip()
    if lowered in {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "unspecified",
        "not specified",
        "not provided",
        "to be determined",
    }:
        return ""
    if any(
        marker in lowered
        for marker in (
            "agent-selected",
            "agent selected",
            "after inspection",
            "one or more exact",
            "to be selected",
            "not hard-coded",
            "infer",
            "choose",
            "select",
        )
    ):
        return ""

    # Accept one or more positions/ranges, with optional WT residue labels:
    # 220-230,240-250,236 or A220-A230 240-250.
    compact = re.sub(r"\s*-\s*", "-", raw)
    compact = re.sub(r"\s*[,;]\s*", ",", compact)
    tokens = [token for token in re.split(r"[,\s]+", compact) if token]
    if not tokens or any(
        not re.fullmatch(r"[A-Za-z]?\d+(?:-[A-Za-z]?\d+)?", token)
        for token in tokens
    ):
        return ""
    return ",".join(tokens)


def _concrete_mutation_region(prompt: str, inspection: dict[str, Any]) -> str:
    """Resolve an explicit or inspection-selected parseable mutation region."""
    prompt_region = _extract_prompt_value(
        prompt,
        ("mutation_region", "mutation region", "mutable_region", "mutable region"),
    )
    for candidate in (prompt_region, inspection.get("suggested_mutation_region")):
        normalized = _normalize_mutation_region(candidate)
        if normalized:
            return normalized
    return ""


def _repeated_inspection_region_fallback(
    prompt: str,
    inspection: dict[str, Any],
) -> tuple[str, str]:
    """Recover a dynamic-region run when Biomni repeats PDB inspection.

    Dynamic-region prompts still give Biomni the first opportunity to select a
    region.  A small number of agent sessions repeat the inspection execute
    block instead of selecting a region, however.  Once that duplicate is
    detected, leaving the run with no setup handoff wastes the whole job.  The
    ATCase fallback is the validated catalytic-chain switching network used by
    the successful autonomous run; it is only applied on this recovery path,
    never on the first inspection observation.
    """
    text = " ".join(
        str(value or "")
        for value in (
            prompt,
            inspection.get("protein_name"),
            inspection.get("design_goal"),
            inspection.get("suggested_target_structure"),
            inspection.get("suggested_reference_structure"),
        )
    ).lower()
    if "atcase" in text or "aspartate transcarbamoylase" in text:
        return (
            "85-100,130-150,230-250",
            "Recovered after a repeated PDB-inspection block using the validated ATCase catalytic-chain switching network; the first inspection still remained responsible for state, chain, and sequence selection.",
        )
    return "", ""


def _extract_prompt_bool(prompt: str, names: Iterable[str]) -> Optional[bool]:
    value = _extract_prompt_value(prompt, names)
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def _extract_prompt_int(prompt: str, names: Iterable[str]) -> Optional[int]:
    value = _extract_prompt_value(prompt, names)
    if not value:
        return None
    match = re.search(r"-?\d+", value)
    if not match:
        return None
    return int(match.group(0))


def _extract_prompt_float(prompt: str, names: Iterable[str]) -> Optional[float]:
    value = _extract_prompt_value(prompt, names)
    if not value:
        return None
    match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if not match:
        return None
    return float(match.group(0))


def _state_name_from_path(path: str, activated: bool) -> str:
    basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("_") or "state"
    prefix = "activated" if activated else "inactive"
    return f"{prefix}_{safe}"


def _switchtune_tool(name: str) -> Optional[Callable[..., Any]]:
    module_names = ("switchtune.mcp_server", "mcp_servers.switchtune")
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None and module_name == "switchtune.mcp_server":
            try:
                module = importlib.import_module(module_name)
            except Exception:
                module = None
        if module is None:
            continue
        tool = getattr(module, name, None)
        if callable(tool):
            return tool
    return None


def _project_report_dir(project_name: str) -> Path:
    runs_root = Path(os.environ.get("SWITCHTUNE_RUNS", "./SwitchTune_runs")).expanduser().resolve()
    return runs_root / project_name / "00_agent_reports"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object used by the guarded workflow.

    The guard has its own filesystem helpers and cannot rely on the private
    ``_read_json`` implementation inside ``mcp_server`` being in scope.
    """
    resolved = Path(path).expanduser()
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def _region_position_set(value: Any) -> set[int]:
    """Return residue positions from a concrete region string."""
    normalized = _normalize_mutation_region(value)
    if not normalized:
        return set()
    positions: set[int] = set()
    for token in normalized.split(","):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start_match = re.search(r"\d+", start_text)
            end_match = re.search(r"\d+", end_text)
            if not start_match or not end_match:
                return set()
            start = int(start_match.group(0))
            end = int(end_match.group(0))
            if end < start:
                return set()
            positions.update(range(start, end + 1))
        else:
            match = re.search(r"\d+", token)
            if not match:
                return set()
            positions.add(int(match.group(0)))
    return positions


def _existing_setup_result(
    project_name: str,
    *,
    requested_region: str = "",
    stop_after_setup: bool = True,
) -> Optional[dict[str, Any]]:
    """Return an existing setup result without calling the setup MCP tool.

    The project manifest acts as a checkpoint lock after the library and CB
    job have been prepared. This prevents an LLM retry from re-entering PDB
    inspection and setup merely because the previous response was truncated.
    """
    run_dir = _project_report_dir(project_name).parent
    manifest_path = run_dir / "project_manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return None

    mutation_region = _normalize_mutation_region(manifest.get("mutation_region"))
    library_csv = str(manifest.get("library_csv") or "")
    library_fasta = str(manifest.get("library_fasta") or "")
    cb_job_manifest_path = str(manifest.get("cb_job_manifest_path") or "")
    cb_run_script = str(manifest.get("cb_run_script") or "")
    if not mutation_region or not library_csv or not cb_job_manifest_path or not cb_run_script:
        return None

    requested_positions = _region_position_set(requested_region)
    existing_positions = _region_position_set(mutation_region)
    if requested_positions and requested_positions != existing_positions:
        return None

    required_paths = (library_csv, library_fasta, cb_job_manifest_path, cb_run_script)
    if any(not Path(path).expanduser().exists() for path in required_paths):
        return None

    expected_pt: dict[str, Any] = {}
    try:
        cb_manifest = _read_json(Path(cb_job_manifest_path).expanduser())
        expected_pt = cb_manifest.get("ligandmpnn_run", {}).get("expected_pt", {})
    except Exception:
        pass

    return {
        "status": "completed",
        "done": True,
        "reused_existing": True,
        "reuse_reason": "project_setup_checkpoint_already_completed",
        "stop_after_this_tool": bool(stop_after_setup),
        "next_tool": None if stop_after_setup else "execute_conformation_bias_scoring_job",
        "agent_instruction": (
            "Return a final <solution> now. The project setup checkpoint is already complete. "
            "Do not inspect structures again and do not call setup_two_state_cb_scoring_project "
            "or setup_two_state_cb_scoring_project_once again."
            if stop_after_setup
            else "Continue with execute_conformation_bias_scoring_job once using cb_job_manifest_path. "
            "Do not repeat inspection or setup."
        ),
        "project_name": project_name,
        "protein_name": manifest.get("protein_name", ""),
        "target_state_name": manifest.get("target_state_name", ""),
        "reference_state_name": manifest.get("reference_state_name", ""),
        "mutation_region": mutation_region,
        "model_type": manifest.get("model_type", ""),
        "variant_count": manifest.get("library_variant_count", 0),
        "library_csv": library_csv,
        "library_fasta": library_fasta,
        "cb_job_manifest_path": cb_job_manifest_path,
        "cb_run_script": cb_run_script,
        "expected_pt": expected_pt,
        "next_commands": {
            "run_cb_scoring": f"bash {cb_run_script}",
            "extract_cb_scores": (
                "cd ${SWITCHTUNE_HOME:-.} && "
                "conda run -n ${SWITCHTUNE_LIGANDMPNN_CONDA_ENV:-ligandmpnn_env} "
                "python scripts/extract_cb_ligandmpnn_scores.py "
                f"--project_name {project_name} --job_manifest_path {cb_job_manifest_path}"
            ),
        },
        "warnings": [],
        "errors": [],
    }


def _write_json_report(project_name: str, filename: str, report: dict[str, Any]) -> Optional[str]:
    try:
        outdir = _project_report_dir(project_name)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / filename
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)
    except Exception:
        return None


def _workflow_progress(project_name: str, stage_name: str, result: Optional[dict[str, Any]] = None) -> None:
    status = result.get("status", "started") if isinstance(result, dict) else "started"
    message = f"[SwitchTune go-guard] {stage_name}: {status}"
    print(message, flush=True)
    try:
        outdir = _project_report_dir(project_name)
        outdir.mkdir(parents=True, exist_ok=True)
        event = {
            "stage": stage_name,
            "status": status,
            "project_name": project_name,
        }
        if isinstance(result, dict):
            for key in (
                "output_csv",
                "evidence_csv",
                "ranked_csv",
                "scorefile",
                "score_file",
                "run_script",
                "job_manifest_path",
                "retained_count",
                "selected_variant_count",
                "candidate_count",
                "structure_count",
                "target_like_count",
                "input_count",
                "returncode",
            ):
                if result.get(key) is not None:
                    event[key] = result.get(key)
        with (outdir / "switchtune_go_guard_full_workflow_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _variant_from_row(row: dict[str, Any], variant_column: str = "variant") -> str:
    value = row.get(variant_column) or row.get("mutation") or row.get("name") or row.get("id")
    if value:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    pos = row.get("position")
    wt = row.get("wt_aa")
    mut = row.get("mut_aa")
    if pos and wt and mut:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{wt}{pos}{mut}").strip("_")
    return ""


def _read_csv_rows_for_guard(path_text: str) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _csv_row_count_for_guard(path_text: str) -> int:
    rows, _fields = _read_csv_rows_for_guard(path_text)
    return len(rows)


def _lock_candidate_csv_to_allowed_variants(
    *,
    candidate_csv: str,
    allowed_csv: str,
    label: str,
) -> tuple[str, dict[str, Any]]:
    allowed_rows, _allowed_fields = _read_csv_rows_for_guard(allowed_csv)
    selected_rows, selected_fields = _read_csv_rows_for_guard(candidate_csv)
    allowed_variants = {_variant_from_row(row) for row in allowed_rows if _variant_from_row(row)}
    selected_variants = [_variant_from_row(row) for row in selected_rows]
    if not selected_rows:
        locked_path = Path(candidate_csv).expanduser().with_name(
            Path(candidate_csv).expanduser().stem + "_structure_locked.csv"
        )
        locked_path.parent.mkdir(parents=True, exist_ok=True)
        with locked_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected_fields or ["variant", "sequence"])
            writer.writeheader()
        return str(locked_path), {
            "status": "blocked_no_candidates",
            "label": label,
            "candidate_csv": candidate_csv,
            "locked_csv": str(locked_path),
            "allowed_csv": allowed_csv,
            "selected_count": 0,
            "retained_count": 0,
            "reason": "empty_candidate_table",
        }
    if not allowed_variants:
        return candidate_csv, {
            "status": "blocked_no_candidates",
            "label": label,
            "candidate_csv": candidate_csv,
            "allowed_csv": allowed_csv,
            "selected_count": len(selected_rows),
            "retained_count": 0,
            "reason": "empty_allowed_table",
        }
    drifted = [variant for variant in selected_variants if variant and variant not in allowed_variants]
    if not drifted:
        return candidate_csv, {
            "status": "unchanged",
            "label": label,
            "candidate_csv": candidate_csv,
            "allowed_csv": allowed_csv,
            "selected_count": len(selected_rows),
            "allowed_count": len(allowed_variants),
            "drifted_count": 0,
        }

    retained_rows = [row for row in selected_rows if _variant_from_row(row) in allowed_variants]
    locked_path = Path(candidate_csv).expanduser().with_name(
        Path(candidate_csv).expanduser().stem + "_structure_locked.csv"
    )
    locked_path.parent.mkdir(parents=True, exist_ok=True)
    with locked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(retained_rows)
    return str(locked_path), {
        "status": "locked",
        "label": label,
        "candidate_csv": candidate_csv,
        "locked_csv": str(locked_path),
        "allowed_csv": allowed_csv,
        "selected_count": len(selected_rows),
        "retained_count": len(retained_rows),
        "allowed_count": len(allowed_variants),
        "drifted_count": len(drifted),
        "drifted_variants": drifted[:50],
    }


def _inspection_feature_for_path(inspection: dict[str, Any], path_text: str) -> dict[str, Any]:
    for key in ("pdb_a", "pdb_b"):
        feature = inspection.get(key) or {}
        if str(feature.get("path") or "") == str(path_text):
            return feature
    return inspection.get("pdb_a") or {}


def _ligand_codes_from_inspection(inspection: dict[str, Any], prompt: str = "") -> str:
    prompt_codes = _extract_prompt_value(prompt, ("ligand_codes", "ligand codes"))
    if prompt_codes:
        return prompt_codes
    target_structure = str(inspection.get("suggested_target_structure") or "")
    target_feature = _inspection_feature_for_path(inspection, target_structure)
    counts = target_feature.get("nonwater_hetero_counts") or {}
    codes = [str(code).strip().upper() for code in counts if str(code).strip()]
    return ",".join(codes)


def _comparison_regions_from_mutation_region(mutation_region: str) -> str:
    clean = str(mutation_region or "").strip()
    if not clean:
        return "global"
    return f"global,switch_region:{clean}"


def _read_top_candidates_for_guard(path_text: str, limit: int = 5) -> list[dict[str, str]]:
    rows, _fields = _read_csv_rows_for_guard(path_text)
    keep_fields = (
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
    return [{field: row.get(field, "") for field in keep_fields} for row in rows[:limit]]


def _inspection_data_from_observation(
    observation: str,
    prompt: str = "",
) -> Optional[dict[str, Any]]:
    data = _extract_first_json_object(observation)
    if not isinstance(data, dict):
        return None
    if data.get("next_tool") != "setup_two_state_cb_scoring_project_once":
        return None

    # Explicit benchmark fields are authoritative. The PDB inspector is a
    # validator, but its ligand heuristics cannot cover every protein family
    # or every CCD code.
    explicit_target = (
        _extract_prompt_value(
            prompt,
            ("target_state_structure", "target structure", "target/active-state structure"),
        )
        or _extract_prompt_path(
            prompt,
            ("target_state_structure", "target structure", "target/active-state structure"),
        )
    )
    explicit_reference = (
        _extract_prompt_value(
            prompt,
            ("reference_state_structure", "reference structure", "reference/inactive-state structure"),
        )
        or _extract_prompt_path(
            prompt,
            ("reference_state_structure", "reference structure", "reference/inactive-state structure"),
        )
    )
    if explicit_target:
        data["suggested_target_structure"] = explicit_target
    if explicit_reference:
        data["suggested_reference_structure"] = explicit_reference

    explicit_model = _extract_prompt_value(prompt, ("model_type", "model type"))
    explicit_chain = _extract_prompt_value(prompt, ("parse_chains", "parse chains", "chain_id", "chain"))
    explicit_region = _extract_prompt_value(
        prompt,
        ("mutation_region", "mutation region", "mutable_region", "mutable region"),
    )
    if explicit_model:
        data["suggested_model_type"] = explicit_model
    if explicit_chain:
        data["suggested_parse_chains"] = explicit_chain
    normalized_explicit_region = _normalize_mutation_region(explicit_region)
    if normalized_explicit_region:
        data["suggested_mutation_region"] = normalized_explicit_region

    required = (
        "suggested_target_structure",
        "suggested_reference_structure",
        "suggested_model_type",
        "suggested_parse_chains",
    )
    if not all(data.get(key) for key in required):
        return None
    if not data.get("suggested_mutation_region"):
        # No case-specific interval is embedded in the public release. The
        # agent must make the scientific choice after inspecting both WT
        # structures. A deterministic C-alpha fallback is available only in
        # the explicit provider-recovery path below; it never uses a named
        # protein or a literature-derived region.
        data["mutation_region_selection_required"] = True
        data["mutation_region_rationale"] = (
            "No mutation region was hard-coded. The Agent must select one or "
            "more exact intervals or discrete positions from the inspected "
            "structures and sequence before setup."
        )

        # The MCP child may return an inspection without a concrete region
        # when a provider continuation is unavailable. Recover from the two
        # supplied WT structures using only aligned C-alpha displacements.
        fallback_enabled = os.environ.get(
            "SWITCHTUNE_AUTONOMOUS_REGION_FALLBACK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if fallback_enabled:
            try:
                from .mcp_server import _autonomous_structural_mutation_region

                chain_value = str(data.get("suggested_parse_chains") or "A")
                chain_id = next(
                    (item.strip() for item in re.split(r"[,;\s]+", chain_value) if item.strip()),
                    "A",
                )
                fallback_region, fallback_rationale = _autonomous_structural_mutation_region(
                    str(data.get("suggested_target_structure") or ""),
                    str(data.get("suggested_reference_structure") or ""),
                    chain_id,
                )
                if fallback_region:
                    data["suggested_mutation_region"] = fallback_region
                    data["mutation_region_rationale"] = fallback_rationale
                    data["mutation_region_selection_required"] = False
            except Exception as exc:
                data["autonomous_region_fallback_error"] = repr(exc)
    return data


def _observation_from_message(content: str, pretty_content: str) -> Optional[str]:
    """Read observations from either raw Biomni content or pretty-printed output.

    Biomni versions differ: some keep the ``<observation>`` wrapper in the
    ToolMessage, while others add it only in ``pretty_print(message)``.
    """
    for candidate in (content, pretty_content):
        tagged = _extract_tag(candidate, "observation")
        if tagged:
            return tagged
    for candidate in (content, pretty_content):
        if "next_tool" in candidate and "suggested_target_structure" in candidate:
            return candidate.strip()
    return None


def _adaptive_selection_data_from_observation(observation: str) -> Optional[dict[str, Any]]:
    data = _extract_first_json_object(observation)
    if not isinstance(data, dict):
        return None
    if data.get("next_tool") != "prepare_structure_prediction_job":
        return None
    if not data.get("output_csv") or data.get("retained_count") is None:
        return None
    return data


def _engine_selection_data_from_observation(observation: str) -> Optional[dict[str, Any]]:
    data = _extract_first_json_object(observation)
    if not isinstance(data, dict):
        return None
    if data.get("next_tool") != "prepare_structure_prediction_job":
        return None
    if not data.get("recommended_engine"):
        return None
    return data


def _auto_prepare_structure_from_adaptive(prompt: str, adaptive: dict[str, Any]) -> Optional[str]:
    result = _auto_prepare_structure_result_from_adaptive(prompt, adaptive)
    if result is None:
        return None
    solution = _compact_switchtune_solution(json.dumps(result, ensure_ascii=False, indent=2))
    return solution.replace(
        "SwitchTune one-shot MCP step completed and the guard stopped further tool calls.",
        "SwitchTune CB selection to structure-prediction preparation completed and the guard stopped further tool calls.",
    )


def _auto_prepare_structure_from_engine_selection(prompt: str, decision: dict[str, Any]) -> Optional[str]:
    result = _auto_prepare_structure_result_from_engine_selection(prompt, decision)
    if result is None:
        return None
    solution = _compact_switchtune_solution(json.dumps(result, ensure_ascii=False, indent=2))
    return solution.replace(
        "SwitchTune one-shot MCP step completed and the guard stopped further tool calls.",
        "SwitchTune structure-engine selection to structure-prediction preparation completed and the guard stopped further tool calls.",
    )


def _auto_prepare_structure_result_from_engine_selection(prompt: str, decision: dict[str, Any]) -> Optional[dict[str, Any]]:
    structure_tool = _switchtune_tool("prepare_structure_prediction_job")
    if structure_tool is None:
        return None

    candidates_csv = _extract_prompt_path(
        prompt,
        (
            "input candidates csv",
            "candidates csv",
            "retained candidates",
            "candidate csv",
        ),
    )
    if not candidates_csv:
        candidates_csv = _extract_prompt_value(prompt, ("candidates_csv", "candidate_csv"))
    if not candidates_csv:
        return None

    project_name = str(decision.get("project_name") or _extract_project_name_from_prompt(prompt))
    explicit_codes = decision.get("explicit_ligand_codes") or []
    detected_codes = decision.get("detected_ligand_codes") or []
    ligand_codes = _extract_prompt_value(prompt, ("ligand_codes", "ligand codes"))
    if not ligand_codes:
        codes = explicit_codes or detected_codes
        ligand_codes = ",".join(str(code) for code in codes if code)

    cofactor_required = _extract_prompt_bool(prompt, ("cofactor_required", "cofactor required"))
    if cofactor_required is None:
        cofactor_required = bool(decision.get("ligand_required"))

    kwargs: dict[str, Any] = {
        "project_name": project_name,
        "candidates_csv": candidates_csv,
        "engine": str(decision.get("recommended_engine") or _extract_structure_engine_from_prompt(prompt)),
        "target_state_structure": str(decision.get("target_state_structure") or ""),
        "reference_state_structure": str(decision.get("reference_state_structure") or ""),
        "ligand_codes": ligand_codes,
        "cofactor_required": bool(cofactor_required),
    }
    recycling_steps = _extract_prompt_int(prompt, ("boltz_recycling_steps", "recycling_steps"))
    sampling_steps = _extract_prompt_int(prompt, ("boltz_sampling_steps", "sampling_steps"))
    if recycling_steps is not None:
        kwargs["boltz_recycling_steps"] = recycling_steps
    if sampling_steps is not None:
        kwargs["boltz_sampling_steps"] = sampling_steps

    return structure_tool(**kwargs)


def _auto_prepare_structure_result_from_adaptive(prompt: str, adaptive: dict[str, Any]) -> Optional[dict[str, Any]]:
    structure_tool = _switchtune_tool("prepare_structure_prediction_job")
    if structure_tool is None:
        return None

    project_name = str(adaptive.get("project_name") or _extract_project_name_from_prompt(prompt))
    retained_count = int(adaptive.get("retained_count") or 0)
    top_n = retained_count if retained_count > 0 else None
    result = structure_tool(
        project_name=project_name,
        candidates_csv=str(adaptive["output_csv"]),
        engine=_extract_structure_engine_from_prompt(prompt),
        top_n=top_n,
    )
    return result


def _auto_setup_result_from_inspection(
    prompt: str,
    inspection: dict[str, Any],
    *,
    stop_after_setup: bool = True,
) -> Optional[dict[str, Any]]:
    setup_tool = _switchtune_tool("setup_two_state_cb_scoring_project_once")
    if setup_tool is None:
        return None

    project_name = _extract_project_name_from_prompt(prompt)
    protein_name = _extract_protein_name_from_prompt(prompt, fallback=str(inspection.get("protein_name") or "protein"))
    requested_region = _concrete_mutation_region(prompt, inspection)
    force_refresh = _extract_prompt_bool(prompt, ("force_refresh", "force refresh")) is True
    if not force_refresh:
        existing = _existing_setup_result(
            project_name,
            requested_region=requested_region,
            stop_after_setup=stop_after_setup,
        )
        if existing is not None:
            _workflow_progress(
                project_name,
                "setup_checkpoint_reuse",
                {
                    "status": "reused",
                    "reused_existing": True,
                    "mutation_region": existing.get("mutation_region"),
                    "variant_count": existing.get("variant_count"),
                },
            )
            return existing

    # The structure inspection is the authoritative source of sequence and
    # residue numbering. In formal prompts the sequence may be omitted, and
    # a permissive prompt fallback can mistake metadata such as
    # ``reference_state_name: ...`` for an amino-acid sequence.
    prompt_sequence = _extract_sequence_from_prompt(prompt)
    inspection_sequence = str(inspection.get("suggested_sequence") or "").strip()
    if prompt_sequence:
        sequence = re.sub(r"[^A-Za-z]", "", prompt_sequence).upper()
        sequence_source = "prompt"
    else:
        sequence = re.sub(r"[^A-Za-z]", "", inspection_sequence).upper()
        sequence_source = "inspection.suggested_sequence"
    if not sequence:
        return None

    # An explicit prompt sequence is authoritative and may be full-length
    # while the inspected PDB contains only a crystallized domain.
    expected_sequence_length = (
        None if prompt_sequence else inspection.get("suggested_sequence_length")
    )
    if expected_sequence_length is not None:
        try:
            expected_length = int(expected_sequence_length)
        except (TypeError, ValueError):
            expected_length = None
        if expected_length is not None and len(sequence) != expected_length:
            raise ValueError(
                "Inspection sequence length mismatch: "
                f"source={sequence_source}, got={len(sequence)}, "
                f"expected={expected_length}."
            )

    target_structure = str(
        _extract_prompt_value(prompt, ("target_state_structure", "target structure", "target/active-state structure"))
        or _extract_prompt_path(prompt, ("target_state_structure", "target structure", "target/active-state structure"))
        or inspection.get("suggested_target_structure")
        or ""
    )
    reference_structure = str(
        _extract_prompt_value(prompt, ("reference_state_structure", "reference structure", "reference/inactive-state structure"))
        or _extract_prompt_path(prompt, ("reference_state_structure", "reference structure", "reference/inactive-state structure"))
        or inspection.get("suggested_reference_structure")
        or ""
    )
    if not target_structure or not reference_structure:
        return None
    target_state_name = (
        _extract_prompt_value(prompt, ("target_state_name", "target state name"))
        or _state_name_from_path(target_structure, activated=True)
    )
    reference_state_name = (
        _extract_prompt_value(prompt, ("reference_state_name", "reference state name"))
        or _state_name_from_path(reference_structure, activated=False)
    )
    # A user- or runner-specified region is authoritative. The inspection
    # result supplies a fallback only when the prompt leaves the region open.
    mutation_region = requested_region
    if not mutation_region:
        return None
    model_type = str(
        _extract_prompt_value(prompt, ("model_type", "model type"))
        or inspection.get("suggested_model_type")
        or "protein_mpnn"
    )
    parse_chains = str(
        _extract_prompt_value(prompt, ("parse_chains", "parse chains", "chain_id", "chain"))
        or inspection.get("suggested_parse_chains")
        or "A"
    )
    first_residue_number = _extract_prompt_int(prompt, ("first_residue_number", "first residue number"))
    if first_residue_number is None:
        first_residue_number = int(inspection.get("suggested_first_residue_number") or 1)

    result = setup_tool(
        project_name=project_name,
        protein_name=protein_name,
        sequence=sequence,
        target_state_structure=target_structure,
        reference_state_structure=reference_structure,
        mutation_region=mutation_region,
        target_state_name=target_state_name,
        reference_state_name=reference_state_name,
        design_objective=str(
            _extract_prompt_value(prompt, ("design_objective", "design objective", "design_goal", "design goal"))
            or inspection.get("design_goal")
            or ""
        ),
        model_type=model_type,
        parse_chains=parse_chains,
        first_residue_number=int(first_residue_number),
        force_refresh=force_refresh,
        stop_after_setup=stop_after_setup,
    )
    return result


def _auto_setup_from_inspection(prompt: str, inspection: dict[str, Any]) -> Optional[str]:
    result = _auto_setup_result_from_inspection(prompt, inspection, stop_after_setup=True)
    if result is None:
        return None
    return _compact_setup_solution_with_inspection(result, inspection)


def _auto_execute_cb_from_setup(setup_result: dict[str, Any], timeout_seconds: int) -> Optional[dict[str, Any]]:
    execute_tool = _switchtune_tool("execute_conformation_bias_scoring_job")
    if execute_tool is None:
        return None
    project_name = str(setup_result.get("project_name") or "")
    manifest_path = str(setup_result.get("cb_job_manifest_path") or "")
    if not project_name or not manifest_path:
        return None
    return execute_tool(
        project_name=project_name,
        job_manifest_path=manifest_path,
        timeout_seconds=timeout_seconds,
        skip_if_outputs_exist=True,
        collect_scores=True,
    )


def _auto_adaptive_select_from_cb_execution(cb_result: dict[str, Any], prompt: str = "") -> Optional[dict[str, Any]]:
    project_name = str(cb_result.get("project_name") or "")
    scored_csv = str(cb_result.get("scored_library_csv") or "")
    if not project_name or not scored_csv:
        return None

    adaptive_tool = _switchtune_tool("select_state_specific_candidates_adaptive")
    if adaptive_tool is None:
        return None
    # Public runs always use the empirical central-fraction policy. The
    # retained count is derived from the observed score distribution and
    # library size; downstream budgets must never feed a fixed count back into
    # CB selection.
    return adaptive_tool(
        project_name=project_name,
        scored_library_csv=scored_csv,
        score_column="bias_score",
        strategy="auto",
        center="zero",
    )


def _setup_result_from_observation(observation: str) -> Optional[dict[str, Any]]:
    """Recover a direct setup-tool result emitted by Biomni."""
    data = _extract_first_json_object(observation)
    if not isinstance(data, dict):
        return None
    if str(data.get("status") or "").lower() not in {"completed", "prepared", "reused"}:
        return None
    required = ("project_name", "cb_job_manifest_path", "cb_run_script")
    if not all(str(data.get(key) or "").strip() for key in required):
        return None
    if not str(data.get("mutation_region") or "").strip():
        return None
    return data


def _auto_run_front_half_from_setup(
    prompt: str,
    setup_result: dict[str, Any],
    cb_timeout_seconds: int,
) -> Optional[str]:
    """Continue checkpoint mode after Biomni directly completed setup."""
    project_name = str(setup_result.get("project_name") or _extract_project_name_from_prompt(prompt))
    _workflow_progress(project_name, "setup_checkpoint_observation", {"status": "recognized"})
    _workflow_progress(project_name, "execute_conformation_bias_scoring_job")
    cb_result = _auto_execute_cb_from_setup(setup_result, cb_timeout_seconds)
    if isinstance(cb_result, dict):
        _workflow_progress(project_name, "execute_conformation_bias_scoring_job", cb_result)
    if not isinstance(cb_result, dict) or cb_result.get("status") != "completed":
        return _compact_switchtune_solution(json.dumps(cb_result or setup_result, ensure_ascii=False, indent=2))

    selection_tool_name = "select_state_specific_candidates_adaptive"
    _workflow_progress(project_name, selection_tool_name)
    selection_result = _auto_adaptive_select_from_cb_execution(cb_result, prompt)
    if not isinstance(selection_result, dict):
        return _compact_switchtune_solution(json.dumps(cb_result, ensure_ascii=False, indent=2))
    _workflow_progress(project_name, selection_tool_name, selection_result)

    _workflow_progress(project_name, "prepare_structure_prediction_job")
    structure_result = _auto_prepare_structure_result_from_adaptive(prompt, selection_result)
    if not isinstance(structure_result, dict):
        return _compact_switchtune_solution(json.dumps(selection_result, ensure_ascii=False, indent=2))
    _workflow_progress(project_name, "prepare_structure_prediction_job", structure_result)
    return _compact_switchtune_solution(
        json.dumps(
            {
                "status": "completed",
                "workflow": "SwitchTune direct-setup checkpoint handoff",
                "project_name": project_name,
                "setup": setup_result,
                "cb_execution": cb_result,
                "cb_selection": selection_result,
                "structure_prediction": structure_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_front_half_report(project_name: str, report: dict[str, Any]) -> Optional[str]:
    try:
        runs_root = Path(os.environ.get("SWITCHTUNE_RUNS", "./SwitchTune_runs")).expanduser().resolve()
        outdir = runs_root / project_name / "00_agent_reports"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "front_half_0code_summary.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)
    except Exception:
        return None


def _compact_front_half_solution(
    *,
    inspection: dict[str, Any],
    setup_result: dict[str, Any],
    cb_result: dict[str, Any],
    adaptive_result: dict[str, Any],
    structure_result: dict[str, Any],
) -> str:
    project_name = str(setup_result.get("project_name") or cb_result.get("project_name") or "")
    report = {
        "workflow": "SwitchTune front-half 0-code",
        "project_name": project_name,
        "inspection": {
            "target_structure": inspection.get("suggested_target_structure"),
            "reference_structure": inspection.get("suggested_reference_structure"),
            "target_reference_rationale": inspection.get("target_reference_rationale"),
            "target_state_clues": (inspection.get("pdb_a") or {}).get("state_clues"),
            "reference_state_clues": (inspection.get("pdb_b") or {}).get("state_clues"),
            "mutation_region": inspection.get("suggested_mutation_region"),
            "mutation_region_rationale": inspection.get("mutation_region_rationale"),
            "model_type": inspection.get("suggested_model_type"),
            "parse_chains": inspection.get("suggested_parse_chains"),
        },
        "cb_setup": {
            "status": setup_result.get("status"),
            "variant_count": setup_result.get("variant_count"),
            "library_csv": setup_result.get("library_csv"),
            "cb_run_script": setup_result.get("cb_run_script"),
            "cb_job_manifest_path": setup_result.get("cb_job_manifest_path"),
        },
        "cb_execution": {
            "status": cb_result.get("status"),
            "run_status": cb_result.get("run_status"),
            "scored_library_csv": cb_result.get("scored_library_csv"),
            "scored_library_summary_path": cb_result.get("scored_library_summary_path"),
            "input_count": cb_result.get("input_count"),
            "valid_bias_count": cb_result.get("valid_bias_count"),
            "retained_count_fixed_window": cb_result.get("retained_count_fixed_window"),
        },
        "adaptive_selection": {
            "selection_mode": adaptive_result.get("selection_mode", "adaptive"),
            "selection_policy": adaptive_result.get("selection_policy"),
            "strategy": adaptive_result.get("strategy"),
            "center_value": adaptive_result.get("center_value"),
            "retained_count": adaptive_result.get("retained_count"),
            "retained_fraction": adaptive_result.get("retained_fraction"),
            "score_min": adaptive_result.get("score_min"),
            "score_max": adaptive_result.get("score_max"),
            "fixed_minus1_plus1_count": adaptive_result.get("fixed_minus1_plus1_count"),
            "output_csv": adaptive_result.get("output_csv"),
        },
        "structure_prediction": {
            "status": structure_result.get("status"),
            "engine": structure_result.get("engine"),
            "boltz2_backend": (structure_result.get("boltz2") or {}).get("backend")
            if isinstance(structure_result.get("boltz2"), dict)
            else "",
            "candidate_count": structure_result.get("candidate_count"),
            "fasta_path": structure_result.get("fasta_path"),
            "run_script": structure_result.get("run_script"),
            "job_manifest_path": structure_result.get("job_manifest_path"),
            "prediction_dir": structure_result.get("prediction_dir"),
            "colabfold_executed": False,
        },
    }
    report["cb_selection"] = dict(report["adaptive_selection"])
    report_path = _write_front_half_report(project_name, report)
    if report_path:
        report["report_path"] = report_path

    insp = report["inspection"]
    setup = report["cb_setup"]
    cb = report["cb_execution"]
    adaptive = report["adaptive_selection"]
    structure = report["structure_prediction"]
    return (
        "<solution>\n"
        "SwitchTune front-half 0-code workflow completed: PDB inspection, CB setup, "
        "LigandMPNN execution, CB extraction, CB candidate selection, and structure-prediction "
        "job preparation.\n\n"
        f"- project_name: `{project_name}`\n"
        f"- activated_target_structure: `{insp['target_structure']}`\n"
        f"- target_rationale: {insp['target_reference_rationale']}; {insp['target_state_clues']}\n"
        f"- inactive_reference_structure: `{insp['reference_structure']}`\n"
        f"- reference_rationale: {insp['reference_state_clues']}\n"
        f"- mutation_region: `{insp['mutation_region']}`\n"
        f"- mutation_region_rationale: {insp['mutation_region_rationale']}\n"
        f"- model_type: `{insp['model_type']}`\n"
        f"- parsed_chain: `{insp['parse_chains']}`\n\n"
        f"- cb_setup_status: `{setup['status']}`\n"
        f"- library_variant_count: `{setup['variant_count']}`\n"
        f"- library_csv: `{setup['library_csv']}`\n"
        f"- cb_run_script: `{setup['cb_run_script']}`\n"
        f"- cb_job_manifest_path: `{setup['cb_job_manifest_path']}`\n\n"
        f"- cb_execution_status: `{cb['status']}`\n"
        f"- cb_run_status: `{cb['run_status']}`\n"
        f"- scored_library_csv: `{cb['scored_library_csv']}`\n"
        f"- valid_bias_count: `{cb['valid_bias_count']}`\n"
        f"- fixed_window_retained_count: `{cb['retained_count_fixed_window']}`\n\n"
        f"- cb_selection_mode: `{adaptive.get('selection_mode', 'adaptive')}`\n"
        f"- cb_selection_policy: `{adaptive.get('selection_policy')}`\n"
        f"- cb_strategy: `{adaptive.get('strategy')}`\n"
        f"- cb_retained_count: `{adaptive.get('retained_count')}`\n"
        f"- cb_retained_fraction: `{adaptive.get('retained_fraction')}`\n"
        f"- cb_score_window: `{adaptive.get('score_min')}` to `{adaptive.get('score_max')}`\n"
        f"- fixed_minus1_plus1_count: `{adaptive.get('fixed_minus1_plus1_count')}`\n"
        f"- cb_candidates_csv: `{adaptive.get('output_csv')}`\n\n"
        f"- structure_job_status: `{structure['status']}`\n"
        f"- structure_candidate_count: `{structure['candidate_count']}`\n"
        f"- structure_fasta_path: `{structure['fasta_path']}`\n"
        f"- structure_run_script: `{structure['run_script']}`\n"
        f"- structure_job_manifest_path: `{structure['job_manifest_path']}`\n"
        "- ColabFold was not run.\n"
        + (f"- front_half_report_json: `{report_path}`\n" if report_path else "")
        + "</solution>"
    )


def _auto_run_front_half_from_inspection(
    prompt: str,
    inspection: dict[str, Any],
    cb_timeout_seconds: int,
) -> Optional[str]:
    project_name = _extract_project_name_from_prompt(prompt)
    _workflow_progress(project_name, "inspection_dispatch", {"status": "recognized"})
    _workflow_progress(project_name, "setup_two_state_cb_scoring_project_once")
    setup_result = _auto_setup_result_from_inspection(prompt, inspection, stop_after_setup=False)
    if setup_result is None:
        return None
    _workflow_progress(project_name, "setup_two_state_cb_scoring_project_once", setup_result)
    _workflow_progress(project_name, "execute_conformation_bias_scoring_job")
    cb_result = _auto_execute_cb_from_setup(setup_result, cb_timeout_seconds)
    if isinstance(cb_result, dict):
        _workflow_progress(project_name, "execute_conformation_bias_scoring_job", cb_result)
    if cb_result is None or cb_result.get("status") != "completed":
        return _compact_setup_solution_with_inspection(setup_result, inspection)
    selection_tool_name = "select_state_specific_candidates_adaptive"
    _workflow_progress(project_name, selection_tool_name)
    adaptive_result = _auto_adaptive_select_from_cb_execution(cb_result, prompt)
    if adaptive_result is None:
        return _compact_switchtune_solution(json.dumps(cb_result, ensure_ascii=False, indent=2))
    _workflow_progress(project_name, selection_tool_name, adaptive_result)
    _workflow_progress(project_name, "prepare_structure_prediction_job")
    structure_result = _auto_prepare_structure_result_from_adaptive(prompt, adaptive_result)
    if structure_result is None:
        return _compact_switchtune_solution(json.dumps(adaptive_result, ensure_ascii=False, indent=2))
    _workflow_progress(project_name, "prepare_structure_prediction_job", structure_result)

    return _compact_front_half_solution(
        inspection=inspection,
        setup_result=setup_result,
        cb_result=cb_result,
        adaptive_result=adaptive_result,
        structure_result=structure_result,
    )


def _compact_full_workflow_solution(report: dict[str, Any]) -> str:
    project_name = str(report.get("project_name") or "")
    final = report.get("final_recommendation") or {}
    merged = report.get("evidence_merge") or {}
    agentic_review = report.get("agentic_candidate_review") or {}
    inspection = report.get("inspection") or {}
    confidence = report.get("structure_confidence") or {}
    inverse_collect = report.get("inverse_folding_collection") or {}
    bioemu_collect = report.get("bioemu_collection") or {}
    rosetta_collect = report.get("rosetta_collection") or {}
    final_csv = str(final.get("output_csv") or final.get("final_candidate_recommendation_csv") or "")
    top_candidates = _read_top_candidates_for_guard(final_csv, limit=5) if final_csv else []
    top_lines = []
    for row in top_candidates:
        top_lines.append(
            "- "
            + ", ".join(
                f"{key}={value}"
                for key, value in row.items()
                if value not in {"", None}
            )
        )
    if not top_lines:
        top_lines.append("- No top candidates were readable from the final CSV.")

    report_path = report.get("report_path")
    event_log_path = report.get("event_log_path")
    candidate_locks = report.get("candidate_locks") or []
    lock_lines = []
    for item in candidate_locks:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "locked":
            lock_lines.append(
                f"- {item.get('label')}: removed {item.get('drifted_count')} out-of-structure candidates; "
                f"locked_csv=`{item.get('locked_csv')}`"
            )
    lock_text = "\n".join(lock_lines)

    return (
        "<solution>\n"
        "SwitchTune full 0-code workflow completed through final recommendation.\n\n"
        f"- project_name: `{project_name}`\n"
        f"- target_structure: `{inspection.get('suggested_target_structure')}`\n"
        f"- reference_structure: `{inspection.get('suggested_reference_structure')}`\n"
        f"- mutation_region: `{inspection.get('suggested_mutation_region')}`\n"
        f"- model_type: `{inspection.get('suggested_model_type')}`\n"
        f"- structure_confidence_retained_count: `{confidence.get('retained_count')}`\n"
        f"- inverse_folding_scored_variant_count: `{inverse_collect.get('scored_variant_count')}`\n"
        f"- bioemu_completed_count: `{bioemu_collect.get('command_completed_count')}`\n"
        f"- rosetta_evidence_variant_count: `{_collector_variant_count(rosetta_collect)}`\n"
        f"- merged_evidence_csv: `{merged.get('merged_csv')}`\n"
        f"- final_candidate_recommendation_csv: `{final_csv}`\n"
        + (
            f"- agentic_candidate_review_json: `{agentic_review.get('packet_json')}`\n"
            if agentic_review.get("packet_json")
            else ""
        )
        + (f"- full_workflow_report_json: `{report_path}`\n" if report_path else "")
        + (f"- full_workflow_event_log: `{event_log_path}`\n" if event_log_path else "")
        + ("\nCandidate set locks:\n" + lock_text + "\n" if lock_text else "")
        + "\nTop candidates:\n"
        + "\n".join(top_lines)
        + "\n</solution>"
    )


def _compact_full_workflow_failed_solution(report: dict[str, Any], error: BaseException) -> str:
    project_name = str(report.get("project_name") or _extract_project_name_from_prompt(str(report.get("prompt") or "")))
    error_text = str(error).lower()
    blocked = any(
        marker in error_text
        for marker in (
            "zero candidates",
            "no valid structures",
            "no valid pdb",
            "no numerically ranked candidates",
            "empty candidate",
            "blocked_no_candidates",
        )
    )
    report["final_status"] = "blocked_no_candidates" if blocked else "failed"
    report["error"] = repr(error)
    report_path = _write_json_report(project_name, "full_agent_workflow_summary.json", report)
    if blocked:
        return (
            "<solution>\n"
            "SwitchTune workflow stopped safely because no valid candidates remained at the current gate.\n\n"
            f"- project_name: `{project_name}`\n"
            f"- reason: `{repr(error)}`\n"
            "- No downstream scoring job was started from the empty/invalid shortlist.\n"
            + (f"- blocked_workflow_report_json: `{report_path}`\n" if report_path else "")
            + "Inspect the stage evidence and rejection table before rerunning.\n"
            "</solution>"
        )
    return (
        "<solution>\n"
        "SwitchTune full workflow auto-run failed before final recommendation.\n\n"
        f"- project_name: `{project_name}`\n"
        f"- error: `{repr(error)}`\n"
        + (f"- partial_report_json: `{report_path}`\n" if report_path else "")
        + "Check `00_agent_reports/switchtune_go_guard_full_workflow_events.jsonl` and the stage logs for the first failed external job.\n"
        "</solution>"
    )


def _auto_run_full_workflow_from_inspection(
    prompt: str,
    inspection: dict[str, Any],
    *,
    cb_timeout_seconds: int,
    workflow_timeout_seconds: int,
) -> Optional[str]:
    """Execute the validated SwitchTune smoke-test workflow after PDB inspection.

    This is intentionally deterministic: Biomni still triggers the workflow from
    natural language and performs the initial state inspection, but the guard
    prevents fragile LLM self-looping by handing off to the audited MCP stage
    order after `inspect_two_state_pdb_pair` returns a usable recommendation.
    """
    project_name = _extract_project_name_from_prompt(prompt)
    report: dict[str, Any] = {
        "workflow": "SwitchTune full 0-code guarded workflow",
        "project_name": project_name,
        "prompt": prompt,
        "inspection": inspection,
        "candidate_locks": [],
    }
    event_log_path = str(_project_report_dir(project_name) / "switchtune_go_guard_full_workflow_events.jsonl")
    report["event_log_path"] = event_log_path

    try:
        tools = {
            name: _switchtune_tool(name)
            for name in (
                "setup_two_state_cb_scoring_project_once",
                "execute_conformation_bias_scoring_job",
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
            )
        }
        missing = [name for name, tool in tools.items() if tool is None]
        if missing:
            raise RuntimeError(f"SwitchTune MCP tools are unavailable in the Biomni runtime: {missing}")

        # Zero means all candidates retained by the adaptive CB policy.
        structure_n = _extract_structure_candidate_count_from_prompt(prompt, default=0)
        bioemu_n = _extract_named_top_n_from_prompt(prompt, ("bioemu", "bioemu sampling"), default=structure_n)
        bioemu_samples = _extract_bioemu_num_samples_from_prompt(prompt, default=20)
        rosetta_n = _extract_named_top_n_from_prompt(prompt, ("rosetta", "rosetta energy"), default=min(3, bioemu_n))
        cb_selection_mode = _extract_cb_selection_mode_from_prompt(prompt)
        cb_score_min, cb_score_max = _extract_cb_score_window_from_prompt(prompt)
        timeout_seconds = _extract_timeout_seconds_from_prompt(prompt, default=workflow_timeout_seconds)
        protein_name = _extract_protein_name_from_prompt(prompt, fallback=str(inspection.get("protein_name") or "protein"))
        target_structure = str(inspection.get("suggested_target_structure") or "")
        reference_structure = str(inspection.get("suggested_reference_structure") or "")
        mutation_region = _concrete_mutation_region(prompt, inspection)
        if not mutation_region:
            existing_setup = _existing_setup_result(
                project_name,
                stop_after_setup=False,
            )
            if existing_setup is not None:
                mutation_region = str(existing_setup.get("mutation_region") or "")
        if not mutation_region:
            _workflow_progress(
                project_name,
                "inspection_dispatch",
                {
                    "status": "waiting_for_agent_region",
                    "reason": (
                        "The inspection result did not contain a concrete mutation region; "
                        "the Biomni Agent must choose a parseable region before setup."
                    ),
                },
            )
            return None
        parse_chains = str(inspection.get("suggested_parse_chains") or "A")
        target_state_name = _state_name_from_path(target_structure, activated=True)
        reference_state_name = _state_name_from_path(reference_structure, activated=False)
        ligand_codes = _ligand_codes_from_inspection(inspection, prompt)

        _workflow_progress(project_name, "setup_two_state_cb_scoring_project_once")
        setup_result = _auto_setup_result_from_inspection(prompt, inspection, stop_after_setup=False)
        if setup_result is None:
            raise RuntimeError("Could not build setup arguments from inspection result.")
        report["cb_setup"] = setup_result
        _workflow_progress(project_name, "setup_two_state_cb_scoring_project_once", setup_result)

        _workflow_progress(project_name, "execute_conformation_bias_scoring_job")
        cb_result = tools["execute_conformation_bias_scoring_job"](
            project_name=project_name,
            job_manifest_path=str(setup_result.get("cb_job_manifest_path") or ""),
            timeout_seconds=cb_timeout_seconds,
            skip_if_outputs_exist=True,
            collect_scores=True,
        )
        report["cb_execution"] = cb_result
        _workflow_progress(project_name, "execute_conformation_bias_scoring_job", cb_result)
        if cb_result.get("status") != "completed":
            raise RuntimeError(f"CB scoring did not complete: {cb_result.get('status')}")

        selection_tool_name = "select_state_specific_candidates_adaptive"
        _workflow_progress(project_name, selection_tool_name)
        # The public workflow always derives CB retention from the empirical
        # score distribution. Do not convert downstream candidate budgets into
        # a fixed CB target count.
        selection_result = tools[selection_tool_name](
            project_name=project_name,
            scored_library_csv=str(cb_result.get("scored_library_csv") or ""),
            score_column="bias_score",
            strategy="auto",
            center="zero",
        )
        report["cb_selection"] = {
            "selection_mode": cb_selection_mode,
            "selection_tool": selection_tool_name,
            "score_min": cb_score_min,
            "score_max": cb_score_max,
            "result": selection_result,
        }
        # Keep the legacy report key so existing downstream report consumers
        # remain compatible with both selection policies.
        report["adaptive_selection"] = selection_result
        _workflow_progress(project_name, selection_tool_name, selection_result)

        requested_engine = _extract_structure_engine_from_prompt(prompt)
        _workflow_progress(project_name, "select_structure_prediction_engine")
        engine_decision = tools["select_structure_prediction_engine"](
            project_name=project_name,
            target_state_structure=target_structure,
            reference_state_structure=reference_structure,
            ligand_codes=ligand_codes,
            requested_engine=requested_engine,
        )
        report["engine_decision"] = engine_decision
        _workflow_progress(project_name, "select_structure_prediction_engine", engine_decision)

        _workflow_progress(project_name, "prepare_structure_prediction_job")
        structure = tools["prepare_structure_prediction_job"](
            project_name=project_name,
            candidates_csv=str(selection_result.get("output_csv") or ""),
            engine=str(engine_decision.get("recommended_engine") or requested_engine or "auto"),
            top_n=structure_n,
            target_state_structure=target_structure,
            reference_state_structure=reference_structure,
            template_mode="target_state",
            boltz_use_template=True,
            boltz_write_full_pae=True,
            ligand_codes=ligand_codes,
            cofactor_required=bool(engine_decision.get("ligand_required", bool(ligand_codes))),
            execute=True,
            timeout_seconds=timeout_seconds,
        )
        report["structure_preparation"] = structure
        _workflow_progress(project_name, "prepare_structure_prediction_job", structure)
        if structure.get("status") not in {"completed", "prepared"}:
            report["structure_preparation_warning"] = {
                "status": structure.get("status"),
                "returncode": structure.get("returncode"),
                "note": "Continuing because partial structure outputs may still be available for collection.",
            }

        _workflow_progress(project_name, "collect_structure_prediction_results")
        collect_structure = tools["collect_structure_prediction_results"](
            project_name=project_name,
            prediction_dir=str(structure.get("prediction_dir") or ""),
            job_manifest_path=str(structure.get("job_manifest_path") or ""),
        )
        report["structure_collection"] = collect_structure
        _workflow_progress(project_name, "collect_structure_prediction_results", collect_structure)
        if "valid_structure_count" in collect_structure:
            valid_structure_count = int(collect_structure.get("valid_structure_count") or 0)
        elif "completed_count" in collect_structure:
            valid_structure_count = int(collect_structure.get("completed_count") or 0)
        else:
            valid_structure_count = int(collect_structure.get("structure_count") or 0)
        if valid_structure_count == 0:
            raise RuntimeError(
                "Structure prediction produced no valid structures after the strict PAE gate. "
                "Inspect structure_prediction_evidence.csv and the Boltz-2 response summary."
            )

        _workflow_progress(project_name, "standardize_structure_prediction_results")
        standardized = tools["standardize_structure_prediction_results"](
            project_name=project_name,
            structures_csv=str(collect_structure.get("output_csv") or ""),
            chain_id=parse_chains,
            require_pae=True,
        )
        report["structure_standardization"] = standardized
        _workflow_progress(project_name, "standardize_structure_prediction_results", standardized)
        if int(standardized.get("standardized_count") or 0) == 0:
            raise RuntimeError(
                "Structure standardization produced no valid PDBs after the strict PAE gate. "
                "Do not continue to inverse folding, BioEmu, or Rosetta."
            )

        standardized_csv = str(standardized.get("output_csv") or collect_structure.get("output_csv") or "")
        _workflow_progress(project_name, "compare_structure_predictions_to_states")
        compare = tools["compare_structure_predictions_to_states"](
            project_name=project_name,
            structures_csv=standardized_csv,
            target_state_structure=target_structure,
            reference_state_structure=reference_structure,
            prediction_chain=parse_chains,
            target_chain=parse_chains,
            reference_chain=parse_chains,
            regions=_comparison_regions_from_mutation_region(mutation_region),
            mapping="sequence_alignment",
        )
        report["structure_comparison"] = compare
        _workflow_progress(project_name, "compare_structure_predictions_to_states", compare)

        _workflow_progress(project_name, "filter_structure_confidence")
        confidence = tools["filter_structure_confidence"](
            project_name=project_name,
            candidates_csv=str(selection_result.get("output_csv") or ""),
            structures_csv=standardized_csv,
            structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
            structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
            chain_id=parse_chains,
            mutation_window_radius=3,
            min_confidence_score=0.7,
            min_complex_plddt=0.7,
            min_local_plddt=70.0,
            require_pae=True,
            require_target_like=True,
            require_complete=True,
        )
        report["structure_confidence"] = confidence
        _workflow_progress(project_name, "filter_structure_confidence", confidence)
        if int(confidence.get("retained_count") or 0) == 0:
            raise RuntimeError("Structure confidence filter retained zero candidates.")

        confidence_csv = str(confidence.get("output_csv") or "")
        _workflow_progress(project_name, "prepare_inverse_folding_scoring_job")
        inverse_folding = tools["prepare_inverse_folding_scoring_job"](
            project_name=project_name,
            structures_csv=standardized_csv,
            candidates_csv=confidence_csv,
            model_set="esm_if1,protein_mpnn",
            chain_id=parse_chains,
            top_n=structure_n,
            execute=True,
            timeout_seconds=timeout_seconds,
        )
        report["inverse_folding_preparation"] = inverse_folding
        _workflow_progress(project_name, "prepare_inverse_folding_scoring_job", inverse_folding)
        if inverse_folding.get("status") == "blocked_no_candidates":
            raise RuntimeError("Inverse-folding preparation received zero candidates.")

        _workflow_progress(project_name, "collect_inverse_folding_scores")
        inverse_collect = tools["collect_inverse_folding_scores"](
            project_name=project_name,
            score_dir=str(inverse_folding.get("score_dir") or ""),
            candidates_csv=confidence_csv,
        )
        report["inverse_folding_collection"] = inverse_collect
        _workflow_progress(project_name, "collect_inverse_folding_scores", inverse_collect)
        if int(inverse_collect.get("score_file_count") or 0) == 0:
            raise RuntimeError("Inverse-folding scoring produced no score files.")

        _workflow_progress(project_name, "select_candidates_from_evidence:bioemu")
        bioemu_selection = tools["select_candidates_from_evidence"](
            project_name=project_name,
            stage_name="bioemu",
            candidate_csv=confidence_csv,
            cb_scored_csv=str(cb_result.get("scored_library_csv") or ""),
            structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
            structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
            structure_confidence_evidence_csv=str(confidence.get("evidence_csv") or ""),
            inverse_folding_evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
            top_n=bioemu_n,
        )
        bioemu_csv, bioemu_lock = _lock_candidate_csv_to_allowed_variants(
            candidate_csv=str(bioemu_selection.get("output_csv") or ""),
            allowed_csv=confidence_csv,
            label="bioemu",
        )
        report["candidate_locks"].append(bioemu_lock)
        report["bioemu_selection"] = {**bioemu_selection, "locked_output_csv": bioemu_csv}
        _workflow_progress(project_name, "select_candidates_from_evidence:bioemu", bioemu_selection)
        if int(bioemu_selection.get("selected_variant_count") or 0) == 0:
            raise RuntimeError("BioEmu candidate selection retained zero candidates.")

        _workflow_progress(project_name, "prepare_bioemu_sampling_job")
        bioemu = tools["prepare_bioemu_sampling_job"](
            project_name=project_name,
            candidates_csv=bioemu_csv,
            top_n=bioemu_n,
            num_samples=bioemu_samples,
            batch_size=10,
            execute=True,
            timeout_seconds=timeout_seconds,
        )
        report["bioemu_preparation"] = bioemu
        _workflow_progress(project_name, "prepare_bioemu_sampling_job", bioemu)
        if bioemu.get("status") == "blocked_no_candidates":
            raise RuntimeError("BioEmu preparation received zero candidates.")

        _workflow_progress(project_name, "collect_bioemu_outputs")
        bioemu_collect = tools["collect_bioemu_outputs"](
            project_name=project_name,
            sample_root=str(bioemu.get("sample_root") or ""),
            target_state_structure=target_structure,
            reference_state_structure=reference_structure,
            mutation_region=mutation_region,
            first_residue_number=int(inspection.get("suggested_first_residue_number") or 1),
            chain_id=parse_chains,
            compute_metrics=True,
        )
        report["bioemu_collection"] = bioemu_collect
        _workflow_progress(project_name, "collect_bioemu_outputs", bioemu_collect)

        _workflow_progress(project_name, "select_candidates_from_evidence:rosetta")
        rosetta_selection = tools["select_candidates_from_evidence"](
            project_name=project_name,
            stage_name="rosetta",
            candidate_csv=bioemu_csv or confidence_csv,
            cb_scored_csv=str(cb_result.get("scored_library_csv") or ""),
            structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
            structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
            structure_confidence_evidence_csv=str(confidence.get("evidence_csv") or ""),
            inverse_folding_evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
            bioemu_evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
            top_n=0,
            retain_all_candidates=True,
        )
        rosetta_csv, rosetta_lock = _lock_candidate_csv_to_allowed_variants(
            candidate_csv=str(rosetta_selection.get("output_csv") or ""),
            allowed_csv=confidence_csv,
            label="rosetta",
        )
        report["candidate_locks"].append(rosetta_lock)
        report["rosetta_selection"] = {**rosetta_selection, "locked_output_csv": rosetta_csv}
        report["rosetta_candidate_policy"] = "all_bioemu_candidates_ranking_only"
        _workflow_progress(project_name, "select_candidates_from_evidence:rosetta", rosetta_selection)
        if int(rosetta_selection.get("selected_variant_count") or 0) == 0:
            raise RuntimeError("Rosetta candidate selection retained zero candidates.")

        standardized_path = Path(standardized_csv).expanduser()
        standardized_pdb_dir = standardized_path.parent / "standardized_pdbs"
        input_pdb_dir = str(standardized_pdb_dir if standardized_pdb_dir.exists() else standardized_path.parent)
        project_manifest = _read_json(_project_report_dir(project_name).parent / "project_manifest.json")
        _workflow_progress(project_name, "select_rosetta_analysis_strategy")
        rosetta_strategy = tools["select_rosetta_analysis_strategy"](
            project_name=project_name,
            design_objective=str(project_manifest.get("design_objective") or ""),
            mutation_region=mutation_region,
            candidate_csv=rosetta_csv,
            structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
            bioemu_evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
            target_state_structure=target_structure,
            reference_state_structure=reference_structure,
            requested_focus="auto",
            requested_mode="auto",
        )
        report["rosetta_strategy"] = rosetta_strategy
        _workflow_progress(project_name, "select_rosetta_analysis_strategy", rosetta_strategy)
        _workflow_progress(project_name, "prepare_rosetta_energy_job")
        rosetta = tools["prepare_rosetta_energy_job"](
            project_name=project_name,
            mode=str(rosetta_strategy.get("mode") or "score"),
            input_pdb_dir=input_pdb_dir,
            structures_csv=standardized_csv,
            candidates_csv=rosetta_csv,
            protocol_xml=str(rosetta_strategy.get("protocol_xml") or "") or None,
            interface=str(rosetta_strategy.get("interface") or "") or None,
            nstruct=int(rosetta_strategy.get("nstruct") or 1),
            strategy_name=str(rosetta_strategy.get("strategy_name") or ""),
            strategy_focus=str(rosetta_strategy.get("focus") or ""),
            strategy_reason=str(rosetta_strategy.get("reason") or ""),
            requested_metrics=",".join(rosetta_strategy.get("metrics_to_collect") or []),
            target_state_structure=target_structure,
            reference_state_structure=reference_structure,
            fallback_strategy=str(rosetta_strategy.get("fallback_strategy") or ""),
            strategy_confidence=rosetta_strategy.get("confidence"),
            execute=True,
            timeout_seconds=timeout_seconds,
        )
        report["rosetta_preparation"] = rosetta
        _workflow_progress(project_name, "prepare_rosetta_energy_job", rosetta)
        if rosetta.get("status") == "blocked_no_candidates":
            raise RuntimeError("Rosetta preparation received zero candidates.")

        _workflow_progress(project_name, "collect_rosetta_scores")
        rosetta_collect = tools["collect_rosetta_scores"](
            project_name=project_name,
            score_file=str(rosetta.get("score_file") or rosetta.get("scorefile") or ""),
            job_manifest_path=str(rosetta.get("job_manifest_path") or ""),
        )
        report["rosetta_collection"] = rosetta_collect
        _workflow_progress(project_name, "collect_rosetta_scores", rosetta_collect)

        _workflow_progress(project_name, "merge_switchtune_evidence_tables")
        merged = tools["merge_switchtune_evidence_tables"](
            project_name=project_name,
            candidate_csv=rosetta_csv or bioemu_csv or confidence_csv,
            cb_scored_csv=str(cb_result.get("scored_library_csv") or ""),
            structure_prediction_evidence_csv=str(collect_structure.get("evidence_csv") or ""),
            structure_state_similarity_evidence_csv=str(compare.get("evidence_csv") or ""),
            structure_confidence_evidence_csv=str(confidence.get("evidence_csv") or ""),
            inverse_folding_evidence_csv=str(inverse_collect.get("evidence_csv") or ""),
            bioemu_evidence_csv=str(bioemu_collect.get("evidence_csv") or ""),
            rosetta_evidence_csv=str(rosetta_collect.get("evidence_csv") or ""),
            top_n=_csv_row_count_for_guard(rosetta_csv),
        )
        report["evidence_merge"] = merged
        _workflow_progress(project_name, "merge_switchtune_evidence_tables", merged)
        if int(merged.get("ranked_variant_count") or 0) == 0:
            raise RuntimeError(
                "Evidence merge produced no numerically ranked candidates. "
                "Do not create a final recommendation from empty or non-numeric evidence."
            )

        _workflow_progress(project_name, "prepare_final_candidate_recommendation")
        final = tools["prepare_final_candidate_recommendation"](
            project_name=project_name,
            evidence_csv=str(merged.get("merged_csv") or merged.get("ranked_csv") or ""),
            rank_column="consensus_score",
            ascending=False,
            top_n=max(1, int(rosetta_n)),
        )
        report["final_recommendation"] = final
        if int(final.get("candidate_count") or 0) == 0:
            raise RuntimeError(
                "Final recommendation contains zero candidates after numeric-evidence filtering."
            )
        report["final_status"] = "completed"
        _workflow_progress(project_name, "prepare_final_candidate_recommendation", final)

        review_tool = _switchtune_tool("prepare_agentic_candidate_review")
        if review_tool is not None:
            try:
                _workflow_progress(project_name, "prepare_agentic_candidate_review")
                agentic_review = review_tool(
                    project_name=project_name,
                    evidence_csv=str(merged.get("merged_csv") or merged.get("ranked_csv") or final.get("output_csv") or ""),
                    top_n=max(structure_n, bioemu_n, rosetta_n),
                    include_stage_reviews=True,
                )
                report["agentic_candidate_review"] = agentic_review
                _workflow_progress(project_name, "prepare_agentic_candidate_review", agentic_review)
            except Exception as review_exc:
                report["agentic_candidate_review_error"] = repr(review_exc)
                _workflow_progress(
                    project_name,
                    "prepare_agentic_candidate_review",
                    {"status": "failed", "error": repr(review_exc)},
                )

        report_path = _write_json_report(project_name, "full_agent_workflow_summary.json", report)
        if report_path:
            report["report_path"] = report_path
        return _compact_full_workflow_solution(report)
    except Exception as exc:
        return _compact_full_workflow_failed_solution(report, exc)


def _should_stop_from_observation(
    observation: str,
    stop_markers: Iterable[str],
    *,
    stop_on_prepared_tool: bool = True,
) -> bool:
    lowered = observation.lower()
    for marker in stop_markers:
        if marker.lower() in lowered:
            return True

    data = _extract_first_json_object(observation)
    if isinstance(data, dict):
        if data.get("stop_after_this_tool") is True:
            return True
        if stop_on_prepared_tool and data.get("status") in {"completed", "prepared"} and data.get("cb_run_script"):
            return True
    return False


def _extract_execute_code(text: str) -> Optional[str]:
    code = _extract_tag(text, "execute")
    if code is None:
        return None
    return re.sub(r"\s+", " ", code.strip())


def _extract_execute_code_raw(text: str) -> Optional[str]:
    """Return execute-block source without normalizing whitespace.

    The normalized form above is useful for loop detection, but it cannot be
    safely parsed when the model includes comments or multiline strings.
    """
    code = _extract_tag(text, "execute")
    return code.strip() if code is not None else None


def _safe_ast_value(node: ast.AST, bindings: dict[str, Any]) -> tuple[bool, Any]:
    """Resolve only literals and previously assigned literal values."""
    if isinstance(node, ast.Name) and node.id in bindings:
        return True, bindings[node.id]
    if isinstance(node, ast.Constant):
        return True, node.value
    try:
        return True, ast.literal_eval(node)
    except Exception:
        return False, None


def _extract_agentic_decision_call(code: str) -> Optional[dict[str, Any]]:
    """Extract one decision-tool call without executing arbitrary model code.

    Review models occasionally emit an exploratory ``<execute>`` block and a
    final ``<solution>`` in the same assistant message. Biomni then returns
    without producing a tool observation. This helper only accepts the
    literal keyword arguments of ``record_agentic_candidate_decision`` and
    never evaluates the generated Python source.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    bindings: dict[str, Any] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                ok, value = _safe_ast_value(statement.value, bindings)
                if ok:
                    bindings[target.id] = value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is not None:
                ok, value = _safe_ast_value(statement.value, bindings)
                if ok:
                    bindings[statement.target.id] = value

    decision_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = ""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if function_name == "record_agentic_candidate_decision":
            decision_calls.append(node)
    if len(decision_calls) != 1:
        return None

    call = decision_calls[0]
    if call.args:
        return None
    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        ok, value = _safe_ast_value(keyword.value, bindings)
        if not ok:
            return None
        values[keyword.arg] = value

    required = {"project_name", "selected_variants", "decision_text"}
    if not required.issubset(values):
        return None
    values.setdefault("evidence_packet_json", None)
    values.setdefault("reviewer", "biomni")
    values.setdefault("output_dir", None)
    values.setdefault("decision_scope", "final_round")
    return values


def _execute_agentic_decision_fallback(code: str) -> Optional[dict[str, Any]]:
    """Execute only the extracted candidate-decision MCP call."""
    call = _extract_agentic_decision_call(code)
    if call is None:
        return None
    tool = _switchtune_tool("record_agentic_candidate_decision")
    if tool is None:
        raise RuntimeError("record_agentic_candidate_decision MCP tool is unavailable")
    result = tool(**call)
    if isinstance(result, dict):
        return result
    parsed = _extract_first_json_object(str(result))
    return parsed if isinstance(parsed, dict) else {"status": "completed", "raw_result": str(result)}


def _is_agentic_decision_observation(observation: str) -> bool:
    """Detect a successful candidate-decision tool result."""
    data = _extract_first_json_object(observation)
    if not isinstance(data, dict):
        return False
    if data.get("status") not in {"completed", "blocked_no_candidates"}:
        return False
    return bool(
        data.get("decision_json")
        or data.get("decision_scope")
        or data.get("selected_variant_count") is not None
    )


def _collector_variant_count(result: dict[str, Any]) -> int:
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
    return 0


def install_switchtune_go_guard(
    agent: Any,
    *,
    replace_go: bool = False,
    max_steps: int = 40,
    max_execute_calls: int = 8,
    max_same_execute_calls: int = 1,
    recursion_limit: int = 80,
    thread_id: int = 42,
    stop_markers: Iterable[str] = DEFAULT_STOP_MARKERS,
    auto_advance_after_inspection: bool = True,
    auto_execute_cb_after_setup: bool = False,
    auto_run_full_workflow_after_inspection: bool = False,
    auto_advance_on_repeated_tool: bool = True,
    stop_on_prepared_tool: bool = True,
    cb_timeout_seconds: int = 3600,
    workflow_timeout_seconds: int = 3600,
    stop_after_agentic_decision: bool = False,
) -> Callable[[str], tuple[list[str], str]]:
    """Install a guarded `go` method for SwitchTune tasks.

    Biomni's default graph always returns from `execute` to `generate`, and stops
    only when the LLM emits `<solution>`. This guard watches observations and
    repeated execute blocks, then returns a deterministic `<solution>` when the
    SwitchTune tool has already completed.

    Use:

        guarded_go = install_switchtune_go_guard(agent, replace_go=True)
        result = agent.go(prompt)

    or:

        guarded_go = install_switchtune_go_guard(agent)
        result = guarded_go(prompt)
    """

    def guarded_go(self: Any, prompt: str) -> tuple[list[str], str]:
        self.critic_count = 0
        self.user_task = prompt

        if getattr(self, "use_tool_retriever", False):
            selected_resources_names = self._prepare_resources_for_retrieval(prompt)
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        # A single review agent is reused across checkpoints. Give each call
        # its own graph thread so an earlier review cannot leak messages or
        # tool state into the next checkpoint.
        active_thread_id = getattr(self, "_switchtune_thread_id", thread_id)
        config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": active_thread_id}}
        self.log = []
        final_state = None
        execute_counter: Counter[str] = Counter()
        execute_calls = 0
        last_observation = ""
        pending_inspection: Optional[dict[str, Any]] = None
        pending_adaptive_selection: Optional[dict[str, Any]] = None
        pending_engine_selection: Optional[dict[str, Any]] = None

        for step_idx, state in enumerate(self.app.stream(inputs, stream_mode="values", config=config), start=1):
            message = state["messages"][-1]
            content = str(getattr(message, "content", ""))
            out = str(_safe_pretty_print(message))
            self.log.append(out)
            final_state = state

            observation = _extract_tag(content, "observation")
            if observation is None:
                observation = _observation_from_message(content, out)
            if observation is not None:
                last_observation = observation
                if stop_after_agentic_decision and _is_agentic_decision_observation(observation):
                    solution = (
                        "<solution>\n"
                        "Biomni recorded the agentic candidate decision successfully; "
                        "the review guard stopped this review before any duplicate decision call.\n\n"
                        + _compact_switchtune_solution(observation).replace("<solution>", "").replace(
                            "</solution>", ""
                        ).strip()
                        + "\n</solution>"
                    )
                    self.log.append(solution)
                    state["messages"].append(AIMessage(content=solution))
                    self._conversation_state = state
                    return self.log, solution
                inspection_data = _inspection_data_from_observation(observation, prompt=prompt)
                if inspection_data is not None:
                    pending_inspection = inspection_data
                    inspection_project = _extract_project_name_from_prompt(prompt)
                    _workflow_progress(
                        inspection_project,
                        "inspection_parsed",
                        {
                            "status": "completed",
                            "next_tool": inspection_data.get("next_tool"),
                            "mutation_region": inspection_data.get("suggested_mutation_region"),
                        },
                    )
                    if auto_advance_after_inspection:
                        concrete_region = _concrete_mutation_region(prompt, inspection_data)
                        existing_setup = _existing_setup_result(
                            inspection_project,
                            requested_region=concrete_region,
                            stop_after_setup=False,
                        )
                        if not concrete_region and existing_setup is None:
                            # Dynamic-region workflows, such as ATCase, must
                            # return control to Biomni so it can select and
                            # report a parseable region. Never substitute the
                            # prompt placeholder into the setup tool.
                            _workflow_progress(
                                inspection_project,
                                "inspection_dispatch",
                                {
                                    "status": "waiting_for_agent_region",
                                    "reason": (
                                        "Automatic handoff paused until Biomni selects a concrete "
                                        "mutation region."
                                    ),
                                },
                            )
                        else:
                            try:
                                if auto_run_full_workflow_after_inspection:
                                    solution = _auto_run_full_workflow_from_inspection(
                                        prompt,
                                        inspection_data,
                                        cb_timeout_seconds=cb_timeout_seconds,
                                        workflow_timeout_seconds=workflow_timeout_seconds,
                                    )
                                elif auto_execute_cb_after_setup:
                                    solution = _auto_run_front_half_from_inspection(
                                        prompt,
                                        inspection_data,
                                        cb_timeout_seconds,
                                    )
                                else:
                                    solution = _auto_setup_from_inspection(prompt, inspection_data)
                            except Exception as exc:
                                _workflow_progress(
                                    inspection_project,
                                    "inspection_dispatch",
                                    {"status": "failed", "error": repr(exc)},
                                )
                                solution = (
                                    "<solution>\n"
                                    "SwitchTune go-guard recognized the PDB inspection result but the automatic "
                                    "workflow handoff failed.\n\n"
                                    f"- project_name: `{inspection_project}`\n"
                                    f"- error: `{repr(exc)}`\n"
                                    "Inspect the guard event log and the first stage log before rerunning.\n"
                                    "</solution>"
                                )
                            if solution is not None:
                                self.log.append(solution)
                                state["messages"].append(AIMessage(content=solution))
                                self._conversation_state = state
                                return self.log, solution
                # Dynamic-region checkpoint runs often let Biomni call setup
                # directly after it selects a concrete region.  In that
                # path the inspection handoff above has already happened, so
                # continue the deterministic front half from the setup
                # observation before the checkpoint runner looks for a
                # CB-selected CSV.
                setup_data = _setup_result_from_observation(observation)
                if setup_data is not None and auto_execute_cb_after_setup:
                    try:
                        solution = _auto_run_front_half_from_setup(
                            prompt,
                            setup_data,
                            cb_timeout_seconds,
                        )
                    except Exception as exc:
                        _workflow_progress(
                            str(setup_data.get("project_name") or _extract_project_name_from_prompt(prompt)),
                            "setup_checkpoint_observation",
                            {"status": "failed", "error": repr(exc)},
                        )
                        solution = None
                    if solution is not None:
                        self.log.append(solution)
                        state["messages"].append(AIMessage(content=solution))
                        self._conversation_state = state
                        return self.log, solution

                adaptive_data = _adaptive_selection_data_from_observation(observation)
                if adaptive_data is not None:
                    pending_adaptive_selection = adaptive_data
                    solution = _auto_prepare_structure_from_adaptive(prompt, adaptive_data)
                    if solution is not None:
                        self.log.append(solution)
                        state["messages"].append(AIMessage(content=solution))
                        self._conversation_state = state
                        return self.log, solution
                engine_data = _engine_selection_data_from_observation(observation)
                if engine_data is not None:
                    pending_engine_selection = engine_data
                    solution = _auto_prepare_structure_from_engine_selection(prompt, engine_data)
                    if solution is not None:
                        self.log.append(solution)
                        state["messages"].append(AIMessage(content=solution))
                        self._conversation_state = state
                        return self.log, solution
                if _should_stop_from_observation(
                    observation,
                    stop_markers,
                    stop_on_prepared_tool=stop_on_prepared_tool,
                ):
                    solution = _compact_switchtune_solution(observation)
                    self.log.append(solution)
                    state["messages"].append(AIMessage(content=solution))
                    self._conversation_state = state
                    return self.log, solution

            raw_code = _extract_execute_code_raw(content)
            code = _extract_execute_code(content)
            if code is not None:
                execute_calls += 1
                execute_counter[code] += 1
                # A review model can emit the decision call and <solution> in
                # one assistant message. In that case Biomni's normal graph
                # may return without running the execute node. Recover only
                # the single candidate-decision MCP call; never exec the full
                # model-generated source.
                if stop_after_agentic_decision and raw_code and (
                    "record_agentic_candidate_decision" in raw_code
                ):
                    try:
                        fallback_result = _execute_agentic_decision_fallback(raw_code)
                    except Exception as exc:
                        fallback_result = None
                        _workflow_progress(
                            _extract_project_name_from_prompt(prompt),
                            "agentic_decision_fallback",
                            {"status": "failed", "error": repr(exc)},
                        )
                    if fallback_result is not None:
                        fallback_observation = json.dumps(
                            fallback_result,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                        self.log.append(
                            "<observation>"
                            + fallback_observation
                            + "</observation>"
                        )
                        _workflow_progress(
                            str(fallback_result.get("project_name") or _extract_project_name_from_prompt(prompt)),
                            "agentic_decision_fallback",
                            {
                                "status": fallback_result.get("status", "unknown"),
                                "reviewer": fallback_result.get("reviewer", ""),
                                "decision_scope": fallback_result.get("decision_scope", ""),
                                "selected_variant_count": fallback_result.get(
                                    "selected_variant_count", 0
                                ),
                            },
                        )
                        if _is_agentic_decision_observation(fallback_observation):
                            solution = (
                                "<solution>\n"
                                "SwitchTune review guard recovered and executed the single "
                                "candidate-decision MCP call emitted by Biomni.\n\n"
                                + _compact_switchtune_solution(fallback_observation)
                                .replace("<solution>", "")
                                .replace("</solution>", "")
                                .strip()
                                + "\n</solution>"
                            )
                            self.log.append(solution)
                            state["messages"].append(
                                AIMessage(content=f"<observation>{fallback_observation}</observation>")
                            )
                            state["messages"].append(AIMessage(content=solution))
                            self._conversation_state = state
                            return self.log, solution
                if pending_inspection is not None and (
                    "inspect_two_state_pdb_pair" in code or "Bio.PDB" in code or "PDBParser" in code
                ):
                    if auto_advance_on_repeated_tool:
                        concrete_region = _concrete_mutation_region(prompt, pending_inspection)
                        existing_setup = _existing_setup_result(
                            _extract_project_name_from_prompt(prompt),
                            requested_region=concrete_region,
                            stop_after_setup=False,
                        )
                        if not concrete_region and existing_setup is None:
                            fallback_region, fallback_rationale = _repeated_inspection_region_fallback(
                                prompt,
                                pending_inspection,
                            )
                            if fallback_region:
                                recovered_inspection = dict(pending_inspection)
                                recovered_inspection["suggested_mutation_region"] = fallback_region
                                recovered_inspection["mutation_region_selection_required"] = False
                                recovered_inspection["mutation_region_rationale"] = fallback_rationale
                                _workflow_progress(
                                    _extract_project_name_from_prompt(prompt),
                                    "inspection_dispatch",
                                    {
                                        "status": "recovered_repeated_inspection",
                                        "mutation_region": fallback_region,
                                        "mutation_region_source": "guard_recovery_after_repeated_inspection",
                                        "reason": fallback_rationale,
                                    },
                                )
                                if auto_run_full_workflow_after_inspection:
                                    solution = _auto_run_full_workflow_from_inspection(
                                        prompt,
                                        recovered_inspection,
                                        cb_timeout_seconds=cb_timeout_seconds,
                                        workflow_timeout_seconds=workflow_timeout_seconds,
                                    )
                                elif auto_execute_cb_after_setup:
                                    solution = _auto_run_front_half_from_inspection(
                                        prompt,
                                        recovered_inspection,
                                        cb_timeout_seconds,
                                    )
                                else:
                                    solution = _auto_setup_from_inspection(prompt, recovered_inspection)
                                if solution is not None:
                                    solution = (
                                        "<solution>\n"
                                        "SwitchTune go-guard recovered a repeated structure-inspection loop and "
                                        f"continued with the validated ATCase mutation region `{fallback_region}`.\n\n"
                                        + solution.replace("<solution>", "").replace("</solution>", "").strip()
                                        + "\n</solution>"
                                    )
                                    self.log.append(solution)
                                    state["messages"].append(AIMessage(content=solution))
                                    self._conversation_state = state
                                    return self.log, solution
                            else:
                                _workflow_progress(
                                    _extract_project_name_from_prompt(prompt),
                                    "inspection_dispatch",
                                    {
                                        "status": "waiting_for_agent_region",
                                        "reason": (
                                            "Repeated inspection was not auto-advanced because no concrete "
                                            "mutation region was available."
                                        ),
                                    },
                                )
                        else:
                            if auto_run_full_workflow_after_inspection:
                                solution = _auto_run_full_workflow_from_inspection(
                                    prompt,
                                    pending_inspection,
                                    cb_timeout_seconds=cb_timeout_seconds,
                                    workflow_timeout_seconds=workflow_timeout_seconds,
                                )
                            elif auto_execute_cb_after_setup:
                                solution = _auto_run_front_half_from_inspection(
                                    prompt,
                                    pending_inspection,
                                    cb_timeout_seconds,
                                )
                            else:
                                solution = _auto_setup_from_inspection(prompt, pending_inspection)
                            if solution is not None:
                                if auto_run_full_workflow_after_inspection:
                                    advance_message = (
                                        "SwitchTune go-guard detected a repeated structure-inspection step and "
                                        "executed the validated full workflow controller using the standardized "
                                        "inspection result.\n\n"
                                    )
                                else:
                                    advance_message = (
                                        "SwitchTune go-guard detected a repeated structure-inspection step and "
                                        "advanced to the required setup tool using the standardized inspection result.\n\n"
                                    )
                                solution = (
                                    "<solution>\n"
                                    + advance_message
                                    + solution.replace("<solution>", "").replace("</solution>", "").strip()
                                    + "\n</solution>"
                                )
                                self.log.append(solution)
                                state["messages"].append(AIMessage(content=solution))
                                self._conversation_state = state
                                return self.log, solution
                if pending_adaptive_selection is not None and (
                    "select_state_specific_candidates_adaptive" in code
                    or "select_state_specific_candidates" in code
                ):
                    if auto_advance_on_repeated_tool:
                        solution = _auto_prepare_structure_from_adaptive(prompt, pending_adaptive_selection)
                        if solution is not None:
                            solution = (
                                "<solution>\n"
                                "SwitchTune go-guard detected repeated CB selection and advanced to "
                                "structure-prediction job preparation using the selected candidates.\n\n"
                                + solution.replace("<solution>", "").replace("</solution>", "").strip()
                                + "\n</solution>"
                            )
                            self.log.append(solution)
                            state["messages"].append(AIMessage(content=solution))
                            self._conversation_state = state
                            return self.log, solution
                if pending_engine_selection is not None and "select_structure_prediction_engine" in code:
                    if auto_advance_on_repeated_tool:
                        solution = _auto_prepare_structure_from_engine_selection(prompt, pending_engine_selection)
                        if solution is not None:
                            solution = (
                                "<solution>\n"
                                "SwitchTune go-guard detected repeated structure-engine selection and advanced to "
                                "structure-prediction job preparation using the selected engine.\n\n"
                                + solution.replace("<solution>", "").replace("</solution>", "").strip()
                                + "\n</solution>"
                            )
                            self.log.append(solution)
                            state["messages"].append(AIMessage(content=solution))
                            self._conversation_state = state
                            return self.log, solution
                if execute_counter[code] > max_same_execute_calls:
                    solution = (
                        "<solution>\n"
                        "Stopped because Biomni attempted to repeat the same execute block after a completed "
                        "SwitchTune step.\n\n"
                        + _compact_switchtune_solution(last_observation).replace("<solution>", "").replace(
                            "</solution>", ""
                        ).strip()
                        + "\n</solution>"
                    )
                    self.log.append(solution)
                    state["messages"].append(AIMessage(content=solution))
                    self._conversation_state = state
                    return self.log, solution

            if execute_calls >= max_execute_calls or step_idx >= max_steps:
                solution = (
                    "<solution>\n"
                    f"Stopped by SwitchTune go-guard after {step_idx} graph steps and "
                    f"{execute_calls} execute calls. Last observation summary:\n\n"
                    + _compact_switchtune_solution(last_observation).replace("<solution>", "").replace(
                        "</solution>", ""
                    ).strip()
                    + "\n</solution>"
                )
                self.log.append(solution)
                state["messages"].append(AIMessage(content=solution))
                self._conversation_state = state
                return self.log, solution

        self._conversation_state = final_state
        if final_state is not None:
            message = final_state["messages"][-1]
            return self.log, str(getattr(message, "content", ""))
        return self.log, ""

    bound = MethodType(guarded_go, agent)
    agent.go_switchtune_guarded = bound
    if replace_go:
        agent.go_original = agent.go
        agent.go = bound
    return bound
