#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher. Set the required variables and the API key before running.
: "${PROJECT_NAME:?Set PROJECT_NAME}"
: "${PROTEIN_NAME:?Set PROTEIN_NAME}"
: "${TARGET_STATE_STRUCTURE:?Set TARGET_STATE_STRUCTURE}"
: "${REFERENCE_STATE_STRUCTURE:?Set REFERENCE_STATE_STRUCTURE}"
: "${SWITCHTUNE_LLM_MODEL:?Set SWITCHTUNE_LLM_MODEL}"
: "${SWITCHTUNE_LLM_API_KEY:?Set SWITCHTUNE_LLM_API_KEY}"

export SWITCHTUNE_HOME="${SWITCHTUNE_HOME:-$(pwd)}"
export SWITCHTUNE_RUNS="${SWITCHTUNE_RUNS:-${SWITCHTUNE_HOME}/SwitchTune_runs}"
export SWITCHTUNE_CONFIG="${SWITCHTUNE_CONFIG:-${SWITCHTUNE_HOME}/config/switchtune.yaml}"
export SWITCHTUNE_TOOL_REGISTRY="${SWITCHTUNE_TOOL_REGISTRY:-${SWITCHTUNE_HOME}/config/tool_registry.yaml}"
export SWITCHTUNE_DECISION_POLICY="${SWITCHTUNE_DECISION_POLICY:-${SWITCHTUNE_HOME}/config/decision_policy.yaml}"
export PYTHONPATH="${SWITCHTUNE_HOME}:${PYTHONPATH:-}"

python -u "${SWITCHTUNE_HOME}/scripts/run_switchtune_agent_benchmark.py" \
  --project_name "$PROJECT_NAME" \
  --protein_name "$PROTEIN_NAME" \
  --target_state_structure "$TARGET_STATE_STRUCTURE" \
  --reference_state_structure "$REFERENCE_STATE_STRUCTURE" \
  --target_state_name "${TARGET_STATE_NAME:-target_state}" \
  --reference_state_name "${REFERENCE_STATE_NAME:-reference_state}" \
  --parse_chains "${PARSE_CHAINS:-A}" \
  --first_residue_number "${FIRST_RESIDUE_NUMBER:-1}" \
  --structure_engine "${SWITCHTUNE_STRUCTURE_ENGINE:-boltz2_local}" \
  --structure_top_n 0 \
  --bioemu_top_n 0 \
  --rosetta_top_n 0 \
  --llm "$SWITCHTUNE_LLM_MODEL" \
  --llm_source "${SWITCHTUNE_LLM_SOURCE:-Custom}" \
  --base_url "${SWITCHTUNE_LLM_BASE_URL:-}" \
  --api_key_env SWITCHTUNE_LLM_API_KEY
