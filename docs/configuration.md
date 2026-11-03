# Configuration

SwitchTune reads configuration from environment variables and the YAML files in `config/`. Start from `.env.example`, then export the variables in the shell or load them with a secret manager.

## Natural-language design prompt

The required `prompt` input is the user's natural-language description of the mutation-design task. It should state the desired state-specific behavior and relevant experimental constraints. The agent selects the mutation region after inspecting both supplied wild-type structures, so the prompt does not need to provide a residue range.

```bash
export SWITCHTUNE_PROMPT="Reduce dark-state leakage while preserving the light-state response and reversible switching."
```

The launcher passes this value to `scripts/run_switchtune_agent_benchmark.py --prompt`. Longer requests can be stored in a UTF-8 file and supplied directly with `--prompt_file`. The legacy `--design_objective` option remains available as a compatibility alias and is used only when `prompt` is omitted.

## LLM connection

```bash
export SWITCHTUNE_LLM_SOURCE=Custom
export SWITCHTUNE_LLM_MODEL=your-model-id
export SWITCHTUNE_LLM_BASE_URL=https://your-provider.example/v1
export SWITCHTUNE_LLM_API_KEY='your_api_key'
```

`Custom` is the Biomni source for an OpenAI-compatible endpoint. If the installed Biomni version provides a native source for the selected provider, use that source and its documented connection parameters. SwitchTune does not assume a particular provider.

## Paths

```bash
export SWITCHTUNE_HOME=/opt/switchtune
export SWITCHTUNE_RUNS=/data/switchtune_runs
export SWITCHTUNE_CONFIG=$SWITCHTUNE_HOME/config/switchtune.yaml
export SWITCHTUNE_TOOL_REGISTRY=$SWITCHTUNE_HOME/config/tool_registry.yaml
export SWITCHTUNE_DECISION_POLICY=$SWITCHTUNE_HOME/config/decision_policy.yaml
```

Tool-specific paths are configured with `LIGANDMPNN_REPO`, `ESM_IF1_REPO`, `BOLTZ_CACHE`, `BIOEMU_HOME` and `ROSETTA_HOME`. The sample registry uses environment expansion and contains no machine-specific user directory.

## Autonomous region selection

The public launcher does not expose `MUTATION_REGION`. The first MCP call inspects both supplied wild-type structures. Biomni then chooses exact residue intervals or discrete positions and records:

- the selected range(s);
- state-dependent structural rationale;
- excluded residues or regions;
- the expected saturation-library size.

The structural fallback is available for provider interruptions, but it is based only on aligned wild-type coordinates and is recorded separately from the agent decision.

## Adaptive CB filtering

The agent first inspects the empirical conformation-bias score distribution. It chooses a central retention fraction according to the score spread and library size. The tool stores the selected fraction, quantile bounds and retained count. A universal score threshold and fixed retained count are not used.

## Adaptive final shortlist

For each downstream stage, the agent may pass an explicit candidate budget. If it leaves the budget unset, SwitchTune derives it from the valid evidence pool with the configured 20-30 range. The derived value is recorded in the manifest and stage summary. This allows larger input libraries to produce larger shortlists while preventing an arbitrary fixed output count.
