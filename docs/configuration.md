<!-- API, model and workflow configuration guide. -->

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

## Structure prediction backend

Run `python3 scripts/configure_structure_backend.py` before installing predictor dependencies. It saves one deployment choice to ignored `config/structure_backend.local.json`. The default location is shared by the launcher, checkpoint runner and MCP server. Set `SWITCHTUNE_STRUCTURE_CONFIG` to an absolute path if using a different file.

The selection is applied to ligand-free and ligand-containing targets alike. An agent request for a conflicting engine raises an actionable error; it cannot silently replace the installed predictor. Selection precedence is an explicit runner `--structure_engine`, the `SWITCHTUNE_STRUCTURE_ENGINE` environment variable, the local deployment file, then the legacy YAML default (`boltz2_local`). An intentional override should point to an already installed backend.

| AF3 setting | Purpose |
| --- | --- |
| `AF3_DEPLOYMENT` | `docker` (default) or `native` |
| `AF3_DOCKER_IMAGE` | Official AF3 image built locally, default `alphafold3` |
| `AF3_MODEL_DIR` | Directory containing model parameters |
| `AF3_DB_DIR` | Database directory; needed for `search` mode |
| `AF3_REPO`, `AF3_PYTHON` | Native AF3 checkout and its Python executable |
| `AF3_MSA_MODE` | `search`, `provided`, or `single_sequence` |
| `AF3_INPUT_TEMPLATE` | Optional official WT JSON for prepared MSAs, complex entities, modifications, covalent bonds and custom CCDs |
| `AF3_SEEDS`, `AF3_NUM_SAMPLES` | Comma-separated seeds and samples per seed; defaults `1` and `5` |

The registry's `alphafold3` block can also set `protein_chain_id` (one uppercase letter, default `A`) and `extra_args` (a list). Managed input/output, pipeline and sample-count flags cannot be overridden through `extra_args`. Environment overrides take precedence over the local file and registry. `search` invokes the official MSA data pipeline; `provided` and `single_sequence` run inference only.

For ordinary ligands, the existing MCP `ligand_codes`, `ligand_smiles` and `ligand_payload_json` inputs work with both backends. AF3 maps CCD codes to `ccdCodes` and SMILES to `smiles`. Detected PDB HETATM codes are suggestions and must be checked for biological relevance. For covalent cofactors, modified residues, extra protein chains or custom CCDs, provide `AF3_INPUT_TEMPLATE`; the template owns all entities and bonds, and generic detected ligands are not appended to it. Only the designed chain sequence changes. Ensure the template describes the intended target complex. Molecular stereochemistry and covalent connectivity are not inferred from PDB HETATM records.

AF3 input templates must use separate, single-letter entity IDs for downstream PDB compatibility. The designed protein ID defaults to `A`. The structural template adapter extracts one protein chain, aligns resolved coordinates to each candidate and writes explicit zero-based query/template mappings. It writes entity/sequence/scheme metadata and a creation date for the derived template, as required by AF3. Missing residues are handled by sequence alignment. Arbitrary multi-chain sequence redesign is outside this single-substitution adapter.

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
