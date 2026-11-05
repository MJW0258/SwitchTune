<div align="center">

# 🧬 SwitchTune

**Natural-language-driven mutation design for conformational-switching proteins**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

[Quick start](#quick-start) · [Installation](docs/installation.md) · [Configuration](docs/configuration.md) · [Workflow](docs/workflow.md) · [Architecture](docs/architecture.md)

</div>

## ⚡ Overview

SwitchTune is a natural-language-driven framework for state-aware mutation design in conformational-switching proteins. It coordinates a Biomni scientific agent with modular MCP tools and records each computational decision, intermediate result and quality-control gate.

<div align="center">
  <img src="assets/switchtune-workflow.png" alt="SwitchTune workflow for state-aware mutation design" width="960">
</div>

## ✨ Key features

- 💬 **Natural-language objectives.** Describe the desired switching behaviour and design constraints in a prompt.
- 🧭 **Autonomous region selection.** The agent inspects both wild-type states and chooses structurally justified mutation sites.
- ⚖️ **Two-state scoring.** Compare sequence compatibility in the target and reference states with ProteinMPNN or LigandMPNN.
- 🎯 **Adaptive candidate selection.** Use the empirical score distribution and valid candidate pool to determine retention and shortlist size.
- 🧩 **Modular scientific tools.** Combine structure, inverse-folding, ensemble and energy evidence through MCP tools.
- 🧾 **Traceable, resumable runs.** Record intermediate evidence and decisions, then resume from completed checkpoints.

## 🔬 Workflow

| Stage | What SwitchTune does |
| --- | --- |
| **Input and library setup** | Inspect both wild-type structures, select a mutation region and generate a saturation single-mutant library. |
| **State-bias scoring** | Score variants in both states with ProteinMPNN or LigandMPNN and retain an adaptive, moderate-bias fraction. |
| **Structure prediction and QC** | Predict target-state variants with Boltz-2, then assess confidence, PAE and similarity to both input states. |
| **Sequence scoring** | Validate sequence–structure compatibility with ESM-IF1 and ProteinMPNN/LigandMPNN. |
| **Ensemble sampling** | Characterize conformational ensembles with BioEmu. |
| **Energy scoring** | Evaluate protein or interface energetics with Rosetta. |
| **Evidence fusion and review** | Integrate evidence, record agent decisions and return a ranked candidate shortlist. |

SwitchTune returns an adaptive shortlist of **20–30 candidates** when enough valid variants remain. The budget is calculated from the valid evidence pool. If fewer than 20 variants pass the available gates, SwitchTune reports the complete valid set.

<a id="quick-start"></a>

## 🚀 Quick start

Use Python 3.10 or newer for the orchestration layer. A Linux server with a CUDA-compatible PyTorch installation is recommended for local scientific backends; follow each model's environment requirements.

### 1. Install the orchestration layer

```bash
git clone https://github.com/MJW0258/SwitchTune.git
cd SwitchTune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Follow the [installation guide](docs/installation.md) to install Biomni and the scientific backends needed for your run. The commands above install the base SwitchTune dependencies.

Edit `.env` with the LLM provider settings and local model paths, then export those variables in the launch shell. API credentials are supplied by the user and are read from environment variables.

### 2. Describe and run a design

Set the project inputs and a natural-language design request:

```bash
export PROJECT_NAME=my_switch
export PROTEIN_NAME=MyProtein
export TARGET_STATE_STRUCTURE=/data/target_state.pdb
export REFERENCE_STATE_STRUCTURE=/data/reference_state.pdb
export TARGET_STATE_NAME=target_state
export REFERENCE_STATE_NAME=reference_state
export SWITCHTUNE_PROMPT="Design mutations that favor the target state while preserving reversible switching."
export SWITCHTUNE_LLM_MODEL=your-model-id
export SWITCHTUNE_LLM_API_KEY='your_api_key'
export SWITCHTUNE_LLM_SOURCE=Custom
export SWITCHTUNE_LLM_BASE_URL=https://your-provider.example/v1
bash scripts/run_switchtune.sh
```

`SWITCHTUNE_PROMPT` (or `SWITCHTUNE_PROMPT_FILE`) is the required natural-language design request. It describes the desired state-specific behavior and constraints; the agent interprets it together with the supplied structures. For a longer request, use `SWITCHTUNE_PROMPT_FILE=/path/to/design_request.md` instead of `SWITCHTUNE_PROMPT`, or pass `--prompt_file` to the Python runner.

For a provider with a native Biomni integration, set `SWITCHTUNE_LLM_SOURCE` to the corresponding Biomni source and omit `SWITCHTUNE_LLM_BASE_URL` when it is not needed. For an OpenAI-compatible endpoint, use `Custom` and provide its base URL.

## 🔌 MCP client configuration

Copy `mcp_config.example.yaml` to the configuration format required by your MCP client. Set `SWITCHTUNE_HOME` and `SWITCHTUNE_RUNS` in that client environment. The server command is:

```bash
python -m switchtune.mcp_server
```

## 🧰 External models and deployment

| Component | Role | Deployment guide |
| --- | --- | --- |
| Biomni | Agent runtime and execution environment | [Base environment](docs/installation.md#1-base-environment) |
| ProteinMPNN / LigandMPNN | Two-state scoring and sequence validation | [MPNN setup](docs/installation.md#2-ligandmpnn-and-proteinmpnn) |
| ESM-IF1 | Inverse-folding validation | [ESM-IF1 setup](docs/installation.md#3-esm-if1) |
| BioEmu | Conformational ensemble sampling | [BioEmu setup](docs/installation.md#4-bioemu) |
| Boltz-2 | Structure prediction | [Boltz-2 setup](docs/installation.md#5-boltz-2) |
| Rosetta | Protein and interface energy evaluation | [Rosetta setup](docs/installation.md#6-rosetta) |

Backends are installed separately. Stages that depend on an unavailable component are recorded as pending or blocked. PyTorch and CUDA versions are chosen for each backend's environment.

## 📖 Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Agent environment, external models and deployment |
| [Configuration](docs/configuration.md) | LLM providers, credentials, paths and prompt inputs |
| [Workflow](docs/workflow.md) | Autonomous region selection, filtering, ranking and checkpoints |
| [Architecture](docs/architecture.md) | Agent boundaries, MCP modules and evidence flow |
| [Validation](docs/validation.md) | Local checks and expected results |

## 🗂️ Repository layout

<details>
<summary><strong>Expand file-by-file descriptions</strong></summary>

- `switchtune/mcp_server.py`: MCP tools for structure inspection, library generation, state-specific scoring, structure prediction, inverse folding, ensemble sampling, energy evaluation and evidence fusion.
- `switchtune/biomni_go_guard.py`: guarded Biomni handoff and checkpoint enforcement.
- `switchtune/biomni_mcp_adapter.py`: normalization layer between Biomni tool calls and SwitchTune MCP results.
- `switchtune/__init__.py`: package metadata and public import surface.
- `scripts/run_switchtune_agent_benchmark.py`: checkpointed agent runner that accepts the user's natural-language design prompt.
- `scripts/run_switchtune.sh`: minimal environment-variable launcher for a complete run.
- `scripts/checkpoint_workflow_runner.py`: resumable downstream checkpoint execution and review.
- `scripts/biomni_runtime.py`: Biomni agent construction and review-session utilities.
- `scripts/run_boltz2_local_batch.py`: local Boltz-2 batch execution helper.
- `scripts/run_boltz2_nim_batch.py`: hosted Boltz-2/NIM batch execution helper.
- `config/project_schema.yaml`: required inputs, including the natural-language `prompt`, and agent-decided fields.
- `config/switchtune.yaml`: default workflow and external-tool settings.
- `config/tool_registry.yaml`: executable names, paths and model backends.
- `config/decision_policy.yaml`: autonomous region, adaptive filtering and candidate-budget policies.
- `docs/installation.md`: environment setup and deployment instructions for Biomni and external models.
- `docs/configuration.md`: provider credentials, paths, prompt input and model settings.
- `docs/workflow.md`: checkpoint sequence, autonomous region selection and adaptive ranking behavior.
- `docs/architecture.md`: component boundaries and data flow.
- `docs/validation.md`: validation commands and expected checks.
- `prompts/switchtune_agent_prompt.md`: reusable blind-test agent instructions.
- `prompts/switchtune_agent_review_prompt.md`: checkpoint and final-review instructions.
- `assets/switchtune-workflow.png`: workflow overview figure.
- `mcp_config.example.yaml`: example MCP client registration.
- `.env.example`: template for user-supplied credentials and local model paths.
- `requirements.txt`: minimal Python dependencies.
- `pyproject.toml`: package metadata and tooling configuration.
- `tests/test_mcp_tools.py`: MCP tool contract tests.
- `tests/test_adaptive_cb_selection_to_structure_prediction.py`: adaptive filtering and handoff tests.
- `tests/test_structure_engine_selection_and_boltz_job.py`: structure-engine and Boltz-2 job tests.

</details>

## 🔐 Security and data policy

Do not commit API keys, local absolute paths, generated run directories, private logs, model checkpoints or structure files whose redistribution is restricted. Keep credentials in an environment manager or an ignored `.env` file.

## 📄 License

SwitchTune is released under the [MIT License](LICENSE). External models, weights and software retain their respective licenses and usage conditions.

## 🛠️ Project status

SwitchTune is research software. Model versions, GPU memory, licenses and external command-line interfaces can change. Every run writes a manifest and decision log so that failures can be inspected and stages can be resumed.
