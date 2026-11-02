# Installation and external-model deployment

SwitchTune has a lightweight orchestration layer and several optional scientific backends. Install the orchestration layer first, then install only the stages required for the intended design.

## 1. Base environment

Use a Linux server with a recent NVIDIA driver, CUDA-compatible PyTorch and enough local storage for model checkpoints and predicted structures. Create a dedicated environment for the agent layer:

```bash
conda create -n switchtune python=3.11 -y
conda activate switchtune
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install biomni
```

Install the package in editable mode when developing locally:

```bash
python -m pip install -e .
```

Biomni is the agent runtime. Its model source, model identifier, base URL and API key are configured by the user through `SWITCHTUNE_LLM_SOURCE`, `SWITCHTUNE_LLM_MODEL`, `SWITCHTUNE_LLM_BASE_URL` and `SWITCHTUNE_LLM_API_KEY`. Use the source names and connection settings supported by the installed Biomni version. The key is never written to a SwitchTune file.

## 2. LigandMPNN and ProteinMPNN

LigandMPNN is used for ligand/cofactor-aware state scoring. ProteinMPNN is used for ligand-free scoring. The LigandMPNN distribution also contains the ProteinMPNN scoring implementation used by many deployments.

```bash
git clone https://github.com/dauparas/LigandMPNN.git external/LigandMPNN
cd external/LigandMPNN
conda env create -f environment.yml
conda activate ligandmpnn
bash get_model_params.sh
python score.py --help
cd ../..
```

If a separate ProteinMPNN checkout is preferred, install it from its official repository and set `PROTEINMPNN_REPO` and `PROTEINMPNN_MODEL_PARAMS`. Set `LIGANDMPNN_REPO` and `LIGANDMPNN_ENV` in `.env` and enable the corresponding entries in `config/tool_registry.yaml`. Do not copy checkpoints into this repository.

## 3. ESM-IF1

ESM-IF1 provides inverse-folding sequence-structure compatibility scores. Install the ESM package in a separate environment or reuse the agent environment if the CUDA and PyTorch versions are compatible:

```bash
git clone https://github.com/facebookresearch/esm.git external/esm
conda create -n esm-if1 python=3.10 -y
conda activate esm-if1
python -m pip install --upgrade pip
python -m pip install -e external/esm
python -m pip install biotite
python external/esm/examples/inverse_folding/score_log_likelihoods.py --help
```

The first model invocation downloads the ESM-IF1 weights through the ESM package. For an offline server, download the weights through the official distribution and set `ESM_IF1_REPO` and `ESM_IF1_SCORE_SCRIPT` to the local paths.

## 4. BioEmu

BioEmu samples conformational ensembles after structure and inverse-folding quality control:

```bash
conda create -n bioemu python=3.11 -y
conda activate bioemu
python -m pip install --upgrade pip
python -m pip install bioemu
python -m bioemu.sample --help
```

Follow the BioEmu release instructions for model-weight access and any required Hugging Face authentication. Store the downloaded checkpoint outside the repository and set `BIOEMU_HOME`, `BIOEMU_ENV` and any local checkpoint variables in `.env` or the tool registry.

## 5. Boltz-2

The default structure backend is a local Boltz-2 CLI. Install it in a CUDA-enabled environment:

```bash
conda create -n boltz2 python=3.11 -y
conda activate boltz2
python -m pip install --upgrade pip
python -m pip install "boltz[cuda]"
boltz --help
```

Set `SWITCHTUNE_STRUCTURE_ENGINE=boltz2_local`, `BOLTZ_CACHE` and the executable path in the tool registry. SwitchTune runs one prediction batch at a time by default, requests a target-state structural template and records full PAE/PDE evidence when the installed Boltz version supports it.

A hosted Boltz-2-compatible backend can be used by setting `SWITCHTUNE_STRUCTURE_ENGINE=boltz2_hosted`, `BOLTZ2_ENDPOINT` and `BOLTZ2_API_KEY`. Hosted backends are optional; their endpoint-specific request schema must match `scripts/run_boltz2_nim_batch.py` or a user-supplied adapter.

## 6. Rosetta

Rosetta requires an academic or commercial license and is not redistributed here. Download the source from the official Rosetta Commons distribution, then build the required binaries:

```bash
# after downloading and extracting Rosetta from the licensed distribution
cd rosetta/main/source
./scons.py -j$(nproc) bin mode=release
```

Set `ROSETTA_HOME` to the extracted `main` directory. Enable `rosetta` in `config/tool_registry.yaml` and verify:

```bash
$ROSETTA_HOME/source/bin/score_jd2.static.linuxgccrelease -help | head
$ROSETTA_HOME/source/bin/relax.static.linuxgccrelease -help | head
```

Use the Rosetta protocol appropriate for the protein and complex. SwitchTune records the selected protocol and score files; it does not distribute Rosetta binaries or protocol XML files.

## 7. Verify installation

Run lightweight checks before an expensive design:

```bash
python -c "import switchtune; print(switchtune.__version__)"
python tests/test_mcp_tools.py
python tests/test_adaptive_cb_selection_to_structure_prediction.py
```

Then validate the external command paths:

```bash
python -c "import shutil; print('boltz:', shutil.which('boltz'))"
python external/esm/examples/inverse_folding/score_log_likelihoods.py --help
```

A missing optional backend should produce a recorded pending/blocked stage with an actionable path, not a fabricated result.
