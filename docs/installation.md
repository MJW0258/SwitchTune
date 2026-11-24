# Installation and external-model deployment

SwitchTune has a lightweight orchestration layer and separately installed scientific backends. **Select the structure predictor before installing model dependencies.** Install either Boltz-2 or AlphaFold3; both feed the same downstream quality-control and scoring workflow.

## 0. Choose a structure predictor

Clone the repository, inspect the server and save your choice. This helper uses only Python's standard library, so no scientific environment is required yet:

```bash
git clone https://github.com/MJW0258/SwitchTune.git
cd SwitchTune
python3 scripts/configure_structure_backend.py --storage-root /data
```

The helper reports NVIDIA GPU memory and available disk space, asks which predictor to use, and writes the ignored file `config/structure_backend.local.json`. SwitchTune and its MCP server read that file automatically. Nothing is downloaded by this step. Use `--inspect-only` to inspect without changing the selection.

| Predictor | Deployment and resource considerations | Choose when |
| --- | --- | --- |
| **Boltz-2** | PyTorch CLI; model/cache storage is much smaller than a full AF3 search installation. Memory use depends on token count, ligands and sampling settings. Small proteins may fit on modest GPUs with serial, low-sampling settings; an 8 GB GPU is not sufficient for every target. Keep free space for model downloads, temporary files, full PAE/PDE and predictions. | You want the smaller local deployment, subject to testing the actual protein on your GPU. |
| **AlphaFold3** | Upstream validates Linux GPU inference on A100/H100 80 GB and recommends at least 64 GB host RAM. Full search databases require about 252 GB compressed / 630 GB extracted; upstream advises allowing up to 1 TB for the full installation. Weights require official access. Smaller targets and other supported hardware can have different requirements; check upstream performance guidance. | You have the necessary hardware/storage and weights, or already have an AF3 installation. |
| **AlphaFold3 with prepared input** | `provided` mode skips the search databases when every chain has explicit MSAs/templates. `single_sequence` also omits MSA search but provides weaker evolutionary evidence. Both still require AF3 weights, its runtime and enough GPU memory. | MSAs/templates have been prepared elsewhere, or single-sequence prediction is an intentional choice. |

These are planning guidelines, not enforced universal minimums. Check `nvidia-smi`, `free -h` and `df -h /data` on the server. Consult the [official AF3 installation](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md), [AF3 performance guidance](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md) and [Boltz repository](https://github.com/jwohlwend/boltz) for the installed release.

Noninteractive examples:

```bash
# Smaller local predictor
python3 scripts/configure_structure_backend.py --engine boltz2_local --storage-root /data

# AF3 with Docker and full database search
python3 scripts/configure_structure_backend.py --engine alphafold3 --storage-root /data \
  --model-dir /data/af3/models --db-dir /data/af3/databases
```

Then install the base environment, shared tools and **only the selected structure backend** below. Shared ESM-IF1/BioEmu/MPNN/Rosetta storage is additional to the predictor estimates.

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

Follow this section if you selected Boltz-2. Install it in a CUDA-enabled environment:

```bash
conda create -n boltz2 python=3.11 -y
conda activate boltz2
python -m pip install --upgrade pip
python -m pip install "boltz[cuda]"
boltz --help
```

The saved selection sets `boltz2_local`. Set `BOLTZ_CACHE` and point `BOLTZ_LOCAL_EXECUTABLE` at the installed executable, so it can be called from the agent environment:

```bash
export BOLTZ_LOCAL_EXECUTABLE="$(command -v boltz)"
export BOLTZ_CACHE=/data/boltz2_cache
conda activate switchtune
```

Persist that absolute executable path in your shell configuration or exported `.env` settings. SwitchTune runs candidates serially by default, writes native YAML with the target-state template, and requires full PAE. PDE is also requested. It checks usable structure/PAE artifacts before downstream scoring.

A hosted Boltz-2-compatible backend can be used by setting `SWITCHTUNE_STRUCTURE_ENGINE=boltz2_hosted`, `BOLTZ2_ENDPOINT` and `BOLTZ2_API_KEY`. Hosted backends are optional; their endpoint-specific request schema must match `scripts/run_boltz2_nim_batch.py` or a user-supplied adapter.

## 5a. AlphaFold3

Follow this section if you selected AlphaFold3. Use the **official local AlphaFold3 implementation**, rather than AlphaFold Server JSON. The default adapter runs the official Docker image and keeps its JAX dependencies separate from SwitchTune's agent and PyTorch tools.

Install Docker and NVIDIA Container Toolkit using the [upstream instructions](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md), and verify GPU access in a container. Obtain model parameters directly through the official AF3 access process and place them in `/data/af3/models`. AF3 source and model parameters have their own licenses/terms; neither is redistributed here.

```bash
# From the SwitchTune repository root; use your selected data volume.
mkdir -p /data/af3/models /data/af3/databases
git clone https://github.com/google-deepmind/alphafold3.git external/alphafold3
docker build -t alphafold3 -f external/alphafold3/docker/Dockerfile external/alphafold3

# Required only for msa_mode=search. This downloads hundreds of GB.
bash external/alphafold3/fetch_databases.sh /data/af3/databases

docker run --rm --gpus all alphafold3 python run_alphafold.py --help
```

The data directory must be outside the AF3 checkout so database files do not enter the Docker build context. Record the AF3 Git revision and Docker image digest used in a study. SwitchTune's AF3 default is **seed 1, five diffusion samples**, configurable with `AF3_SEEDS` and `AF3_NUM_SAMPLES`.

For an existing native AF3 installation:

```bash
python scripts/configure_structure_backend.py --engine alphafold3 --deployment native \
  --repo /opt/alphafold3 --python /opt/af3-env/bin/python \
  --model-dir /data/af3/models --db-dir /data/af3/databases --storage-root /data
```

SwitchTune's generated batch runner uses that interpreter; AF3 is not installed into the agent environment. Follow upstream native installation instructions for JAX/CUDA and AF3 compilation.

For prepared MSA inputs without the full local search databases:

```bash
python scripts/configure_structure_backend.py --engine alphafold3 --msa-mode provided \
  --model-dir /data/af3/models --storage-root /data
export AF3_INPUT_TEMPLATE=/data/prepared_wild_type_af3.json
```

Supply official `alphafold3`-dialect JSON with an individual protein chain `A`, explicit `unpairedMsa`, `pairedMsa` (possibly empty) and `templates` for each protein; RNA chains also need explicit `unpairedMsa`. The adapter substitutes the candidate sequence, updates MSA query rows, and inserts the aligned target-state template for the designed chain. Other entities, covalent bonds and user CCD definitions are preserved. Relative MSA/mmCIF file paths are embedded before container execution. See [configuration](configuration.md#structure-prediction-backend) for ligand and complex inputs.

Use `--msa-mode single_sequence` only when intentionally omitting MSA evidence. It sends explicit empty MSAs; it is not a substitute for full-MSA prediction. Neither model selection nor identical sample settings guarantees replication of the iLID study: match the actual input structures, MSAs, model version and study-specific screening settings.

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
python tests/test_structure_engine_selection_and_boltz_job.py
python -m pip install pytest
python -m pytest tests/test_alphafold3_backend.py -q
```

Then validate the external command paths:

```bash
python -c "import shutil; print('boltz:', shutil.which('boltz'))"
python external/esm/examples/inverse_folding/score_log_likelihoods.py --help
```

A missing optional backend should produce a recorded pending/blocked stage with an actionable path, not a fabricated result.

Finally run **real predictor inference** on a small control before a full agent design:

```bash
conda activate switchtune
python scripts/smoke_structure_backend.py --structure /data/target_state.pdb --chain A --execute
```

This uses the saved predictor, prepares one WT control, runs inference, validates its structures/PAE and creates a standardized PDB. Inspect `smoke_report.json` and the prediction logs. Without `--execute` it prepares inputs only. The adapter tests use synthetic outputs and do not establish that model inference works on your hardware.
