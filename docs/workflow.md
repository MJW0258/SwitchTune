<!-- Design stages, evidence records and checkpoints. -->

# Workflow and checkpoints

A run is a sequence of auditable checkpoints. Every stage writes a manifest, candidate table and decision record under `$SWITCHTUNE_RUNS/<project_name>/`.

## Input and region selection

Provide the protein name, target-state structure, reference-state structure, state labels and a natural-language `prompt` describing the desired design outcome. The agent must inspect both wild-type structures before selecting a mutation region. The target and reference labels in the user request remain authoritative; ligand clues are validation evidence rather than an automatic state swap. The prompt is written to the invocation record and included in the generated agent request for auditability.

## Library and conformation bias

SwitchTune generates all permitted single substitutions within the selected region, excluding self substitutions and any user-declared fixed residues. ProteinMPNN scores ligand-free structures; LigandMPNN scores structures containing ligands or cofactors. The state bias is computed from the difference between standardized target and reference log-probability changes.

The agent evaluates the score distribution and selects a central fraction. The resulting table records both the selected candidates and the empirical score bounds. The fraction is adaptive to the available library, so a larger input pool can retain more candidates.

## Structure and quality control

The deployment-selected predictor processes each retained variant with the target-state structure as a template. Boltz-2 receives native YAML; AlphaFold3 receives official-dialect JSON with explicit template mappings, MSAs and ligand entities as configured. A run records the actual engine in its manifest, and downstream collection uses that engine's output adapter.

For AF3, all configured seed/sample outputs must contain a sequence-matched mmCIF, full confidences JSON, summary JSON and valid PAE. Missing or malformed samples mark that variant incomplete. Among complete samples without severe clashes, the sample with the highest AF3 ranking score supplies a matched structure/metric pair. Raw ranking score, clash/disorder metrics, sample count, mean pLDDT across samples and its standard deviation remain in evidence tables. The normalized generic confidence is mean designed-chain residue pLDDT divided by 100; it is **not** the AF3 ranking score or numerically equivalent to Boltz confidence. The native files are retained.

The AF3 adapter extracts the designed-chain PAE submatrix using `token_chain_ids`, so ligand tokens cannot shift residue windows. Boltz PAE/PDE use its existing parser. AF3 has no PDE output and no synthetic PDE value is inserted. Standardized PDB structures are compared with both supplied states. Confidence, local pLDDT, PAE and target-state similarity are combined in the structure gate. Study-specific iLID thresholds are not imposed on every protein.

## Inverse folding and ensemble evidence

ESM-IF1 and ProteinMPNN/LigandMPNN provide complementary sequence-structure compatibility evidence. Lower negative log-likelihood indicates better compatibility for the corresponding model. BioEmu samples ensembles for the candidates that pass the checkpoint review. Rosetta then evaluates the selected candidates with the configured physical-energy protocol.

## Evidence fusion and output

The workflow merges state-bias, structural, inverse-folding, ensemble and energy evidence. The agent chooses the final shortlist budget from the valid evidence pool. The intended range is 20-30 candidates; when fewer valid variants are available, the complete valid set is returned. No result is silently filled with an invented score or candidate.

## Resume behavior

Long-running jobs are prepared as manifests and shell commands. If a job is interrupted, inspect `project_manifest.json`, the stage summary and the generated job manifest. Re-run only the incomplete stage or use the checkpoint runner's resume options. Completed outputs are detected and reused where the evidence schema allows it.

AF3 output directories include an input/settings fingerprint. Changed sequences, templates, seeds or sampling settings use a different directory. Rerunning `scripts/run_alphafold3_batch.py --manifest /path/to/structure_prediction_job_manifest.json` validates and skips complete variants, then retries incomplete ones. Use `--collect-only` to rebuild the summary without loading AF3. MCP collection independently rescans native outputs, so an interrupted summary write does not invalidate completed predictions. Failed variants stay visible and are excluded from downstream scoring; complete variants can continue.
