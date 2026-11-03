# Workflow and checkpoints

A run is a sequence of auditable checkpoints. Every stage writes a manifest, candidate table and decision record under `$SWITCHTUNE_RUNS/<project_name>/`.

## Input and region selection

Provide the protein name, target-state structure, reference-state structure, state labels and a natural-language `prompt` describing the desired design outcome. The agent must inspect both wild-type structures before selecting a mutation region. The target and reference labels in the user request remain authoritative; ligand clues are validation evidence rather than an automatic state swap. The prompt is written to the invocation record and included in the generated agent request for auditability.

## Library and conformation bias

SwitchTune generates all permitted single substitutions within the selected region, excluding self substitutions and any user-declared fixed residues. ProteinMPNN scores ligand-free structures; LigandMPNN scores structures containing ligands or cofactors. The state bias is computed from the difference between standardized target and reference log-probability changes.

The agent evaluates the score distribution and selects a central fraction. The resulting table records both the selected candidates and the empirical score bounds. The fraction is adaptive to the available library, so a larger input pool can retain more candidates.

## Structure and quality control

Boltz-2 predicts each retained variant with the target-state structure as a template. The collection stage validates output structures, PAE/PDE availability and completion status. Standardized structures are compared with both supplied states. Confidence, local pLDDT, PAE and target-state similarity are combined in the structure gate.

## Inverse folding and ensemble evidence

ESM-IF1 and ProteinMPNN/LigandMPNN provide complementary sequence-structure compatibility evidence. Lower negative log-likelihood indicates better compatibility for the corresponding model. BioEmu samples ensembles for the candidates that pass the checkpoint review. Rosetta then evaluates the selected candidates with the configured physical-energy protocol.

## Evidence fusion and output

The workflow merges state-bias, structural, inverse-folding, ensemble and energy evidence. The agent chooses the final shortlist budget from the valid evidence pool. The intended range is 20-30 candidates; when fewer valid variants are available, the complete valid set is returned. No result is silently filled with an invented score or candidate.

## Resume behavior

Long-running jobs are prepared as manifests and shell commands. If a job is interrupted, inspect `project_manifest.json`, the stage summary and the generated job manifest. Re-run only the incomplete stage or use the checkpoint runner's resume options. Completed outputs are detected and reused where the evidence schema allows it.
