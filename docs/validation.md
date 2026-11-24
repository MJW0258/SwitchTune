<!-- Validation coverage and deployment checks. -->

# Validation Summary

SwitchTune has been developed and tested incrementally rather than validated as a single opaque command.

## Validated capabilities

- Biomni-to-MCP tool registration and dictionary-compatible tool results.
- Input validation and saturation single-mutant library generation.
- Two-state PDB inspection and target/reference state assignment.
- Conformation-bias scoring job preparation and candidate selection.
- Target-state template input for Boltz-2 NIM.
- Full PAE request and strict handling of missing PAE responses.
- Structure standardization, target-state similarity comparison, and confidence filtering.
- ESM-IF1 and ProteinMPNN/LigandMPNN inverse-folding scoring integration.
- BioEmu sampling job preparation and per-variant output tracking.
- Rosetta energy-job preparation and score-file collection.
- Evidence merging, checkpoint/resume behavior, and agentic review records.

## Selectable predictor integration tests

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_alphafold3_backend.py -q
python tests/test_structure_engine_selection_and_boltz_job.py
python tests/test_adaptive_cb_selection_to_structure_prediction.py
python tests/test_mcp_tools.py
```

The AF3 tests generate synthetic official-layout mmCIF/JSON fixtures. They check deployment selection, target-state template mappings (including gaps and non-default input chain/numbering), ligand inputs, prepared MSAs/covalent bonds, matched best-sample selection, PAE token indexing, incomplete/malformed outputs, resumable execution and the handoff through structure standardization, state comparison, confidence filtering and inverse-folding job preparation. They do not load AF3 weights or test biological prediction accuracy.

For actual inference on the deployment server, use a small target:

```bash
python scripts/smoke_structure_backend.py --structure /data/target_state.pdb --chain A --execute
```

A passing `smoke_report.json` requires one validated prediction and one standardized PDB. AF3 checks every requested seed/sample; a directory containing only a root-ranked model is insufficient. Keep the report, CLI logs, native files, model/software versions and selected settings with the run.

## Development benchmark system

Development runs used compact two-state proteins with experimentally characterized alternative structures. The public workflow does not depend on a case-specific protein, structure pair or predefined mutation interval.

Each run supplies:

- a target-state structure;
- a reference structure representing the alternate state;
- an agent-selected mutation region justified by the two-state comparison;
- target-state template constraints during structure prediction;
- multiple downstream compatibility and stability checks.

## Reproducibility requirements

Each released benchmark should record:

1. input sequence and residue numbering;
2. target/reference structure identifiers and chain mapping;
3. candidate counts after every stage;
4. model and tool versions;
5. execution environment and hardware;
6. threshold and ranking policy;
7. failure, retry, and resume events;
8. final evidence tables and decision rationale.

The current repository intentionally does not include private run directories or unpublished benchmark data.
