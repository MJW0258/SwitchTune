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

## Development benchmark system

CheY has been used as the primary two-state development system because it provides a compact receiver-domain model with experimentally characterized structural states and a well-defined switching region.

The development workflow uses:

- an activation-like target structure;
- a reference structure representing the alternate state;
- a mutation region near the conformational switching surface;
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
