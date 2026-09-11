# SwitchTune Architecture

## Overview

SwitchTune separates scientific reasoning from computational execution:

```text
Natural-language request
        |
        v
Biomni agent
        |
        v
MCP tool modules
        |
        +--> state inspection and project setup
        +--> mutation-library generation
        +--> conformation-bias scoring
        +--> structure prediction and confidence QC
        +--> inverse-folding compatibility
        +--> BioEmu sampling
        +--> Rosetta energy evaluation
        +--> evidence fusion and candidate review
        |
        v
Ranked candidates with evidence and decision trace
```

## Design principles

### State-aware design

The system represents a target state and a reference state explicitly. This avoids treating an allosteric protein as a single static structure and allows candidate mutations to be evaluated for both target-state preference and retained switchability.

### Modular execution

Each computational stage is exposed as a bounded tool or job preparation step. Long-running programs are normally represented by a manifest and an execution script rather than being run directly inside an LLM call.

### Evidence before recommendation

Candidate selection is staged. A mutation is not promoted solely because it has a favorable score in one model. Structural confidence, target-state similarity, inverse-folding compatibility, conformational sampling, and energy evidence can be combined before the final recommendation.

### Checkpointed workflow

Manifests, status tables, and decision logs make the workflow resumable. A completed stage is not repeated merely because the agent is restarted. A downstream stage receives the exact candidate table and evidence produced by the preceding checkpoint.

### Guarded autonomy

The agent can inspect results, choose the next valid stage, and recommend thresholds within defined boundaries. Expensive external jobs remain subject to explicit execution policies and can be resumed after interruption.

## Typical MCP module groups

| Module | Responsibility |
|---|---|
| Project operations | Project creation, prompt parsing, status, summaries, decision logs |
| State inspection and bootstrap | PDB inspection, input validation, library creation, initial job setup |
| Conformation-bias scoring | Bias scoring, score-distribution analysis, adaptive candidate selection |
| Structure prediction and QC | Engine selection, template-constrained prediction, pLDDT/PAE QC, state comparison |
| Inverse folding | ESM-IF1, ProteinMPNN, and LigandMPNN compatibility scoring |
| BioEmu sampling | Sampling job preparation and output collection |
| Rosetta scoring | Energy-job preparation and score collection |
| Evidence fusion and ranking | Cross-stage evidence merging and ranked recommendations |
| Agentic review | Stage review, candidate review, rationale recording, and checkpoint decisions |

## Deployment model

The intended deployment is local or on a research server where the required scientific tools, model weights, GPU resources, and API credentials are available. The agent layer can run separately from expensive jobs, but the execution environment must be able to access the relevant structures and tools.

## Public release boundary

The public repository should contain portable documentation, configuration templates, tests that do not expose private data, and source code after a security review. Machine-specific paths, credentials, generated runs, private logs, and unlicensed structural data should remain outside version control.
