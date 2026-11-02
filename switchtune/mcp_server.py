from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

_MCP_IMPORT_ERROR = None
try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    _MCP_IMPORT_ERROR = exc

    class FastMCP:  # type: ignore[no-redef]
        """Small fallback that allows direct function tests without MCP installed."""

        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(func):
                return func

            return decorator

        def run(self):
            raise RuntimeError(
                "The MCP package is required to run the server. "
                "Install with: pip install -r requirements-mcp.txt"
            ) from _MCP_IMPORT_ERROR

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = set(DEFAULT_AA)
BOLTZ2_NIM_DEFAULT_ENDPOINT = ""
WATER_HETERO_CODES = {"HOH", "WAT", "DOD"}
STANDARD_PDB_PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
PDB_RESNAME_TO_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AUTO_TEMPLATE_CHAIN_IDS = {"", "auto", "best", "none", "omit", "null", "*"}
EMPTY_CANDIDATE_TOKENS = {"NONE", "EMPTY", "NO_CANDIDATES", "REJECT_ALL", "REJECT-ALL"}


def adaptive_candidate_budget(input_count: int, *, minimum: int = 20, maximum: int = 30) -> int:
    """Choose a 20--30 candidate budget from the available evidence.

    The budget grows with the size of the valid input pool and is capped to
    keep downstream sampling experimentally manageable.  When fewer than
    ``minimum`` valid candidates exist, all available candidates are retained.
    This function is intentionally deterministic so the selected budget is
    recorded and reproducible in the project manifest.
    """
    count = max(0, int(input_count))
    if count == 0:
        return 0
    if count <= minimum:
        return count
    # A square-root schedule increases the shortlist for larger libraries
    # without making the budget proportional to a single fixed library size.
    proposed = int(math.ceil(2.0 * math.sqrt(count)))
    return max(minimum, min(maximum, proposed))

mcp = FastMCP("SwitchTune")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "switchtune_project").strip())
    return cleaned.strip("_") or "switchtune_project"


def _normalize_selected_variant_input(value: Any) -> List[str]:
    """Parse agent shortlist output, including JSON list and explicit [] forms."""
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = str(value or "").strip()
        if text in {"[]", "[ ]", "[\n]"}:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            raw_values = parsed
        else:
            cleaned = text.strip("[]")
            cleaned = cleaned.replace('"', " ").replace("'", " ")
            raw_values = re.split(r"[,;\s]+", cleaned) if cleaned else []

    normalized: List[str] = []
    for item in raw_values:
        token = str(item or "").strip().strip("[]").strip("\"'")
        if not token:
            continue
        if token.upper() in EMPTY_CANDIDATE_TOKENS:
            continue
        normalized.append(token)
    return normalized


def _runs_root() -> Path:
    return Path(os.environ.get("SWITCHTUNE_RUNS", "./SwitchTune_runs")).expanduser().resolve()


def _project_dir(project_name: str) -> Path:
    path = _runs_root() / _sanitize_name(project_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "jobs").mkdir(exist_ok=True)
    return path


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or yaml is None:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _load_default_config() -> Dict[str, Any]:
    return _load_yaml(os.environ.get("SWITCHTUNE_CONFIG"))


def _write_json(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _append_jsonl(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=True) + "\n")
    return str(path)


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except Exception:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _csv_data_row_count(path_text: Optional[str]) -> Optional[int]:
    """Return the number of data rows in a CSV, or None when it is unavailable."""
    if not path_text:
        return None
    path = Path(str(path_text)).expanduser()
    if not path.exists() or not path.is_file():
        return None
    try:
        rows, _fields = _read_csv_rows(path)
    except Exception:
        return None
    return len(rows)


def _first_nonempty_csv_path(*path_texts: Optional[str]) -> str:
    """Return the first existing CSV that contains at least one data row."""
    first_existing = ""
    for path_text in path_texts:
        if not path_text:
            continue
        if not first_existing:
            first_existing = str(path_text)
        count = _csv_data_row_count(str(path_text))
        if count is not None and count > 0:
            return str(path_text)
    return first_existing


def _is_true_marker(value: Any) -> bool:
    """Interpret common CSV/JSON truth markers consistently."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "pass", "passed", "completed"}


def _is_structure_prediction_row_valid(
    row: Dict[str, Any],
    *,
    required_pae_override: Optional[bool] = None,
) -> bool:
    """Return whether a structure row is safe to pass to downstream tools."""
    status = str(row.get("prediction_status") or row.get("status") or "").strip().lower()
    if status.startswith("failed") or status in {
        "completed_no_structure_detected",
        "blocked_no_candidates",
    }:
        return False

    structure_path = str(row.get("structure_path") or "").strip()
    if not structure_path or not Path(structure_path).expanduser().exists():
        return False

    complete_value = row.get("structure_prediction_complete")
    if complete_value not in {"", None} and not _is_true_marker(complete_value):
        return False

    required_pae = (
        bool(required_pae_override)
        if required_pae_override is not None
        else _is_true_marker(
            row.get("required_pae")
            or row.get("structure_prediction_required_pae")
        )
    )
    if required_pae:
        pae_usable = row.get("pae_matrix_usable")
        if pae_usable in {"", None}:
            pae_usable = row.get("structure_prediction_pae_matrix_usable")
        if not _is_true_marker(pae_usable):
            return False
    return True


def _required_pae_from_structure_job_manifest(job_manifest: Dict[str, Any]) -> bool:
    """Return the strict PAE setting recorded for a structure job.

    Older toy fixtures may not contain a Boltz-2 configuration block. Those
    fixtures remain permissive, while every current Boltz-2 job writes
    ``require_pae=true`` into its manifest.
    """
    boltz = job_manifest.get("boltz2")
    if not isinstance(boltz, dict):
        return False
    if "require_pae" in boltz:
        # The production SwitchTune workflow never permits a Boltz-2 job to
        # disable the PAE gate. Keep the field visible for auditability, but
        # enforce the hard requirement at collection time as well.
        return True
    # Runs created just before the explicit require_pae manifest field was
    # added still recorded write_full_pae=true. Treat those runs strictly so
    # a stale response with pae=null cannot enter downstream scoring.
    return _is_true_marker(boltz.get("write_full_pae"))


def _inspect_pdb_file_basic(path: str) -> Dict[str, Any]:
    """Inspect a PDB file using only fixed-column records.

    This intentionally avoids BioPython so the MCP server can provide stable,
    deterministic structural-context features to the LLM agent.
    """
    pdb_path = Path(path).expanduser()
    if not pdb_path.exists():
        return {"path": str(pdb_path), "exists": False, "errors": [f"PDB file not found: {path}"]}

    atom_residues: set[Tuple[str, str, str]] = set()
    het_residues: List[Tuple[str, str, str]] = []
    het_counts: Counter[str] = Counter()
    het_by_chain: Dict[str, Counter[str]] = {}
    atom_chains: set[str] = set()
    het_chains: set[str] = set()
    protein_residues_by_chain: Dict[str, Dict[Tuple[int, str], str]] = {}
    model_count = 0
    seen_model = 0
    in_first_model = True
    selected_model: Optional[str] = None

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_count += 1
                seen_model += 1
                if seen_model == 1:
                    in_first_model = True
                    selected_model = line[10:14].strip() or "1"
                else:
                    in_first_model = False
                continue
            if record == "ENDMDL":
                if seen_model == 1:
                    in_first_model = False
                continue
            if model_count and not in_first_model:
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            chain = (line[21:22].strip() or "_")
            resname = line[17:20].strip()
            resseq = line[22:27].strip()
            if record == "ATOM":
                atom_chains.add(chain)
                atom_residues.add((chain, resname, resseq))
                if resname in PDB_RESNAME_TO_AA:
                    resseq_text = line[22:26].strip()
                    icode = line[26:27].strip()
                    try:
                        resseq_number = int(resseq_text)
                    except ValueError:
                        continue
                    protein_residues_by_chain.setdefault(chain, {}).setdefault((resseq_number, icode), resname)
            else:
                het_chains.add(chain)
                het_residues.append((chain, resname, resseq))
                het_counts[resname] += 1
                het_by_chain.setdefault(chain, Counter())[resname] += 1

    residue_counts_by_chain: Dict[str, int] = {}
    for chain, _resname, _resseq in atom_residues:
        residue_counts_by_chain[chain] = residue_counts_by_chain.get(chain, 0) + 1

    nonwater_het_counts = {k: v for k, v in sorted(het_counts.items()) if k not in {"HOH", "WAT"}}
    nonwater_het_by_chain = {
        chain: {k: v for k, v in sorted(counts.items()) if k not in {"HOH", "WAT"}}
        for chain, counts in sorted(het_by_chain.items())
    }
    nonwater_het_by_chain = {chain: counts for chain, counts in nonwater_het_by_chain.items() if counts}

    state_clues = []
    nonwater_resnames = set(nonwater_het_counts)
    if {"BEF", "MN"}.issubset(nonwater_resnames) or {"BEF", "MG"}.issubset(nonwater_resnames):
        state_clues.append("contains BEF plus a divalent metal ion, consistent with a phosphorylation/activation mimic")
    if "BEF" in nonwater_resnames:
        state_clues.append("contains BEF activation-mimic ligand")
    if "MN" in nonwater_resnames or "MG" in nonwater_resnames:
        state_clues.append("contains divalent metal ion")
    if "IPT" in nonwater_resnames:
        state_clues.append("contains IPTG inducer ligand (IPT)")
    if nonwater_resnames and nonwater_resnames <= {"SO4"}:
        state_clues.append("contains sulfate only among non-water hetero residues")

    sequence_by_chain: Dict[str, str] = {}
    first_residue_number_by_chain: Dict[str, int] = {}
    residue_number_span_by_chain: Dict[str, str] = {}
    for chain, residue_map in sorted(protein_residues_by_chain.items()):
        ordered = sorted(residue_map.items(), key=lambda item: (item[0][0], item[0][1]))
        if not ordered:
            continue
        sequence_by_chain[chain] = "".join(PDB_RESNAME_TO_AA[resname] for _key, resname in ordered)
        first_residue_number_by_chain[chain] = ordered[0][0][0]
        residue_number_span_by_chain[chain] = f"{ordered[0][0][0]}-{ordered[-1][0][0]}"

    return {
        "path": str(pdb_path),
        "exists": True,
        "model_count": model_count,
        "selected_model": selected_model,
        "model_selection_mode": "first_model_only" if model_count else "no_model_records",
        "chains": sorted(atom_chains),
        "hetero_chains": sorted(het_chains),
        "residue_counts_by_chain": dict(sorted(residue_counts_by_chain.items())),
        "sequence_by_chain": sequence_by_chain,
        "sequence_lengths_by_chain": {chain: len(seq) for chain, seq in sequence_by_chain.items()},
        "first_residue_number_by_chain": first_residue_number_by_chain,
        "residue_number_span_by_chain": residue_number_span_by_chain,
        "nonwater_hetero_counts": nonwater_het_counts,
        "nonwater_hetero_by_chain": nonwater_het_by_chain,
        "state_clues": state_clues,
        "warnings": [],
        "errors": [],
    }


def _load_numpy() -> Any:
    """Load NumPy lazily so inventory-only collection stays lightweight."""
    try:
        import numpy as np

        return np
    except Exception:
        return None


def _pdb_ca_coordinate_table(path: str, chain_id: str = "A") -> Dict[str, Any]:
    """Read one protein chain's CA coordinates without requiring BioPython."""
    pdb_path = Path(path).expanduser()
    result: Dict[str, Any] = {
        "path": str(pdb_path),
        "exists": pdb_path.exists(),
        "chain_id": str(chain_id or "A"),
        "sequence": "",
        "coordinates": [],
        "residue_numbers": [],
        "mapping_mode": "unavailable",
        "error": "",
    }
    if not pdb_path.exists():
        result["error"] = f"structure file not found: {pdb_path}"
        return result
    if pdb_path.suffix.lower() not in {".pdb", ".ent"}:
        result["error"] = "coordinate extraction currently supports PDB input; mmCIF coordinate parsing is deferred."
        return result

    requested_chain = str(chain_id or "").strip() or "A"
    residues: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    try:
        with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                record = raw_line[:6].strip()
                if record != "ATOM":
                    continue
                line = raw_line.rstrip("\r\n").ljust(80)
                chain = line[21:22].strip() or "_"
                if chain != requested_chain:
                    continue
                atom_name = line[12:16].strip().upper()
                if atom_name != "CA":
                    continue
                altloc = line[16:17].strip()
                if altloc not in {"", "A"}:
                    continue
                resname = line[17:20].strip().upper()
                if resname not in PDB_RESNAME_TO_AA:
                    continue
                try:
                    resseq = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (TypeError, ValueError):
                    continue
                icode = line[26:27].strip()
                key = (chain, resseq, icode)
                residues.setdefault(
                    key,
                    {
                        "resname": resname,
                        "coord": [x, y, z],
                    },
                )
    except OSError as exc:
        result["error"] = str(exc)
        return result

    ordered = sorted(residues.items(), key=lambda item: (item[0][1], item[0][2]))
    result["sequence"] = "".join(PDB_RESNAME_TO_AA[item[1]["resname"]] for item in ordered)
    result["coordinates"] = [item[1]["coord"] for item in ordered]
    result["residue_numbers"] = [item[0][1] for item in ordered]
    result["mapping_mode"] = "pdb_ca"
    if not ordered:
        result["error"] = f"no CA atoms found for chain {requested_chain!r}"
    return result


def _autonomous_structural_mutation_region(
    pdb_a: str,
    pdb_b: str,
    chain_id: str = "A",
    *,
    window_size: int = 11,
    max_windows: int = 3,
) -> Tuple[Optional[str], str]:
    """Select mutation windows from two-state C-alpha displacement.

    This is a deterministic fallback for an upstream LLM continuation failure.
    It uses only the supplied wild-type structures and never uses mutation
    databases or literature. Windows are reported in PDB residue numbering.
    """
    np = _load_numpy()
    if np is None:
        return None, "NumPy is unavailable; structural region fallback was skipped."
    table_a = _pdb_ca_coordinate_table(pdb_a, chain_id)
    table_b = _pdb_ca_coordinate_table(pdb_b, chain_id)
    numbers_a = list(table_a.get("residue_numbers") or [])
    numbers_b = list(table_b.get("residue_numbers") or [])
    common_numbers = sorted(set(numbers_a).intersection(numbers_b))
    if len(common_numbers) < max(2 * int(window_size), 20):
        return None, "Too few common C-alpha residues for an autonomous region fallback."

    index_a = {number: index for index, number in enumerate(numbers_a)}
    index_b = {number: index for index, number in enumerate(numbers_b)}
    mobile = np.asarray(
        [table_a["coordinates"][index_a[number]] for number in common_numbers],
        dtype=float,
    )
    reference = np.asarray(
        [table_b["coordinates"][index_b[number]] for number in common_numbers],
        dtype=float,
    )
    rmsd, aligned = _kabsch_transform(mobile, reference)
    if rmsd is None or aligned is None:
        return None, "C-alpha superposition failed; structural region fallback was skipped."
    displacement = np.linalg.norm(np.asarray(aligned) - reference, axis=1)
    width = min(int(window_size), max(5, len(common_numbers) // 4))
    candidates: List[Tuple[float, int, int]] = []
    for start in range(2, len(common_numbers) - width - 2):
        stop = start + width - 1
        score = float(np.mean(displacement[start : stop + 1]))
        candidates.append((score, start, stop))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    selected: List[Tuple[float, int, int]] = []
    for candidate in candidates:
        _score, start, stop = candidate
        if any(
            not (stop < old_start - 2 or start > old_stop + 2)
            for _s, old_start, old_stop in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(max_windows):
            break
    if not selected:
        return None, "No non-overlapping structural windows were available."

    selected.sort(key=lambda item: item[1])
    ranges = [
        f"{common_numbers[start]}-{common_numbers[stop]}"
        for _score, start, stop in selected
    ]
    rationale = (
        f"Selected {len(ranges)} non-overlapping {width}-residue windows on chain "
        f"{chain_id or 'A'} from the largest aligned C-alpha displacement between "
        f"the two supplied structures (common_residues={len(common_numbers)}, "
        f"global_ca_rmsd={float(rmsd):.3f} A)."
    )
    return ",".join(ranges), rationale


def _bioemu_npz_ensemble(variant_dir: Path) -> Dict[str, Any]:
    """Load BioEmu coordinate batches and return one residue-coordinate ensemble."""
    np = _load_numpy()
    result: Dict[str, Any] = {
        "available": False,
        "pos": None,
        "sequence": "",
        "npz_files": [],
        "sample_count": 0,
        "residue_count": 0,
        "nan_count": 0,
        "invalid_coordinate_count": 0,
        "invalid_sample_count": 0,
        "error": "",
    }
    if np is None:
        result["error"] = "NumPy is not installed; NPZ metrics were not computed."
        return result

    files = sorted(path for path in variant_dir.rglob("*.npz") if path.is_file())
    result["npz_files"] = [str(path) for path in files]
    if not files:
        result["error"] = "no NPZ coordinate batches found"
        return result

    chunks = []
    sequence = ""
    residue_count: Optional[int] = None
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                if "pos" not in data.files:
                    continue
                pos = np.asarray(data["pos"], dtype=float)
                if pos.ndim == 2 and pos.shape[-1] == 3:
                    pos = pos[None, ...]
                if pos.ndim != 3 or pos.shape[-1] != 3:
                    continue
                if residue_count is None:
                    residue_count = int(pos.shape[1])
                if int(pos.shape[1]) != residue_count:
                    result["error"] = "NPZ batches contain inconsistent residue counts"
                    return result
                chunks.append(pos)
                if not sequence and "sequence" in data.files:
                    value = data["sequence"]
                    try:
                        sequence = str(value.item() if getattr(value, "ndim", 0) == 0 else value)
                    except Exception:
                        sequence = str(value)
        except Exception as exc:
            result["error"] = f"failed to read {path.name}: {exc}"
            return result

    if not chunks:
        result["error"] = "NPZ files did not contain a usable pos array"
        return result
    pos = np.concatenate(chunks, axis=0)
    result.update(
        {
            "available": True,
            "pos": pos,
            "sequence": sequence.strip().upper(),
            "sample_count": int(pos.shape[0]),
            "residue_count": int(pos.shape[1]),
            "nan_count": int(np.isnan(pos).sum()),
            "invalid_coordinate_count": int((~np.isfinite(pos)).sum()),
            "invalid_sample_count": int((~np.isfinite(pos).all(axis=(1, 2))).sum()),
        }
    )
    return result


def _global_sequence_alignment_pairs(source: str, target: str) -> List[Tuple[int, int]]:
    """Return source/target index pairs using a small deterministic global aligner."""
    source = str(source or "")
    target = str(target or "")
    if not source or not target:
        return []
    if len(source) == len(target):
        return list(zip(range(len(source)), range(len(target))))

    match_score = 2
    mismatch_score = -1
    gap_score = -2
    rows = len(source) + 1
    cols = len(target) + 1
    scores = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        scores[i][0] = scores[i - 1][0] + gap_score
    for j in range(1, cols):
        scores[0][j] = scores[0][j - 1] + gap_score
    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = scores[i - 1][j - 1] + (
                match_score if source[i - 1] == target[j - 1] else mismatch_score
            )
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            scores[i][j] = max(diagonal, up, left)

    pairs: List[Tuple[int, int]] = []
    i, j = len(source), len(target)
    while i > 0 or j > 0:
        diagonal = (
            scores[i - 1][j - 1]
            + (match_score if source[i - 1] == target[j - 1] else mismatch_score)
            if i > 0 and j > 0
            else -10**9
        )
        up = scores[i - 1][j] + gap_score if i > 0 else -10**9
        left = scores[i][j - 1] + gap_score if j > 0 else -10**9
        best = max(diagonal, up, left)
        if i > 0 and j > 0 and best == diagonal:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and best == up:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _kabsch_transform(mobile: Any, reference: Any) -> Tuple[Optional[float], Any]:
    """Align mobile coordinates onto reference and return RMSD plus aligned coords."""
    np = _load_numpy()
    if np is None:
        return None, None
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.ndim != 2 or reference.ndim != 2 or mobile.shape != reference.shape:
        return None, None
    finite = np.isfinite(mobile).all(axis=1) & np.isfinite(reference).all(axis=1)
    if int(finite.sum()) == 0:
        return None, None
    mobile_valid = mobile[finite]
    reference_valid = reference[finite]
    mobile_center = mobile_valid.mean(axis=0)
    reference_center = reference_valid.mean(axis=0)
    mobile_centered = mobile_valid - mobile_center
    reference_centered = reference_valid - reference_center
    if int(finite.sum()) < 2:
        aligned = mobile - mobile_center + reference_center
        diff = aligned[finite] - reference_valid
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), aligned
    try:
        covariance = mobile_centered.T @ reference_centered
        u, _s, vt = np.linalg.svd(covariance)
        correction = np.eye(3)
        if np.linalg.det(u @ vt) < 0:
            correction[-1, -1] = -1.0
        rotation = u @ correction @ vt
        aligned = (mobile - mobile_center) @ rotation + reference_center
    except Exception:
        aligned = mobile - mobile_center + reference_center
    diff = aligned[finite] - reference_valid
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), aligned


def _coordinate_rmsd(mobile: Any, reference: Any, pairs: List[Tuple[int, int]]) -> Optional[float]:
    np = _load_numpy()
    if np is None or not pairs:
        return None
    mobile_idx = [pair[0] for pair in pairs]
    reference_idx = [pair[1] for pair in pairs]
    try:
        rmsd, _aligned = _kabsch_transform(
            np.asarray(mobile)[mobile_idx],
            np.asarray(reference)[reference_idx],
        )
        return rmsd
    except Exception:
        return None


def _bioemu_mutation_indices(
    mutation_region: Optional[str],
    *,
    first_residue_number: int,
    residue_count: int,
) -> List[int]:
    indices: List[int] = []
    if mutation_region:
        try:
            positions = [position for _wt, position in _parse_region(str(mutation_region))]
        except Exception:
            positions = []
        for position in positions:
            index = int(position) - int(first_residue_number)
            if 0 <= index < residue_count and index not in indices:
                indices.append(index)
    return indices


def _bioemu_log_metrics(log_path: Optional[str]) -> Dict[str, Any]:
    """Extract physicality summaries already emitted by BioEmu's converter."""
    if not log_path:
        return {}
    path = Path(str(log_path)).expanduser()
    if not path.exists() or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    output: Dict[str, Any] = {}
    names = (
        "ca_break_mean",
        "ca_clash_mean",
        "ca_ca_dist_mean",
        "clash_distances_mean",
    )
    for name in names:
        matches = re.findall(
            rf"(?:physicality/)?{re.escape(name)}\s*:\s*([-+0-9.eE]+)",
            text,
        )
        if matches:
            value = _safe_float(matches[-1])
            if value is not None:
                output[f"bioemu_physicality_{name}"] = value
    return output


def _bioemu_log_error_summary(log_path: Optional[str]) -> str:
    """Return a short, actionable error extracted from a BioEmu log."""
    if not log_path:
        return ""
    path = Path(str(log_path)).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except OSError:
        return ""
    if not lines:
        return ""

    patterns = (
        ("CUDA OOM", re.compile(r"(?:outofmemoryerror|cuda out of memory)", re.IGNORECASE)),
        ("CUDA error", re.compile(r"\bcuda error\b|\bcuda initialization\b", re.IGNORECASE)),
        ("network/download error", re.compile(
            r"timeout|connecttimeout|connection reset|huggingface|hf_hub_download",
            re.IGNORECASE,
        )),
        ("missing module", re.compile(r"modulenotfounderror|no module named", re.IGNORECASE)),
        ("invalid argument", re.compile(r"unrecognized arguments|usage:", re.IGNORECASE)),
        ("file/path error", re.compile(r"filenotfounderror|not a directory|permission denied", re.IGNORECASE)),
    )
    for label, pattern in patterns:
        matches = [line for line in lines if pattern.search(line)]
        if matches:
            return f"{label}: {matches[-1][:500]}"

    for line in reversed(lines):
        if re.search(r"error|exception|failed|killed", line, re.IGNORECASE):
            return line[:500]
    return ""


def _compute_bioemu_metrics(
    variant_dir: Path,
    *,
    target_state_structure: Optional[str],
    reference_state_structure: Optional[str],
    mutation_region: Optional[str],
    first_residue_number: int,
    chain_id: str,
    log_path: Optional[str],
) -> Dict[str, Any]:
    """Compute structure-ensemble metrics from BioEmu NPZ batches."""
    np = _load_numpy()
    metrics: Dict[str, Any] = {}
    metrics.update(_bioemu_log_metrics(log_path))
    metrics["bioemu_error_summary"] = _bioemu_log_error_summary(log_path)
    ensemble = _bioemu_npz_ensemble(variant_dir)
    metrics.update(
        {
            "bioemu_metrics_status": "unavailable",
            "bioemu_metrics_error": ensemble.get("error", ""),
            "bioemu_sample_count": ensemble.get("sample_count", 0),
            "bioemu_complete_sample_count": 0,
            "bioemu_complete_sample_fraction": 0.0,
            "bioemu_coordinate_residue_count": ensemble.get("residue_count", 0),
            "bioemu_coordinate_nan_count": ensemble.get("nan_count", 0),
            "bioemu_coordinate_invalid_count": ensemble.get("invalid_coordinate_count", 0),
            "bioemu_coordinate_invalid_sample_count": ensemble.get("invalid_sample_count", 0),
            "bioemu_coordinate_source": "pos" if ensemble.get("available") else "",
            "bioemu_coordinate_semantics": (
                "residue_node_coordinates_unverified" if ensemble.get("available") else ""
            ),
            "bioemu_npz_file_count": len(ensemble.get("npz_files") or []),
            "bioemu_sequence_length": len(str(ensemble.get("sequence") or "")),
            "bioemu_target_structure_available": False,
            "bioemu_reference_structure_available": False,
            "bioemu_target_coordinate_match_count": 0,
            "bioemu_reference_coordinate_match_count": 0,
            "bioemu_coordinate_mapping_mode": "unavailable",
            "bioemu_mutation_window_residue_count": 0,
        }
    )
    if np is None or not ensemble.get("available"):
        return metrics

    pos = np.asarray(ensemble["pos"], dtype=float)
    residue_count = int(pos.shape[1])
    complete_mask = np.isfinite(pos).all(axis=(1, 2))
    valid_pos = pos[complete_mask]
    metrics["bioemu_complete_sample_count"] = int(valid_pos.shape[0])
    metrics["bioemu_complete_sample_fraction"] = (
        float(valid_pos.shape[0] / pos.shape[0]) if pos.shape[0] else 0.0
    )
    if valid_pos.shape[0] == 0:
        metrics["bioemu_metrics_error"] = "all coordinate samples contain NaN or infinite values"
        return metrics

    sequence = str(ensemble.get("sequence") or "")
    target_table = _pdb_ca_coordinate_table(str(target_state_structure or ""), chain_id)
    reference_table = _pdb_ca_coordinate_table(str(reference_state_structure or ""), chain_id)
    target_coords = np.asarray(target_table.get("coordinates") or [], dtype=float)
    reference_coords = np.asarray(reference_table.get("coordinates") or [], dtype=float)
    target_sequence = str(target_table.get("sequence") or "")
    reference_sequence = str(reference_table.get("sequence") or "")
    metrics["bioemu_target_structure_available"] = bool(target_coords.size)
    metrics["bioemu_reference_structure_available"] = bool(reference_coords.size)
    metrics["bioemu_target_structure_path"] = str(target_state_structure or "")
    metrics["bioemu_reference_structure_path"] = str(reference_state_structure or "")

    target_pairs: List[Tuple[int, int]] = []
    reference_pairs: List[Tuple[int, int]] = []
    if target_coords.size:
        pairs = _global_sequence_alignment_pairs(sequence, target_sequence)
        target_pairs = [pair for pair in pairs if pair[0] < residue_count and pair[1] < len(target_coords)]
    if reference_coords.size:
        pairs = _global_sequence_alignment_pairs(sequence, reference_sequence)
        reference_pairs = [
            pair for pair in pairs if pair[0] < residue_count and pair[1] < len(reference_coords)
        ]
    metrics["bioemu_target_coordinate_match_count"] = len(target_pairs)
    metrics["bioemu_reference_coordinate_match_count"] = len(reference_pairs)
    metrics["bioemu_coordinate_mapping_mode"] = (
        "sequence_alignment" if sequence and (target_pairs or reference_pairs) else "index_fallback"
    )

    target_rmsds: List[float] = []
    reference_rmsds: List[float] = []
    for sample in valid_pos:
        target_rmsd = _coordinate_rmsd(sample, target_coords, target_pairs)
        reference_rmsd = _coordinate_rmsd(sample, reference_coords, reference_pairs)
        if target_rmsd is not None:
            target_rmsds.append(target_rmsd)
        if reference_rmsd is not None:
            reference_rmsds.append(reference_rmsd)
    if target_rmsds:
        metrics.update(
            {
                "bioemu_rmsd_to_target_mean_A": mean(target_rmsds),
                "bioemu_rmsd_to_target_median_A": median(target_rmsds),
                "bioemu_rmsd_to_target_sd_A": pstdev(target_rmsds) if len(target_rmsds) > 1 else 0.0,
                "bioemu_rmsd_to_target_min_A": min(target_rmsds),
                "bioemu_rmsd_to_target_max_A": max(target_rmsds),
            }
        )
    if reference_rmsds:
        metrics.update(
            {
                "bioemu_rmsd_to_reference_mean_A": mean(reference_rmsds),
                "bioemu_rmsd_to_reference_median_A": median(reference_rmsds),
                "bioemu_rmsd_to_reference_sd_A": pstdev(reference_rmsds) if len(reference_rmsds) > 1 else 0.0,
                "bioemu_rmsd_to_reference_min_A": min(reference_rmsds),
                "bioemu_rmsd_to_reference_max_A": max(reference_rmsds),
            }
        )
    if target_rmsds and reference_rmsds and len(target_rmsds) == len(reference_rmsds):
        margins = [reference - target for reference, target in zip(reference_rmsds, target_rmsds)]
        metrics["bioemu_target_like_fraction"] = sum(margin > 0 for margin in margins) / len(margins)
        metrics["bioemu_target_preference_margin_A"] = mean(margins)
        metrics["bioemu_target_like_sample_count"] = sum(margin > 0 for margin in margins)

    mutation_indices = _bioemu_mutation_indices(
        mutation_region,
        first_residue_number=first_residue_number,
        residue_count=residue_count,
    )
    metrics["bioemu_mutation_window_residue_count"] = len(mutation_indices)

    # Align each sample to the first valid sample before estimating ensemble spread.
    alignment_reference = valid_pos[0]
    aligned_samples = []
    for sample in valid_pos:
        _rmsd, aligned = _kabsch_transform(sample, alignment_reference)
        if aligned is not None:
            aligned_samples.append(aligned)
    if aligned_samples:
        aligned_array = np.asarray(aligned_samples, dtype=float)
        center = aligned_array.mean(axis=0)
        rmsf = np.sqrt(np.mean(np.sum((aligned_array - center) ** 2, axis=2), axis=0))
        metrics["bioemu_rmsf_global_mean_A"] = float(np.mean(rmsf))
        metrics["bioemu_rmsf_global_median_A"] = float(np.median(rmsf))
        metrics["bioemu_rmsf_global_max_A"] = float(np.max(rmsf))
        if mutation_indices:
            window_rmsf = rmsf[mutation_indices]
            metrics["bioemu_mutation_window_rmsf_mean_A"] = float(np.mean(window_rmsf))
            metrics["bioemu_mutation_window_rmsf_max_A"] = float(np.max(window_rmsf))

        sample_count_for_pairs = min(100, len(aligned_array))
        if sample_count_for_pairs > 1:
            selected_indices = [
                int(index)
                for index in np.linspace(0, len(aligned_array) - 1, sample_count_for_pairs)
            ]
            pairwise_values: List[float] = []
            for left_index in range(len(selected_indices)):
                for right_index in range(left_index + 1, len(selected_indices)):
                    value = _coordinate_rmsd(
                        aligned_array[selected_indices[left_index]],
                        aligned_array[selected_indices[right_index]],
                        list(zip(range(residue_count), range(residue_count))),
                    )
                    if value is not None:
                        pairwise_values.append(value)
            if pairwise_values:
                metrics["bioemu_pairwise_rmsd_mean_A"] = mean(pairwise_values)
                metrics["bioemu_pairwise_rmsd_median_A"] = median(pairwise_values)
                metrics["bioemu_pairwise_rmsd_sd_A"] = (
                    pstdev(pairwise_values) if len(pairwise_values) > 1 else 0.0
                )
                metrics["bioemu_pairwise_rmsd_pair_count"] = len(pairwise_values)

        radius_values: List[float] = []
        for sample in valid_pos:
            sample_center = sample.mean(axis=0)
            radius_values.append(float(np.sqrt(np.mean(np.sum((sample - sample_center) ** 2, axis=1)))))
        if radius_values:
            metrics["bioemu_radius_gyration_mean_A"] = mean(radius_values)
            metrics["bioemu_radius_gyration_sd_A"] = (
                pstdev(radius_values) if len(radius_values) > 1 else 0.0
            )

    metrics["bioemu_metrics_status"] = "completed"
    metrics["bioemu_metrics_error"] = ""
    return metrics


BIOEMU_COMPUTED_METRIC_FIELDS = [
    "bioemu_metrics_status",
    "bioemu_metrics_error",
    "bioemu_error_summary",
    "bioemu_sample_count",
    "bioemu_complete_sample_count",
    "bioemu_complete_sample_fraction",
    "bioemu_coordinate_residue_count",
    "bioemu_coordinate_nan_count",
    "bioemu_coordinate_invalid_count",
    "bioemu_coordinate_invalid_sample_count",
    "bioemu_coordinate_source",
    "bioemu_coordinate_semantics",
    "bioemu_npz_file_count",
    "bioemu_sequence_length",
    "bioemu_target_structure_available",
    "bioemu_reference_structure_available",
    "bioemu_target_structure_path",
    "bioemu_reference_structure_path",
    "bioemu_target_coordinate_match_count",
    "bioemu_reference_coordinate_match_count",
    "bioemu_coordinate_mapping_mode",
    "bioemu_rmsd_to_target_mean_A",
    "bioemu_rmsd_to_target_median_A",
    "bioemu_rmsd_to_target_sd_A",
    "bioemu_rmsd_to_target_min_A",
    "bioemu_rmsd_to_target_max_A",
    "bioemu_rmsd_to_reference_mean_A",
    "bioemu_rmsd_to_reference_median_A",
    "bioemu_rmsd_to_reference_sd_A",
    "bioemu_rmsd_to_reference_min_A",
    "bioemu_rmsd_to_reference_max_A",
    "bioemu_target_like_fraction",
    "bioemu_target_preference_margin_A",
    "bioemu_target_like_sample_count",
    "bioemu_pairwise_rmsd_mean_A",
    "bioemu_pairwise_rmsd_median_A",
    "bioemu_pairwise_rmsd_sd_A",
    "bioemu_pairwise_rmsd_pair_count",
    "bioemu_rmsf_global_mean_A",
    "bioemu_rmsf_global_median_A",
    "bioemu_rmsf_global_max_A",
    "bioemu_mutation_window_residue_count",
    "bioemu_mutation_window_rmsf_mean_A",
    "bioemu_mutation_window_rmsf_max_A",
    "bioemu_radius_gyration_mean_A",
    "bioemu_radius_gyration_sd_A",
    "bioemu_physicality_ca_break_mean",
    "bioemu_physicality_ca_clash_mean",
    "bioemu_physicality_ca_ca_dist_mean",
    "bioemu_physicality_clash_distances_mean",
]


def _empty_bioemu_metrics(error: str = "") -> Dict[str, Any]:
    """Return stable metric columns when an ensemble is unavailable."""
    values: Dict[str, Any] = {
        "bioemu_metrics_status": "unavailable",
        "bioemu_metrics_error": error,
        "bioemu_error_summary": "",
        "bioemu_complete_sample_count": 0,
        "bioemu_complete_sample_fraction": 0.0,
        "bioemu_coordinate_source": "",
        "bioemu_coordinate_semantics": "",
        "bioemu_coordinate_mapping_mode": "unavailable",
    }
    for field in BIOEMU_COMPUTED_METRIC_FIELDS:
        values.setdefault(field, "")
    return values


def _project_manifest_path(project_name: str) -> Path:
    return _project_dir(project_name) / "project_manifest.json"


def _load_project_manifest(project_name: str) -> Dict[str, Any]:
    return _read_json(_project_manifest_path(project_name))


def _save_project_manifest(project_name: str, manifest: Dict[str, Any]) -> str:
    manifest.setdefault("project_name", project_name)
    manifest["updated_at"] = _utc_now()
    return _write_json(_project_manifest_path(project_name), manifest)


def _update_project_manifest(project_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _load_project_manifest(project_name)
    manifest.update(updates)
    _save_project_manifest(project_name, manifest)
    return manifest


def _read_sequence(sequence: str) -> str:
    """Accept a raw sequence, a FASTA string, or a FASTA file path."""
    if sequence is None:
        return ""
    raw = str(sequence).strip()
    text = raw

    # A full amino-acid sequence is not a filesystem path. Avoid calling
    # Path.exists() on long raw sequences because Linux raises
    # ``OSError: [Errno 36] File name too long`` before the sequence validator
    # can process it.
    if raw and "\n" not in raw and "\r" not in raw and len(raw) < 4096:
        try:
            candidate = Path(raw).expanduser()
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8")
        except (OSError, ValueError):
            text = raw
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    return "".join(lines).upper().replace(" ", "")


def _parse_position_token(token: str) -> Tuple[Optional[str], int]:
    token = token.strip()
    match = re.fullmatch(r"([A-Za-z]?)(\d+)", token)
    if not match:
        raise ValueError(f"Cannot parse residue token: {token}")
    wt = match.group(1).upper() or None
    return wt, int(match.group(2))


def _parse_region(mutation_region: str) -> List[Tuple[Optional[str], int]]:
    raw_region = str(mutation_region or "").strip()
    if not raw_region:
        return []
    lowered = re.sub(r"\s+", " ", raw_region.lower()).strip()
    if any(
        marker in lowered
        for marker in (
            "agent-selected",
            "agent selected",
            "after inspection",
            "one or more exact",
            "to be selected",
            "not hard-coded",
            "infer",
            "choose",
            "select",
        )
    ):
        raise ValueError(
            "mutation_region is unresolved; provide concrete residue positions or ranges "
            "such as 220-230,240-250,236."
        )
    positions: List[Tuple[Optional[str], int]] = []
    for raw_token in re.split(r"[,;\s]+", raw_region):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start_wt, start_pos = _parse_position_token(start_text)
            _, end_pos = _parse_position_token(end_text)
            if end_pos < start_pos:
                raise ValueError(f"Region range is reversed: {token}")
            for pos in range(start_pos, end_pos + 1):
                positions.append((start_wt if pos == start_pos else None, pos))
        else:
            positions.append(_parse_position_token(token))
    seen = set()
    unique: List[Tuple[Optional[str], int]] = []
    for wt, pos in positions:
        if pos in seen:
            continue
        seen.add(pos)
        unique.append((wt, pos))
    return unique


def _position_to_index(position: int, first_residue_number: int) -> int:
    return position - int(first_residue_number)


def _validate_sequence(seq: str) -> List[str]:
    warnings = []
    if not seq:
        warnings.append("Sequence is empty.")
        return warnings
    invalid = sorted(set(seq) - VALID_AA)
    if invalid:
        warnings.append(f"Sequence contains non-canonical residues: {''.join(invalid)}")
    return warnings


def _job_record(project_name: str, tool_name: str, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = f"{tool_name}_{uuid.uuid4().hex[:12]}"
    record = {
        "job_id": job_id,
        "project_name": project_name,
        "tool_name": tool_name,
        "status": status,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "payload": payload,
    }
    project = _project_dir(project_name)
    _write_json(project / "jobs" / f"{job_id}.json", record)
    return record


@mcp.tool()
def create_project(
    project_name: str,
    protein_name: str,
    design_objective: str = "",
    target_state_name: str = "target_state",
    work_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or refresh the top-level SwitchTune project manifest."""
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    manifest.update(
        {
            "project_name": project_name,
            "protein_name": protein_name,
            "target_state_name": target_state_name,
            "design_objective": design_objective,
            "requested_work_dir": work_dir,
            "metadata": metadata or {},
            "status": manifest.get("status", "created"),
            "created_at": manifest.get("created_at", _utc_now()),
        }
    )
    manifest["stage"] = "project_created"
    manifest_path = _save_project_manifest(project_name, manifest)
    return {
        "status": "created",
        "project_name": project_name,
        "project_dir": str(project),
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


@mcp.tool()
def parse_design_prompt(prompt: str) -> Dict[str, Any]:
    """Heuristically extract design-relevant fields from a natural-language prompt."""
    parsed = _infer_prompt_fields(prompt)
    missing = [
        key
        for key in ["protein_name", "mutation_region"]
        if not parsed.get(key)
        and not (key == "mutation_region" and parsed.get("mutation_region_selection_required"))
    ]
    parsed["missing_required_fields"] = missing
    parsed["status"] = "parsed"
    return parsed


@mcp.tool()
def write_decision_log(
    project_name: str,
    step_name: str,
    decision: str,
    rationale: str,
    input_summary: Optional[Dict[str, Any]] = None,
    output_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a structured decision entry for the agent trace."""
    project = _project_dir(project_name)
    record = {
        "timestamp": _utc_now(),
        "project_name": project_name,
        "step_name": step_name,
        "decision": decision,
        "rationale": rationale,
        "input_summary": input_summary or {},
        "output_summary": output_summary or {},
    }
    log_path = _append_jsonl(project / "decision_log.jsonl", record)
    return {
        "status": "completed",
        "project_name": project_name,
        "decision_log_path": log_path,
        "record": record,
    }


@mcp.tool()
def summarize_library(
    project_name: str,
    library_csv: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize a saturation library by position and amino-acid mutation counts."""
    csv_path = Path(library_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"library_csv not found: {library_csv}")

    rows, fieldnames = _read_csv_rows(csv_path)
    if "position" not in fieldnames:
        raise ValueError("library_csv must contain a 'position' column.")

    position_counts = Counter()
    aa_counts = Counter()
    variants = []
    for row in rows:
        position_counts[row.get("position", "")] += 1
        aa_counts[row.get("mut_aa", "")] += 1
        if row.get("variant"):
            variants.append(row["variant"])

    outdir = Path(output_dir).expanduser().resolve() if output_dir else _project_dir(project_name) / "01_saturation_library"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(csv_path),
        "variant_count": len(rows),
        "unique_positions": sorted({row.get("position") for row in rows if row.get("position") is not None}),
        "position_counts": dict(position_counts),
        "mut_aa_counts": dict(aa_counts),
        "examples": variants[:10],
    }
    summary_path = _write_json(outdir / "library_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "library_summarized",
            "library_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def filter_library_by_constraints(
    project_name: str,
    library_csv: str,
    output_dir: Optional[str] = None,
    exclude_positions: Optional[str] = None,
    allowed_amino_acids: str = DEFAULT_AA,
    exclude_self_mutations: bool = True,
) -> Dict[str, Any]:
    """Filter a mutation library by residue positions or amino-acid constraints."""
    csv_path = Path(library_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"library_csv not found: {library_csv}")

    exclude_set = {int(x) for x in re.split(r"[,;\s]+", exclude_positions or "") if x.strip().isdigit()}
    allowed_set = {aa for aa in allowed_amino_acids.upper() if aa in VALID_AA}
    rows, fieldnames = _read_csv_rows(csv_path)

    filtered = []
    removed = 0
    for row in rows:
        try:
            position = int(row.get("position", ""))
        except Exception:
            position = None
        mut_aa = (row.get("mut_aa") or "").upper()
        wt_aa = (row.get("wt_aa") or "").upper()
        if position is not None and position in exclude_set:
            removed += 1
            continue
        if mut_aa and mut_aa not in allowed_set:
            removed += 1
            continue
        if exclude_self_mutations and wt_aa and mut_aa == wt_aa:
            removed += 1
            continue
        filtered.append(row)

    outdir = Path(output_dir).expanduser().resolve() if output_dir else _project_dir(project_name) / "01_saturation_library"
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / "filtered_saturation_single_mutant_library.csv"
    output_fields = fieldnames or (list(filtered[0].keys()) if filtered else [])
    _write_csv_rows(output_csv, filtered, output_fields)
    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
        "input_count": len(rows),
        "retained_count": len(filtered),
        "removed_count": removed,
        "exclude_positions": sorted(exclude_set),
        "allowed_amino_acids": "".join(sorted(allowed_set)),
    }
    summary_path = _write_json(outdir / "filtered_library_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "library_filtered",
            "filtered_library_csv": str(output_csv),
            "filtered_library_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def prepare_bias_scoring_inputs(
    project_name: str,
    library_csv: str,
    target_state_structure: str,
    target_state_name: str = "target_state",
    first_residue_number: int = 1,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a manifest pairing each library sequence with the target-state structure."""
    csv_path = Path(library_csv).expanduser()
    structure_path = Path(target_state_structure).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"library_csv not found: {library_csv}")
    if not structure_path.exists():
        raise FileNotFoundError(f"target_state_structure not found: {target_state_structure}")

    rows, fieldnames = _read_csv_rows(csv_path)
    if "sequence" not in fieldnames:
        raise ValueError("library_csv must contain a 'sequence' column.")

    outdir = Path(output_dir).expanduser().resolve() if output_dir else _project_dir(project_name) / "02_conformation_bias_scoring"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_csv = outdir / "bias_scoring_manifest.csv"
    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {
                "project_name": project_name,
                "variant": row.get("variant", ""),
                "sequence": row.get("sequence", ""),
                "target_state_name": target_state_name,
                "target_state_structure": str(structure_path),
                "mutation_region": row.get("position", ""),
                "first_residue_number": first_residue_number,
            }
        )
    _write_csv_rows(
        manifest_csv,
        manifest_rows,
        [
            "project_name",
            "variant",
            "sequence",
            "target_state_name",
            "target_state_structure",
            "mutation_region",
            "first_residue_number",
        ],
    )
    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(csv_path),
        "target_state_structure": str(structure_path),
        "target_state_name": target_state_name,
        "output_csv": str(manifest_csv),
        "row_count": len(manifest_rows),
        "variant_count": len(manifest_rows),
    }
    summary_path = _write_json(outdir / "bias_scoring_manifest_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "bias_scoring_prepared",
            "bias_scoring_manifest_csv": str(manifest_csv),
            "bias_scoring_manifest_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def inspect_bias_score_distribution(
    project_name: str,
    scored_csv: str,
    score_column: str = "bias_score",
    output_dir: Optional[str] = None,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize the distribution of a bias-score table."""
    csv_path = Path(scored_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"scored_csv not found: {scored_csv}")
    rows, fieldnames = _read_csv_rows(csv_path)
    if score_column not in fieldnames:
        raise ValueError(f"Score column '{score_column}' not found in {scored_csv}.")

    scores = []
    for row in rows:
        try:
            scores.append(float(row[score_column]))
        except Exception:
            continue
    if not scores:
        raise ValueError("No numeric scores could be parsed from scored_csv.")

    retained = None
    if score_min is not None and score_max is not None:
        retained = sum(1 for x in scores if score_min <= x <= score_max)

    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(csv_path),
        "score_column": score_column,
        "distribution": _score_distribution_summary(scores),
        "threshold_window": {"score_min": score_min, "score_max": score_max},
        "retained_within_window": retained,
        "retained_fraction_within_window": (retained / len(scores)) if retained is not None else None,
    }
    outdir = Path(output_dir).expanduser().resolve() if output_dir else _project_dir(project_name) / "02_conformation_bias_scoring"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_json(outdir / "bias_score_distribution_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "bias_distribution_inspected",
            "bias_distribution_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def recommend_bias_threshold(
    project_name: str,
    scored_csv: str,
    score_column: str = "bias_score",
    target_retained_fraction: float = 0.5,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend a score window based on the empirical score distribution."""
    csv_path = Path(scored_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"scored_csv not found: {scored_csv}")
    rows, fieldnames = _read_csv_rows(csv_path)
    if score_column not in fieldnames:
        raise ValueError(f"Score column '{score_column}' not found in {scored_csv}.")

    scores = []
    for row in rows:
        try:
            scores.append(float(row[score_column]))
        except Exception:
            continue
    if not scores:
        raise ValueError("No numeric scores could be parsed from scored_csv.")

    scores_sorted = sorted(scores)
    retained_fraction = min(max(float(target_retained_fraction), 0.05), 0.95)
    tail = (1.0 - retained_fraction) / 2.0
    lower = _quantile(scores_sorted, tail)
    upper = _quantile(scores_sorted, 1.0 - tail)
    recommended = {
        "score_min": lower,
        "score_max": upper,
        "target_retained_fraction": retained_fraction,
        "empirical_retained_fraction": sum(1 for x in scores if lower <= x <= upper) / len(scores),
    }
    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(csv_path),
        "score_column": score_column,
        "recommendation": recommended,
        "distribution": _score_distribution_summary(scores),
        "rationale": "Central score window chosen from empirical quantiles to preserve moderate-bias candidates.",
    }
    outdir = Path(output_dir).expanduser().resolve() if output_dir else _project_dir(project_name) / "02_conformation_bias_scoring"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_json(outdir / "bias_threshold_recommendation.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "bias_threshold_recommended",
            "bias_threshold_recommendation_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def select_state_specific_candidates_adaptive(
    project_name: str,
    scored_library_csv: str,
    score_column: str = "bias_score",
    strategy: str = "auto",
    target_retained_fraction: Optional[float] = None,
    target_count: Optional[int] = None,
    min_candidates: int = 0,
    max_candidates: int = 0,
    center: str = "zero",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Adaptively retain moderate-bias SwitchTune candidates.

    This tool avoids a fixed universal threshold such as -1 <= score <= 1.
    Instead it chooses an empirical central window from the observed score
    distribution, optionally constrained by target retained fraction or target
    candidate count. This is intended for LLM/agent decision-making across
    proteins with different CB score ranges.

    Strategies:
    - `auto`: choose a central fraction from library size unless target_count or
      target_retained_fraction is supplied.
    - `central_fraction`: keep the empirical central fraction by quantiles.
    - `target_count`: keep the N variants closest to the chosen center.
    """
    input_path = Path(scored_library_csv).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"scored_library_csv not found: {scored_library_csv}")
    rows, fieldnames = _read_csv_rows(input_path)
    if score_column not in fieldnames:
        raise ValueError(f"Score column '{score_column}' not found in {scored_library_csv}.")

    parsed: List[Tuple[int, Dict[str, str], float]] = []
    rejected_missing_score = 0
    for idx, row in enumerate(rows):
        value = _safe_float(row.get(score_column))
        if value is None:
            rejected_missing_score += 1
            continue
        parsed.append((idx, row, value))
    if not parsed:
        raise ValueError("No numeric scores could be parsed from scored_library_csv.")

    scores = [value for _idx, _row, value in parsed]
    n = len(scores)
    scores_sorted = sorted(scores)
    strategy_clean = (strategy or "auto").strip().lower()
    if strategy_clean not in {"auto", "central_fraction", "target_count"}:
        raise ValueError("strategy must be one of: auto, central_fraction, target_count.")

    if center.strip().lower() == "median":
        center_value = median(scores)
    elif center.strip().lower() in {"mean", "average"}:
        center_value = mean(scores)
    elif center.strip().lower() == "zero":
        center_value = 0.0
    else:
        try:
            center_value = float(center)
        except Exception as exc:
            raise ValueError("center must be 'zero', 'median', 'mean', or a numeric value.") from exc

    auto_rationale = []
    if strategy_clean == "auto":
        if target_count is not None:
            strategy_clean = "target_count"
            auto_rationale.append("target_count was provided, so auto selected target_count strategy.")
        else:
            strategy_clean = "central_fraction"
            if target_retained_fraction is None:
                # Use a smooth, monotone library-size schedule. The retained
                # fraction decreases for larger libraries, while the retained
                # count does not drop when the input pool grows.
                target_retained_fraction = max(
                    0.15,
                    min(0.70, 7.0 / math.sqrt(max(1, n))),
                )
                auto_rationale.append(
                    f"auto selected target_retained_fraction={target_retained_fraction:.3f} "
                    f"from the monotone library-size schedule for n={n}."
                )

    if target_count is None and target_retained_fraction is not None:
        target_count = max(1, int(round(n * float(target_retained_fraction))))
    if target_count is None:
        target_count = max(1, int(round(n * 0.5)))

    lower_bound = max(1, int(min_candidates)) if int(min_candidates) > 0 else 1
    upper_bound = min(int(max_candidates), n) if int(max_candidates) > 0 else n
    target_count = max(lower_bound, min(upper_bound, int(target_count), n))
    retained: List[Tuple[int, Dict[str, str], float]]
    score_min: float
    score_max: float

    if strategy_clean == "central_fraction":
        retained_fraction = target_count / n
        tail = (1.0 - retained_fraction) / 2.0
        lower = _quantile(scores_sorted, tail)
        upper = _quantile(scores_sorted, 1.0 - tail)
        retained = [(idx, row, value) for idx, row, value in parsed if lower <= value <= upper]
        # Quantile ties can retain more than the target. Trim by distance to center
        # while preserving the central-window interpretation.
        if len(retained) > target_count:
            retained = sorted(retained, key=lambda item: (abs(item[2] - center_value), item[0]))[:target_count]
        elif len(retained) < target_count:
            # Quantile ties can leave a central window short of its requested
            # proportion. Backfill the closest remaining scores so the retained
            # count follows the adaptive library-size schedule monotonically.
            retained_keys = {item[0] for item in retained}
            backfill = [item for item in parsed if item[0] not in retained_keys]
            backfill = sorted(backfill, key=lambda item: (abs(item[2] - center_value), item[0]))
            retained.extend(backfill[: target_count - len(retained)])
        score_min = min(value for _idx, _row, value in retained)
        score_max = max(value for _idx, _row, value in retained)
    else:
        retained = sorted(parsed, key=lambda item: (abs(item[2] - center_value), item[0]))[:target_count]
        score_min = min(value for _idx, _row, value in retained)
        score_max = max(value for _idx, _row, value in retained)

    retained_sorted = sorted(retained, key=lambda item: (abs(item[2] - center_value), item[0]))
    output_rows: List[Dict[str, Any]] = []
    for rank, (_idx, row, value) in enumerate(retained_sorted, start=1):
        out = dict(row)
        out["adaptive_bias_rank"] = rank
        out["adaptive_abs_bias_from_center"] = abs(value - center_value)
        out["adaptive_center"] = center_value
        output_rows.append(out)

    outdir = Path(output_dir).expanduser().resolve() if output_dir else _project_dir(project_name) / "03_state_specific_candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / "state_specific_retained_candidates_adaptive.csv"
    output_fields = _csv_fieldnames_union(output_rows, fieldnames + ["adaptive_bias_rank", "adaptive_abs_bias_from_center", "adaptive_center"])
    _write_csv_rows(output_csv, output_rows, output_fields)

    fixed_minus1_plus1_count = sum(1 for value in scores if -1.0 <= value <= 1.0)
    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(input_path),
        "output_csv": str(output_csv),
        "selection_mode": "adaptive",
        "selection_policy": "empirical_central_window",
        "score_column": score_column,
        "strategy": strategy_clean,
        "center_mode": center,
        "center_value": center_value,
        "input_count": len(rows),
        "numeric_score_count": n,
        "rejected_missing_score": rejected_missing_score,
        "target_count": target_count,
        "retained_count": len(output_rows),
        "retained_fraction": len(output_rows) / n,
        "score_min": score_min,
        "score_max": score_max,
        "fixed_minus1_plus1_count": fixed_minus1_plus1_count,
        "fixed_minus1_plus1_fraction": fixed_minus1_plus1_count / n,
        "distribution": _score_distribution_summary(scores),
        "rationale": (
            "Adaptive SwitchTune threshold: retain central moderate-bias candidates "
            "from the empirical score distribution rather than using a universal fixed window."
        ),
        "auto_rationale": auto_rationale,
        "next_tool": "prepare_structure_prediction_job",
        "agent_instruction": (
            "Use output_csv as candidates_csv for prepare_structure_prediction_job. "
            "Do not repeat adaptive selection unless the retained count violates the downstream budget."
        ),
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "adaptive_selection_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "state_specific_candidates_selected_adaptive",
            "state_specific_candidates_adaptive_csv": str(output_csv),
            "state_specific_candidates_adaptive_summary_path": summary_path,
            "state_specific_selection_mode": "adaptive",
            "adaptive_bias_score_min": score_min,
            "adaptive_bias_score_max": score_max,
        },
    )
    return {**summary, "summary_path": summary_path}


def _infer_prompt_fields(prompt: str) -> Dict[str, Any]:
    prompt = prompt or ""
    fields: Dict[str, Any] = {
        "raw_prompt": prompt,
        "protein_name": None,
        "target_state_name": None,
        "target_state_structure": None,
        "mutation_region": None,
        "design_objective": None,
        "ligand_or_cofactor": None,
        "first_residue_number": 1,
        "notes": [],
    }

    region_selection_required = bool(
        re.search(
            r"`?mutation_region`?\s*[:=]\s*(?:agent[- ]selected|infer|choose|select)",
            prompt,
            flags=re.IGNORECASE,
        )
    )
    region_match = (
        None
        if region_selection_required
        else re.search(r"(\d+\s*-\s*\d+|\d+(?:\s*,\s*\d+)+)", prompt)
    )
    if region_match:
        fields["mutation_region"] = region_match.group(1).replace(" ", "")
    fields["mutation_region_selection_required"] = region_selection_required

    # Simple keyword heuristics for common dynamic proteins in this project.
    lower = prompt.lower()
    heuristics = {
        "iLID": ["ilid", "lov2", "ja helix"],
        "CaM": ["calmodulin", "cam", "ef-hand"],
        "TetRts": ["tetrts", "tet repressor", "temperature-sensitive"],
        "NtrC": ["ntrc", "response regulator"],
        "PcaV": ["pcav", "ligand-binding"],
    }
    for protein, keys in heuristics.items():
        if any(k in lower for k in keys):
            fields["protein_name"] = protein
            break

    if not fields["protein_name"]:
        direct_name_patterns = [
            r"\bfor\s+([A-Z][A-Za-z0-9_-]{2,})\b",
            r"\bprotein\s+([A-Z][A-Za-z0-9_-]{2,})\b",
            r"\bfor\s+the\s+([A-Z][A-Za-z0-9_-]{2,})\b",
        ]
        for pattern in direct_name_patterns:
            m = re.search(pattern, prompt)
            if m:
                token = m.group(1)
                if token.lower() not in {"design", "state", "switch", "target"}:
                    fields["protein_name"] = token
                    break

    if "dark state" in lower or "inactive state" in lower:
        fields["target_state_name"] = "dark_state" if "dark" in lower else "inactive_state"
    elif "active state" in lower or "light state" in lower:
        fields["target_state_name"] = "active_state" if "active" in lower else "light_state"

    if any(token in lower for token in ["ca2+", "calcium"]):
        fields["ligand_or_cofactor"] = "Ca2+"
    if any(token in lower for token in ["fmn", "flavin"]):
        fields["ligand_or_cofactor"] = "FMN"

    if fields["mutation_region"] and "preserve switchability" in lower:
        fields["design_objective"] = "State-specific tuning with switchability preserved."
    elif fields["mutation_region"]:
        fields["design_objective"] = "State-specific screening around the requested region."

    if not fields["protein_name"]:
        fields["notes"].append("protein_name not confidently inferred")
    if not fields["mutation_region"]:
        fields["notes"].append(
            "mutation_region will be selected by the Agent after structure inspection"
            if region_selection_required
            else "mutation_region not found"
        )

    fields["confidence"] = 0.5 + 0.15 * int(bool(fields["protein_name"])) + 0.15 * int(bool(fields["mutation_region"]))
    fields["confidence"] = min(fields["confidence"], 0.95)
    return fields


def _score_distribution_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    q25 = sorted_values[int(round(0.25 * (len(sorted_values) - 1)))]
    q75 = sorted_values[int(round(0.75 * (len(sorted_values) - 1)))]
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "q25": q25,
        "q75": q75,
    }


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty list.")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _z_scale(values: List[Optional[float]]) -> List[Optional[float]]:
    numeric = [v for v in values if v is not None and math.isfinite(v)]
    if not numeric:
        return [None for _ in values]
    mu = mean(numeric)
    sigma = pstdev(numeric) if len(numeric) > 1 else 0.0
    if sigma == 0:
        return [0.0 if v is not None and math.isfinite(v) else None for v in values]
    return [((v - mu) / sigma) if v is not None and math.isfinite(v) else None for v in values]


def _csv_fieldnames_union(rows: List[Dict[str, Any]], preferred: Optional[List[str]] = None) -> List[str]:
    fields: List[str] = []
    for name in preferred or []:
        if name not in fields:
            fields.append(name)
    for row in rows:
        for name in row.keys():
            if name not in fields:
                fields.append(name)
    return fields


def _load_tool_registry() -> Dict[str, Any]:
    return _load_yaml(os.environ.get("SWITCHTUNE_TOOL_REGISTRY"))


def _tool_config(tool_name: str) -> Dict[str, Any]:
    registry = _load_tool_registry()
    tools = registry.get("tools", {}) if isinstance(registry.get("tools", {}), dict) else {}
    config = tools.get(tool_name, {})
    return config if isinstance(config, dict) else {}


def _quote_bash(text: str) -> str:
    return "'" + str(text).replace("'", "'\"'\"'") + "'"


def _expand_path_text(path_text: str) -> str:
    text = str(path_text)

    def replace_default(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        return os.environ.get(name, default)

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]-(.*?)\}", replace_default, text)
    return os.path.expandvars(os.path.expanduser(text))


def _join_path_text(base: str, *parts: str) -> str:
    base_text = _expand_path_text(base).rstrip("/\\")
    separator = "/" if base_text.startswith("/") else os.sep
    return separator.join([base_text] + [str(part).strip("/\\") for part in parts])


def _normalize_mpnn_model_type(model_type: str) -> str:
    cleaned = (model_type or "protein_mpnn").strip().lower().replace("-", "_")
    aliases = {
        "protein": "protein_mpnn",
        "proteinmpnn": "protein_mpnn",
        "protein_mpnn": "protein_mpnn",
        "ligand": "ligand_mpnn",
        "ligandmpnn": "ligand_mpnn",
        "ligand_mpnn": "ligand_mpnn",
    }
    if cleaned not in aliases:
        raise ValueError("model_type must be 'protein_mpnn' or 'ligand_mpnn'.")
    return aliases[cleaned]


def _default_ligandmpnn_paths(model_type: str) -> Dict[str, str]:
    tool_name = "ligandmpnn" if model_type == "ligand_mpnn" else "proteinmpnn"
    config = _tool_config(tool_name)
    repo = str(config.get("repo") or "${LIGANDMPNN_REPO:-./external/LigandMPNN}")
    score_script = str(config.get("score_script") or _join_path_text(repo, "score.py"))
    model_params = str(config.get("model_params") or _join_path_text(repo, "model_params"))
    checkpoint_name = "ligandmpnn_v_32_010_25.pt" if model_type == "ligand_mpnn" else "proteinmpnn_v_48_020.pt"
    checkpoint_key = "checkpoint_ligand_mpnn" if model_type == "ligand_mpnn" else "checkpoint_protein_mpnn"
    checkpoint = str(config.get(checkpoint_key) or _join_path_text(model_params, checkpoint_name))
    conda_env = str(config.get("conda_env") or "ligandmpnn_env")
    return {
        "repo": repo,
        "score_script": score_script,
        "model_params": model_params,
        "checkpoint": checkpoint,
        "checkpoint_arg": f"--{checkpoint_key}",
        "conda_env": conda_env,
    }


def _select_first_pdb_model_for_scoring(text: str) -> Tuple[str, Dict[str, Any]]:
    """Return a deterministic first-model PDB copy for CB scoring.

    LigandMPNN/ProteinMPNN are run on one state conformation.  NMR PDB files
    contain multiple MODEL blocks, so retain only MODEL 1 while preserving its
    ATOM and HETATM records (including state-defining ions).
    """
    model_count = sum(1 for raw_line in text.splitlines() if raw_line[:6].strip() == "MODEL")
    if model_count == 0:
        return text, {
            "input_model_count": 0,
            "selected_model": None,
            "model_selection_mode": "no_model_records",
            "selected_atom_count": sum(1 for line in text.splitlines() if line[:6].strip() == "ATOM"),
            "selected_hetero_atom_count": sum(1 for line in text.splitlines() if line[:6].strip() == "HETATM"),
        }

    selected_model: Optional[str] = None
    seen_model = 0
    in_first_model = False
    kept_lines: List[str] = []
    for raw_line in text.splitlines():
        record = raw_line[:6].strip()
        if record == "MODEL":
            seen_model += 1
            if seen_model == 1:
                in_first_model = True
                selected_model = raw_line[10:14].strip() or "1"
            else:
                in_first_model = False
            continue
        if record == "ENDMDL":
            if seen_model == 1:
                in_first_model = False
            continue
        if in_first_model:
            kept_lines.append(raw_line.rstrip("\r\n"))

    atom_count = sum(1 for line in kept_lines if line[:6].strip() == "ATOM")
    hetero_count = sum(1 for line in kept_lines if line[:6].strip() == "HETATM")
    if atom_count == 0:
        return text, {
            "input_model_count": model_count,
            "selected_model": selected_model,
            "model_selection_mode": "first_model_only_failed_no_atoms",
            "selected_atom_count": 0,
            "selected_hetero_atom_count": hetero_count,
            "warnings": ["First PDB model contained no ATOM records; original scoring input was retained."],
        }
    return "\n".join(kept_lines).rstrip() + "\n", {
        "input_model_count": model_count,
        "selected_model": selected_model,
        "model_selection_mode": "first_model_only",
        "selected_atom_count": atom_count,
        "selected_hetero_atom_count": hetero_count,
    }


def _copy_state_pdbs(
    states: List[Tuple[str, str]],
    outdir: Path,
) -> List[Dict[str, Any]]:
    pdb_dir = outdir / "state_pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for state_name, pdb_path_text in states:
        src = Path(pdb_path_text).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"State structure not found: {pdb_path_text}")
        safe_name = _sanitize_name(state_name)
        dst = pdb_dir / f"{safe_name}.pdb"
        model_info: Dict[str, Any] = {
            "input_model_count": 0,
            "selected_model": None,
            "model_selection_mode": "not_applicable",
        }
        if src.suffix.lower() in {".pdb", ".ent"}:
            source_text = src.read_text(encoding="utf-8", errors="replace")
            scoring_text, model_info = _select_first_pdb_model_for_scoring(source_text)
            if src.resolve() == dst.resolve() and model_info.get("input_model_count", 0):
                dst = pdb_dir / f"{safe_name}_first_model.pdb"
            if src.resolve() != dst.resolve() or model_info.get("input_model_count", 0):
                dst.write_text(scoring_text, encoding="utf-8")
        elif src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied.append({"state_name": state_name, "input_pdb": str(src), "scoring_pdb": str(dst), **model_info})
    return copied


def _write_ligandmpnn_cb_script(
    *,
    outdir: Path,
    state_records: List[Dict[str, str]],
    model_type: str,
    parse_chains: str,
    chains_to_design: Optional[str],
    batch_size: int,
    number_of_batches: int,
    use_sequence: int,
    ligand_mpnn_use_atom_context: int,
    ligand_mpnn_use_side_chain_context: int,
    ligand_mpnn_cutoff_for_score: float,
    parse_atoms_with_zero_occupancy: int,
    seed: int,
    verbose: int,
    ligandmpnn_repo: Optional[str],
    score_script: Optional[str],
    checkpoint_path: Optional[str],
    conda_env: Optional[str],
) -> Dict[str, Any]:
    paths = _default_ligandmpnn_paths(model_type)
    repo = _expand_path_text(ligandmpnn_repo or paths["repo"])
    script = _expand_path_text(score_script or paths["score_script"])
    checkpoint = _expand_path_text(checkpoint_path or paths["checkpoint"])
    env_name = conda_env or paths["conda_env"]

    score_out = outdir / f"{model_type}_state_pt"
    score_out.mkdir(parents=True, exist_ok=True)
    pdb_multi = outdir / f"{model_type}_state_pdbs.json"
    pdb_payload = {record["scoring_pdb"]: "" for record in state_records}
    _write_json(pdb_multi, pdb_payload)

    expected_pt = {}
    for record in state_records:
        expected_pt[record["state_name"]] = str(score_out / f"{Path(record['scoring_pdb']).stem}.pt")

    cmd = [
        "python",
        script,
        "--model_type",
        model_type,
        paths["checkpoint_arg"],
        checkpoint,
        "--pdb_path_multi",
        str(pdb_multi),
        "--out_folder",
        str(score_out),
        "--use_sequence",
        str(int(use_sequence)),
        "--single_aa_score",
        "1",
        "--autoregressive_score",
        "0",
        "--batch_size",
        str(int(batch_size)),
        "--number_of_batches",
        str(int(number_of_batches)),
        "--seed",
        str(int(seed)),
        "--verbose",
        str(int(verbose)),
        "--ligand_mpnn_use_atom_context",
        str(int(ligand_mpnn_use_atom_context)),
        "--ligand_mpnn_use_side_chain_context",
        str(int(ligand_mpnn_use_side_chain_context)),
        "--ligand_mpnn_cutoff_for_score",
        str(float(ligand_mpnn_cutoff_for_score)),
        "--parse_atoms_with_zero_occupancy",
        str(int(parse_atoms_with_zero_occupancy)),
    ]
    if parse_chains:
        cmd.extend(["--parse_these_chains_only", parse_chains])
    if chains_to_design:
        cmd.extend(["--chains_to_design", chains_to_design])

    command_text = " ".join(_quote_bash(x) for x in cmd)
    log_path = outdir / f"{model_type}_score.log"
    run_script = outdir / f"run_{model_type}_cb_score.sh"
    run_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"LIGAND_MPNN_DIR={_quote_bash(repo)}",
                f"CONDA_ENV={_quote_bash(env_name)}",
                "",
                "if command -v conda >/dev/null 2>&1; then",
                "  CONDA_RUN=(conda run -n \"$CONDA_ENV\")",
                "else",
                "  CONDA_RUN=()",
                "fi",
                "",
                "cd \"$LIGAND_MPNN_DIR\"",
                f"${{CONDA_RUN[@]}} {command_text} 2>&1 | tee {_quote_bash(str(log_path))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        run_script.chmod(0o755)
    except Exception:
        pass

    return {
        "script_path": str(run_script),
        "log_path": str(log_path),
        "pdb_path_multi": str(pdb_multi),
        "score_output_dir": str(score_out),
        "expected_pt": expected_pt,
        "command": command_text,
        "repo": repo,
        "score_script": script,
        "checkpoint": checkpoint,
        "conda_env": env_name,
    }


def _natural_residue_sort_key(encoded: str) -> Tuple[str, int, str]:
    match = re.match(r"^([A-Za-z]*)(-?\d+)(.*)$", str(encoded))
    if not match:
        return (str(encoded), 0, "")
    return (match.group(1), int(match.group(2)), match.group(3))


def _load_ligandmpnn_pt_profile(pt_path: str) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyTorch is required to parse LigandMPNN .pt score files.") from exc

    path = Path(pt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"LigandMPNN .pt file not found: {pt_path}")

    # LigandMPNN score.py writes numpy-containing .pt score profiles.
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True), which rejects
    # these locally generated files unless the legacy pickle loader is enabled.
    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(str(path), map_location="cpu")
    mean_of_probs = data.get("mean_of_probs")
    if not isinstance(mean_of_probs, dict) or not mean_of_probs:
        raise ValueError(f"{pt_path} does not contain a usable mean_of_probs dictionary.")

    residue_names_raw = data.get("residue_names", {})
    residue_names: List[str] = []
    if isinstance(residue_names_raw, dict) and residue_names_raw:
        residue_names = [str(residue_names_raw[k]) for k in sorted(residue_names_raw, key=lambda x: int(x))]
    else:
        residue_names = sorted([str(k) for k in mean_of_probs.keys()], key=_natural_residue_sort_key)

    sequence = data.get("sequence", [])
    if sequence is not None:
        sequence = [str(x) for x in list(sequence)]
    else:
        sequence = []

    return {
        "path": str(path),
        "mean_of_probs": mean_of_probs,
        "residue_names": residue_names,
        "sequence": sequence,
        "length": len(residue_names),
    }


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _run_external_cb_extract(
    *,
    project_name: str,
    job_manifest_path: str,
    output_csv: Optional[str],
    score_min: Optional[float],
    score_max: Optional[float],
    score_mode: str,
    chain_id: Optional[str],
    position_column: str,
    sequence_index_column: str,
    wt_column: str,
    mut_column: str,
    sequence_column: str,
) -> Dict[str, Any]:
    manifest = _read_json(Path(job_manifest_path).expanduser())
    run_info = manifest.get("ligandmpnn_run", {}) if isinstance(manifest.get("ligandmpnn_run", {}), dict) else {}
    conda_env = str(run_info.get("conda_env") or _default_ligandmpnn_paths(str(manifest.get("model_type") or "protein_mpnn"))["conda_env"])
    project_root = Path(os.environ.get("SWITCHTUNE_HOME", Path(__file__).resolve().parents[1])).expanduser().resolve()
    script = project_root / "scripts" / "extract_cb_ligandmpnn_scores.py"
    if not script.exists():
        raise FileNotFoundError(f"External CB extraction script not found: {script}")

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(script),
        "--project_name",
        project_name,
        "--job_manifest_path",
        str(job_manifest_path),
        "--score_mode",
        score_mode,
        "--position_column",
        position_column,
        "--sequence_index_column",
        sequence_index_column,
        "--wt_column",
        wt_column,
        "--mut_column",
        mut_column,
        "--sequence_column",
        sequence_column,
    ]
    if output_csv:
        cmd.extend(["--output_csv", output_csv])
    if score_min is not None:
        cmd.extend(["--score_min", str(score_min)])
    if score_max is not None:
        cmd.extend(["--score_max", str(score_max)])
    if chain_id:
        cmd.extend(["--chain_id", chain_id])

    env = os.environ.copy()
    env["SWITCHTUNE_CB_ALLOW_EXTERNAL_EXTRACT"] = "0"
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            "External CB extraction failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )

    marker = "SWITCHTUNE_CB_EXTRACT_JSON:"
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(
        "External CB extraction did not return a JSON marker.\n"
        f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
    )


def _profile_log_prob(profile: Dict[str, Any], residue_key: str, aa: str, eps: float = 1e-12) -> Optional[float]:
    mean_probs = profile["mean_of_probs"]
    probs = mean_probs.get(residue_key)
    if not isinstance(probs, dict):
        return None
    value = _safe_float(probs.get(aa))
    if value is None:
        return None
    return math.log(max(value, eps))


def _resolve_profile_residue_key(
    profile: Dict[str, Any],
    row: Dict[str, Any],
    *,
    chain_id: Optional[str],
    first_residue_number: int,
    position_column: str,
    sequence_index_column: str,
) -> Tuple[Optional[str], str]:
    residues = list(profile["residue_names"])
    by_exact = {name: name for name in residues}
    by_position: Dict[int, List[str]] = {}
    for name in residues:
        match = re.match(r"^([A-Za-z]*)(-?\d+)", str(name))
        if match:
            if chain_id and match.group(1) and match.group(1) != chain_id:
                continue
            by_position.setdefault(int(match.group(2)), []).append(name)

    raw_pos = row.get(position_column)
    pos = None
    try:
        pos = int(str(raw_pos))
    except Exception:
        pos = None

    if pos is not None:
        if chain_id:
            exact = f"{chain_id}{pos}"
            if exact in by_exact:
                return exact, "pdb_number_exact"
        matches = by_position.get(pos, [])
        if len(matches) == 1:
            return matches[0], "pdb_number"

    raw_seq_index = row.get(sequence_index_column)
    seq_index = None
    try:
        seq_index = int(str(raw_seq_index))
    except Exception:
        seq_index = None

    if seq_index is None and pos is not None:
        seq_index = pos - int(first_residue_number) + 1

    if seq_index is not None and 1 <= seq_index <= len(residues):
        return residues[seq_index - 1], "sequence_index_1based"

    return None, "unresolved"


def _score_single_mutant_delta(
    profile: Dict[str, Any],
    row: Dict[str, Any],
    *,
    chain_id: Optional[str],
    first_residue_number: int,
    position_column: str,
    sequence_index_column: str,
    wt_column: str,
    mut_column: str,
) -> Dict[str, Any]:
    residue_key, mapping_mode = _resolve_profile_residue_key(
        profile,
        row,
        chain_id=chain_id,
        first_residue_number=first_residue_number,
        position_column=position_column,
        sequence_index_column=sequence_index_column,
    )
    wt_aa = str(row.get(wt_column, "")).upper()
    mut_aa = str(row.get(mut_column, "")).upper()
    result: Dict[str, Any] = {
        "residue_key": residue_key,
        "mapping_mode": mapping_mode,
        "wt_log_prob": None,
        "mut_log_prob": None,
        "delta_log_prob": None,
        "delta_nll": None,
        "score_status": "unresolved_residue" if residue_key is None else "ok",
    }
    if residue_key is None:
        return result
    if wt_aa not in VALID_AA or mut_aa not in VALID_AA:
        result["score_status"] = "invalid_amino_acid"
        return result
    wt_lp = _profile_log_prob(profile, residue_key, wt_aa)
    mut_lp = _profile_log_prob(profile, residue_key, mut_aa)
    result["wt_log_prob"] = wt_lp
    result["mut_log_prob"] = mut_lp
    if wt_lp is None or mut_lp is None:
        result["score_status"] = "missing_probability"
        return result
    delta = mut_lp - wt_lp
    result["delta_log_prob"] = delta
    result["delta_nll"] = -delta
    return result


def _score_full_sequence_delta(
    profile: Dict[str, Any],
    row: Dict[str, Any],
    *,
    sequence_column: str,
) -> Dict[str, Any]:
    sequence = str(row.get(sequence_column, "")).upper()
    residues = list(profile["residue_names"])
    native_sequence = "".join(profile.get("sequence") or [])
    result: Dict[str, Any] = {
        "residue_key": "",
        "mapping_mode": "full_sequence",
        "wt_log_prob": None,
        "mut_log_prob": None,
        "delta_log_prob": None,
        "delta_nll": None,
        "score_status": "ok",
    }
    if len(sequence) != len(residues):
        result["score_status"] = f"sequence_length_mismatch:{len(sequence)}!={len(residues)}"
        return result

    mut_total = 0.0
    wt_total = 0.0
    for residue_key, aa in zip(residues, sequence):
        lp = _profile_log_prob(profile, residue_key, aa)
        if lp is None:
            result["score_status"] = "missing_mut_probability"
            return result
        mut_total += lp

    if len(native_sequence) == len(residues):
        for residue_key, aa in zip(residues, native_sequence):
            lp = _profile_log_prob(profile, residue_key, aa)
            if lp is None:
                result["score_status"] = "missing_wt_probability"
                return result
            wt_total += lp
        result["wt_log_prob"] = wt_total

    result["mut_log_prob"] = mut_total
    result["delta_log_prob"] = mut_total - wt_total if result["wt_log_prob"] is not None else mut_total
    result["delta_nll"] = -result["delta_log_prob"]
    return result


@mcp.tool()
def validate_inputs(
    project_name: str,
    protein_name: str,
    sequence: str,
    target_state_structure: str,
    mutation_region: str,
    design_objective: str = "",
    target_state_name: str = "target_state",
    first_residue_number: int = 1,
) -> Dict[str, Any]:
    """Validate minimal SwitchTune project inputs."""
    seq = _read_sequence(sequence)
    warnings = _validate_sequence(seq)
    errors: List[str] = []

    if not project_name:
        errors.append("project_name is required.")
    if not protein_name:
        errors.append("protein_name is required.")
    if not seq:
        errors.append("sequence is required.")
    if not target_state_structure:
        errors.append("target_state_structure is required.")
    else:
        structure_path = Path(target_state_structure).expanduser()
        if not structure_path.exists():
            warnings.append(f"target_state_structure does not exist on this machine: {target_state_structure}")
    if not mutation_region:
        errors.append("mutation_region is required.")

    parsed_positions: List[Tuple[Optional[str], int]] = []
    region_warnings: List[str] = []
    try:
        parsed_positions = _parse_region(mutation_region)
        if not parsed_positions:
            errors.append("mutation_region did not produce any residue positions.")
        for expected_wt, pos in parsed_positions:
            idx = _position_to_index(pos, first_residue_number)
            if idx < 0 or idx >= len(seq):
                errors.append(
                    f"Residue position {pos} is outside sequence range "
                    f"{first_residue_number}-{first_residue_number + len(seq) - 1}."
                )
                continue
            observed = seq[idx]
            if expected_wt and expected_wt != observed:
                region_warnings.append(
                    f"Residue label mismatch at {pos}: region says {expected_wt}, sequence has {observed}."
                )
    except Exception as exc:
        errors.append(str(exc))

    warnings.extend(region_warnings)
    status = "valid" if not errors else "invalid"
    project = _project_dir(project_name)
    manifest = {
        "project_name": project_name,
        "protein_name": protein_name,
        "target_state_name": target_state_name,
        "target_state_structure": target_state_structure,
        "design_objective": design_objective,
        "sequence_length": len(seq),
        "first_residue_number": first_residue_number,
        "mutation_region": mutation_region,
        "parsed_positions": [pos for _, pos in parsed_positions],
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "updated_at": _utc_now(),
    }
    existing = _load_project_manifest(project_name)
    if existing:
        existing.update(manifest)
        manifest = existing
    manifest["stage"] = "input_validation"
    manifest_path = _save_project_manifest(project_name, manifest)
    return {
        "status": status,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "sequence_length": len(seq),
        "parsed_positions": [pos for _, pos in parsed_positions],
        "project_dir": str(project),
        "manifest_path": manifest_path,
    }


@mcp.tool()
def generate_saturation_library(
    project_name: str,
    sequence: str,
    mutation_region: str,
    first_residue_number: int = 1,
    output_dir: Optional[str] = None,
    aa_alphabet: str = DEFAULT_AA,
    exclude_self_mutations: bool = True,
) -> Dict[str, Any]:
    """Generate a single-mutant saturation library for the requested residue region."""
    seq = _read_sequence(sequence)
    if not seq:
        raise ValueError("sequence is required")
    positions = _parse_region(mutation_region)
    if not positions:
        raise ValueError("mutation_region produced no positions")

    alphabet = "".join([aa for aa in aa_alphabet.upper() if aa in VALID_AA])
    if not alphabet:
        raise ValueError("aa_alphabet contains no canonical amino acids")

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "01_saturation_library"
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for expected_wt, pos in positions:
        idx = _position_to_index(pos, first_residue_number)
        if idx < 0 or idx >= len(seq):
            warnings.append(
                f"Skipping position {pos}: outside sequence range "
                f"{first_residue_number}-{first_residue_number + len(seq) - 1}."
            )
            continue
        wt_aa = seq[idx]
        if expected_wt and expected_wt != wt_aa:
            warnings.append(f"Residue label mismatch at {pos}: region says {expected_wt}, sequence has {wt_aa}.")
        for mut_aa in alphabet:
            if exclude_self_mutations and mut_aa == wt_aa:
                continue
            mut_seq = seq[:idx] + mut_aa + seq[idx + 1 :]
            rows.append(
                {
                    "variant": f"{wt_aa}{pos}{mut_aa}",
                    "position": pos,
                    "sequence_index_1based": idx + 1,
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "sequence": mut_seq,
                }
            )

    csv_path = outdir / "saturation_single_mutant_library.csv"
    fasta_path = outdir / "saturation_single_mutant_library.fasta"
    fields = ["variant", "position", "sequence_index_1based", "wt_aa", "mut_aa", "sequence"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['variant']}\n{row['sequence']}\n")

    summary = {
        "status": "completed",
        "project_name": project_name,
        "sequence_length": len(seq),
        "first_residue_number": first_residue_number,
        "mutation_region": mutation_region,
        "positions_requested": [pos for _, pos in positions],
        "positions_used": sorted({int(row["position"]) for row in rows}),
        "variant_count": len(rows),
        "num_variants": len(rows),
        "csv_path": str(csv_path),
        "output_csv": str(csv_path),
        "fasta_path": str(fasta_path),
        "output_fasta": str(fasta_path),
        "warnings": warnings,
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "library_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "library_generated",
            "library_csv": str(csv_path),
            "library_fasta": str(fasta_path),
            "library_variant_count": len(rows),
            "library_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def setup_saturation_screening_project(
    project_name: str,
    protein_name: str,
    sequence: str,
    target_state_structure: str,
    mutation_region: str,
    design_objective: str = "",
    target_state_name: str = "target_state",
    first_residue_number: int = 1,
    output_dir: Optional[str] = None,
    aa_alphabet: str = DEFAULT_AA,
    exclude_self_mutations: bool = True,
) -> Dict[str, Any]:
    """Validate inputs and generate the single-mutant saturation library in one call.

    This composite tool is intentionally useful for Biomni smoke tests and early
    workflow setup, because it avoids fragile multi-step agent loops where the
    model repeatedly imports or re-calls individual MCP tools.
    """
    validation = validate_inputs(
        project_name=project_name,
        protein_name=protein_name,
        sequence=sequence,
        target_state_structure=target_state_structure,
        mutation_region=mutation_region,
        design_objective=design_objective,
        target_state_name=target_state_name,
        first_residue_number=first_residue_number,
    )

    if not validation.get("valid", False):
        return {
            "status": "invalid",
            "project_name": project_name,
            "validation": validation,
            "library": None,
            "summary": {
                "validation_status": validation.get("status", "invalid"),
                "library_status": "not_started",
                "variant_count": 0,
                "warnings": validation.get("warnings", []),
                "errors": validation.get("errors", []),
            },
        }

    library = generate_saturation_library(
        project_name=project_name,
        sequence=sequence,
        mutation_region=mutation_region,
        first_residue_number=first_residue_number,
        output_dir=output_dir,
        aa_alphabet=aa_alphabet,
        exclude_self_mutations=exclude_self_mutations,
    )

    warnings = []
    warnings.extend(validation.get("warnings", []))
    warnings.extend(library.get("warnings", []))

    return {
        "status": "completed",
        "project_name": project_name,
        "validation": validation,
        "library": library,
        "summary": {
            "validation_status": validation.get("status", "unknown"),
            "library_status": library.get("status", "unknown"),
            "variant_count": library.get("variant_count", 0),
            "num_variants": library.get("num_variants", 0),
            "csv_path": library.get("csv_path"),
            "output_csv": library.get("output_csv"),
            "fasta_path": library.get("fasta_path"),
            "output_fasta": library.get("output_fasta"),
            "warnings": warnings,
            "errors": validation.get("errors", []),
        },
    }


@mcp.tool()
def run_conformation_bias_scoring(
    project_name: str,
    library_csv: str,
    target_state_structure: str,
    reference_state_structure: Optional[str] = None,
    target_state_name: str = "target_state",
    reference_state_name: str = "reference_state",
    output_dir: Optional[str] = None,
    model_type: str = "protein_mpnn",
    parse_chains: str = "",
    chains_to_design: Optional[str] = None,
    first_residue_number: int = 1,
    ligandmpnn_repo: Optional[str] = None,
    score_script: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    conda_env: Optional[str] = None,
    batch_size: int = 1,
    number_of_batches: int = 1,
    use_sequence: int = 1,
    ligand_mpnn_use_atom_context: int = 1,
    ligand_mpnn_use_side_chain_context: int = 0,
    ligand_mpnn_cutoff_for_score: float = 8.0,
    parse_atoms_with_zero_occupancy: int = 0,
    seed: int = 0,
    verbose: int = 1,
    timeout_seconds: int = 0,
    execute: bool = False,
    collect_scores: bool = False,
) -> Dict[str, Any]:
    """Prepare or optionally run real CB scoring with LigandMPNN/ProteinMPNN.

    The scorer runs LigandMPNN's `score.py` in `single_aa_score` mode once per
    conformational state. `extract_conformation_bias_scores` then scores every
    single mutant as logP(mutAA)-logP(WTAA) on each structure, standardizes the
    state-wise deltas, and computes:

        bias_score = target_state_scaled - reference_state_scaled

    SwitchTune should retain mutations inside the configured bias-score window
    (for example -1 <= bias_score <= 1) to avoid over-biasing switch proteins.
    """
    model_type = _normalize_mpnn_model_type(model_type)
    library_path = Path(library_csv).expanduser()
    if not library_path.exists():
        raise FileNotFoundError(f"library_csv not found: {library_csv}")

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "02_conformation_bias_scoring"
    outdir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = _read_csv_rows(library_path)
    required = {"variant", "position", "wt_aa", "mut_aa"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"library_csv is missing required columns for CB scoring: {missing}")

    states: List[Tuple[str, str]] = [(target_state_name, target_state_structure)]
    if reference_state_structure:
        states.append((reference_state_name, reference_state_structure))

    state_records = _copy_state_pdbs(states, outdir)
    run_info = _write_ligandmpnn_cb_script(
        outdir=outdir,
        state_records=state_records,
        model_type=model_type,
        parse_chains=parse_chains,
        chains_to_design=chains_to_design,
        batch_size=batch_size,
        number_of_batches=number_of_batches,
        use_sequence=use_sequence,
        ligand_mpnn_use_atom_context=ligand_mpnn_use_atom_context,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
        ligand_mpnn_cutoff_for_score=ligand_mpnn_cutoff_for_score,
        parse_atoms_with_zero_occupancy=parse_atoms_with_zero_occupancy,
        seed=seed,
        verbose=verbose,
        ligandmpnn_repo=ligandmpnn_repo,
        score_script=score_script,
        checkpoint_path=checkpoint_path,
        conda_env=conda_env,
    )

    score_min = _load_default_config().get("screening", {}).get("bias_score_min", -1.0)
    score_max = _load_default_config().get("screening", {}).get("bias_score_max", 1.0)
    job_payload = {
        "library_csv": str(library_path),
        "output_dir": str(outdir),
        "model_type": model_type,
        "state_records": state_records,
        "target_state_name": target_state_name,
        "reference_state_name": reference_state_name if reference_state_structure else None,
        "parse_chains": parse_chains,
        "chains_to_design": chains_to_design,
        "first_residue_number": first_residue_number,
        "score_min": score_min,
        "score_max": score_max,
        "ligandmpnn_run": run_info,
        "formula": {
            "state_delta_log_prob": "logP(mutAA|state_structure)-logP(WTAA|state_structure)",
            "state_scaled": "zscore(state_delta_log_prob across library variants)",
            "bias_score": "target_state_scaled-reference_state_scaled",
            "switchtune_retention": (
            "select_state_specific_candidates_adaptive uses the empirical central "
            "fraction; score_min/score_max are retained only as legacy diagnostics"
        ),
        },
    }
    job_manifest_path = outdir / "cb_ligandmpnn_job_manifest.json"
    _write_json(job_manifest_path, job_payload)
    job = _job_record(project_name, "conformation_bias_scoring", "prepared", job_payload)

    status = "prepared"
    return_payload: Dict[str, Any] = {
        "status": status,
        "project_name": project_name,
        "job_id": job["job_id"],
        "job_record_path": str(_project_dir(project_name) / "jobs" / f"{job['job_id']}.json"),
        "job_manifest_path": str(job_manifest_path),
        "model_type": model_type,
        "variant_count": len(rows),
        "states": state_records,
        "run_script": run_info["script_path"],
        "score_output_dir": run_info["score_output_dir"],
        "expected_pt": run_info["expected_pt"],
        "formula": job_payload["formula"],
        "warnings": [],
        "next_step": "Run run_script, then call extract_conformation_bias_scores on job_manifest_path.",
    }
    if not reference_state_structure:
        return_payload["warnings"].append(
            "reference_state_structure was not provided; extraction can report target-state deltas but cannot compute CB bias_score."
        )

    if execute:
        completed = subprocess.run(
            ["bash", run_info["script_path"]],
            cwd=str(outdir),
            text=True,
            capture_output=True,
            timeout=None if int(timeout_seconds) <= 0 else int(timeout_seconds),
        )
        return_payload["status"] = "completed" if completed.returncode == 0 else "failed"
        return_payload["returncode"] = completed.returncode
        return_payload["stdout_tail"] = completed.stdout[-4000:]
        return_payload["stderr_tail"] = completed.stderr[-4000:]
        if completed.returncode != 0:
            return return_payload

    if collect_scores:
        try:
            extracted = extract_conformation_bias_scores(
                project_name=project_name,
                job_manifest_path=str(job_manifest_path),
                score_min=float(score_min),
                score_max=float(score_max),
            )
            return_payload["extraction"] = extracted
            return_payload["scored_csv"] = extracted.get("output_csv")
            return_payload["status"] = "completed"
            return_payload["next_step"] = "Call select_state_specific_candidates on scored_csv, or inspect the CB distribution first."
        except Exception as exc:
            return_payload["warnings"].append(f"collect_scores failed: {exc}")

    _update_project_manifest(
        project_name,
        {
            "stage": "conformation_bias_scoring_prepared",
            "cb_job_manifest_path": str(job_manifest_path),
            "cb_run_script": run_info["script_path"],
            "cb_model_type": model_type,
        },
    )
    return return_payload


@mcp.tool()
def setup_two_state_cb_scoring_project(
    project_name: str,
    protein_name: str,
    sequence: str,
    target_state_structure: str,
    reference_state_structure: str,
    mutation_region: str,
    target_state_name: str = "target_state",
    reference_state_name: str = "reference_state",
    design_objective: str = "",
    model_type: str = "protein_mpnn",
    parse_chains: str = "",
    chains_to_design: Optional[str] = None,
    first_residue_number: int = 1,
    aa_alphabet: str = DEFAULT_AA,
    exclude_self_mutations: bool = True,
    execute: bool = False,
    collect_scores: bool = False,
) -> Dict[str, Any]:
    """Create a two-state SwitchTune CB scoring project in one reliable call.

    This composite tool is intended for Biomni after it has made three decisions:
    target/reference state assignment, focused mutation region, and MPNN model type.
    It reduces fragile multi-step natural-language loops while keeping those
    scientific decisions explicit in the tool inputs and decision log.
    """
    try:
        parsed_region = _parse_region(mutation_region)
    except Exception as exc:
        return {
            "status": "invalid",
            "project_name": project_name,
            "errors": [str(exc)],
            "summary": {
                "validation_status": "invalid",
                "library_status": "not_started",
                "cb_status": "not_started",
                "warnings": [],
                "errors": [str(exc)],
            },
        }
    if not parsed_region:
        error = (
            "mutation_region must contain at least one concrete residue position "
            "or range, for example 220-230,240-250,236."
        )
        return {
            "status": "invalid",
            "project_name": project_name,
            "errors": [error],
            "summary": {
                "validation_status": "invalid",
                "library_status": "not_started",
                "cb_status": "not_started",
                "warnings": [],
                "errors": [error],
            },
        }

    reference_path = Path(reference_state_structure).expanduser()
    if not reference_path.exists():
        return {
            "status": "invalid",
            "project_name": project_name,
            "errors": [f"reference_state_structure does not exist: {reference_state_structure}"],
        }

    created = create_project(
        project_name=project_name,
        protein_name=protein_name,
        target_state_name=target_state_name,
        design_objective=design_objective,
        metadata={
            "workflow_stage": "two_state_cb_scoring_project",
            "reference_state_name": reference_state_name,
            "model_type": model_type,
            "parse_chains": parse_chains,
        },
    )

    validation = validate_inputs(
        project_name=project_name,
        protein_name=protein_name,
        sequence=sequence,
        target_state_structure=target_state_structure,
        target_state_name=target_state_name,
        mutation_region=mutation_region,
        design_objective=design_objective,
        first_residue_number=first_residue_number,
    )

    if not validation.get("valid", False):
        return {
            "status": "invalid",
            "project_name": project_name,
            "created": created,
            "validation": validation,
            "library": None,
            "cb_job": None,
            "summary": {
                "validation_status": validation.get("status", "invalid"),
                "library_status": "not_started",
                "cb_status": "not_started",
                "warnings": validation.get("warnings", []),
                "errors": validation.get("errors", []),
            },
        }

    decision = write_decision_log(
        project_name=project_name,
        step_name="two_state_cb_setup",
        decision="prepare_cb_job",
        rationale=(
            f"Target={target_state_name}; reference={reference_state_name}; "
            f"mutation_region={mutation_region}; model_type={model_type}; parse_chains={parse_chains}."
        ),
        input_summary={
            "protein_name": protein_name,
            "target_state_structure": target_state_structure,
            "reference_state_structure": reference_state_structure,
            "mutation_region": mutation_region,
            "model_type": model_type,
            "parse_chains": parse_chains,
        },
        output_summary={},
    )

    library = generate_saturation_library(
        project_name=project_name,
        sequence=sequence,
        mutation_region=mutation_region,
        first_residue_number=first_residue_number,
        aa_alphabet=aa_alphabet,
        exclude_self_mutations=exclude_self_mutations,
    )

    cb_job = run_conformation_bias_scoring(
        project_name=project_name,
        library_csv=library["csv_path"],
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        target_state_name=target_state_name,
        reference_state_name=reference_state_name,
        model_type=model_type,
        parse_chains=parse_chains,
        chains_to_design=chains_to_design,
        first_residue_number=first_residue_number,
        execute=execute,
        collect_scores=collect_scores,
    )

    warnings = []
    warnings.extend(validation.get("warnings", []))
    warnings.extend(library.get("warnings", []))
    warnings.extend(cb_job.get("warnings", []))

    summary = {
        "validation_status": validation.get("status"),
        "library_status": library.get("status"),
        "cb_status": cb_job.get("status"),
        "variant_count": library.get("variant_count"),
        "library_csv": library.get("csv_path"),
        "library_fasta": library.get("fasta_path"),
        "cb_job_manifest_path": cb_job.get("job_manifest_path"),
        "cb_run_script": cb_job.get("run_script"),
        "expected_pt": cb_job.get("expected_pt"),
        "warnings": warnings,
        "errors": [],
    }
    return {
        "status": "completed" if cb_job.get("status") in {"prepared", "completed"} else cb_job.get("status", "unknown"),
        "project_name": project_name,
        "created": created,
        "validation": validation,
        "decision": decision,
        "library": library,
        "cb_job": cb_job,
        "summary": summary,
    }


@mcp.tool()
def setup_two_state_cb_scoring_project_once(
    project_name: str,
    protein_name: str,
    sequence: str,
    target_state_structure: str,
    reference_state_structure: str,
    mutation_region: str,
    target_state_name: str = "target_state",
    reference_state_name: str = "reference_state",
    design_objective: str = "",
    model_type: str = "protein_mpnn",
    parse_chains: str = "",
    chains_to_design: Optional[str] = None,
    first_residue_number: int = 1,
    aa_alphabet: str = DEFAULT_AA,
    exclude_self_mutations: bool = True,
    force_refresh: bool = False,
    stop_after_setup: bool = True,
) -> Dict[str, Any]:
    """Compact one-shot two-state CB setup for LLM agents.

    Biomni's `agent.go()` can keep planning after a large nested tool result.
    This wrapper calls the full setup tool once but returns only the fields
    needed for the final answer, plus an explicit stop instruction.
    """
    def compact_response(
        status: str,
        variant_count: Any,
        library_csv: Any,
        library_fasta: Any,
        cb_job_manifest_path: Any,
        cb_run_script: Any,
        expected_pt: Any,
        warnings: Any,
        errors: Any,
        reused_existing: bool = False,
    ) -> Dict[str, Any]:
        extract_command = None
        if cb_job_manifest_path:
            extract_command = (
                "cd ${SWITCHTUNE_HOME:-.} && "
                "conda run -n ${SWITCHTUNE_LIGANDMPNN_CONDA_ENV:-ligandmpnn_env} "
                "python scripts/extract_cb_ligandmpnn_scores.py "
                f"--project_name {project_name} "
                f"--job_manifest_path {cb_job_manifest_path}"
            )

        return {
            "status": status,
            "done": status in {"completed", "prepared"},
            "reused_existing": reused_existing,
            "stop_after_this_tool": bool(stop_after_setup),
            "next_tool": None if stop_after_setup else "execute_conformation_bias_scoring_job",
            "agent_instruction": (
                "Return a final <solution> now. Do not call setup_two_state_cb_scoring_project "
                "or setup_two_state_cb_scoring_project_once again. Do not inspect structures again."
                if stop_after_setup
                else "Next call execute_conformation_bias_scoring_job once using cb_job_manifest_path. "
                "Do not repeat setup or inspect structures again."
            ),
            "project_name": project_name,
            "protein_name": protein_name,
            "target_state_name": target_state_name,
            "reference_state_name": reference_state_name,
            "mutation_region": mutation_region,
            "model_type": model_type,
            "variant_count": variant_count,
            "library_csv": library_csv,
            "library_fasta": library_fasta,
            "cb_job_manifest_path": cb_job_manifest_path,
            "cb_run_script": cb_run_script,
            "expected_pt": expected_pt,
            "next_commands": {
                "run_cb_scoring": f"bash {cb_run_script}" if cb_run_script else None,
                "extract_cb_scores": extract_command,
                "inspect_summary": (
                    f"cat {_project_dir(project_name) / '02_conformation_bias_scoring' / 'cb_scored_library_summary.json'}"
                ),
                "inspect_scores": (
                    f"head {_project_dir(project_name) / '02_conformation_bias_scoring' / 'cb_scored_library.csv'}"
                ),
            },
            "warnings": warnings or [],
            "errors": errors or [],
        }

    try:
        parsed_region = _parse_region(mutation_region)
    except Exception as exc:
        error = str(exc)
        return compact_response(
            status="invalid",
            variant_count=0,
            library_csv=None,
            library_fasta=None,
            cb_job_manifest_path=None,
            cb_run_script=None,
            expected_pt={},
            warnings=[],
            errors=[error],
        )
    if not parsed_region:
        error = (
            "mutation_region must contain at least one concrete residue position "
            "or range, for example 220-230,240-250,236."
        )
        return compact_response(
            status="invalid",
            variant_count=0,
            library_csv=None,
            library_fasta=None,
            cb_job_manifest_path=None,
            cb_run_script=None,
            expected_pt={},
            warnings=[],
            errors=[error],
        )

    if not force_refresh:
        manifest = _load_project_manifest(project_name)
        cb_job_manifest_path = manifest.get("cb_job_manifest_path")
        cb_run_script = manifest.get("cb_run_script")
        library_csv = manifest.get("library_csv")
        library_fasta = manifest.get("library_fasta")
        if cb_job_manifest_path and cb_run_script and library_csv:
            existing_region = str(manifest.get("mutation_region") or "")
            try:
                existing_positions = {pos for _wt, pos in _parse_region(existing_region)}
            except Exception:
                existing_positions = set()
            requested_positions = {pos for _wt, pos in parsed_region}
            if existing_positions and existing_positions != requested_positions:
                return compact_response(
                    status="conflict",
                    variant_count=manifest.get("library_variant_count", 0),
                    library_csv=library_csv,
                    library_fasta=library_fasta,
                    cb_job_manifest_path=cb_job_manifest_path,
                    cb_run_script=cb_run_script,
                    expected_pt={},
                    warnings=[],
                    errors=[
                        "Project setup checkpoint already exists for mutation_region="
                        f"{existing_region}, but the new request selected "
                        f"mutation_region={mutation_region}. Use force_refresh=true "
                        "to intentionally rebuild the project."
                    ],
                    reused_existing=True,
                )
            cb_manifest_path = Path(str(cb_job_manifest_path)).expanduser()
            expected_pt: Dict[str, Any] = {}
            if cb_manifest_path.exists():
                cb_manifest = _read_json(cb_manifest_path)
                expected_pt = cb_manifest.get("ligandmpnn_run", {}).get("expected_pt", {})
            return compact_response(
                status="completed",
                variant_count=manifest.get("library_variant_count"),
                library_csv=library_csv,
                library_fasta=library_fasta,
                cb_job_manifest_path=cb_job_manifest_path,
                cb_run_script=cb_run_script,
                expected_pt=expected_pt,
                warnings=[],
                errors=[],
                reused_existing=True,
            )

    result = setup_two_state_cb_scoring_project(
        project_name=project_name,
        protein_name=protein_name,
        sequence=sequence,
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        mutation_region=mutation_region,
        target_state_name=target_state_name,
        reference_state_name=reference_state_name,
        design_objective=design_objective,
        model_type=model_type,
        parse_chains=parse_chains,
        chains_to_design=chains_to_design,
        first_residue_number=first_residue_number,
        aa_alphabet=aa_alphabet,
        exclude_self_mutations=exclude_self_mutations,
        execute=False,
        collect_scores=False,
    )
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    errors = summary.get("errors", [])
    warnings = summary.get("warnings", [])
    cb_run_script = summary.get("cb_run_script")
    cb_job_manifest_path = summary.get("cb_job_manifest_path")
    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"

    return compact_response(
        status=status,
        variant_count=summary.get("variant_count"),
        library_csv=summary.get("library_csv"),
        library_fasta=summary.get("library_fasta"),
        cb_job_manifest_path=cb_job_manifest_path,
        cb_run_script=cb_run_script,
        expected_pt=summary.get("expected_pt"),
        warnings=warnings,
        errors=errors,
        reused_existing=False,
    )


@mcp.tool()
def inspect_two_state_pdb_pair(
    pdb_a: str,
    pdb_b: str,
    protein_name: str = "",
    design_goal: str = "",
) -> Dict[str, Any]:
    """Inspect two PDB files and return state-assignment clues for SwitchTune.

    This tool does not decide the full design by itself; it supplies stable
    structural evidence so the agent can choose target/reference, MPNN mode,
    chain, and mutation region without repeatedly writing ad-hoc Bio.PDB code.

    ``pdb_a`` and ``pdb_b`` are the only accepted PDB argument names.  Keeping
    the public schema strict prevents the upstream agent API from receiving
    ambiguous duplicate state-path fields.
    """
    feature_a = _inspect_pdb_file_basic(pdb_a)
    feature_b = _inspect_pdb_file_basic(pdb_b)

    def activation_score(feature: Dict[str, Any]) -> int:
        het = set((feature.get("nonwater_hetero_counts") or {}).keys())
        score = 0
        if "BEF" in het:
            score += 3
        if "IPT" in het:
            # IPT is the PDB CCD code for the IPTG inducer in LacI structures.
            score += 3
        if "MN" in het or "MG" in het:
            score += 2
        if "SO4" in het and not ({"BEF", "MN", "MG"} & het):
            score -= 1
        return score

    score_a = activation_score(feature_a)
    score_b = activation_score(feature_b)
    if score_a > score_b:
        suggested_target = pdb_a
        suggested_reference = pdb_b
        target_reason = "first structure has stronger activation-state ligand evidence"
    elif score_b > score_a:
        suggested_target = pdb_b
        suggested_reference = pdb_a
        target_reason = "second structure has stronger activation-state ligand evidence"
    else:
        suggested_target = None
        suggested_reference = None
        target_reason = "no clear ligand-based activation-state difference detected"

    all_nonwater = set((feature_a.get("nonwater_hetero_counts") or {}).keys()) | set(
        (feature_b.get("nonwater_hetero_counts") or {}).keys()
    )
    suggested_model = "ligand_mpnn" if all_nonwater else "protein_mpnn"
    if {"BEF", "MN", "MG"} & all_nonwater:
        suggested_model = "ligand_mpnn"

    chains_a = feature_a.get("chains") or []
    chains_b = feature_b.get("chains") or []
    common_chains = [chain for chain in chains_a if chain in set(chains_b)]
    suggested_chain = common_chains[0] if common_chains else ((chains_a or chains_b or [""])[0])
    target_feature = feature_a if suggested_target == pdb_a else feature_b if suggested_target == pdb_b else feature_a
    reference_feature = feature_b if target_feature is feature_a else feature_a
    suggested_sequence = ""
    suggested_first_residue_number = 1
    if suggested_chain:
        target_sequences = target_feature.get("sequence_by_chain") or {}
        reference_sequences = reference_feature.get("sequence_by_chain") or {}
        suggested_sequence = str(target_sequences.get(suggested_chain) or reference_sequences.get(suggested_chain) or "")
        first_numbers = target_feature.get("first_residue_number_by_chain") or {}
        suggested_first_residue_number = int(first_numbers.get(suggested_chain) or 1)

    suggested_mutation_region = None
    mutation_region_rationale = ""

    # Use the structure-only fallback by default for dynamic-region runs. It
    # prevents a provider-side continuation failure after the initial PDB
    # observation. The selection remains blind: only the supplied wild-type
    # structures and their aligned C-alpha displacements are used.
    if not suggested_mutation_region and os.environ.get(
        "SWITCHTUNE_AUTONOMOUS_REGION_FALLBACK", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        fallback_region, fallback_rationale = _autonomous_structural_mutation_region(
            pdb_a,
            pdb_b,
            suggested_chain,
        )
        if fallback_region:
            suggested_mutation_region = fallback_region
            mutation_region_rationale = fallback_rationale

    if suggested_mutation_region:
        region_instruction = (
            "Use these features to make a final decision, then call "
            "setup_two_state_cb_scoring_project_once exactly once using the "
            "suggested_sequence as the sequence and the suggested_mutation_region "
            "unless the user prompt provides an explicit region. Do not inspect "
            "PDB files again."
        )
    else:
        region_instruction = (
            "Use these features to choose one or more exact mutation intervals "
            "or discrete positions from the inspected structures and sequence. "
            "Encode them as a comma-separated parseable string such as "
            "`220-230,240-250,236`, then call "
            "setup_two_state_cb_scoring_project_once exactly once using the "
            "suggested_sequence as the sequence. Do not inspect PDB files again."
        )

    return {
        "status": "completed",
        "protein_name": protein_name,
        "design_goal": design_goal,
        "pdb_a": feature_a,
        "pdb_b": feature_b,
        "suggested_target_structure": suggested_target,
        "suggested_reference_structure": suggested_reference,
        "target_reference_rationale": target_reason,
        "suggested_model_type": suggested_model,
        "suggested_parse_chains": suggested_chain,
        "suggested_sequence": suggested_sequence,
        "suggested_sequence_length": len(suggested_sequence),
        "suggested_first_residue_number": suggested_first_residue_number,
        "suggested_mutation_region": suggested_mutation_region,
        "mutation_region_rationale": mutation_region_rationale,
        "agent_instruction": region_instruction,
        "mutation_region_selection_required": not bool(suggested_mutation_region),
        "next_tool": "setup_two_state_cb_scoring_project_once",
        "warnings": [],
        "errors": feature_a.get("errors", []) + feature_b.get("errors", []),
    }


@mcp.tool()
def execute_conformation_bias_scoring_job(
    project_name: str,
    job_manifest_path: str,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    timeout_seconds: int = 3600,
    skip_if_outputs_exist: bool = True,
    collect_scores: bool = True,
) -> Dict[str, Any]:
    """Execute a prepared LigandMPNN/ProteinMPNN CB job and extract scores.

    This is the 0-code execution bridge for SwitchTune. It runs the
    generated bash script under the configured conda environment, then extracts
    CB scores from the produced `.pt` profiles. It is intentionally manifest-
    based so the execution is auditable and not arbitrary shell code.
    """
    manifest_path = Path(job_manifest_path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"job_manifest_path not found: {job_manifest_path}")
    manifest = _read_json(manifest_path)
    run_info = manifest.get("ligandmpnn_run", {})
    if not isinstance(run_info, dict):
        raise ValueError("job manifest does not contain ligandmpnn_run.")
    script_path = Path(str(run_info.get("script_path") or "")).expanduser()
    if not script_path.exists():
        raise FileNotFoundError(f"CB run script not found: {script_path}")

    expected_pt = run_info.get("expected_pt", {})
    expected_paths = [Path(str(path)).expanduser() for path in expected_pt.values()] if isinstance(expected_pt, dict) else []
    outputs_exist = bool(expected_paths) and all(path.exists() and path.stat().st_size > 0 for path in expected_paths)
    run_status = "skipped_existing_outputs" if skip_if_outputs_exist and outputs_exist else "not_started"
    returncode: Optional[int] = None
    stdout_tail = ""
    stderr_tail = ""

    if run_status != "skipped_existing_outputs":
        completed = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(script_path.parent),
            text=True,
            capture_output=True,
            timeout=None if int(timeout_seconds) <= 0 else int(timeout_seconds),
        )
        returncode = completed.returncode
        stdout_tail = completed.stdout[-4000:]
        stderr_tail = completed.stderr[-4000:]
        run_status = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            result = {
                "status": "failed",
                "project_name": project_name,
                "job_manifest_path": str(manifest_path),
                "run_script": str(script_path),
                "run_status": run_status,
                "returncode": returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "expected_pt": {k: str(v) for k, v in expected_pt.items()} if isinstance(expected_pt, dict) else expected_pt,
                "errors": ["CB scoring script failed. Inspect stderr_tail and log file."],
                "warnings": [],
            }
            _job_record(project_name, "conformation_bias_scoring_execute", "failed", result)
            return result

    extraction = None
    if collect_scores:
        extraction = extract_conformation_bias_scores(
            project_name=project_name,
            job_manifest_path=str(manifest_path),
            score_min=score_min,
            score_max=score_max,
        )

    result = {
        "status": "completed",
        "project_name": project_name,
        "job_manifest_path": str(manifest_path),
        "run_script": str(script_path),
        "run_status": run_status,
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "expected_pt": {k: str(v) for k, v in expected_pt.items()} if isinstance(expected_pt, dict) else expected_pt,
        "extraction": extraction,
        "scored_library_csv": extraction.get("output_csv") if isinstance(extraction, dict) else None,
        "scored_library_summary_path": extraction.get("summary_path") if isinstance(extraction, dict) else None,
        "input_count": extraction.get("input_count") if isinstance(extraction, dict) else None,
        "valid_bias_count": extraction.get("valid_bias_count") if isinstance(extraction, dict) else None,
        "retained_count_fixed_window": extraction.get("retained_count") if isinstance(extraction, dict) else None,
        "next_tool": "select_state_specific_candidates_adaptive",
        "agent_instruction": (
            "Next call select_state_specific_candidates_adaptive using scored_library_csv. "
            "Do not rerun CB scoring unless outputs are missing or failed."
        ),
        "warnings": [],
        "errors": [],
    }
    _job_record(project_name, "conformation_bias_scoring_execute", "completed", result)
    _update_project_manifest(
        project_name,
        {
            "stage": "conformation_bias_scoring_executed",
            "cb_scored_library_csv": result.get("scored_library_csv"),
            "cb_scored_library_summary_path": result.get("scored_library_summary_path"),
            "cb_execute_run_status": run_status,
        },
    )
    return result


@mcp.tool()
def extract_conformation_bias_scores(
    project_name: str,
    job_manifest_path: str,
    output_csv: Optional[str] = None,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    score_mode: str = "single_mutant_delta",
    chain_id: Optional[str] = None,
    position_column: str = "position",
    sequence_index_column: str = "sequence_index_1based",
    wt_column: str = "wt_aa",
    mut_column: str = "mut_aa",
    sequence_column: str = "sequence",
) -> Dict[str, Any]:
    """Extract real CB scores from LigandMPNN/ProteinMPNN `.pt` state profiles."""
    if not _torch_available() and os.environ.get("SWITCHTUNE_CB_ALLOW_EXTERNAL_EXTRACT", "1") != "0":
        return _run_external_cb_extract(
            project_name=project_name,
            job_manifest_path=job_manifest_path,
            output_csv=output_csv,
            score_min=score_min,
            score_max=score_max,
            score_mode=score_mode,
            chain_id=chain_id,
            position_column=position_column,
            sequence_index_column=sequence_index_column,
            wt_column=wt_column,
            mut_column=mut_column,
            sequence_column=sequence_column,
        )

    manifest_path = Path(job_manifest_path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"job_manifest_path not found: {job_manifest_path}")
    manifest = _read_json(manifest_path)

    library_csv = Path(str(manifest["library_csv"])).expanduser()
    if not library_csv.exists():
        raise FileNotFoundError(f"library_csv from manifest not found: {library_csv}")
    rows, fieldnames = _read_csv_rows(library_csv)

    target_state_name = str(manifest.get("target_state_name") or "target_state")
    reference_state_name = manifest.get("reference_state_name")
    state_records = manifest.get("state_records", [])
    run_info = manifest.get("ligandmpnn_run", {})
    expected_pt = run_info.get("expected_pt", {})
    first_residue_number = int(manifest.get("first_residue_number", 1))
    if chain_id is None:
        parse_chains = str(manifest.get("parse_chains") or "")
        chain_id = parse_chains[0] if len(parse_chains) == 1 else None

    state_profiles: Dict[str, Dict[str, Any]] = {}
    state_pt_paths: Dict[str, str] = {}
    for record in state_records:
        state_name = str(record["state_name"])
        pt_path = expected_pt.get(state_name)
        if not pt_path:
            pt_path = str(Path(run_info.get("score_output_dir", "")) / f"{Path(record['scoring_pdb']).stem}.pt")
        state_profiles[state_name] = _load_ligandmpnn_pt_profile(pt_path)
        state_pt_paths[state_name] = pt_path

    if target_state_name not in state_profiles:
        raise ValueError(f"Target state {target_state_name!r} was not found in expected .pt files.")

    scored_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        for state_name, profile in state_profiles.items():
            if score_mode == "full_sequence_delta":
                score = _score_full_sequence_delta(profile, row, sequence_column=sequence_column)
            else:
                score = _score_single_mutant_delta(
                    profile,
                    row,
                    chain_id=chain_id,
                    first_residue_number=first_residue_number,
                    position_column=position_column,
                    sequence_index_column=sequence_index_column,
                    wt_column=wt_column,
                    mut_column=mut_column,
                )
            prefix = _sanitize_name(state_name)
            out[f"{prefix}_residue_key"] = score.get("residue_key")
            out[f"{prefix}_mapping_mode"] = score.get("mapping_mode")
            out[f"{prefix}_wt_log_prob"] = score.get("wt_log_prob")
            out[f"{prefix}_mut_log_prob"] = score.get("mut_log_prob")
            out[f"{prefix}_delta_log_prob"] = score.get("delta_log_prob")
            out[f"{prefix}_delta_nll"] = score.get("delta_nll")
            out[f"{prefix}_score_status"] = score.get("score_status")
        scored_rows.append(out)

    target_prefix = _sanitize_name(target_state_name)
    target_values = [_safe_float(row.get(f"{target_prefix}_delta_log_prob")) for row in scored_rows]
    target_scaled = _z_scale(target_values)
    for row, scaled in zip(scored_rows, target_scaled):
        row[f"{target_prefix}_scaled"] = scaled
        row["target_state_delta_log_prob"] = row.get(f"{target_prefix}_delta_log_prob")
        row["target_state_scaled"] = scaled

    reference_prefix = None
    if reference_state_name and str(reference_state_name) in state_profiles:
        reference_prefix = _sanitize_name(str(reference_state_name))
        reference_values = [_safe_float(row.get(f"{reference_prefix}_delta_log_prob")) for row in scored_rows]
        reference_scaled = _z_scale(reference_values)
        for row, scaled in zip(scored_rows, reference_scaled):
            row[f"{reference_prefix}_scaled"] = scaled
            row["reference_state_delta_log_prob"] = row.get(f"{reference_prefix}_delta_log_prob")
            row["reference_state_scaled"] = scaled
            t = _safe_float(row.get("target_state_scaled"))
            r = _safe_float(row.get("reference_state_scaled"))
            row["bias_score"] = (t - r) if t is not None and r is not None else None
            row["bias_score_abs"] = abs(row["bias_score"]) if row["bias_score"] is not None else None
            row["bias_score_direction"] = (
                target_state_name if row["bias_score"] is not None and row["bias_score"] > 0
                else str(reference_state_name) if row["bias_score"] is not None and row["bias_score"] < 0
                else "neutral_or_unresolved"
            )
    else:
        for row in scored_rows:
            row["bias_score"] = None
            row["bias_score_abs"] = None
            row["bias_score_direction"] = "reference_state_missing"

    if score_min is None:
        score_min = float(manifest.get("score_min", -1.0))
    if score_max is None:
        score_max = float(manifest.get("score_max", 1.0))
    retained = 0
    valid_bias = 0
    for row in scored_rows:
        score = _safe_float(row.get("bias_score"))
        if score is None:
            row["switchtune_cb_pass"] = ""
            continue
        valid_bias += 1
        passed = float(score_min) <= score <= float(score_max)
        retained += int(passed)
        row["switchtune_cb_pass"] = passed

    out_csv = Path(output_csv).expanduser() if output_csv else manifest_path.parent / "cb_scored_library.csv"
    preferred_fields = list(fieldnames) + [
        "target_state_delta_log_prob",
        "target_state_scaled",
        "reference_state_delta_log_prob",
        "reference_state_scaled",
        "bias_score",
        "bias_score_abs",
        "bias_score_direction",
        "switchtune_cb_pass",
    ]
    _write_csv_rows(out_csv, scored_rows, _csv_fieldnames_union(scored_rows, preferred_fields))

    bias_values = [_safe_float(row.get("bias_score")) for row in scored_rows]
    bias_numeric = [x for x in bias_values if x is not None]
    summary = {
        "status": "completed",
        "project_name": project_name,
        "job_manifest_path": str(manifest_path),
        "library_csv": str(library_csv),
        "output_csv": str(out_csv),
        "model_type": manifest.get("model_type"),
        "score_mode": score_mode,
        "target_state_name": target_state_name,
        "reference_state_name": reference_state_name,
        "state_pt_paths": state_pt_paths,
        "input_count": len(scored_rows),
        "valid_bias_count": valid_bias,
        "retained_count": retained,
        "score_min": score_min,
        "score_max": score_max,
        "bias_distribution": _score_distribution_summary(bias_numeric) if bias_numeric else {"count": 0},
        "formula": manifest.get("formula", {}),
    }
    summary_path = _write_json(out_csv.with_name("cb_scored_library_summary.json"), summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "conformation_bias_scores_extracted",
            "cb_scored_library_csv": str(out_csv),
            "cb_scored_library_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def select_state_specific_candidates(
    project_name: str,
    scored_library_csv: str,
    score_column: str = "bias_score",
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Legacy explicit score-window selector.

    The public launcher never calls this tool. Public runs use
    ``select_state_specific_candidates_adaptive`` and an empirical central
    fraction; this function remains for compatibility with older projects.
    """
    config = _load_default_config()
    screening = config.get("screening", {}) if isinstance(config.get("screening", {}), dict) else {}
    if score_min is None:
        score_min = float(screening.get("bias_score_min", -1.0))
    if score_max is None:
        score_max = float(screening.get("bias_score_max", 1.0))
    score_min = float(score_min)
    score_max = float(score_max)
    if score_min > score_max:
        raise ValueError(f"score_min ({score_min}) must be <= score_max ({score_max}).")

    input_path = Path(scored_library_csv).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"scored_library_csv not found: {scored_library_csv}")

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "03_state_specific_candidates"
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if score_column not in fieldnames:
        raise ValueError(
            f"Score column '{score_column}' not found in {scored_library_csv}. "
            "Run the real conformation-bias scorer first."
        )

    retained = []
    rejected_missing_score = 0
    for row in rows:
        try:
            score = float(row[score_column])
        except Exception:
            rejected_missing_score += 1
            continue
        if float(score_min) <= score <= float(score_max):
            retained.append(row)

    output_csv = outdir / "state_specific_retained_candidates.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)

    summary = {
        "status": "completed",
        "project_name": project_name,
        "input_csv": str(input_path),
        "output_csv": str(output_csv),
        "selection_mode": "fixed_window",
        "selection_policy": "legacy_fixed_score_window",
        "score_column": score_column,
        "score_min": score_min,
        "score_max": score_max,
        "input_count": len(rows),
        "retained_count": len(retained),
        "rejected_missing_score": rejected_missing_score,
        "next_tool": "prepare_structure_prediction_job",
        "agent_instruction": (
            "Use output_csv as candidates_csv for prepare_structure_prediction_job. "
            "This selection used the explicitly requested fixed score window; do not "
            "replace it with adaptive selection unless the user changes the policy."
        ),
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "selection_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "state_specific_candidates_selected",
            "state_specific_candidates_csv": str(output_csv),
            "state_specific_candidates_summary_path": summary_path,
            "state_specific_selection_mode": "fixed_window",
            "state_specific_selection_score_min": score_min,
            "state_specific_selection_score_max": score_max,
        },
    )
    return {**summary, "summary_path": summary_path}


def _candidate_rows_from_csv(
    csv_path: str,
    top_n: Optional[int] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> Tuple[List[Dict[str, str]], List[str]]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"candidate CSV not found: {csv_path}")
    rows, fieldnames = _read_csv_rows(path)
    if sort_by and sort_by in fieldnames:
        def key_func(row: Dict[str, str]) -> Tuple[int, float]:
            value = _safe_float(row.get(sort_by))
            return (1, 0.0) if value is None else (0, value)

        rows = sorted(rows, key=key_func, reverse=not ascending)
    if top_n is not None and int(top_n) > 0:
        rows = rows[: int(top_n)]
    return rows, fieldnames


def _variant_id(row: Dict[str, Any], variant_column: str = "variant") -> str:
    value = row.get(variant_column) or row.get("mutation") or row.get("name") or row.get("id")
    if value:
        return _sanitize_name(str(value))
    pos = row.get("position")
    wt = row.get("wt_aa")
    mut = row.get("mut_aa")
    if pos and wt and mut:
        return _sanitize_name(f"{wt}{pos}{mut}")
    return _sanitize_name(f"variant_{uuid.uuid4().hex[:8]}")


def mutation_site_from_variant(variant: Any) -> str:
    """Return a stable mutation-site key for single- or multi-mutants.

    Examples:
      ``V83Y`` -> ``83``
      ``T87I`` -> ``87``
      ``D13K;Y106W`` -> ``13;106``

    The site key deliberately ignores the wild-type and mutant amino acids.
    This makes the diversity constraint apply to the biological residue site,
    rather than to a particular substitution notation.
    """
    text = str(variant or "").strip().upper()
    if not text:
        return ""
    positions = re.findall(
        r"(?<![A-Z0-9])[A-Z*](\d+)[A-Z*](?=$|[^A-Z0-9])",
        text,
    )
    if not positions:
        # Be permissive for legacy IDs that use separators not covered by the
        # bounded mutation-token pattern above.
        positions = re.findall(r"[A-Z*](\d+)[A-Z*]", text)
    unique_positions: List[str] = []
    for position in positions:
        if position not in unique_positions:
            unique_positions.append(position)
    return ";".join(unique_positions)


def _row_mutation_site(
    row: Dict[str, Any],
    *,
    variant_column: str = "variant",
) -> str:
    site = mutation_site_from_variant(row.get(variant_column) or row.get("variant"))
    if site:
        return site
    position = str(row.get("position") or "").strip()
    return position


def select_rows_with_site_diversity(
    rows: List[Dict[str, Any]],
    *,
    target_count: int,
    max_per_mutation_site: int,
    preferred_variants: Optional[List[str]] = None,
    variant_column: str = "variant",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select rows with site coverage first and a hard per-site cap.

    The first pass takes one high-priority row from every available site.
    Subsequent passes take the next row from each site in round-robin order
    until the requested count is reached. This is intentionally different
    from a global top-N sort: a single mutation site cannot monopolize the
    downstream BioEmu budget.

    ``preferred_variants`` is used by checkpoint repair to preserve the
    Biomni-selected variants whenever they do not violate the site cap. Rows
    added from the broader valid pool are reported as diversity backfill.
    """
    if int(max_per_mutation_site) <= 0:
        raise ValueError("max_per_mutation_site must be positive.")

    target = max(0, min(int(target_count), len(rows)))
    preferred = [str(item).strip() for item in (preferred_variants or []) if str(item).strip()]
    preferred_order = {variant: index for index, variant in enumerate(preferred)}
    preferred_set = set(preferred)

    site_queues: Dict[str, List[Dict[str, Any]]] = {}
    seen_variants: set[str] = set()
    for row_index, row in enumerate(rows):
        variant = _variant_id(row, variant_column)
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        site = _row_mutation_site(row, variant_column=variant_column) or f"variant:{variant}"
        row["mutation_site"] = site
        site_queues.setdefault(site, []).append(
            {
                "_row": row,
                "_row_index": row_index,
                "_variant": variant,
            }
        )

    quality_fields = (
        "selection_score",
        "structure_confidence_score",
        "structure_prediction_confidence_score",
        "structure_target_similarity_score",
        "inverse_folding_compatibility_score",
    )

    def quality_value(row: Dict[str, Any]) -> Optional[float]:
        for field in quality_fields:
            value = _safe_float(row.get(field))
            if value is not None and math.isfinite(value):
                return value
        return None

    def bias_value(row: Dict[str, Any]) -> float:
        value = _safe_float(row.get("bias_score_abs"))
        return value if value is not None and math.isfinite(value) else float("inf")

    def quality_sort_value(row: Dict[str, Any]) -> float:
        value = quality_value(row)
        return -value if value is not None else float("inf")

    for queue in site_queues.values():
        queue.sort(
            key=lambda item: (
                0 if item["_variant"] in preferred_set else 1,
                preferred_order.get(item["_variant"], len(preferred_order)),
                0 if quality_value(item["_row"]) is not None else 1,
                quality_sort_value(item["_row"]),
                bias_value(item["_row"]),
                item["_row_index"],
            )
        )

    # Preserve the broader candidate pool's quality/order when deciding which
    # sites get the first coverage pass.
    site_order = list(site_queues)
    selected: List[Dict[str, Any]] = []
    selected_variants: set[str] = set()
    for round_index in range(int(max_per_mutation_site)):
        for site in site_order:
            queue = site_queues[site]
            if round_index >= len(queue) or len(selected) >= target:
                continue
            item = queue[round_index]
            variant = item["_variant"]
            if variant in selected_variants:
                continue
            selected_variants.add(variant)
            row = item["_row"]
            row["mutation_site"] = site
            row["site_diversity_round"] = str(round_index + 1)
            selected.append(row)
        if len(selected) >= target:
            break

    before_counts = Counter(
        str(row.get("mutation_site") or _row_mutation_site(row, variant_column=variant_column) or "unknown")
        for row in rows
    )
    after_counts = Counter(
        str(row.get("mutation_site") or _row_mutation_site(row, variant_column=variant_column) or "unknown")
        for row in selected
    )
    selected_variant_list = [_variant_id(row, variant_column) for row in selected]
    backfill_variants = [
        variant for variant in selected_variant_list
        if variant not in preferred_set
    ]
    replaced_variants = [
        variant for variant in preferred
        if variant not in selected_variants
    ]
    info = {
        "policy": "site_round_robin_then_rank",
        "target_count": target,
        "max_per_mutation_site": int(max_per_mutation_site),
        "input_variant_count": len(rows),
        "input_site_count": len(before_counts),
        "selected_variant_count": len(selected),
        "selected_site_count": len(after_counts),
        "site_counts_before": dict(sorted(before_counts.items())),
        "site_counts_after": dict(sorted(after_counts.items())),
        "maximum_selected_site_count": max(after_counts.values()) if after_counts else 0,
        "preferred_variant_count": len(preferred),
        "selected_variants": selected_variant_list,
        "diversity_backfill_variants": backfill_variants,
        "diversity_backfill_count": len(backfill_variants),
        "replaced_preferred_variants": replaced_variants,
        "replaced_preferred_count": len(replaced_variants),
        "site_cap_limited": len(selected) < target,
    }
    return selected, info


def _write_candidate_fasta(
    rows: List[Dict[str, str]],
    fasta_path: Path,
    sequence_column: str = "sequence",
    variant_column: str = "variant",
) -> str:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            seq = str(row.get(sequence_column) or "").strip().upper()
            if not seq:
                continue
            handle.write(f">{_variant_id(row, variant_column)}\n{seq}\n")
    return str(fasta_path)


def _write_shell_script(path: Path, lines: List[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass
    return str(path)


def _conda_run_prefix(conda_env: str) -> List[str]:
    env = str(conda_env or "").strip()
    if env.lower() in {"", "current", "active", "none"}:
        return [
            "CONDA_RUN=()",
        ]
    return [
        f"CONDA_ENV={_quote_bash(env)}",
        "if command -v conda >/dev/null 2>&1; then",
        "  CONDA_RUN=(conda run --no-capture-output -n \"$CONDA_ENV\")",
        "else",
        "  CONDA_RUN=()",
        "fi",
    ]


def _split_codes(value: Optional[str]) -> List[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [parsed]
    except Exception:
        items = re.split(r"[,;\s]+", text)
    codes: List[str] = []
    for item in items:
        code = str(item).strip().upper()
        if code and code not in codes:
            codes.append(code)
    return codes


def _parse_ligand_smiles(value: Optional[str]) -> List[Dict[str, str]]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = [parsed]
        else:
            records = [parsed]
    except Exception:
        records = [item for item in re.split(r"[,;]\s*", text) if item.strip()]

    ligands: List[Dict[str, str]] = []
    for idx, record in enumerate(records, start=1):
        if isinstance(record, dict):
            smiles = str(record.get("smiles") or "").strip()
            ligand_id = str(record.get("id") or record.get("name") or f"SM{idx}").strip()
        else:
            item = str(record).strip()
            if ":" in item:
                ligand_id, smiles = [part.strip() for part in item.split(":", 1)]
            else:
                ligand_id, smiles = f"SM{idx}", item
        if smiles:
            ligands.append({"id": _sanitize_name(ligand_id), "smiles": smiles})
    return ligands


def _parse_ligand_payload_json(value: Optional[str]) -> List[Dict[str, Any]]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("ligand_payload_json must be a JSON object or list of objects.")


def _infer_state_structures_from_manifest(project_name: str) -> Tuple[str, str]:
    manifest = _load_project_manifest(project_name)
    target = str(
        manifest.get("target_state_structure")
        or manifest.get("suggested_target_structure")
        or ""
    )
    reference = str(
        manifest.get("reference_state_structure")
        or manifest.get("suggested_reference_structure")
        or ""
    )

    cb_manifest_path = manifest.get("cb_job_manifest_path")
    if (not target or not reference) and cb_manifest_path:
        cb_manifest = _read_json(Path(str(cb_manifest_path)).expanduser())
        for key in ("target_state_structure", "target_structure"):
            if not target and cb_manifest.get(key):
                target = str(cb_manifest.get(key))
        for key in ("reference_state_structure", "reference_structure"):
            if not reference and cb_manifest.get(key):
                reference = str(cb_manifest.get(key))
        state_records = cb_manifest.get("state_records") or cb_manifest.get("states")
        if isinstance(state_records, list):
            paths = []
            for record in state_records:
                if isinstance(record, dict):
                    path = record.get("input_pdb") or record.get("scoring_pdb") or record.get("path")
                    if path:
                        paths.append(str(path))
            if paths and not target:
                target = paths[0]
            if len(paths) > 1 and not reference:
                reference = paths[1]
    return target, reference


def _structure_ligand_context(
    *,
    project_name: str,
    target_state_structure: Optional[str],
    reference_state_structure: Optional[str],
    ligand_codes: Optional[str],
    ligand_smiles: Optional[str],
    ligand_payload_json: Optional[str],
    cofactor_required: bool,
) -> Dict[str, Any]:
    target = str(target_state_structure or "").strip()
    reference = str(reference_state_structure or "").strip()
    if not target or not reference:
        manifest_target, manifest_reference = _infer_state_structures_from_manifest(project_name)
        target = target or manifest_target
        reference = reference or manifest_reference

    features: Dict[str, Any] = {}
    detected_codes: List[str] = []
    for label, path in (("target", target), ("reference", reference)):
        if path:
            info = _inspect_pdb_file_basic(path)
            features[label] = info
            for code in (info.get("nonwater_hetero_counts") or {}).keys():
                code_clean = str(code).upper()
                if code_clean not in WATER_HETERO_CODES and code_clean not in detected_codes:
                    detected_codes.append(code_clean)

    explicit_codes = _split_codes(ligand_codes)
    smiles_ligands = _parse_ligand_smiles(ligand_smiles)
    custom_ligands = _parse_ligand_payload_json(ligand_payload_json)
    explicit_ligand_required = bool(explicit_codes or smiles_ligands or custom_ligands or cofactor_required)
    detected_ligand_required = bool(detected_codes)

    ligand_required = explicit_ligand_required or detected_ligand_required
    return {
        "target_state_structure": target,
        "reference_state_structure": reference,
        "features": features,
        "detected_ligand_codes": detected_codes,
        "explicit_ligand_codes": explicit_codes,
        "smiles_ligand_count": len(smiles_ligands),
        "custom_ligand_count": len(custom_ligands),
        "cofactor_required": bool(cofactor_required),
        "ligand_required": ligand_required,
    }


def _normalize_structure_engine(engine: str) -> str:
    cleaned = (engine or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "colabfold": "colabfold",
        "af2": "colabfold",
        "alphafold2": "colabfold",
        "boltz": "boltz2_nim",
        "boltz2": "boltz2_nim",
        "boltz_2": "boltz2_nim",
        "boltz2_nim": "boltz2_nim",
        "boltz_2_nim": "boltz2_nim",
        "nim": "boltz2_nim",
        "boltz2_local": "boltz2_local",
        "boltz_2_local": "boltz2_local",
        "boltz_local": "boltz2_local",
        "local_boltz2": "boltz2_local",
        "boltz2_cli": "boltz2_local",
        "local_cli": "boltz2_local",
    }
    if cleaned not in aliases:
        raise ValueError(
            "Supported structure prediction engines: auto, colabfold/af2, boltz2_nim, boltz2_local."
        )
    return aliases[cleaned]


def _select_structure_engine_decision(
    *,
    project_name: str,
    engine: str,
    target_state_structure: Optional[str],
    reference_state_structure: Optional[str],
    ligand_codes: Optional[str],
    ligand_smiles: Optional[str],
    ligand_payload_json: Optional[str],
    cofactor_required: bool,
) -> Dict[str, Any]:
    requested = _normalize_structure_engine(engine)
    context = _structure_ligand_context(
        project_name=project_name,
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        ligand_codes=ligand_codes,
        ligand_smiles=ligand_smiles,
        ligand_payload_json=ligand_payload_json,
        cofactor_required=cofactor_required,
    )
    if requested == "auto":
        config = _load_default_config()
        structure_cfg = config.get("structure_prediction", {}) if isinstance(config.get("structure_prediction", {}), dict) else {}
        env_engine = str(os.environ.get("SWITCHTUNE_STRUCTURE_ENGINE") or "").strip()
        env_normalized = _normalize_structure_engine(env_engine) if env_engine else "auto"
        if env_normalized != "auto":
            recommended = env_normalized
        else:
            ligand_engine = _normalize_structure_engine(
                str(structure_cfg.get("ligand_or_cofactor_engine") or "boltz2_local")
            )
            ligand_free_engine = _normalize_structure_engine(
                str(structure_cfg.get("ligand_free_engine") or "boltz2_local")
            )
            recommended = ligand_engine if context["ligand_required"] else ligand_free_engine
    else:
        recommended = requested

    if recommended == "boltz2_local" and context["ligand_required"]:
        rationale = (
            "Ligand/cofactor/ion context detected or declared; use the locally deployed "
            "Boltz-2 CLI for ligand-aware prediction."
        )
    elif recommended == "boltz2_local":
        rationale = "The current SwitchTune configuration uses the locally deployed Boltz-2 CLI as the default structure predictor."
    elif recommended == "boltz2_nim" and context["ligand_required"]:
        rationale = "Ligand/cofactor/ion context detected or declared; use Boltz-2 NIM for ligand-aware prediction."
    elif recommended == "boltz2_nim":
        rationale = "No ligand/cofactor was detected, but the selected SwitchTune backend is Boltz-2 NIM."
    else:
        rationale = "No ligand/cofactor/ion requirement detected or user explicitly requested AF2; use ColabFold."
    warnings: List[str] = []
    if recommended == "colabfold" and context["ligand_required"]:
        warnings.append("ColabFold is ligand-agnostic here; use only for smoke tests unless intentionally overriding.")
    if recommended in {"boltz2_nim", "boltz2_local"} and context["detected_ligand_codes"] and not (
        context["explicit_ligand_codes"] or context["smiles_ligand_count"] or context["custom_ligand_count"]
    ):
        warnings.append(
            "Boltz-2 was selected from detected PDB HETATM codes. Review ligand_codes/ligand_smiles before production."
        )
    return {
        "status": "completed",
        "project_name": project_name,
        "requested_engine": requested,
        "recommended_engine": recommended,
        "ligand_required": context["ligand_required"],
        "detected_ligand_codes": context["detected_ligand_codes"],
        "explicit_ligand_codes": context["explicit_ligand_codes"],
        "target_state_structure": context["target_state_structure"],
        "reference_state_structure": context["reference_state_structure"],
        "rationale": rationale,
        "warnings": warnings,
        "features": context["features"],
        "next_tool": "prepare_structure_prediction_job",
        "agent_instruction": (
            "Use recommended_engine as engine for prepare_structure_prediction_job. "
            "If recommended_engine is boltz2_nim or boltz2_local, provide ligand_codes or ligand_smiles when known."
        ),
    }


def _boltz2_ligands(
    *,
    ligand_codes: Optional[str],
    ligand_smiles: Optional[str],
    ligand_payload_json: Optional[str],
    detected_ligand_codes: List[str],
    include_detected_ligands: bool,
) -> List[Dict[str, Any]]:
    ligands: List[Dict[str, Any]] = []
    for idx, code in enumerate(_split_codes(ligand_codes), start=1):
        ligands.append({"id": f"L{idx}", "ccd": code})
    ligands.extend(_parse_ligand_smiles(ligand_smiles))
    ligands.extend(_parse_ligand_payload_json(ligand_payload_json))

    if not ligands and include_detected_ligands:
        for idx, code in enumerate(detected_ligand_codes, start=1):
            if code not in WATER_HETERO_CODES:
                ligands.append({"id": f"L{idx}", "ccd": code})
    return ligands


# Some local Boltz-2 CCD bundles do not contain a usable preprocessed PAL
# component even though PAL is a valid RCSB chemical component.  Supplying the
# equivalent SMILES keeps the local CLI independent of that optional CCD
# entry.  The stereochemistry is the L-aspartate configuration reported by
# PubChem (CID 39981).
_LOCAL_BOLTZ_LIGAND_SMILES: Dict[str, str] = {
    "PAL": "C([C@@H](C(=O)O)NC(=O)CP(=O)(O)O)C(=O)O",
}


def _local_boltz_ligands(ligands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Use SMILES for local CCDs known to be absent/incomplete in some bundles."""
    normalized: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for ligand in ligands:
        if not isinstance(ligand, dict):
            continue
        code = str(ligand.get("ccd") or "").strip().upper()
        smiles = _LOCAL_BOLTZ_LIGAND_SMILES.get(code)
        if smiles:
            replacement = {key: value for key, value in ligand.items() if key not in {"ccd", "source_ccd"}}
            replacement["smiles"] = smiles
            replacement["source_ccd"] = code
            replacement["representation"] = "smiles"
            normalized.append(replacement)
            warnings.append(
                f"Local Boltz ligand {code} was emitted as SMILES because some CCD bundles omit a usable {code}."
            )
        else:
            normalized.append(dict(ligand))
    return normalized, warnings


def _sanitize_local_boltz_template(
    template_info: Dict[str, Any],
    *,
    template_chain_id: str,
    polymer_chain_id: str,
    cleaning_mode: str = "protein_atom_only",
) -> Tuple[Dict[str, Any], List[str]]:
    """Make a local Boltz template independent of template HETATM CCD records."""
    if not template_info.get("enabled"):
        return template_info, []
    path = Path(str(template_info.get("template_path") or "")).expanduser()
    if path.suffix.lower() != ".pdb" or str(cleaning_mode).strip().lower() != "protein_atom_only":
        return template_info, []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned, sanitize_info = _sanitize_pdb_template_for_boltz(
            text,
            requested_chain_id=template_chain_id,
            output_chain_id=polymer_chain_id,
        )
        path.write_text(cleaned, encoding="utf-8")
    except Exception as exc:
        warning = f"Local Boltz template sanitization failed; original template retained: {exc}"
        return template_info, [warning]

    updated = dict(template_info)
    updated["template_cleaning_mode"] = "protein_atom_only"
    updated["template_sanitization"] = sanitize_info
    warnings = list(updated.get("warnings") or [])
    warnings.extend(sanitize_info.get("warnings") or [])
    updated["warnings"] = warnings
    return updated, [
        "Local Boltz template was reduced to the requested standard-protein ATOM chain; HETATM ligands remain explicit YAML ligands."
    ]


def _normalize_template_mode(template_mode: Optional[str]) -> str:
    cleaned = (template_mode or "target_state").strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "no_template": "none",
        "target": "target_state",
        "target_state": "target_state",
        "template": "target_state",
    }
    if cleaned not in aliases:
        raise ValueError("template_mode must be 'target_state' or 'none'.")
    return aliases[cleaned]


def _structure_format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return "pdb"
    return "cif"


def _pdb_chain_residue_counts(text: str) -> Dict[str, int]:
    residues_by_chain: Dict[str, set[Tuple[str, str, str]]] = {}
    for line in text.splitlines():
        if line[:6].strip() != "ATOM":
            continue
        resname = line[17:20].strip().upper()
        if resname not in STANDARD_PDB_PROTEIN_RESNAMES:
            continue
        chain = line[21:22].strip() or "_"
        resseq = line[22:27].strip()
        residues_by_chain.setdefault(chain, set()).add((resname, resseq, line[26:27].strip()))
    return {chain: len(residues) for chain, residues in residues_by_chain.items()}


def _choose_pdb_template_chain(text: str, requested_chain_id: Optional[str]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    counts = _pdb_chain_residue_counts(text)
    cleaned = str(requested_chain_id or "").strip()
    if cleaned.lower() in AUTO_TEMPLATE_CHAIN_IDS:
        cleaned = ""
    if cleaned and cleaned in counts:
        return cleaned, warnings
    if cleaned and cleaned not in counts:
        warnings.append(
            f"Requested template_chain_id={cleaned!r} was not found in protein ATOM records; using the largest protein chain instead."
        )
    if not counts:
        return "", ["No standard protein ATOM records were found in the PDB template."]
    chosen = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return chosen, warnings


def _sanitize_pdb_template_for_boltz(
    text: str,
    *,
    requested_chain_id: Optional[str],
    output_chain_id: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """Return a single-model, single-chain protein PDB template for Boltz-2.

    Boltz-2 NIM separates protein polymers from ligands/cofactors in the request.
    Passing a full PDB with HETATM records as a protein structural template can
    trigger parse_boltz_schema failures, so the template is narrowed to a single
    standard-residue protein chain. NMR entries commonly contain many MODEL
    records; only the first model is retained because a template represents one
    coordinate realization. SEQRES is rebuilt because Gemmi cannot infer the
    full entity sequence from ATOM records alone.
    """
    chosen_chain, warnings = _choose_pdb_template_chain(text, requested_chain_id)
    target_chain = (str(output_chain_id or "A").strip() or "A")[:1]
    kept_lines: List[str] = []
    kept_residues: set[Tuple[str, str, str]] = set()
    residue_names: List[str] = []
    residue_keys: set[Tuple[str, str, str]] = set()
    skipped_hetero = 0
    skipped_nonstandard = 0
    skipped_other_chain = 0

    # PDB NMR entries can contain dozens of MODEL/ENDMDL blocks.  Feeding all
    # blocks into a single Boltz template duplicates every residue and can make
    # Gemmi interpret the template inconsistently.  Select the first model
    # before applying the regular ATOM/HETATM filtering below.  For ordinary
    # single-model PDB files this leaves the input behavior unchanged.
    model_count = sum(1 for raw_line in text.splitlines() if raw_line[:6].strip() == "MODEL")
    in_first_model = model_count == 0
    seen_model = 0
    selected_model: Optional[str] = None

    for raw_line in text.splitlines():
        record = raw_line[:6].strip()
        if record == "MODEL":
            seen_model += 1
            if seen_model == 1:
                in_first_model = True
                model_number = raw_line[10:14].strip()
                selected_model = model_number or "1"
            else:
                in_first_model = False
            continue
        if record == "ENDMDL":
            if seen_model == 1:
                in_first_model = False
            continue
        if not in_first_model:
            continue
        if record == "HETATM":
            skipped_hetero += 1
            continue
        if record != "ATOM":
            continue
        line = raw_line.rstrip("\n\r")
        resname = line[17:20].strip().upper()
        if resname not in STANDARD_PDB_PROTEIN_RESNAMES:
            skipped_nonstandard += 1
            continue
        chain = line[21:22].strip() or "_"
        if chosen_chain and chain != chosen_chain:
            skipped_other_chain += 1
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        if len(line) < 80:
            line = line.ljust(80)
        # Normalize altloc and output chain ID to match the requested polymer.
        line = line[:16] + " " + line[17:21] + target_chain + line[22:]
        kept_lines.append(line)
        residue_key = (line[22:26].strip(), line[26:27].strip(), line[17:20].strip())
        kept_residues.add((line[17:20].strip(), line[22:27].strip(), line[26:27].strip()))
        if residue_key not in residue_keys:
            residue_keys.add(residue_key)
            residue_names.append(residue_key[2])

    if kept_lines:
        # Gemmi does not infer an entity's full sequence from ATOM records
        # alone.  Boltz aligns the template polymer against that sequence,
        # so an ATOM-only PDB without SEQRES produces an empty sequence and
        # eventually IndexError in parse_polymer(). Rebuild canonical SEQRES
        # records for the retained chain before the coordinates.
        seqres_lines: List[str] = []
        total_residues = len(residue_names)
        for offset in range(0, total_residues, 13):
            chunk = residue_names[offset : offset + 13]
            line = f"SEQRES {offset // 13 + 1:3d} {target_chain:1s} {total_residues:4d}  "
            line += " ".join(f"{name:>3s}" for name in chunk)
            seqres_lines.append(line.ljust(80))
        kept_lines = seqres_lines + kept_lines
        kept_lines.append("TER".ljust(80))
        kept_lines.append("END".ljust(80))
    else:
        warnings.append("Boltz-2 PDB template sanitization produced no ATOM records; falling back to the original template text.")
        return text, {
            "sanitized": False,
            "chosen_chain_id": chosen_chain,
            "output_chain_id": target_chain,
            "input_model_count": model_count,
            "selected_model": selected_model,
            "model_selection_mode": "first_model_only" if model_count else "no_model_records",
            "residue_count": 0,
            "atom_count": 0,
            "skipped_hetero_atom_count": skipped_hetero,
            "skipped_nonstandard_atom_count": skipped_nonstandard,
            "skipped_other_chain_atom_count": skipped_other_chain,
            "warnings": warnings,
        }

    return "\n".join(kept_lines) + "\n", {
        "sanitized": True,
        "chosen_chain_id": chosen_chain,
        "output_chain_id": target_chain,
        "input_model_count": model_count,
        "selected_model": selected_model,
        "model_selection_mode": "first_model_only" if model_count else "no_model_records",
        "residue_count": len(kept_residues),
        "atom_count": sum(1 for line in kept_lines if line.startswith("ATOM  ")),
        "skipped_hetero_atom_count": skipped_hetero,
        "skipped_nonstandard_atom_count": skipped_nonstandard,
        "skipped_other_chain_atom_count": skipped_other_chain,
        "warnings": warnings,
    }


def _infer_pdb_element(atom_name: str, element: str) -> str:
    cleaned = str(element or "").strip()
    if cleaned:
        return cleaned.upper()
    letters = re.sub(r"[^A-Za-z]", "", str(atom_name or ""))
    if not letters:
        return "C"
    # PDB atom names can be right/left justified; for proteins this simple
    # fallback is sufficient for N/CA/C/O/S backbone and side-chain atoms.
    if letters[:2].upper() in {"CL", "BR", "FE", "MG", "MN", "ZN", "CA"}:
        return letters[:2].upper()
    return letters[0].upper()


def _cif_value(value: Any) -> str:
    text = str(value)
    if text == "":
        return "?"
    if re.search(r"\s|['\";#]", text):
        return "'" + text.replace("'", "''") + "'"
    return text


def _pdb_template_to_minimal_mmcif(pdb_text: str, *, data_name: str) -> Tuple[str, Dict[str, Any]]:
    residue_order: Dict[Tuple[str, str, str], int] = {}
    residue_names: List[str] = []
    rows: List[List[str]] = []
    atom_count = 0
    for raw_line in pdb_text.splitlines():
        if raw_line[:6].strip() != "ATOM":
            continue
        line = raw_line.rstrip("\n\r").ljust(80)
        atom_count += 1
        atom_name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "A"
        auth_seq_id = line[22:26].strip()
        icode = line[26:27].strip()
        residue_key = (chain, auth_seq_id, icode)
        if residue_key not in residue_order:
            residue_order[residue_key] = len(residue_order) + 1
            residue_names.append(resname)
        label_seq_id = residue_order[residue_key]
        element = _infer_pdb_element(atom_name, line[76:78].strip())
        try:
            x = f"{float(line[30:38]):.3f}"
            y = f"{float(line[38:46]):.3f}"
            z = f"{float(line[46:54]):.3f}"
        except ValueError:
            continue
        try:
            occupancy = f"{float(line[54:60] or 1.0):.2f}"
        except ValueError:
            occupancy = "1.00"
        try:
            bfactor = f"{float(line[60:66] or 0.0):.2f}"
        except ValueError:
            bfactor = "0.00"
        rows.append(
            [
                "ATOM",
                str(len(rows) + 1),
                element,
                atom_name,
                ".",
                resname,
                chain,
                "1",
                str(label_seq_id),
                icode or "?",
                x,
                y,
                z,
                occupancy,
                bfactor,
                "?",
                auth_seq_id or str(label_seq_id),
                resname,
                chain,
                atom_name,
                "1",
            ]
        )

    safe_data_name = _sanitize_name(data_name or "boltz_template")
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M",
    }
    sequence = "".join(three_to_one.get(name, "X") for name in residue_names)
    chains = sorted({key[0] for key in residue_order}) or ["A"]
    chain_text = ",".join(chains)
    header = [
        f"data_{safe_data_name}",
        "#",
        "loop_",
        "_entity.id",
        "_entity.type",
        "1 polymer",
        "#",
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "_entity_poly.pdbx_strand_id",
        "_entity_poly.pdbx_seq_one_letter_code",
        f"1 polypeptide(L) {chain_text} {sequence or '?'}",
        "#",
        "loop_",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        f"{chains[0]} 1",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    body = [" ".join(_cif_value(value) for value in row) for row in rows]
    return "\n".join(header + body + ["#"]) + "\n", {
        "converted": True,
        "source_format": "pdb",
        "target_format": "cif",
        "atom_count": len(rows),
        "input_atom_count": atom_count,
        "residue_count": len(residue_order),
        "data_name": safe_data_name,
    }


def _prepare_target_template(
    *,
    input_dir: Path,
    template_mode: Optional[str],
    template_structure: Optional[str],
    target_state_structure: Optional[str],
    template_name: str,
) -> Dict[str, Any]:
    mode = _normalize_template_mode(template_mode)
    if mode == "none":
        return {"template_mode": "none", "enabled": False, "warnings": []}

    source_text = str(template_structure or target_state_structure or "").strip()
    if not source_text:
        return {
            "template_mode": mode,
            "enabled": False,
            "warnings": ["template_mode=target_state but no template_structure/target_state_structure was available."],
        }
    source = Path(source_text).expanduser()
    if not source.exists():
        return {
            "template_mode": mode,
            "enabled": False,
            "source_path": str(source),
            "warnings": [f"Template structure not found: {source_text}"],
        }

    template_dir = input_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_name(template_name or source.stem or "target_state_template")
    dst = template_dir / f"{safe_name}{source.suffix.lower() or '.cif'}"
    if source.resolve() != dst.resolve():
        shutil.copy2(source, dst)

    fmt = _structure_format_from_path(dst)
    return {
        "template_mode": mode,
        "enabled": True,
        "source_path": str(source),
        "template_path": str(dst),
        "template_dir": str(template_dir),
        "template_name": safe_name,
        "template_format": fmt,
        "warnings": (
            [
                "ColabFold custom templates are most reliable as mmCIF. A PDB template was supplied; convert to mmCIF if ColabFold rejects it."
            ]
            if fmt == "pdb"
            else []
        ),
    }


def _boltz2_structural_templates(
    template_info: Dict[str, Any],
    *,
    template_chain_id: str,
    polymer_chain_id: str,
    include_chain_id: bool,
    cleaning_mode: str,
    format_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not template_info.get("enabled"):
        return [], {"enabled": False, "count": 0, "warnings": list(template_info.get("warnings") or [])}
    path = Path(str(template_info["template_path"])).expanduser()
    structure_text = path.read_text(encoding="utf-8", errors="replace")
    fmt = str(template_info.get("template_format") or _structure_format_from_path(path))
    metadata: Dict[str, Any] = {
        "enabled": True,
        "source_path": str(path),
        "format": fmt,
        "name": str(template_info.get("template_name") or path.stem)[:64],
        "warnings": list(template_info.get("warnings") or []),
    }
    cleaning_clean = str(cleaning_mode or "raw").strip().lower().replace("-", "_")
    format_clean = str(format_mode or "source").strip().lower().replace("-", "_")
    if cleaning_clean not in {"raw", "protein_atom_only"}:
        raise ValueError("boltz_template_cleaning_mode must be 'raw' or 'protein_atom_only'.")
    if format_clean not in {"source", "minimal_cif"}:
        raise ValueError("boltz_template_format_mode must be 'source' or 'minimal_cif'.")
    metadata["cleaning_mode"] = cleaning_clean
    metadata["format_mode"] = format_clean

    if fmt == "pdb" and cleaning_clean == "protein_atom_only":
        structure_text, sanitize_info = _sanitize_pdb_template_for_boltz(
            structure_text,
            requested_chain_id=template_chain_id,
            output_chain_id=polymer_chain_id,
        )
        metadata["pdb_sanitization"] = sanitize_info
        metadata["warnings"].extend(sanitize_info.get("warnings") or [])
    if fmt == "pdb" and format_clean == "minimal_cif":
        cif_text, conversion_info = _pdb_template_to_minimal_mmcif(
            structure_text,
            data_name=str(template_info.get("template_name") or path.stem),
        )
        if conversion_info.get("atom_count", 0) > 0:
            structure_text = cif_text
            fmt = "cif"
            metadata["format"] = fmt
            metadata["pdb_to_cif_conversion"] = conversion_info
        else:
            metadata["warnings"].append(
                "PDB-to-mmCIF conversion produced no atoms; sending sanitized PDB template instead."
            )

    record: Dict[str, Any] = {
        "structure": structure_text,
        "format": fmt,
        "name": str(template_info.get("template_name") or path.stem)[:64],
    }
    if include_chain_id:
        chain_id = str(template_chain_id or polymer_chain_id or "").strip()
        if chain_id.lower() not in AUTO_TEMPLATE_CHAIN_IDS and chain_id:
            record["chain_id"] = chain_id
            metadata["chain_id_field"] = chain_id
    metadata["count"] = 1
    metadata["chain_id_field_included"] = "chain_id" in record
    return [record], metadata


def _config_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "y", "on"}:
        return True
    if cleaned in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _config_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_boltz2_payloads(
    *,
    rows: List[Dict[str, str]],
    input_dir: Path,
    sequence_column: str,
    variant_column: str,
    chain_id: str,
    molecule_type: str,
    ligands: List[Dict[str, Any]],
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    step_scale: float,
    cyclic: bool,
    constraints: List[Dict[str, Any]],
    structural_templates: Optional[List[Dict[str, Any]]],
    output_format: str,
    write_full_pae: bool,
    extra_payload_json: Optional[str],
) -> Tuple[List[Dict[str, Any]], str]:
    payload_dir = input_dir / "boltz2_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    extra_payload: Dict[str, Any] = {}
    if extra_payload_json and str(extra_payload_json).strip():
        parsed = json.loads(str(extra_payload_json))
        if not isinstance(parsed, dict):
            raise ValueError("extra_payload_json must be a JSON object.")
        extra_payload = parsed

    records: List[Dict[str, Any]] = []
    for row in rows:
        variant = _variant_id(row, variant_column)
        seq = str(row.get(sequence_column) or "").strip().upper()
        payload: Dict[str, Any] = {
            "polymers": [
                {
                    "id": chain_id or "A",
                    "molecule_type": molecule_type or "protein",
                    "sequence": seq,
                    "cyclic": bool(cyclic),
                    "modifications": [],
                }
            ],
            "ligands": ligands,
            "constraints": constraints,
            "recycling_steps": int(recycling_steps),
            "sampling_steps": int(sampling_steps),
            "diffusion_samples": int(diffusion_samples),
            "step_scale": float(step_scale),
            "write_full_pae": bool(write_full_pae),
        }
        if structural_templates:
            payload["polymers"][0]["structural_templates"] = structural_templates
        if output_format:
            payload["output_format"] = output_format
        payload.update(extra_payload)
        payload_path = payload_dir / f"{variant}.json"
        _write_json(payload_path, payload)
        records.append(
            {
                "variant": variant,
                "sequence": seq,
                "payload_path": str(payload_path),
                "source_row": row,
            }
        )
    return records, str(payload_dir)


def _write_boltz2_local_inputs(
    *,
    rows: List[Dict[str, str]],
    input_dir: Path,
    sequence_column: str,
    variant_column: str,
    chain_id: str,
    ligands: List[Dict[str, Any]],
    template_info: Dict[str, Any],
    template_chain_id: str,
    constraints: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """Write Boltz CLI YAML inputs while keeping template/ligand semantics explicit."""
    yaml_dir = input_dir / "boltz2_local_inputs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    protein_chain = str(chain_id or "A").strip() or "A"
    requested_template_chain = str(template_chain_id or "").strip()
    if requested_template_chain.lower() in AUTO_TEMPLATE_CHAIN_IDS:
        requested_template_chain = protein_chain

    for row in rows:
        variant = _variant_id(row, variant_column)
        sequence = str(row.get(sequence_column) or "").strip().upper()
        input_data: Dict[str, Any] = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": protein_chain,
                        "sequence": sequence,
                        "msa": "empty",
                    }
                }
            ],
        }
        for ligand in ligands:
            if not isinstance(ligand, dict):
                continue
            ligand_id = str(ligand.get("id") or f"L{len(input_data['sequences'])}").strip()
            ligand_record: Dict[str, Any] = {"id": ligand_id}
            if ligand.get("ccd"):
                ligand_record["ccd"] = str(ligand["ccd"])
            elif ligand.get("smiles"):
                ligand_record["smiles"] = str(ligand["smiles"])
            if len(ligand_record) > 1:
                input_data["sequences"].append({"ligand": ligand_record})

        if template_info.get("enabled"):
            input_data["templates"] = [
                {
                    "pdb": str(template_info["template_path"]),
                    "chain_id": requested_template_chain,
                    "template_id": f"{protein_chain}1",
                }
            ]
        if constraints:
            input_data["constraints"] = constraints

        yaml_path = yaml_dir / f"{variant}.yaml"
        if yaml is not None:
            yaml_path.write_text(
                yaml.safe_dump(input_data, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
        else:
            # JSON is a valid YAML 1.2 document and keeps this fallback dependency-free.
            yaml_path.write_text(json.dumps(input_data, indent=2, ensure_ascii=True), encoding="utf-8")
        records.append(
            {
                "variant": variant,
                "sequence": sequence,
                "yaml_path": str(yaml_path),
                "source_row": row,
            }
        )
    return records, str(yaml_dir)


def _resolve_structure_job_manifest(outdir: Path, job_manifest_path: Optional[str]) -> Tuple[Dict[str, Any], str]:
    manifest_candidates: List[Path] = []
    if job_manifest_path:
        manifest_candidates.append(Path(job_manifest_path).expanduser())
    manifest_candidates.append(outdir / "structure_prediction_job_manifest.json")
    for candidate in manifest_candidates:
        if candidate.exists():
            return _read_json(candidate), str(candidate)
    return {}, ""


def _resolve_prediction_summary_csv(
    *,
    outdir: Path,
    pred_dir: Path,
    prediction_summary_csv: Optional[str],
    job_manifest: Dict[str, Any],
) -> str:
    candidates: List[Path] = []
    if prediction_summary_csv:
        candidates.append(Path(prediction_summary_csv).expanduser())
    boltz = job_manifest.get("boltz2")
    if isinstance(boltz, dict) and boltz.get("summary_csv"):
        candidates.append(Path(str(boltz["summary_csv"])).expanduser())
    if job_manifest.get("prediction_summary_csv"):
        candidates.append(Path(str(job_manifest["prediction_summary_csv"])).expanduser())
    candidates.extend(
        [
            pred_dir.parent / "boltz2_nim_prediction_summary.csv",
            outdir / "boltz2_nim_prediction_summary.csv",
            pred_dir.parent / "boltz2_local_prediction_summary.csv",
            outdir / "boltz2_local_prediction_summary.csv",
            pred_dir.parent / "structure_prediction_summary.csv",
            outdir / "structure_prediction_summary.csv",
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)
    return ""


def _candidate_map_variants(candidate_map_path: Optional[str]) -> List[str]:
    if not candidate_map_path:
        return []
    candidate_map = _read_json(Path(candidate_map_path).expanduser())
    return [str(item.get("variant")) for item in candidate_map.get("candidates", []) if item.get("variant")]


def _payload_paths_by_variant(job_manifest: Dict[str, Any]) -> Dict[str, str]:
    boltz = job_manifest.get("boltz2")
    if not isinstance(boltz, dict):
        return {}
    records = boltz.get("payload_records") or boltz.get("input_yaml_records") or []
    out: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        variant = str(record.get("variant") or "").strip()
        payload_path = str(
            record.get("payload_path")
            or record.get("yaml_path")
            or record.get("input_yaml_path")
            or ""
        ).strip()
        if variant and payload_path:
            out[_sanitize_name(variant)] = payload_path
    return out


def _summarize_structure_prediction_payload(payload_path: str) -> Dict[str, Any]:
    path = Path(payload_path).expanduser()
    if not path.exists():
        return {"structure_prediction_payload_json": str(path), "structure_prediction_payload_found": False}
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = _load_yaml(str(path))
            if not payload:
                # The local job writer intentionally falls back to JSON, which
                # is valid YAML 1.2 and keeps preparation usable without PyYAML.
                payload = _read_json(path)
        else:
            payload = _read_json(path)
    except Exception as exc:
        return {
            "structure_prediction_payload_json": str(path),
            "structure_prediction_payload_found": False,
            "structure_prediction_payload_error": str(exc),
        }

    polymers = payload.get("polymers") if isinstance(payload.get("polymers"), list) else []
    sequences = payload.get("sequences") if isinstance(payload.get("sequences"), list) else []
    template_count = 0
    template_formats: List[str] = []
    template_names: List[str] = []
    template_chars = 0
    for polymer in polymers:
        if not isinstance(polymer, dict):
            continue
        templates = polymer.get("structural_templates") if isinstance(polymer.get("structural_templates"), list) else []
        template_count += len(templates)
        for template in templates:
            if not isinstance(template, dict):
                continue
            fmt = str(template.get("format") or "").strip()
            name = str(template.get("name") or "").strip()
            structure = str(template.get("structure") or "")
            if fmt:
                template_formats.append(fmt)
            if name:
                template_names.append(name)
            template_chars += len(structure)
    local_templates = payload.get("templates") if isinstance(payload.get("templates"), list) else []
    template_count += len(local_templates)
    for template in local_templates:
        if not isinstance(template, dict):
            continue
        template_path = str(template.get("pdb") or template.get("cif") or "").strip()
        if template_path:
            template_formats.append(Path(template_path).suffix.lower().lstrip(".") or "pdb")
            template_names.append(str(template.get("template_id") or Path(template_path).stem))

    ligands = payload.get("ligands") if isinstance(payload.get("ligands"), list) else []
    for item in sequences:
        if not isinstance(item, dict):
            continue
        ligand = item.get("ligand")
        if isinstance(ligand, dict):
            ligands.append(ligand)
    ligand_labels: List[str] = []
    for ligand in ligands:
        if not isinstance(ligand, dict):
            continue
        label = str(ligand.get("ccd") or ligand.get("smiles") or ligand.get("id") or "").strip()
        if label:
            ligand_labels.append(label)

    return {
        "structure_prediction_payload_json": str(path),
        "structure_prediction_payload_found": True,
        "structure_prediction_has_template": bool(template_count),
        "structure_prediction_template_count": template_count,
        "structure_prediction_template_formats": ";".join(sorted(set(template_formats))),
        "structure_prediction_template_names": ";".join(sorted(set(template_names))),
        "structure_prediction_template_structure_chars": template_chars,
        "structure_prediction_ligand_count": len(ligands),
        "structure_prediction_ligands": ";".join(ligand_labels),
        "structure_prediction_diffusion_samples": payload.get("diffusion_samples"),
        "structure_prediction_sampling_steps": payload.get("sampling_steps"),
        "structure_prediction_recycling_steps": payload.get("recycling_steps"),
        "structure_prediction_step_scale": payload.get("step_scale"),
        "structure_prediction_write_full_pae": payload.get("write_full_pae"),
        "structure_prediction_msa_mode": (
            "empty"
            if any(
                isinstance(item, dict)
                and isinstance(item.get("protein"), dict)
                and str(item["protein"].get("msa") or "").lower() == "empty"
                for item in sequences
            )
            else ""
        ),
    }


def _merge_nonempty_structure_metadata(
    target: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """Fill missing structure metadata without overwriting runner evidence.

    The local Boltz-2 runner records execution flags such as
    ``structure_prediction_write_full_pae`` in its summary CSV.  The local
    YAML input does not contain those CLI-only flags, so its payload summary
    may legitimately return ``None``.  Summary values are authoritative;
    payload metadata only fills fields that are absent or empty.
    """
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        if target.get(key) is None or target.get(key) == "":
            target[key] = value


def _structure_prediction_evidence_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence_rows: List[Dict[str, Any]] = []
    for row in rows:
        variant = _variant_id(row)
        status = str(row.get("prediction_status") or row.get("status") or "").strip()
        structure_path = str(row.get("structure_path") or "").strip()
        confidence = _safe_float(
            row.get("structure_prediction_confidence_score")
            or row.get("prediction_confidence_score")
            or row.get("confidence_score")
            or row.get("confidence")
        )
        completed = bool(structure_path) and Path(structure_path).expanduser().exists()
        if status and not (
            status in {"completed", "skipped_existing"}
            or status.startswith("completed_")
        ):
            completed = False
        required_pae = _is_true_marker(
            row.get("required_pae")
            or row.get("structure_prediction_required_pae")
        )
        pae_usable_value = row.get("pae_matrix_usable")
        if pae_usable_value in {"", None}:
            pae_usable_value = row.get("structure_prediction_pae_matrix_usable")
        pae_usable = _is_true_marker(pae_usable_value)
        if required_pae and not pae_usable:
            completed = False
        out: Dict[str, Any] = {
            "variant": variant,
            "structure_prediction_status": status or ("completed" if completed else "missing"),
            "structure_prediction_complete": completed,
            "structure_prediction_valid": completed,
            "structure_prediction_required_pae": required_pae,
            "structure_prediction_pae_matrix_usable": pae_usable if required_pae or pae_usable_value not in {"", None} else "",
            "structure_prediction_structure_path": structure_path,
            "structure_prediction_structure_format": row.get("structure_format") or "",
            "structure_prediction_confidence_score": confidence,
            "structure_prediction_http_status": row.get("prediction_http_status") or row.get("http_status") or "",
            "structure_prediction_error": row.get("prediction_error") or row.get("error") or "",
        }
        # Preserve local CLI diagnostics and any controlled fallback decision
        # in the evidence table; otherwise a failed primary attempt is reduced
        # to the unhelpful ``failed_missing_structure`` label.
        for source_key, target_key in {
            "attempt_mode": "structure_prediction_attempt_mode",
            "attempted_input_yaml": "structure_prediction_attempted_input_yaml",
            "failure_class": "structure_prediction_failure_class",
            "fallback_used": "structure_prediction_fallback_used",
            "fallback_warning": "structure_prediction_fallback_warning",
            "return_code": "structure_prediction_cli_return_code",
            "log_path": "structure_prediction_cli_log_path",
            "log_tail": "structure_prediction_cli_log_tail",
            "cli_error_visible": "structure_prediction_cli_error_visible",
        }.items():
            if source_key in row:
                out[target_key] = row.get(source_key)
        for key, value in row.items():
            if key.startswith("structure_prediction_") and key not in out:
                out[key] = value
            elif key.startswith("metric_"):
                out[f"structure_prediction_{key}"] = value
        evidence_rows.append(out)
    return evidence_rows


def _first_list_value(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _nested_shape(value: Any) -> List[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value)] + _nested_shape(value[0])
    return []


def _numeric_matrix_from_response(value: Any) -> Optional[List[List[float]]]:
    """Return a 2D numeric matrix from Boltz-style [sample, n, n] or [n, n]."""
    if value is None:
        return None
    matrix_like = value
    if (
        isinstance(matrix_like, list)
        and matrix_like
        and isinstance(matrix_like[0], list)
        and matrix_like[0]
        and isinstance(matrix_like[0][0], list)
    ):
        matrix_like = matrix_like[0]
    if not isinstance(matrix_like, list) or not matrix_like:
        return None
    matrix: List[List[float]] = []
    for row in matrix_like:
        if not isinstance(row, list) or not row:
            return None
        numeric_row: List[float] = []
        for item in row:
            value_float = _safe_float(item)
            if value_float is None or not math.isfinite(value_float):
                return None
            numeric_row.append(value_float)
        matrix.append(numeric_row)
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return None
    return matrix


def _resolve_project_output_path(path_text: str, project_name: str) -> Path:
    """Resolve copied server paths when a run directory was mirrored locally."""
    path = Path(str(path_text or "")).expanduser()
    if path.exists():
        return path
    normalized = str(path_text or "").replace("\\", "/")
    marker = f"/{_sanitize_name(project_name)}/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        candidate = _project_dir(project_name) / rel
        if candidate.exists():
            return candidate
    return path


def _structure_response_summary(response_path: Path) -> Dict[str, Any]:
    if not response_path.exists():
        return {
            "structure_confidence_response_json": str(response_path),
            "structure_confidence_response_found": False,
        }
    try:
        response = _read_json(response_path)
    except Exception as exc:
        return {
            "structure_confidence_response_json": str(response_path),
            "structure_confidence_response_found": False,
            "structure_confidence_response_error": str(exc),
        }

    pae = response.get("pae")
    pde = response.get("pde")
    pae_matrix = _numeric_matrix_from_response(pae)
    pde_matrix = _numeric_matrix_from_response(pde)
    return {
        "structure_confidence_response_json": str(response_path),
        "structure_confidence_response_found": True,
        "structure_confidence_response_top_keys": ";".join(sorted(response.keys())),
        "structure_confidence_confidence_score": _first_list_value(response.get("confidence_scores")),
        "structure_confidence_ptm_score": _first_list_value(response.get("ptm_scores")),
        "structure_confidence_iptm_score": _first_list_value(response.get("iptm_scores")),
        "structure_confidence_ligand_iptm_score": _first_list_value(response.get("ligand_iptm_scores")),
        "structure_confidence_protein_iptm_score": _first_list_value(response.get("protein_iptm_scores")),
        "structure_confidence_complex_plddt_score_0_1": _first_list_value(response.get("complex_plddt_scores")),
        "structure_confidence_complex_iplddt_score_0_1": _first_list_value(response.get("complex_iplddt_scores")),
        "structure_confidence_complex_pde_score": _first_list_value(response.get("complex_pde_scores")),
        "structure_confidence_complex_ipde_score": _first_list_value(response.get("complex_ipde_scores")),
        "structure_confidence_pae_present": pae is not None,
        "structure_confidence_pae_shape": json.dumps(_nested_shape(pae), ensure_ascii=True) if pae is not None else "",
        "structure_confidence_pae_matrix_usable": pae_matrix is not None,
        "structure_confidence_pae_matrix_shape": (
            json.dumps(_nested_shape(pae_matrix), ensure_ascii=True) if pae_matrix is not None else ""
        ),
        "structure_confidence_pde_present": pde is not None,
        "structure_confidence_pde_shape": json.dumps(_nested_shape(pde), ensure_ascii=True) if pde is not None else "",
        "structure_confidence_pde_matrix_usable": pde_matrix is not None,
        "structure_confidence_metric_total_time_seconds": (response.get("metrics") or {}).get("total_time_seconds")
        if isinstance(response.get("metrics"), dict)
        else None,
        "structure_confidence_metric_model_inference_time_seconds": (response.get("metrics") or {}).get(
            "model_inference_time_seconds"
        )
        if isinstance(response.get("metrics"), dict)
        else None,
    }


def _extract_structure_response_matrix(response_path: Path, key: str) -> Optional[List[List[float]]]:
    if not response_path.exists():
        return None
    try:
        response = _read_json(response_path)
    except Exception:
        return None
    return _numeric_matrix_from_response(response.get(key))


def _parse_structure_b_factor_confidence(
    structure_path: Path,
    *,
    chain_id: str,
) -> Dict[str, Any]:
    """Parse CIF/PDB B-factors as atom/residue-level confidence values."""
    if not structure_path.exists():
        return {
            "structure_confidence_structure_path": str(structure_path),
            "structure_confidence_structure_found": False,
        }

    selected_chains = _chain_selector_set(chain_id)
    atom_values: List[float] = []
    residue_values: Dict[int, List[float]] = {}

    if structure_path.suffix.lower() in {".cif", ".mmcif"}:
        lines = structure_path.read_text(encoding="utf-8", errors="replace").splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() != "loop_":
                i += 1
                continue
            i += 1
            columns: List[str] = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                columns.append(lines[i].strip())
                i += 1
            if not columns or not any(column.startswith("_atom_site.") for column in columns):
                while i < len(lines) and lines[i].strip() not in {"#", "loop_"}:
                    i += 1
                continue

            chain_i = _cif_column_index(columns, "_atom_site.label_asym_id", "_atom_site.auth_asym_id")
            seq_i = _cif_column_index(columns, "_atom_site.label_seq_id", "_atom_site.auth_seq_id")
            b_i = _cif_column_index(columns, "_atom_site.B_iso_or_equiv")
            group_i = _cif_column_index(columns, "_atom_site.group_PDB")
            if seq_i is None or b_i is None:
                break

            while i < len(lines):
                raw = lines[i].strip()
                if not raw or raw == "#":
                    i += 1
                    break
                if raw == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                    break
                tokens = _cif_tokens(raw)
                if len(tokens) < len(columns):
                    i += 1
                    continue
                group = tokens[group_i] if group_i is not None and len(tokens) > group_i else "ATOM"
                if not str(group).upper().startswith("ATOM"):
                    i += 1
                    continue
                parsed_chain = tokens[chain_i] if chain_i is not None and len(tokens) > chain_i else ""
                if selected_chains and parsed_chain not in selected_chains:
                    i += 1
                    continue
                residue_number = _parse_int_like(tokens[seq_i], -1)
                b_value = _parse_float_like(tokens[b_i], float("nan"))
                if residue_number > 0 and math.isfinite(b_value):
                    atom_values.append(b_value)
                    residue_values.setdefault(residue_number, []).append(b_value)
                i += 1
            break
    elif structure_path.suffix.lower() == ".pdb":
        for raw in structure_path.read_text(encoding="utf-8", errors="replace").splitlines():
            record = raw[:6].strip()
            if record != "ATOM":
                continue
            parsed_chain = raw[21:22].strip()
            if selected_chains and parsed_chain not in selected_chains:
                continue
            residue_number = _parse_int_like(raw[22:26].strip(), -1)
            b_value = _parse_float_like(raw[60:66].strip(), float("nan")) if len(raw) >= 66 else float("nan")
            if residue_number > 0 and math.isfinite(b_value):
                atom_values.append(b_value)
                residue_values.setdefault(residue_number, []).append(b_value)

    raw_residue_means = {
        residue_number: mean(values)
        for residue_number, values in residue_values.items()
        if values
    }
    raw_atom_values = list(atom_values)
    raw_atom_min = min(raw_atom_values) if raw_atom_values else None
    raw_atom_max = max(raw_atom_values) if raw_atom_values else None
    if raw_atom_values and all(0.0 <= value <= 1.0 for value in raw_atom_values):
        plddt_scale = "normalized_0_1_to_0_100"
        atom_values = [value * 100.0 for value in raw_atom_values]
        residue_means = {
            residue_number: value * 100.0 for residue_number, value in raw_residue_means.items()
        }
    elif raw_atom_values and all(0.0 <= value <= 100.0 for value in raw_atom_values):
        plddt_scale = "native_0_100"
        residue_means = raw_residue_means
    elif raw_atom_values:
        plddt_scale = "unrecognized"
        residue_means = raw_residue_means
    else:
        plddt_scale = "unavailable"
        residue_means = {}
    return {
        "structure_confidence_structure_path": str(structure_path),
        "structure_confidence_structure_found": True,
        "structure_confidence_plddt_scale": plddt_scale,
        "structure_confidence_atom_plddt_raw_min": raw_atom_min,
        "structure_confidence_atom_plddt_raw_max": raw_atom_max,
        "structure_confidence_atom_plddt_count": len(atom_values),
        "structure_confidence_atom_plddt_min": min(atom_values) if atom_values else None,
        "structure_confidence_atom_plddt_mean": mean(atom_values) if atom_values else None,
        "structure_confidence_atom_plddt_median": median(atom_values) if atom_values else None,
        "structure_confidence_atom_plddt_max": max(atom_values) if atom_values else None,
        "structure_confidence_residue_count": len(residue_means),
        "_residue_plddt_means": residue_means,
        "_residue_order": sorted(residue_means),
    }


def _candidate_mutation_position(row: Dict[str, Any], variant: str) -> Optional[int]:
    for key in ("sequence_index_1based", "position", "mutation_position", "residue_position"):
        value = _safe_int(row.get(key))
        if value is not None and value > 0:
            return value
    match = re.search(r"(\d+)", str(variant or ""))
    return int(match.group(1)) if match else None


def _mean_matrix_values(matrix: List[List[float]], row_indices: List[int], col_indices: List[int]) -> Optional[float]:
    values: List[float] = []
    n = len(matrix)
    for i in row_indices:
        if i < 0 or i >= n:
            continue
        for j in col_indices:
            if j < 0 or j >= n:
                continue
            value = matrix[i][j]
            if math.isfinite(value):
                values.append(value)
    return mean(values) if values else None


def _pae_region_summary(
    matrix: Optional[List[List[float]]],
    residue_order: List[int],
    mutation_position: Optional[int],
    *,
    window_radius: int,
) -> Dict[str, Any]:
    if matrix is None:
        return {
            "structure_confidence_pae_matrix_usable": False,
            "structure_confidence_mutation_mean_pae_to_protein": None,
            "structure_confidence_mutation_window_mean_pae_to_protein": None,
            "structure_confidence_mutation_window_internal_mean_pae": None,
        }
    residue_to_index = {residue: idx for idx, residue in enumerate(residue_order)}
    protein_indices = [idx for idx in range(len(residue_order)) if idx < len(matrix)]
    if mutation_position is None or mutation_position not in residue_to_index:
        return {
            "structure_confidence_pae_matrix_usable": True,
            "structure_confidence_pae_token_count": len(matrix),
            "structure_confidence_protein_token_count": len(protein_indices),
            "structure_confidence_mutation_mean_pae_to_protein": None,
            "structure_confidence_mutation_window_mean_pae_to_protein": None,
            "structure_confidence_mutation_window_internal_mean_pae": None,
        }

    mutation_index = residue_to_index[mutation_position]
    window_residues = [
        residue
        for residue in range(mutation_position - int(window_radius), mutation_position + int(window_radius) + 1)
        if residue in residue_to_index
    ]
    window_indices = [residue_to_index[residue] for residue in window_residues]
    mutation_to_protein = _mean_matrix_values(
        matrix,
        [mutation_index],
        [idx for idx in protein_indices if idx != mutation_index],
    )
    protein_to_mutation = _mean_matrix_values(
        matrix,
        [idx for idx in protein_indices if idx != mutation_index],
        [mutation_index],
    )
    directional = [
        value
        for value in [mutation_to_protein, protein_to_mutation]
        if value is not None and math.isfinite(value)
    ]
    window_to_protein = _mean_matrix_values(matrix, window_indices, protein_indices)
    window_internal = _mean_matrix_values(matrix, window_indices, window_indices)
    return {
        "structure_confidence_pae_matrix_usable": True,
        "structure_confidence_pae_token_count": len(matrix),
        "structure_confidence_protein_token_count": len(protein_indices),
        "structure_confidence_mutation_pae_token_index": mutation_index,
        "structure_confidence_mutation_window_positions": ";".join(str(item) for item in window_residues),
        "structure_confidence_mutation_mean_pae_to_protein": mean(directional) if directional else None,
        "structure_confidence_mutation_window_mean_pae_to_protein": window_to_protein,
        "structure_confidence_mutation_window_internal_mean_pae": window_internal,
    }


@mcp.tool()
def select_structure_prediction_engine(
    project_name: str,
    target_state_structure: Optional[str] = None,
    reference_state_structure: Optional[str] = None,
    ligand_codes: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    ligand_payload_json: Optional[str] = None,
    cofactor_required: bool = False,
    requested_engine: str = "auto",
) -> Dict[str, Any]:
    """Select ColabFold or a configured Boltz-2 backend for structure prediction."""
    result = _select_structure_engine_decision(
        project_name=project_name,
        engine=requested_engine,
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        ligand_codes=ligand_codes,
        ligand_smiles=ligand_smiles,
        ligand_payload_json=ligand_payload_json,
        cofactor_required=cofactor_required,
    )
    _update_project_manifest(
        project_name,
        {
            "structure_prediction_engine_decision": {
                "recommended_engine": result["recommended_engine"],
                "ligand_required": result["ligand_required"],
                "detected_ligand_codes": result["detected_ligand_codes"],
                "explicit_ligand_codes": result["explicit_ligand_codes"],
                "rationale": result["rationale"],
                "warnings": result["warnings"],
                "updated_at": _utc_now(),
            }
        },
    )
    return result


@mcp.tool()
def prepare_structure_prediction_job(
    project_name: str,
    candidates_csv: str,
    engine: str = "auto",
    sequence_column: str = "sequence",
    variant_column: str = "variant",
    top_n: Optional[int] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    output_dir: Optional[str] = None,
    num_models: int = 5,
    num_recycles: int = 3,
    extra_args: str = "",
    execute: bool = False,
    timeout_seconds: int = 0,
    target_state_structure: Optional[str] = None,
    reference_state_structure: Optional[str] = None,
    template_mode: str = "target_state",
    template_structure: Optional[str] = None,
    template_name: str = "target_state_template",
    template_chain_id: str = "auto",
    colabfold_msa_mode: str = "single_sequence",
    colabfold_use_templates: bool = True,
    colabfold_custom_template_path: Optional[str] = None,
    ligand_codes: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    ligand_payload_json: Optional[str] = None,
    cofactor_required: bool = False,
    boltz_chain_id: str = "A",
    boltz_molecule_type: str = "protein",
    boltz_recycling_steps: int = 3,
    boltz_sampling_steps: int = 20,
    boltz_diffusion_samples: int = 1,
    boltz_step_scale: float = 1.638,
    boltz_cyclic: bool = False,
    boltz_constraints_json: Optional[str] = None,
    boltz_extra_payload_json: Optional[str] = None,
    boltz_include_detected_ligands: bool = True,
    boltz_use_template: Optional[bool] = None,
    boltz_template_include_chain_id: bool = False,
    boltz_template_cleaning_mode: str = "raw",
    boltz_template_format_mode: str = "source",
    boltz_output_format: str = "",
    boltz_write_full_pae: Optional[bool] = None,
    boltz_api_endpoint: Optional[str] = None,
    boltz_api_key_env: str = "BOLTZ2_API_KEY",
    boltz_rate_limit_sleep_seconds: float = 3.0,
    boltz_max_retries: int = 5,
    boltz_request_timeout_seconds: int = 600,
    boltz_pae_missing_retries: Optional[int] = None,
    boltz_cache_dir: Optional[str] = None,
    boltz_devices: int = 1,
    boltz_accelerator: str = "gpu",
    boltz_max_parallel_samples: Optional[int] = None,
    boltz_num_workers: Optional[int] = None,
    boltz_preprocessing_threads: Optional[int] = None,
    boltz_no_kernels: Optional[bool] = None,
    boltz_write_full_pde: Optional[bool] = None,
) -> Dict[str, Any]:
    """Prepare a structure-prediction job for retained SwitchTune variants.

    With `engine="auto"`, the configured Boltz-2 backend is selected for
    ligand-free and ligand/cofactor/ion-aware projects. The tool writes inputs,
    a job manifest, and a shell script. It does not run the expensive job unless
    `execute=True` is explicitly passed.
    """
    rows, _fieldnames = _candidate_rows_from_csv(candidates_csv, top_n=top_n, sort_by=sort_by, ascending=ascending)
    engine_decision = _select_structure_engine_decision(
        project_name=project_name,
        engine=engine,
        target_state_structure=target_state_structure,
        reference_state_structure=reference_state_structure,
        ligand_codes=ligand_codes,
        ligand_smiles=ligand_smiles,
        ligand_payload_json=ligand_payload_json,
        cofactor_required=cofactor_required,
    )
    engine_clean = str(engine_decision["recommended_engine"])
    resolved_target_state_structure = str(engine_decision.get("target_state_structure") or target_state_structure or "")

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "04_structure_prediction"
    input_dir = outdir / "input"
    prediction_dir = outdir / "predictions"
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = Path(_write_candidate_fasta(rows, input_dir / "structure_prediction_input.fasta", sequence_column, variant_column))
    candidate_map = []
    for row in rows:
        candidate_map.append(
            {
                "variant": _variant_id(row, variant_column),
                "sequence": row.get(sequence_column, ""),
                "source_row": row,
            }
        )
    candidate_map_path = _write_json(input_dir / "candidate_sequence_map.json", {"candidates": candidate_map})
    template_info = _prepare_target_template(
        input_dir=input_dir,
        template_mode=template_mode,
        template_structure=template_structure,
        target_state_structure=resolved_target_state_structure,
        template_name=template_name,
    )

    if engine_clean == "colabfold":
        config = _tool_config("colabfold")
        conda_env = str(config.get("conda_env") or "colabfold")
        executable = str(config.get("executable") or "colabfold_batch")
        cmd = [
            executable,
            str(fasta_path),
            str(prediction_dir),
            "--msa-mode",
            str(colabfold_msa_mode or "single_sequence"),
            "--num-models",
            str(int(num_models)),
            "--num-recycle",
            str(int(num_recycles)),
        ]
        colabfold_template_path = ""
        if colabfold_use_templates:
            if colabfold_custom_template_path:
                colabfold_template_path = _expand_path_text(colabfold_custom_template_path)
            elif template_info.get("enabled"):
                colabfold_template_path = str(template_info["template_dir"])
        if colabfold_template_path:
            cmd.extend(["--templates", "--custom-template-path", colabfold_template_path])
        if extra_args.strip():
            cmd.append(extra_args.strip())
        command_text = " ".join(_quote_bash(part) for part in cmd)
        log_path = outdir / "colabfold_batch.log"
        script_path = outdir / "run_colabfold_structure_prediction.sh"
        _write_shell_script(
            script_path,
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                *_conda_run_prefix(conda_env),
                "",
                f"${{CONDA_RUN[@]}} {command_text} 2>&1 | tee {_quote_bash(str(log_path))}",
            ],
        )
        payload = {
            "engine": engine_clean,
            "engine_decision": engine_decision,
            "candidates_csv": str(Path(candidates_csv).expanduser()),
            "candidate_count": len(rows),
            "fasta_path": str(fasta_path),
            "candidate_map_path": candidate_map_path,
            "prediction_dir": str(prediction_dir),
            "run_script": str(script_path),
            "log_path": str(log_path),
            "conda_env": conda_env,
            "command": command_text,
            "template_info": template_info,
            "msa_mode": str(colabfold_msa_mode or "single_sequence"),
            "colabfold_custom_template_path": colabfold_template_path,
        }
        agent_instruction = "Return a final <solution> now. Do not run ColabFold inside the agent call."
    elif engine_clean == "boltz2_local":
        config = _tool_config("boltz2_local")
        conda_env = str(config.get("conda_env") or "current")
        executable = str(os.environ.get("BOLTZ_LOCAL_EXECUTABLE") or config.get("executable") or "boltz")
        runner_script = _expand_path_text(
            str(
                config.get("runner_script")
                or "${SWITCHTUNE_HOME:-.}/scripts/run_boltz2_local_batch.py"
            )
        )
        resolved_boltz_use_template = (
            _config_bool(config, "use_templates", True)
            if boltz_use_template is None
            else bool(boltz_use_template)
        )
        resolved_boltz_write_full_pae = (
            _config_bool(config, "write_full_pae", True)
            if boltz_write_full_pae is None
            else bool(boltz_write_full_pae)
        )
        resolved_boltz_write_full_pde = (
            _config_bool(config, "write_full_pde", True)
            if boltz_write_full_pde is None
            else bool(boltz_write_full_pde)
        )
        if not resolved_boltz_write_full_pae:
            raise ValueError(
                "require_pae=true is mandatory for SwitchTune structure prediction; "
                "local Boltz-2 must use write_full_pae=true."
            )
        ligands = _boltz2_ligands(
            ligand_codes=ligand_codes,
            ligand_smiles=ligand_smiles,
            ligand_payload_json=ligand_payload_json,
            detected_ligand_codes=list(engine_decision.get("detected_ligand_codes") or []),
            include_detected_ligands=bool(boltz_include_detected_ligands),
        )
        ligands, local_ligand_warnings = _local_boltz_ligands(ligands)
        local_template_info = template_info
        local_template_warnings: List[str] = []
        if resolved_boltz_use_template:
            local_template_info, local_template_warnings = _sanitize_local_boltz_template(
                template_info,
                template_chain_id=template_chain_id,
                polymer_chain_id=boltz_chain_id,
                cleaning_mode=str(config.get("template_cleaning_mode") or "protein_atom_only"),
            )
        constraints: List[Dict[str, Any]] = []
        if boltz_constraints_json and str(boltz_constraints_json).strip():
            parsed_constraints = json.loads(str(boltz_constraints_json))
            if not isinstance(parsed_constraints, list):
                raise ValueError("boltz_constraints_json must be a JSON list.")
            constraints = [item for item in parsed_constraints if isinstance(item, dict)]
        local_input_records, local_yaml_dir = _write_boltz2_local_inputs(
            rows=rows,
            input_dir=input_dir,
            sequence_column=sequence_column,
            variant_column=variant_column,
            chain_id=boltz_chain_id,
            ligands=ligands,
            template_info=local_template_info if resolved_boltz_use_template else {"enabled": False},
            template_chain_id=template_chain_id,
            constraints=constraints,
        )
        cache_dir = _expand_path_text(
            str(boltz_cache_dir or config.get("cache_dir") or "${BOLTZ_CACHE:-./models/boltz2}")
        )
        resolved_recycling_steps = int(
            _config_int(config, "default_recycling_steps", 1)
            if int(boltz_recycling_steps) == 3
            else int(boltz_recycling_steps)
        )
        resolved_sampling_steps = int(
            _config_int(config, "default_sampling_steps", int(boltz_sampling_steps))
            if int(boltz_sampling_steps) == 20
            else int(boltz_sampling_steps)
        )
        resolved_diffusion_samples = int(
            _config_int(config, "default_diffusion_samples", int(boltz_diffusion_samples))
            if int(boltz_diffusion_samples) == 1
            else int(boltz_diffusion_samples)
        )
        resolved_max_parallel_samples = int(
            boltz_max_parallel_samples
            if boltz_max_parallel_samples is not None
            else _config_int(config, "default_max_parallel_samples", 1)
        )
        resolved_num_workers = int(
            boltz_num_workers
            if boltz_num_workers is not None
            else _config_int(config, "default_num_workers", 1)
        )
        resolved_preprocessing_threads = int(
            boltz_preprocessing_threads
            if boltz_preprocessing_threads is not None
            else _config_int(config, "default_preprocessing_threads", 1)
        )
        resolved_no_kernels = (
            _config_bool(config, "no_kernels", True)
            if boltz_no_kernels is None
            else bool(boltz_no_kernels)
        )
        resolved_output_format = str(
            boltz_output_format or config.get("output_format") or "mmcif"
        )
        log_path = outdir / "boltz2_local_batch.log"
        script_path = outdir / "run_boltz2_local_structure_prediction.sh"
        command_text = (
            f"python {_quote_bash(runner_script)} "
            f"--manifest {_quote_bash(str(outdir / 'structure_prediction_job_manifest.json'))}"
        )
        _write_shell_script(
            script_path,
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                *_conda_run_prefix(conda_env),
                "",
                f'"${{CONDA_RUN[@]}}" {command_text} 2>&1 | tee {_quote_bash(str(log_path))}',
            ],
        )
        payload = {
            "engine": engine_clean,
            "engine_decision": engine_decision,
            "candidates_csv": str(Path(candidates_csv).expanduser()),
            "candidate_count": len(rows),
            "fasta_path": str(fasta_path),
            "candidate_map_path": candidate_map_path,
            "prediction_dir": str(prediction_dir),
            "run_script": str(script_path),
            "log_path": str(log_path),
            "command": command_text,
            "template_info": local_template_info,
            "boltz2": {
                "backend": "boltz2_local",
                "executable": executable,
                "conda_env": conda_env,
                "runner_script": runner_script,
                "input_yaml_dir": local_yaml_dir,
                "input_yaml_records": local_input_records,
                "payload_records": local_input_records,
                "ligands": ligands,
                "ligand_warnings": local_ligand_warnings,
                "constraints": constraints,
                "template_info": local_template_info,
                "template_warnings": local_template_warnings,
                "template_cleaning_mode": str(config.get("template_cleaning_mode") or "protein_atom_only"),
                "boltz_use_template": resolved_boltz_use_template,
                "template_chain_id": template_chain_id,
                "polymer_chain_id": boltz_chain_id,
                "cache_dir": cache_dir,
                "devices": int(boltz_devices),
                "accelerator": str(boltz_accelerator),
                "recycling_steps": resolved_recycling_steps,
                "sampling_steps": resolved_sampling_steps,
                "diffusion_samples": resolved_diffusion_samples,
                "max_parallel_samples": resolved_max_parallel_samples,
                "num_workers": resolved_num_workers,
                "preprocessing_threads": resolved_preprocessing_threads,
                "no_kernels": resolved_no_kernels,
                "output_format": resolved_output_format,
                "write_full_pae": resolved_boltz_write_full_pae,
                "write_full_pde": resolved_boltz_write_full_pde,
                "require_pae": True,
                # A failed ligand/template parse must be diagnosable and must
                # not discard an otherwise usable protein prediction. The
                # runner tries these modes only after the ligand-aware primary
                # command fails and records which mode produced the structure.
                "fallback_modes": [
                    "no_template",
                    "no_ligands_keep_template",
                    "no_ligands",
                ],
                # CPU fallback is opt-in because it can be much slower than a
                # GPU retry. The local runner enables it only when this field
                # is explicitly set in the manifest after a GPU OOM.
                "oom_cpu_fallback": False,
                "response_dir": str(outdir / "boltz2_responses"),
                "summary_csv": str(outdir / "boltz2_local_prediction_summary.csv"),
            },
        }
        agent_instruction = (
            "Return a final <solution> now. Do not run local Boltz-2 inside the agent call. "
            "Run the generated local CLI script outside the agent call, then collect the outputs."
        )
    elif engine_clean == "boltz2_nim":
        config = _tool_config("boltz2_nim")
        endpoint = str(
            boltz_api_endpoint
            or os.environ.get("BOLTZ2_ENDPOINT")
            or config.get("endpoint")
            or BOLTZ2_NIM_DEFAULT_ENDPOINT
        )
        api_key_env = str(boltz_api_key_env or config.get("api_key_env") or "BOLTZ2_API_KEY")
        resolved_boltz_use_template = (
            _config_bool(config, "use_structural_templates", True)
            if boltz_use_template is None
            else bool(boltz_use_template)
        )
        resolved_boltz_write_full_pae = (
            _config_bool(config, "write_full_pae", True)
            if boltz_write_full_pae is None
            else bool(boltz_write_full_pae)
        )
        resolved_boltz_pae_missing_retries = max(
            0,
            _config_int(config, "pae_missing_retries", 2)
            if boltz_pae_missing_retries is None
            else int(boltz_pae_missing_retries),
        )
        if not resolved_boltz_write_full_pae:
            raise ValueError(
                "require_pae=true is mandatory for SwitchTune structure prediction; "
                "boltz_write_full_pae cannot be disabled."
            )
        runner_script = _expand_path_text(
            str(
                config.get("runner_script")
                or "${SWITCHTUNE_HOME:-.}/scripts/run_boltz2_nim_batch.py"
            )
        )
        ligands = _boltz2_ligands(
            ligand_codes=ligand_codes,
            ligand_smiles=ligand_smiles,
            ligand_payload_json=ligand_payload_json,
            detected_ligand_codes=list(engine_decision.get("detected_ligand_codes") or []),
            include_detected_ligands=bool(boltz_include_detected_ligands),
        )
        constraints: List[Dict[str, Any]] = []
        if boltz_constraints_json and str(boltz_constraints_json).strip():
            parsed_constraints = json.loads(str(boltz_constraints_json))
            if not isinstance(parsed_constraints, list):
                raise ValueError("boltz_constraints_json must be a JSON list.")
            constraints = [item for item in parsed_constraints if isinstance(item, dict)]
        if resolved_boltz_use_template:
            structural_templates, structural_template_metadata = _boltz2_structural_templates(
                template_info,
                template_chain_id=template_chain_id,
                polymer_chain_id=boltz_chain_id,
                include_chain_id=bool(boltz_template_include_chain_id),
                cleaning_mode=boltz_template_cleaning_mode,
                format_mode=boltz_template_format_mode,
            )
        else:
            structural_templates, structural_template_metadata = (
                [],
                {
                    "enabled": False,
                    "count": 0,
                    "warnings": [
                        "Boltz-2 NIM structural_templates were disabled for this job because boltz_use_template=False.",
                        "Use this fallback only when the hosted NIM template path is unavailable or for ligand-aware no-template control runs.",
                    ],
                },
            )
        payload_records, payload_dir = _write_boltz2_payloads(
            rows=rows,
            input_dir=input_dir,
            sequence_column=sequence_column,
            variant_column=variant_column,
            chain_id=boltz_chain_id,
            molecule_type=boltz_molecule_type,
            ligands=ligands,
            recycling_steps=boltz_recycling_steps,
            sampling_steps=boltz_sampling_steps,
            diffusion_samples=boltz_diffusion_samples,
            step_scale=boltz_step_scale,
            cyclic=boltz_cyclic,
            constraints=constraints,
            structural_templates=structural_templates,
            output_format=boltz_output_format,
            write_full_pae=resolved_boltz_write_full_pae,
            extra_payload_json=boltz_extra_payload_json,
        )
        log_path = outdir / "boltz2_nim_batch.log"
        script_path = outdir / "run_boltz2_nim_structure_prediction.sh"
        command_text = (
            f"python {_quote_bash(runner_script)} "
            f"--manifest {_quote_bash(str(outdir / 'structure_prediction_job_manifest.json'))}"
        )
        _write_shell_script(
            script_path,
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"API_KEY_ENV={_quote_bash(api_key_env)}",
                "if [[ -z \"${!API_KEY_ENV:-}\" ]]; then",
                "  echo \"ERROR: ${API_KEY_ENV} is not set for Boltz-2 NIM API.\" >&2",
                "  exit 1",
                "fi",
                "",
                f"{command_text} 2>&1 | tee {_quote_bash(str(log_path))}",
            ],
        )
        payload = {
            "engine": engine_clean,
            "engine_decision": engine_decision,
            "candidates_csv": str(Path(candidates_csv).expanduser()),
            "candidate_count": len(rows),
            "fasta_path": str(fasta_path),
            "candidate_map_path": candidate_map_path,
            "prediction_dir": str(prediction_dir),
            "run_script": str(script_path),
            "log_path": str(log_path),
            "command": command_text,
            "boltz2": {
                "endpoint": endpoint,
                "api_key_env": api_key_env,
                "runner_script": runner_script,
                "payload_dir": payload_dir,
                "payload_records": payload_records,
                "ligands": ligands,
                "constraints": constraints,
                "template_info": template_info,
                "structural_template_count": len(structural_templates),
                "structural_template_metadata": structural_template_metadata,
                "template_chain_id": template_chain_id,
                "polymer_chain_id": boltz_chain_id,
                "boltz_use_template": resolved_boltz_use_template,
                "template_chain_id_field_included": bool(boltz_template_include_chain_id),
                "template_cleaning_mode": boltz_template_cleaning_mode,
                "template_format_mode": boltz_template_format_mode,
                "output_format": boltz_output_format,
                "write_full_pae": resolved_boltz_write_full_pae,
                "require_pae": True,
                "diffusion_samples": int(boltz_diffusion_samples),
                "step_scale": float(boltz_step_scale),
                "cyclic": bool(boltz_cyclic),
                "response_dir": str(outdir / "boltz2_responses"),
                "summary_csv": str(outdir / "boltz2_nim_prediction_summary.csv"),
                "rate_limit_sleep_seconds": float(boltz_rate_limit_sleep_seconds),
                "max_retries": int(boltz_max_retries),
                "request_timeout_seconds": int(boltz_request_timeout_seconds),
                "pae_missing_retries": resolved_boltz_pae_missing_retries,
            },
        }
        agent_instruction = (
            "Return a final <solution> now. Do not call Boltz-2 NIM inside the agent call. "
            "Boltz-2 NIM API is prepared with Build UI-compatible inputs; run the generated script outside the agent call."
        )
    else:
        raise ValueError(f"Unsupported structure prediction engine after normalization: {engine_clean}")

    manifest_path = _write_json(outdir / "structure_prediction_job_manifest.json", payload)
    job = _job_record(project_name, "structure_prediction", "prepared", payload)

    result: Dict[str, Any] = {
        "status": "prepared",
        "project_name": project_name,
        "job_id": job["job_id"],
        "job_manifest_path": manifest_path,
        **payload,
        "stop_after_this_tool": True,
        "agent_instruction": agent_instruction,
        "next_step": "Run run_script, then call collect_structure_prediction_results.",
    }
    if execute:
        completed = None
        try:
            completed = subprocess.run(
                ["bash", str(script_path)],
                cwd=str(outdir),
                text=True,
                capture_output=True,
                timeout=None if int(timeout_seconds) <= 0 else int(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            def _timeout_tail(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return str(value)[-4000:]

            result.update(
                {
                    "status": "timeout",
                    "returncode": None,
                    "timed_out": True,
                    "timeout_seconds": int(timeout_seconds),
                    "stdout_tail": _timeout_tail(exc.stdout),
                    "stderr_tail": _timeout_tail(exc.stderr),
                    "error": (
                        f"Structure prediction script exceeded the external timeout "
                        f"of {int(timeout_seconds)} seconds."
                    ),
                }
            )

        execute_status = (
            "completed"
            if completed is not None and completed.returncode == 0
            else "failed"
            if completed is not None
            else "timeout"
        )
        batch_summary_name = (
            "boltz2_local_prediction_summary.summary.json"
            if engine_clean == "boltz2_local"
            else "boltz2_nim_prediction_summary.summary.json"
        )
        batch_summary_path = outdir / batch_summary_name
        batch_summary = _read_json(batch_summary_path) if batch_summary_path.exists() else {}
        if isinstance(batch_summary, dict) and batch_summary.get("status"):
            batch_status = str(batch_summary.get("status") or "").strip()
            if batch_status:
                execute_status = batch_status
                result["boltz2_batch_summary"] = batch_summary
                result["boltz2_batch_summary_path"] = str(batch_summary_path)
                result["boltz2_batch_failure_count"] = batch_summary.get("failure_count")
                result["boltz2_batch_candidate_count"] = batch_summary.get("candidate_count")
                if batch_status == "completed_with_failures":
                    result["partial_failure"] = True
        result.update(
            {
                "status": execute_status,
                "returncode": completed.returncode if completed is not None else None,
                "stdout_tail": (
                    completed.stdout[-4000:]
                    if completed is not None
                    else result.get("stdout_tail", "")
                ),
                "stderr_tail": (
                    completed.stderr[-4000:]
                    if completed is not None
                    else result.get("stderr_tail", "")
                ),
            }
        )

    _update_project_manifest(
        project_name,
        {
            "stage": "structure_prediction_prepared",
            "structure_prediction_job_manifest_path": manifest_path,
            "structure_prediction_run_script": str(script_path),
            "structure_prediction_input_fasta": str(fasta_path),
            "structure_prediction_output_dir": str(prediction_dir),
            "structure_prediction_engine": engine_clean,
            "structure_prediction_template_info": template_info,
            "target_state_structure": resolved_target_state_structure,
            "reference_state_structure": str(reference_state_structure or ""),
        },
    )
    return result


@mcp.tool()
def collect_structure_prediction_results(
    project_name: str,
    prediction_dir: str,
    output_dir: Optional[str] = None,
    candidate_map_path: Optional[str] = None,
    prediction_summary_csv: Optional[str] = None,
    job_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan structure-prediction outputs and build result/evidence tables.

    For Boltz-2 NIM jobs this merges the API summary; for local Boltz-2 jobs
    it merges the normalized CLI summary and response JSON. Both backends
    expose the same confidence, PAE, template, and runtime fields downstream.
    """
    pred_dir = Path(prediction_dir).expanduser()
    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction_dir not found: {prediction_dir}")

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "04_structure_prediction"
    outdir.mkdir(parents=True, exist_ok=True)

    job_manifest, resolved_job_manifest_path = _resolve_structure_job_manifest(outdir, job_manifest_path)
    if not candidate_map_path and job_manifest.get("candidate_map_path"):
        candidate_map_path = str(job_manifest["candidate_map_path"])
    known_variants = _candidate_map_variants(candidate_map_path)
    payload_paths = _payload_paths_by_variant(job_manifest)

    prediction_summary_path = _resolve_prediction_summary_csv(
        outdir=outdir,
        pred_dir=pred_dir,
        prediction_summary_csv=prediction_summary_csv,
        job_manifest=job_manifest,
    )
    prediction_summary_rows: List[Dict[str, str]] = []
    prediction_summary_fields: List[str] = []
    if prediction_summary_path:
        prediction_summary_rows, prediction_summary_fields = _read_csv_rows(Path(prediction_summary_path).expanduser())

    summary_by_variant: Dict[str, Dict[str, str]] = {}
    for summary_row in prediction_summary_rows:
        variant = _variant_id(summary_row)
        summary_by_variant[variant] = summary_row

    structure_files = sorted(
        [path for path in pred_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".pdb", ".cif"}]
    )
    json_files = sorted([path for path in pred_dir.rglob("*.json") if path.is_file()])
    rows_by_variant: Dict[str, Dict[str, Any]] = {}
    for path in structure_files:
        stem = path.stem
        variant = ""
        for candidate in known_variants:
            if candidate and candidate in stem:
                variant = candidate
                break
        if not variant:
            variant = stem
        rows_by_variant[_sanitize_name(variant)] = {
            "variant": _sanitize_name(variant),
            "structure_path": str(path),
            "structure_format": path.suffix.lower().lstrip("."),
            "source_file": path.name,
        }

    for variant, summary_row in summary_by_variant.items():
        structure_path = str(summary_row.get("structure_path") or "").strip()
        row = rows_by_variant.setdefault(
            variant,
            {
                "variant": variant,
                "structure_path": structure_path,
                "structure_format": Path(structure_path).suffix.lower().lstrip(".") if structure_path else "",
                "source_file": Path(structure_path).name if structure_path else "",
            },
        )
        if structure_path and not row.get("structure_path"):
            row["structure_path"] = structure_path
            row["structure_format"] = Path(structure_path).suffix.lower().lstrip(".")
            row["source_file"] = Path(structure_path).name
        for key, value in summary_row.items():
            if key == "variant":
                continue
            if key == "status":
                row["prediction_status"] = value
            elif key == "http_status":
                row["prediction_http_status"] = value
            elif key == "response_json":
                row["prediction_response_json"] = value
            elif key == "error":
                row["prediction_error"] = value
            elif key == "confidence_score":
                row["structure_prediction_confidence_score"] = value
            elif key in {"structure_path", "structure_format", "source_file"}:
                continue
            else:
                row[key] = value
        payload_path = str(summary_row.get("submitted_payload_json") or "").strip() or payload_paths.get(variant, "")
        if payload_path:
            _merge_nonempty_structure_metadata(
                row,
                _summarize_structure_prediction_payload(payload_path),
            )

    for variant, payload_path in payload_paths.items():
        if variant in rows_by_variant and "structure_prediction_has_template" not in rows_by_variant[variant]:
            _merge_nonempty_structure_metadata(
                rows_by_variant[variant],
                _summarize_structure_prediction_payload(payload_path),
            )

    required_pae = _required_pae_from_structure_job_manifest(job_manifest)
    for row in rows_by_variant.values():
        row["required_pae"] = required_pae
        response_text = str(
            row.get("prediction_response_json")
            or row.get("response_json")
            or ""
        ).strip()
        if response_text:
            response_path = _resolve_project_output_path(response_text, project_name)
            response_summary = _structure_response_summary(response_path)
            if "pae_matrix_usable" not in row:
                row["pae_matrix_usable"] = response_summary.get(
                    "structure_confidence_pae_matrix_usable",
                    False,
                )
            if "pae_matrix_shape" not in row:
                row["pae_matrix_shape"] = response_summary.get(
                    "structure_confidence_pae_matrix_shape",
                    "",
                )

    rows = sorted(rows_by_variant.values(), key=lambda row: _natural_residue_sort_key(str(row.get("variant") or "")))
    evidence_rows = _structure_prediction_evidence_rows(rows)

    output_csv = outdir / "structure_prediction_results.csv"
    fields = _csv_fieldnames_union(
        rows,
        [
            "variant",
            "structure_path",
            "structure_format",
            "source_file",
            "prediction_status",
            "structure_prediction_confidence_score",
            "structure_prediction_has_template",
            "structure_prediction_ligand_count",
        ],
    )
    _write_csv_rows(output_csv, rows, fields)
    evidence_csv = outdir / "structure_prediction_evidence.csv"
    _write_csv_rows(
        evidence_csv,
        evidence_rows,
        _csv_fieldnames_union(
            evidence_rows,
            [
                "variant",
                "structure_prediction_complete",
                "structure_prediction_confidence_score",
                "structure_prediction_has_template",
                "structure_prediction_ligand_count",
                "structure_prediction_status",
            ],
        ),
    )

    confidence_values = [
        _safe_float(row.get("structure_prediction_confidence_score"))
        for row in evidence_rows
        if _safe_float(row.get("structure_prediction_confidence_score")) is not None
    ]
    raw_structure_count = sum(1 for row in rows if str(row.get("structure_path") or "").strip())
    completed_count = sum(1 for row in evidence_rows if row.get("structure_prediction_complete") is True)
    valid_structure_count = sum(
        1
        for row in evidence_rows
        if _is_structure_prediction_row_valid(
            {
                "variant": row.get("variant"),
                "structure_path": row.get("structure_prediction_structure_path"),
                "prediction_status": row.get("structure_prediction_status"),
                "structure_prediction_complete": row.get("structure_prediction_complete"),
                "required_pae": row.get("structure_prediction_required_pae"),
                "pae_matrix_usable": row.get("structure_prediction_pae_matrix_usable"),
            },
            required_pae_override=required_pae,
        )
    )
    missing_required_pae_count = sum(
        1
        for row in evidence_rows
        if required_pae
        and row.get("structure_prediction_complete") is not True
        and not _is_true_marker(row.get("structure_prediction_pae_matrix_usable"))
    )
    failed_count = sum(
        1
        for row in evidence_rows
        if str(row.get("structure_prediction_status") or "").lower().startswith("failed")
        or row.get("structure_prediction_complete") is not True
    )
    failure_examples: List[Dict[str, Any]] = []
    for row in rows:
        status_text = str(row.get("prediction_status") or row.get("status") or "").lower()
        if status_text.startswith("failed") or not str(row.get("structure_path") or "").strip():
            failure_examples.append(
                {
                    "variant": row.get("variant"),
                    "status": row.get("prediction_status") or row.get("status"),
                    "error": row.get("prediction_error") or row.get("error"),
                    "return_code": row.get("return_code"),
                    "failure_class": row.get("failure_class"),
                    "attempt_mode": row.get("attempt_mode"),
                    "log_path": row.get("log_path"),
                    "log_tail": row.get("log_tail"),
                }
            )
            if len(failure_examples) >= 3:
                break
    template_count = sum(1 for row in evidence_rows if str(row.get("structure_prediction_has_template")).lower() == "true")
    summary = {
        "status": "completed" if valid_structure_count else "blocked_no_candidates",
        "project_name": project_name,
        "prediction_dir": str(pred_dir),
        "output_csv": str(output_csv),
        "evidence_csv": str(evidence_csv),
        "structure_count": valid_structure_count,
        "raw_structure_count": raw_structure_count,
        "result_row_count": len(rows),
        "completed_count": completed_count,
        "valid_structure_count": valid_structure_count,
        "failed_count": failed_count,
        "failure_examples": failure_examples,
        "missing_required_pae_count": missing_required_pae_count,
        "required_pae": required_pae,
        "template_constrained_count": template_count,
        "prediction_summary_csv": prediction_summary_path,
        "prediction_summary_row_count": len(prediction_summary_rows),
        "prediction_summary_fields": prediction_summary_fields,
        "job_manifest_path": resolved_job_manifest_path,
        "candidate_map_path": candidate_map_path or "",
        "confidence_distribution": _score_distribution_summary(confidence_values),
        "json_file_count": len(json_files),
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "structure_prediction_results_summary.json", summary)
    collection_stage = (
        "structure_prediction_collected"
        if valid_structure_count
        else "structure_prediction_blocked_no_valid_structures"
    )
    _update_project_manifest(
        project_name,
        {
            "stage": collection_stage,
            "structure_prediction_results_csv": str(output_csv),
            "structure_prediction_evidence_csv": str(evidence_csv),
            "structure_prediction_results_summary_path": summary_path,
            "structure_prediction_valid_structure_count": valid_structure_count,
            "structure_prediction_required_pae": required_pae,
        },
    )
    return {**summary, "summary_path": summary_path}


def _cif_tokens(line: str) -> List[str]:
    import shlex

    return shlex.split(line, posix=True)


def _cif_column_index(columns: List[str], *names: str) -> Optional[int]:
    by_name = {name: idx for idx, name in enumerate(columns)}
    for name in names:
        if name in by_name:
            return by_name[name]
    return None


def _chain_selector_set(chain_id: Any) -> set[str]:
    """Normalize a chain selector that may contain a comma-separated list.

    The formal HbI workflow uses ``parse_chains=A,B`` for the two-chain WT
    inputs, while Boltz-2 predicts the designed polymer on chain A. The old
    converters compared the literal string ``A,B`` with each one-character
    chain ID and consequently discarded every atom during standardization.
    """
    text = str(chain_id or "").strip()
    if not text:
        return set()
    return {
        token.strip()
        for token in re.split(r"[,;\s]+", text)
        if token.strip()
    }


def _format_pdb_atom_name(atom_name: str, element: str) -> str:
    atom = str(atom_name or "").strip()[:4]
    elem = str(element or "").strip()
    if len(atom) < 4 and len(elem) == 1 and atom and atom[0].isalpha():
        return f" {atom:<3}"[:4]
    return f"{atom:<4}"[:4]


def _parse_int_like(value: str, default: int) -> int:
    try:
        cleaned = str(value).strip()
        if cleaned in {"", ".", "?"}:
            return default
        return int(float(cleaned))
    except Exception:
        return default


def _parse_float_like(value: str, default: float) -> float:
    try:
        cleaned = str(value).strip()
        if cleaned in {"", ".", "?"}:
            return default
        return float(cleaned)
    except Exception:
        return default


def _mmcif_to_pdb(
    source: Path,
    destination: Path,
    *,
    chain_id: str = "",
    include_hetero: bool = True,
) -> Dict[str, Any]:
    """Convert a simple ModelCIF/mmCIF atom_site loop to PDB.

    This converter is intentionally narrow but dependency-light. It is meant for
    hosted Boltz/AF-style prediction outputs where atom_site rows are present.
    """
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    selected_chains = _chain_selector_set(chain_id)
    out_lines: List[str] = []
    atom_count = 0
    hetero_count = 0
    chain_counts: Dict[str, int] = {}
    serial = 1
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        columns: List[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            columns.append(lines[i].strip())
            i += 1
        if not columns or not any(col.startswith("_atom_site.") for col in columns):
            while i < len(lines) and lines[i].strip() not in {"#", "loop_"}:
                i += 1
            continue

        group_i = _cif_column_index(columns, "_atom_site.group_PDB")
        atom_i = _cif_column_index(columns, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        comp_i = _cif_column_index(columns, "_atom_site.auth_comp_id", "_atom_site.label_comp_id")
        chain_i = _cif_column_index(columns, "_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        seq_i = _cif_column_index(columns, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        x_i = _cif_column_index(columns, "_atom_site.Cartn_x")
        y_i = _cif_column_index(columns, "_atom_site.Cartn_y")
        z_i = _cif_column_index(columns, "_atom_site.Cartn_z")
        occ_i = _cif_column_index(columns, "_atom_site.occupancy")
        b_i = _cif_column_index(columns, "_atom_site.B_iso_or_equiv")
        element_i = _cif_column_index(columns, "_atom_site.type_symbol")
        alt_i = _cif_column_index(columns, "_atom_site.label_alt_id", "_atom_site.auth_alt_id")
        icode_i = _cif_column_index(columns, "_atom_site.pdbx_PDB_ins_code")

        required = [atom_i, comp_i, chain_i, x_i, y_i, z_i]
        if any(index is None for index in required):
            raise ValueError(f"Missing required atom_site columns in {source}")

        while i < len(lines):
            raw = lines[i].strip()
            if not raw or raw == "#":
                i += 1
                break
            if raw == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                break
            tokens = _cif_tokens(raw)
            if len(tokens) < len(columns):
                i += 1
                continue

            group = tokens[group_i] if group_i is not None else "ATOM"
            record = "ATOM" if str(group).upper().startswith("ATOM") else "HETATM"
            if record == "HETATM" and not include_hetero:
                i += 1
                continue

            atom_name = tokens[atom_i]
            resname = str(tokens[comp_i]).upper()[:3]
            chain = str(tokens[chain_i]).strip()[:1] or "A"
            if selected_chains and chain not in selected_chains:
                i += 1
                continue

            resseq = _parse_int_like(tokens[seq_i] if seq_i is not None else "", 1)
            x = _parse_float_like(tokens[x_i], 0.0)
            y = _parse_float_like(tokens[y_i], 0.0)
            z = _parse_float_like(tokens[z_i], 0.0)
            occ = _parse_float_like(tokens[occ_i], 1.0) if occ_i is not None else 1.0
            b = _parse_float_like(tokens[b_i], 0.0) if b_i is not None else 0.0
            element = str(tokens[element_i]).strip()[:2] if element_i is not None else re.sub(r"[^A-Za-z]", "", atom_name)[:2]
            altloc = str(tokens[alt_i]).strip()[:1] if alt_i is not None else ""
            if altloc in {".", "?"}:
                altloc = ""
            icode = str(tokens[icode_i]).strip()[:1] if icode_i is not None else ""
            if icode in {".", "?"}:
                icode = ""

            out_lines.append(
                f"{record:<6}{serial:5d} {_format_pdb_atom_name(atom_name, element)}{altloc:1}"
                f"{resname:>3} {chain:1}{resseq:4d}{icode:1}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {element:>2}"
            )
            serial += 1
            atom_count += 1
            hetero_count += int(record == "HETATM")
            chain_counts[chain] = chain_counts.get(chain, 0) + 1
            i += 1

    if not out_lines:
        raise ValueError(f"No atom_site coordinates were converted from {source}")
    out_lines.append("END")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "source_path": str(source),
        "output_path": str(destination),
        "atom_count": atom_count,
        "hetero_atom_count": hetero_count,
        "chain_counts": dict(sorted(chain_counts.items())),
    }


def _copy_or_filter_pdb(
    source: Path,
    destination: Path,
    *,
    chain_id: str = "",
    include_hetero: bool = True,
) -> Dict[str, Any]:
    selected_chains = _chain_selector_set(chain_id)
    copied_lines: List[str] = []
    atom_count = 0
    hetero_count = 0
    chain_counts: Dict[str, int] = {}
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        record = raw[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        if record == "HETATM" and not include_hetero:
            continue
        chain = raw[21:22].strip() or "A"
        if selected_chains and chain not in selected_chains:
            continue
        copied_lines.append(raw[:80])
        atom_count += 1
        hetero_count += int(record == "HETATM")
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
    if not copied_lines:
        raise ValueError(f"No PDB ATOM/HETATM records were copied from {source}")
    copied_lines.append("END")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(copied_lines) + "\n", encoding="utf-8")
    return {
        "source_path": str(source),
        "output_path": str(destination),
        "atom_count": atom_count,
        "hetero_atom_count": hetero_count,
        "chain_counts": dict(sorted(chain_counts.items())),
    }


def _remap_standardized_pdb_chain(
    path: Path,
    *,
    source_chain: str,
    target_chain: str,
) -> Dict[str, Any]:
    """Remap one unambiguous predicted polymer to the workflow chain ID."""
    source = (str(source_chain or "").strip() or "A")[:1]
    target = (str(target_chain or "").strip() or source)[:1]
    if source == target:
        return {"remapped": False, "source_chain": source, "target_chain": target}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rewritten: List[str] = []
    atom_count = 0
    for raw in lines:
        record = raw[:6].strip()
        if record in {"ATOM", "HETATM"}:
            line = raw.ljust(80)
            if line[21:22].strip() == source:
                line = line[:21] + target + line[22:]
                atom_count += 1
            rewritten.append(line[:80])
        else:
            rewritten.append(raw)
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return {
        "remapped": True,
        "source_chain": source,
        "target_chain": target,
        "atom_count": atom_count,
    }


@mcp.tool()
def standardize_structure_prediction_results(
    project_name: str,
    structures_csv: str,
    output_dir: Optional[str] = None,
    chain_id: str = "",
    include_hetero: bool = True,
    variant_column: str = "variant",
    structure_column: str = "structure_path",
    require_pae: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convert/copy predicted structures to downstream-friendly PDB files.

    Boltz-2 NIM commonly returns mmCIF. Many downstream tools in SwitchTune
    (ESM-IF1, ProteinMPNN/LigandMPNN, Rosetta) are more predictable with PDB,
    so this tool creates a standardized PDB table before later modules.
    """
    rows, fieldnames = _read_csv_rows(Path(structures_csv).expanduser())
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "04_structure_prediction"
    pdb_dir = outdir / "standardized_pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    converted_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    skipped_invalid_pae_count = 0
    skipped_failed_prediction_count = 0
    for row in rows:
        variant = _variant_id(row, variant_column)
        source = Path(str(row.get(structure_column) or "")).expanduser()
        row_required_pae = (
            bool(require_pae)
            if require_pae is not None
            else _is_true_marker(
                row.get("required_pae")
                or row.get("structure_prediction_required_pae")
            )
        )
        status = str(row.get("prediction_status") or row.get("status") or "").strip().lower()
        if status.startswith("failed") or status in {
            "completed_no_structure_detected",
            "blocked_no_candidates",
        }:
            skipped_failed_prediction_count += 1
            errors.append(
                {
                    "variant": variant,
                    "source_path": str(source),
                    "error": f"prediction status is not usable: {status or 'unknown'}",
                }
            )
            continue
        if row_required_pae:
            pae_usable = row.get("pae_matrix_usable")
            if pae_usable in {"", None}:
                pae_usable = row.get("structure_prediction_pae_matrix_usable")
            if not _is_true_marker(pae_usable):
                skipped_invalid_pae_count += 1
                errors.append(
                    {
                        "variant": variant,
                        "source_path": str(source),
                        "error": "required PAE matrix is missing or unusable",
                    }
                )
                continue
        if not source.exists():
            errors.append({"variant": variant, "source_path": str(source), "error": "source structure not found"})
            continue
        destination = pdb_dir / f"{variant}.pdb"
        try:
            if source.suffix.lower() in {".cif", ".mmcif"}:
                try:
                    info = _mmcif_to_pdb(source, destination, chain_id=chain_id, include_hetero=include_hetero)
                except ValueError as exc:
                    # Boltz-2 writes the designed polymer on its own chain
                    # (normally A), while the WT template may use another
                    # chain (C for Dronpa). Retry only for a single
                    # unambiguous output chain, then map it to the requested
                    # workflow chain so downstream tools remain consistent.
                    if "No atom_site coordinates were converted" not in str(exc) or not _chain_selector_set(chain_id):
                        raise
                    fallback = _mmcif_to_pdb(source, destination, chain_id="", include_hetero=include_hetero)
                    available = sorted(str(k) for k in (fallback.get("chain_counts") or {}))
                    requested = next(iter(sorted(_chain_selector_set(chain_id))), "A")
                    if len(available) != 1:
                        raise
                    remap = _remap_standardized_pdb_chain(
                        destination,
                        source_chain=available[0],
                        target_chain=requested,
                    )
                    fallback["chain_counts"] = {requested: int(fallback.get("atom_count") or 0)}
                    fallback["chain_id_remap"] = remap
                    info = fallback
                conversion_status = "converted_mmcif_to_pdb"
            elif source.suffix.lower() == ".pdb":
                try:
                    info = _copy_or_filter_pdb(source, destination, chain_id=chain_id, include_hetero=include_hetero)
                except ValueError as exc:
                    if "No PDB ATOM/HETATM records were copied" not in str(exc) or not _chain_selector_set(chain_id):
                        raise
                    fallback = _copy_or_filter_pdb(source, destination, chain_id="", include_hetero=include_hetero)
                    available = sorted(str(k) for k in (fallback.get("chain_counts") or {}))
                    requested = next(iter(sorted(_chain_selector_set(chain_id))), "A")
                    if len(available) != 1:
                        raise
                    remap = _remap_standardized_pdb_chain(
                        destination,
                        source_chain=available[0],
                        target_chain=requested,
                    )
                    fallback["chain_counts"] = {requested: int(fallback.get("atom_count") or 0)}
                    fallback["chain_id_remap"] = remap
                    info = fallback
                conversion_status = "copied_pdb"
            else:
                errors.append({"variant": variant, "source_path": str(source), "error": "unsupported structure format"})
                continue
        except Exception as exc:
            errors.append({"variant": variant, "source_path": str(source), "error": str(exc)})
            continue

        out = dict(row)
        out["variant"] = variant
        out["source_structure_path"] = str(source)
        out["structure_path"] = str(destination)
        out["structure_format"] = "pdb"
        out["standardization_status"] = conversion_status
        out["atom_count"] = info.get("atom_count")
        out["hetero_atom_count"] = info.get("hetero_atom_count")
        out["chain_counts_json"] = json.dumps(info.get("chain_counts", {}), ensure_ascii=True)
        converted_rows.append(out)

    output_csv = outdir / "structure_prediction_results_pdb.csv"
    preferred = ["variant", "structure_path", "structure_format", "source_structure_path", "standardization_status"]
    _write_csv_rows(output_csv, converted_rows, _csv_fieldnames_union(converted_rows, preferred + fieldnames))
    error_csv = outdir / "structure_standardization_errors.csv"
    _write_csv_rows(error_csv, errors, _csv_fieldnames_union(errors, ["variant", "source_path", "error"]))

    if not converted_rows:
        standardization_status = "blocked_no_candidates"
    elif errors:
        standardization_status = "completed_with_warnings"
    else:
        standardization_status = "completed"
    summary = {
        "status": standardization_status,
        "project_name": project_name,
        "input_csv": str(Path(structures_csv).expanduser()),
        "output_csv": str(output_csv),
        "error_csv": str(error_csv),
        "pdb_dir": str(pdb_dir),
        "input_count": len(rows),
        "standardized_count": len(converted_rows),
        "error_count": len(errors),
        "skipped_invalid_pae_count": skipped_invalid_pae_count,
        "skipped_failed_prediction_count": skipped_failed_prediction_count,
        "require_pae": require_pae,
        "chain_id": chain_id,
        "include_hetero": include_hetero,
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "structure_standardization_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "structure_predictions_standardized",
            "structure_prediction_pdb_results_csv": str(output_csv),
            "structure_prediction_standardization_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path, "errors": errors[:20]}


def _state_similarity_wide_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        variant = _sanitize_name(str(row.get("variant") or "unknown"))
        region = _sanitize_name(str(row.get("region") or "global"))
        out = by_variant.setdefault(variant, {"variant": variant})
        for key in (
            "rmsd_to_target_A",
            "rmsd_to_reference_A",
            "delta_reference_minus_target_A",
            "closer_state",
            "aligned_ca_to_target",
            "aligned_ca_to_reference",
            "comparison_residue_policy",
            "compared_prediction_residue_numbers",
            "pairwise_rmsd_to_target_A",
            "pairwise_rmsd_to_reference_A",
            "pairwise_aligned_ca_to_target",
            "pairwise_aligned_ca_to_reference",
        ):
            out[f"state_similarity_{region}_{key}"] = row.get(key, "")

    for out in by_variant.values():
        deltas = []
        target_count = 0
        reference_count = 0
        for key, value in list(out.items()):
            if key.endswith("_delta_reference_minus_target_A"):
                numeric = _safe_float(value)
                if numeric is not None:
                    deltas.append(numeric)
            if key.endswith("_closer_state"):
                if value == "target":
                    target_count += 1
                elif value == "reference":
                    reference_count += 1
        selected = None
        selected_region = ""
        for region in ("switch_region", "switch_87_110", "global"):
            selected = _safe_float(out.get(f"state_similarity_{region}_delta_reference_minus_target_A"))
            if selected is not None:
                selected_region = region
                break
        if selected is None and deltas:
            selected = mean(deltas)
            selected_region = "mean_of_available_regions"
        out["structure_target_similarity_score"] = selected
        out["structure_target_similarity_region"] = selected_region
        out["structure_target_like_region_count"] = target_count
        out["structure_reference_like_region_count"] = reference_count
        out["structure_state_qc"] = "target_like" if selected is not None and selected > 0 else "reference_like_or_unresolved"
    return list(by_variant.values())


def _structure_comparison_region_specs(regions: str) -> List[str]:
    """Keep comma-separated spans attached to their named comparison region.

    For example, global,switch_region:10,12-47,49-60 must produce two
    --region arguments, not four. Another named region or global starts
    a new argument; unlabelled intervals retain their legacy behaviour.
    """
    specs: List[str] = []
    for token in str(regions).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token or token.lower() == "global":
            specs.append(token)
        elif specs and ":" in specs[-1]:
            specs[-1] += f",{token}"
        else:
            specs.append(token)
    return specs


@mcp.tool()
def compare_structure_predictions_to_states(
    project_name: str,
    structures_csv: str,
    target_state_structure: str,
    reference_state_structure: str,
    output_dir: Optional[str] = None,
    prediction_chain: str = "A",
    target_chain: str = "A",
    reference_chain: str = "A",
    regions: str = "global,switch_87_110:87-110",
    mapping: str = "sequence_alignment",
) -> Dict[str, Any]:
    """Compare predicted structures against target/reference conformations.

    Positive `delta_reference_minus_target_A` means the prediction is closer to
    the target state after CA superposition on the same homologous residues
    present in the prediction and both states. Regions use target numbering.
    Named regions may include single sites and disjoint spans, for example
    `global,switch_region:10,12-47,49-60`.
    """
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "04_structure_prediction"
    outdir.mkdir(parents=True, exist_ok=True)
    long_csv = outdir / "prediction_state_similarity.csv"
    evidence_csv = outdir / "structure_state_similarity_evidence.csv"

    project_root = Path(os.environ.get("SWITCHTUNE_HOME", Path(__file__).resolve().parents[1])).expanduser().resolve()
    script = project_root / "scripts" / "compare_boltz_predictions_to_states.py"
    if not script.exists():
        raise FileNotFoundError(f"State comparison script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--structures-csv",
        str(Path(structures_csv).expanduser()),
        "--target",
        str(Path(target_state_structure).expanduser()),
        "--reference",
        str(Path(reference_state_structure).expanduser()),
        "--output-csv",
        str(long_csv),
        "--prediction-chain",
        prediction_chain,
        "--target-chain",
        target_chain,
        "--reference-chain",
        reference_chain,
        "--mapping",
        mapping,
    ]
    region_specs = _structure_comparison_region_specs(regions)
    for region in region_specs:
        cmd.extend(["--region", region])

    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Structure state comparison failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
        )

    long_rows, _long_fields = _read_csv_rows(long_csv)
    evidence_rows = _state_similarity_wide_rows(long_rows)
    _write_csv_rows(
        evidence_csv,
        evidence_rows,
        _csv_fieldnames_union(
            evidence_rows,
            [
                "variant",
                "structure_target_similarity_score",
                "structure_state_qc",
                "structure_target_like_region_count",
                "structure_reference_like_region_count",
            ],
        ),
    )
    target_like = sum(1 for row in evidence_rows if row.get("structure_state_qc") == "target_like")
    summary = {
        "status": "completed",
        "project_name": project_name,
        "structures_csv": str(Path(structures_csv).expanduser()),
        "target_state_structure": str(Path(target_state_structure).expanduser()),
        "reference_state_structure": str(Path(reference_state_structure).expanduser()),
        "long_csv": str(long_csv),
        "evidence_csv": str(evidence_csv),
        "variant_count": len(evidence_rows),
        "target_like_count": target_like,
        "reference_like_or_unresolved_count": len(evidence_rows) - target_like,
        "regions": region_specs,
        "mapping": mapping,
        "comparison_residue_policy": "shared_across_prediction_target_reference",
        "primary_region_policy": "switch_region_then_legacy_switch_then_global",
        "stdout_tail": completed.stdout[-4000:],
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "prediction_state_similarity_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "structure_state_similarity_compared",
            "structure_state_similarity_csv": str(long_csv),
            "structure_state_similarity_evidence_csv": str(evidence_csv),
            "structure_state_similarity_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def filter_structure_confidence(
    project_name: str,
    candidates_csv: Optional[str] = None,
    structures_csv: Optional[str] = None,
    structure_prediction_evidence_csv: Optional[str] = None,
    structure_state_similarity_evidence_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    variant_column: str = "variant",
    sequence_column: str = "sequence",
    structure_column: str = "structure_path",
    response_json_column: str = "prediction_response_json",
    chain_id: str = "A",
    mutation_window_radius: int = 3,
    min_confidence_score: Optional[float] = 0.70,
    min_complex_plddt: Optional[float] = 0.70,
    min_local_plddt: Optional[float] = 70.0,
    max_mutation_mean_pae: Optional[float] = None,
    adaptive_pae_quantile: Optional[float] = 0.80,
    require_pae: bool = True,
    require_target_like: bool = True,
    require_complete: bool = True,
    enable_agent_rescue_review: bool = False,
    rescue_max_mutation_mean_pae: Optional[float] = 4.0,
    rescue_min_confidence_score: Optional[float] = 0.90,
    rescue_min_complex_plddt: Optional[float] = 0.85,
    rescue_min_local_plddt: Optional[float] = 80.0,
    rescue_max_mutation_mean_pae_to_protein: Optional[float] = 2.0,
    rescue_max_mutation_window_internal_pae: Optional[float] = 2.0,
    rescue_require_target_like: bool = True,
) -> Dict[str, Any]:
    """Filter predicted structures before inverse-folding validation.

    The filter combines Boltz-2 response metrics, full PAE when available,
    CIF/PDB B-factor pLDDT, and optional target/reference similarity evidence.
    By default it is a hard gate. When ``enable_agent_rescue_review`` is true,
    candidates that fail only the adaptive PAE ranking gate can be written to a
    separate review pool. An explicit
    ``SWITCHTUNE_ALLOW_RELATIVE_STRUCTURE_FALLBACK=1`` additionally permits a
    complete, PAE-usable, target-like pool when every candidate misses the
    absolute thresholds. These candidates remain labelled review rows and are
    never automatically passed downstream: Biomni must explicitly select them
    at the structure checkpoint.
    """
    # SwitchTune uses PAE as a mandatory structure-confidence gate. Keep this
    # hard requirement even if an older caller omits or disables the flag.
    require_pae = True
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "04b_structure_confidence_filter"
    outdir.mkdir(parents=True, exist_ok=True)

    candidates_csv = (
        candidates_csv
        or manifest.get("state_specific_candidates_adaptive_csv")
        or manifest.get("state_specific_candidates_csv")
        or _candidate_csv_from_structure_map(project, manifest)
    )
    structures_csv = (
        structures_csv
        or manifest.get("structure_prediction_pdb_results_csv")
        or manifest.get("structure_prediction_results_csv")
    )
    structure_prediction_evidence_csv = (
        structure_prediction_evidence_csv or manifest.get("structure_prediction_evidence_csv")
    )
    structure_state_similarity_evidence_csv = (
        structure_state_similarity_evidence_csv or manifest.get("structure_state_similarity_evidence_csv")
    )

    if not structures_csv:
        raise ValueError("structures_csv is required, or structure prediction results must be registered in the manifest.")

    structure_rows, _structure_fields = _read_csv_rows(_resolve_project_output_path(str(structures_csv), project_name))
    structure_by_variant = {_variant_id(row, variant_column): row for row in structure_rows}

    candidate_rows: List[Dict[str, str]]
    candidate_fields: List[str]
    if candidates_csv:
        candidate_rows, candidate_fields = _read_csv_rows(_resolve_project_output_path(str(candidates_csv), project_name))
    else:
        candidate_rows = [{"variant": variant} for variant in sorted(structure_by_variant)]
        candidate_fields = ["variant"]

    prediction_evidence_by_variant: Dict[str, Dict[str, str]] = {}
    if structure_prediction_evidence_csv:
        evidence_path = _resolve_project_output_path(str(structure_prediction_evidence_csv), project_name)
        if evidence_path.exists():
            prediction_rows, _prediction_fields = _read_csv_rows(evidence_path)
            prediction_evidence_by_variant = {_variant_id(row, variant_column): row for row in prediction_rows}

    state_evidence_by_variant: Dict[str, Dict[str, str]] = {}
    if structure_state_similarity_evidence_csv:
        state_path = _resolve_project_output_path(str(structure_state_similarity_evidence_csv), project_name)
        if state_path.exists():
            state_rows, _state_fields = _read_csv_rows(state_path)
            state_evidence_by_variant = {_variant_id(row, variant_column): row for row in state_rows}

    evidence_rows: List[Dict[str, Any]] = []
    retained_rows: List[Dict[str, Any]] = []
    agent_review_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    pae_primary_values: List[float] = []

    for candidate in candidate_rows:
        variant = _variant_id(candidate, variant_column)
        structure_row = structure_by_variant.get(variant, {})
        prediction_evidence = prediction_evidence_by_variant.get(variant, {})
        state_evidence = state_evidence_by_variant.get(variant, {})

        structure_path_text = str(structure_row.get(structure_column) or structure_row.get("structure_path") or "").strip()
        if not structure_path_text and prediction_evidence.get("structure_prediction_structure_path"):
            structure_path_text = str(prediction_evidence.get("structure_prediction_structure_path") or "").strip()
        structure_path = _resolve_project_output_path(structure_path_text, project_name) if structure_path_text else Path("")
        confidence_structure_path_text = str(
            structure_row.get("source_structure_path")
            or structure_row.get("source_file_path")
            or structure_path_text
        ).strip()
        confidence_structure_path = (
            _resolve_project_output_path(confidence_structure_path_text, project_name)
            if confidence_structure_path_text
            else structure_path
        )

        response_json_text = str(
            structure_row.get(response_json_column)
            or structure_row.get("response_json")
            or structure_row.get("structure_prediction_response_json")
            or prediction_evidence.get("prediction_response_json")
            or prediction_evidence.get("structure_prediction_response_json")
            or ""
        ).strip()
        response_path = _resolve_project_output_path(response_json_text, project_name) if response_json_text else Path("")

        response_summary = _structure_response_summary(response_path) if response_json_text else {}
        bfactor_summary = (
            _parse_structure_b_factor_confidence(confidence_structure_path, chain_id=chain_id)
            if confidence_structure_path_text
            else {}
        )
        residue_plddt = bfactor_summary.pop("_residue_plddt_means", {})
        residue_order = bfactor_summary.pop("_residue_order", [])

        mutation_position = _candidate_mutation_position(candidate, variant)
        window_positions = [
            position
            for position in range(
                int(mutation_position or 0) - int(mutation_window_radius),
                int(mutation_position or 0) + int(mutation_window_radius) + 1,
            )
            if position in residue_plddt
        ]
        mutation_residue_plddt = residue_plddt.get(mutation_position) if mutation_position is not None else None
        mutation_window_plddt = (
            mean([residue_plddt[position] for position in window_positions]) if window_positions else None
        )

        pae_matrix = _extract_structure_response_matrix(response_path, "pae") if response_json_text else None
        pae_summary = _pae_region_summary(
            pae_matrix,
            residue_order,
            mutation_position,
            window_radius=int(mutation_window_radius),
        )

        confidence_score = _safe_float(
            response_summary.get("structure_confidence_confidence_score")
            or prediction_evidence.get("structure_prediction_confidence_score")
            or structure_row.get("structure_prediction_confidence_score")
            or structure_row.get("confidence_score")
        )
        complex_plddt = _safe_float(response_summary.get("structure_confidence_complex_plddt_score_0_1"))
        complex_iplddt = _safe_float(response_summary.get("structure_confidence_complex_iplddt_score_0_1"))
        complex_pde = _safe_float(response_summary.get("structure_confidence_complex_pde_score"))
        complex_ipde = _safe_float(response_summary.get("structure_confidence_complex_ipde_score"))

        pae_primary = _safe_float(pae_summary.get("structure_confidence_mutation_window_mean_pae_to_protein"))
        if pae_primary is None:
            pae_primary = _safe_float(pae_summary.get("structure_confidence_mutation_mean_pae_to_protein"))
        if pae_primary is not None and math.isfinite(pae_primary):
            pae_primary_values.append(pae_primary)

        complete_value = str(
            prediction_evidence.get("structure_prediction_complete")
            or structure_row.get("structure_prediction_complete")
            or ""
        ).strip().lower()
        structure_complete = (
            complete_value in {"true", "1", "yes", "completed"}
            if complete_value
            else (bool(structure_path_text) and structure_path.exists())
        )
        state_qc = str(state_evidence.get("structure_state_qc") or "").strip()
        target_like = state_qc == "target_like" if state_qc else None

        quality_components: List[float] = []
        for value in [confidence_score, complex_plddt, complex_iplddt]:
            if value is not None and math.isfinite(value):
                quality_components.append(max(0.0, min(1.0, value)))
        for value in [mutation_residue_plddt, mutation_window_plddt]:
            if value is not None and math.isfinite(value):
                quality_components.append(max(0.0, min(1.0, value / 100.0)))
        for value in [complex_pde, complex_ipde, pae_primary]:
            if value is not None and math.isfinite(value):
                quality_components.append(1.0 / (1.0 + max(0.0, value)))
        if target_like is not None:
            quality_components.append(1.0 if target_like else 0.0)

        row: Dict[str, Any] = dict(candidate)
        row.update(
            {
                "variant": variant,
                "structure_confidence_complete": structure_complete,
                "structure_confidence_pass": True,
                "structure_confidence_fail_reasons": "",
                "structure_confidence_score": mean(quality_components) if quality_components else None,
                "structure_confidence_quality_component_count": len(quality_components),
                "structure_confidence_mutation_position": mutation_position,
                "structure_confidence_mutation_window_radius": int(mutation_window_radius),
                "structure_confidence_mutation_residue_plddt_mean": mutation_residue_plddt,
                "structure_confidence_mutation_window_plddt_mean": mutation_window_plddt,
                "structure_confidence_mutation_window_pm3_plddt_mean": mutation_window_plddt,
                "structure_confidence_pae_primary_metric": pae_primary,
                "structure_confidence_state_qc": state_qc,
                "structure_confidence_target_like": target_like,
                "structure_path": structure_path_text,
            }
        )
        row.update(response_summary)
        row.update(bfactor_summary)
        row.update(pae_summary)

        fail_reasons: List[str] = []
        if require_complete and not structure_complete:
            fail_reasons.append("missing_or_incomplete_structure")
        if min_confidence_score is not None and (
            confidence_score is None or confidence_score < float(min_confidence_score)
        ):
            fail_reasons.append("low_global_confidence")
        if min_complex_plddt is not None and (
            complex_plddt is not None and complex_plddt < float(min_complex_plddt)
        ):
            fail_reasons.append("low_complex_plddt")
        if min_local_plddt is not None and (
            mutation_window_plddt is not None and mutation_window_plddt < float(min_local_plddt)
        ):
            fail_reasons.append("low_mutation_window_plddt")
        if min_local_plddt is not None and (
            mutation_residue_plddt is not None and mutation_residue_plddt < float(min_local_plddt)
        ):
            fail_reasons.append("low_mutation_residue_plddt")
        if require_pae and not row.get("structure_confidence_pae_matrix_usable"):
            if row.get("structure_confidence_pae_present"):
                fail_reasons.append("invalid_pae")
            else:
                fail_reasons.append("missing_pae")
        if max_mutation_mean_pae is not None and (
            pae_primary is None or pae_primary > float(max_mutation_mean_pae)
        ):
            fail_reasons.append("high_mutation_region_pae")
        if require_target_like and state_evidence_by_variant and target_like is not True:
            fail_reasons.append("not_target_like")

        row["structure_confidence_pass"] = not fail_reasons
        row["structure_confidence_fail_reasons"] = ";".join(fail_reasons)
        evidence_rows.append(row)

    adaptive_pae_cutoff: Optional[float] = None
    if max_mutation_mean_pae is None and adaptive_pae_quantile is not None and pae_primary_values:
        q = min(max(float(adaptive_pae_quantile), 0.0), 1.0)
        adaptive_pae_cutoff = _quantile(sorted(pae_primary_values), q)
        for row in evidence_rows:
            pae_primary = _safe_float(row.get("structure_confidence_pae_primary_metric"))
            if pae_primary is not None and pae_primary > adaptive_pae_cutoff:
                reasons = [
                    item
                    for item in str(row.get("structure_confidence_fail_reasons") or "").split(";")
                    if item
                ]
                reasons.append("high_adaptive_mutation_region_pae")
                row["structure_confidence_pass"] = False
                row["structure_confidence_fail_reasons"] = ";".join(reasons)
            if pae_primary is not None:
                row["structure_confidence_adaptive_pae_cutoff"] = adaptive_pae_cutoff

    for row in evidence_rows:
        passed_strict = row.get("structure_confidence_pass") is True
        fail_reasons = {
            item
            for item in str(row.get("structure_confidence_fail_reasons") or "").split(";")
            if item
        }
        rescue_reasons: List[str] = []
        rescue_eligible = False
        if not passed_strict and enable_agent_rescue_review:
            pae_only_failures = {
                "high_adaptive_mutation_region_pae",
                "high_mutation_region_pae",
            }
            hard_failures = sorted(fail_reasons - pae_only_failures)
            pae_primary = _safe_float(
                row.get("structure_confidence_pae_primary_metric")
            )
            mutation_pae = _safe_float(
                row.get("structure_confidence_mutation_mean_pae_to_protein")
            )
            internal_pae = _safe_float(
                row.get("structure_confidence_mutation_window_internal_mean_pae")
            )
            confidence_value = _safe_float(
                row.get("structure_confidence_confidence_score")
            )
            complex_plddt_value = _safe_float(
                row.get("structure_confidence_complex_plddt_score_0_1")
            )
            local_plddt_values = [
                _safe_float(
                    row.get("structure_confidence_mutation_residue_plddt_mean")
                ),
                _safe_float(
                    row.get("structure_confidence_mutation_window_plddt_mean")
                ),
            ]
            target_like = row.get("structure_confidence_target_like") is True

            if hard_failures:
                rescue_reasons.append(
                    "hard_failures:" + ",".join(hard_failures)
                )
            if pae_primary is None or (
                rescue_max_mutation_mean_pae is not None
                and pae_primary > float(rescue_max_mutation_mean_pae)
            ):
                rescue_reasons.append("mutation_region_pae_above_rescue_limit")
            if mutation_pae is None or (
                rescue_max_mutation_mean_pae_to_protein is not None
                and mutation_pae
                > float(rescue_max_mutation_mean_pae_to_protein)
            ):
                rescue_reasons.append("mutation_pae_to_protein_above_rescue_limit")
            if internal_pae is None or (
                rescue_max_mutation_window_internal_pae is not None
                and internal_pae
                > float(rescue_max_mutation_window_internal_pae)
            ):
                rescue_reasons.append("internal_window_pae_above_rescue_limit")
            if confidence_value is None or (
                rescue_min_confidence_score is not None
                and confidence_value < float(rescue_min_confidence_score)
            ):
                rescue_reasons.append("global_confidence_below_rescue_limit")
            if complex_plddt_value is None or (
                rescue_min_complex_plddt is not None
                and complex_plddt_value < float(rescue_min_complex_plddt)
            ):
                rescue_reasons.append("complex_plddt_below_rescue_limit")
            if any(value is None for value in local_plddt_values) or (
                rescue_min_local_plddt is not None
                and any(
                    value < float(rescue_min_local_plddt)
                    for value in local_plddt_values
                    if value is not None
                )
            ):
                rescue_reasons.append("local_plddt_below_rescue_limit")
            if rescue_require_target_like and not target_like:
                rescue_reasons.append("not_target_like")

            rescue_eligible = bool(fail_reasons) and not rescue_reasons

        row["structure_confidence_rescue_eligible"] = rescue_eligible
        row["structure_confidence_rescue_reasons"] = ";".join(rescue_reasons)
        if passed_strict:
            row["structure_confidence_gate"] = "strict_pass"
            retained_rows.append(row)
        elif rescue_eligible:
            row["structure_confidence_gate"] = "agent_review"
            agent_review_rows.append(row)
        else:
            row["structure_confidence_gate"] = "hard_fail"
            rejected_rows.append(row)

    # Small proteins and ligand-free conformational proxies can produce a
    # complete Boltz-2 prediction set whose absolute confidence is below the
    # generic 0.70/70 thresholds, even when the prediction is target-like and
    # has usable PAE.  Keep the default hard gate unchanged, but allow a
    # caller to request an explicitly labelled, relative fallback review pool.
    # The pool is never auto-passed: the downstream Biomni checkpoint must
    # select candidates and the final report retains the fallback label.
    relative_fallback_enabled = str(
        os.environ.get("SWITCHTUNE_ALLOW_RELATIVE_STRUCTURE_FALLBACK", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    relative_fallback_rows: List[Dict[str, Any]] = []
    if not retained_rows and not agent_review_rows and relative_fallback_enabled:
        fallback_pool: List[Dict[str, Any]] = []
        for row in evidence_rows:
            complete = _bool_or_numeric(row.get("structure_confidence_complete")) == 1.0
            pae_ok = _bool_or_numeric(row.get("structure_confidence_pae_matrix_usable")) == 1.0
            target_like = row.get("structure_confidence_target_like") is True
            confidence_value = _safe_float(
                row.get("structure_confidence_confidence_score")
            )
            local_value = _safe_float(
                row.get("structure_confidence_mutation_window_plddt_mean")
                or row.get("structure_confidence_mutation_window_pm3_plddt_mean")
            )
            pae_value = _safe_float(row.get("structure_confidence_pae_primary_metric"))
            if not (complete and pae_ok and target_like):
                continue
            # Rank by the available structure evidence, preserving a usable
            # PAE and local confidence as tie-breakers.  Do not invent a new
            # absolute quality threshold in this explicitly opt-in mode.
            row["_relative_fallback_rank"] = (
                confidence_value if confidence_value is not None else -1.0,
                local_value if local_value is not None else -1.0,
                -(pae_value if pae_value is not None else float("inf")),
            )
            fallback_pool.append(row)
        fallback_pool.sort(key=lambda item: item.get("_relative_fallback_rank", (-1.0, -1.0, float("-inf"))), reverse=True)
        fallback_limit = min(50, len(fallback_pool))
        if fallback_limit >= 20:
            for row in fallback_pool[:fallback_limit]:
                row.pop("_relative_fallback_rank", None)
                row["structure_confidence_gate"] = "relative_fallback_review"
                row["structure_confidence_fallback_reason"] = (
                    "All candidates failed the absolute confidence gate; complete, PAE-usable, "
                    "target-like candidates retained for explicit relative-quality review."
                )
                row["structure_confidence_rescue_eligible"] = True
                row["structure_confidence_rescue_reasons"] = "relative_quality_review_required"
                relative_fallback_rows.append(row)
                agent_review_rows.append(row)
            fallback_ids = {id(row) for row in relative_fallback_rows}
            rejected_rows = [row for row in rejected_rows if id(row) not in fallback_ids]

    review_rows = retained_rows + agent_review_rows

    evidence_csv = outdir / "structure_confidence_evidence.csv"
    retained_csv = outdir / "structure_confidence_retained_candidates.csv"
    review_csv = outdir / "structure_confidence_review_candidates.csv"
    rejected_csv = outdir / "structure_confidence_rejected_candidates.csv"
    preferred_fields = [
        "variant",
        sequence_column,
        "structure_confidence_pass",
        "structure_confidence_score",
        "structure_confidence_fail_reasons",
        "structure_confidence_confidence_score",
        "structure_confidence_complex_plddt_score_0_1",
        "structure_confidence_complex_iplddt_score_0_1",
        "structure_confidence_mutation_residue_plddt_mean",
        "structure_confidence_mutation_window_plddt_mean",
        "structure_confidence_mutation_window_pm3_plddt_mean",
        "structure_confidence_pae_present",
        "structure_confidence_pae_shape",
        "structure_confidence_pae_matrix_usable",
        "structure_confidence_pae_matrix_shape",
        "structure_confidence_pae_primary_metric",
        "structure_confidence_plddt_scale",
        "structure_confidence_state_qc",
        "structure_confidence_target_like",
        "structure_confidence_gate",
        "structure_confidence_fallback_reason",
        "structure_confidence_rescue_eligible",
        "structure_confidence_rescue_reasons",
        "structure_path",
    ]
    fields = _csv_fieldnames_union(evidence_rows, preferred_fields + candidate_fields)
    _write_csv_rows(evidence_csv, evidence_rows, fields)
    _write_csv_rows(retained_csv, retained_rows, fields)
    _write_csv_rows(review_csv, review_rows, fields)
    _write_csv_rows(rejected_csv, rejected_rows, fields)

    filter_status = (
        "completed"
        if retained_rows
        else "completed_with_agent_review"
        if agent_review_rows
        else "blocked_no_candidates"
    )
    summary = {
        "status": filter_status,
        "project_name": project_name,
        "input_candidates_csv": str(candidates_csv or ""),
        "input_structures_csv": str(structures_csv),
        "structure_prediction_evidence_csv": str(structure_prediction_evidence_csv or ""),
        "structure_state_similarity_evidence_csv": str(structure_state_similarity_evidence_csv or ""),
        "evidence_csv": str(evidence_csv),
        "output_csv": str(retained_csv),
        "retained_csv": str(retained_csv),
        "review_csv": str(review_csv),
        "rejected_csv": str(rejected_csv),
        "input_count": len(evidence_rows),
        "retained_count": len(retained_rows),
        "review_count": len(review_rows),
        "agent_review_count": len(agent_review_rows),
        "rejected_count": len(rejected_rows),
        "failure_reason_counts": dict(Counter(
            reason
            for row in evidence_rows
            for reason in str(row.get("structure_confidence_fail_reasons") or "").split(";")
            if reason
        )),
        "require_complete": bool(require_complete),
        "require_pae": bool(require_pae),
        "require_target_like": bool(require_target_like),
        "min_confidence_score": min_confidence_score,
        "min_complex_plddt": min_complex_plddt,
        "min_local_plddt": min_local_plddt,
        "max_mutation_mean_pae": max_mutation_mean_pae,
        "adaptive_pae_quantile": adaptive_pae_quantile,
        "adaptive_pae_cutoff": adaptive_pae_cutoff,
        "enable_agent_rescue_review": bool(enable_agent_rescue_review),
        "rescue_max_mutation_mean_pae": rescue_max_mutation_mean_pae,
        "rescue_min_confidence_score": rescue_min_confidence_score,
        "rescue_min_complex_plddt": rescue_min_complex_plddt,
        "rescue_min_local_plddt": rescue_min_local_plddt,
        "rescue_max_mutation_mean_pae_to_protein": rescue_max_mutation_mean_pae_to_protein,
        "rescue_max_mutation_window_internal_pae": rescue_max_mutation_window_internal_pae,
        "rescue_require_target_like": bool(rescue_require_target_like),
        "agent_review_variants": [
            str(row.get("variant") or "") for row in agent_review_rows
        ],
        "relative_fallback_enabled": bool(relative_fallback_enabled),
        "relative_fallback_count": len(relative_fallback_rows),
        "confidence_distribution": _score_distribution_summary(
            [
                value
                for value in [_safe_float(row.get("structure_confidence_confidence_score")) for row in evidence_rows]
                if value is not None
            ]
        ),
        "local_plddt_distribution": _score_distribution_summary(
            [
                value
                for value in [
                    _safe_float(
                        row.get("structure_confidence_mutation_window_plddt_mean")
                        or row.get("structure_confidence_mutation_window_pm3_plddt_mean")
                    )
                    for row in evidence_rows
                ]
                if value is not None
            ]
        ),
        "pae_primary_distribution": _score_distribution_summary(pae_primary_values),
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "structure_confidence_filter_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "structure_confidence_filtered",
            "structure_confidence_evidence_csv": str(evidence_csv),
            "structure_confidence_retained_candidates_csv": str(retained_csv),
            "structure_confidence_review_candidates_csv": str(review_csv),
            "structure_confidence_rejected_candidates_csv": str(rejected_csv),
            "structure_confidence_filter_summary_path": summary_path,
        },
    )
    downstream_candidate_csv = retained_csv if retained_rows else review_csv
    return {
        **summary,
        "summary_path": summary_path,
        # A relative-fallback or agent-rescue pool is a valid input to the
        # explicit structure checkpoint.  It must not be silently promoted to
        # a strict pass, but returning ``none`` here would strand a completed
        # prediction stage before that checkpoint can review the pool.
        "next_tool": "prepare_inverse_folding_scoring_job" if review_rows else "none",
        "agent_instruction": (
            (
                f"Use {downstream_candidate_csv} as candidates_csv for prepare_inverse_folding_scoring_job. "
                "Use structure_prediction_pdb_results_csv as structures_csv. Do not repeat structure confidence filtering "
                "unless thresholds or structure predictions change. "
                + (
                    f"{len(agent_review_rows)} additional candidates are in review_csv and must be "
                    "explicitly evaluated by the Biomni structure checkpoint; they are not auto-passed."
                    if agent_review_rows
                    else ""
                )
            )
            if review_rows
            else (
                (
                    f"No candidates passed the strict structure-confidence gate, but "
                    f"{len(agent_review_rows)} candidates are eligible for Biomni rescue review in review_csv. "
                    "Do not pass them downstream without an explicit Biomni checkpoint decision. "
                    "Candidates labelled relative_fallback_review were retained only because the caller explicitly enabled "
                    "relative fallback review; this does not change the strict pass criteria."
                    if agent_review_rows
                    else
                    "No candidates passed the strict structure-confidence gate. Stop the workflow and inspect "
                    "structure_confidence_evidence.csv/rejected_candidates.csv; do not prepare downstream jobs."
                )
            )
        ),
    }


@mcp.tool()
def prepare_inverse_folding_scoring_job(
    project_name: str,
    structures_csv: str,
    candidates_csv: str,
    model_set: str = "esm_if1,protein_mpnn",
    sequence_column: str = "sequence",
    variant_column: str = "variant",
    structure_column: str = "structure_path",
    chain_id: str = "A",
    output_dir: Optional[str] = None,
    top_n: Optional[int] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    execute: bool = False,
    timeout_seconds: int = 0,
) -> Dict[str, Any]:
    """Prepare inverse-folding validation jobs for predicted structures.

    This writes per-variant FASTA files and a shell script for ESM-IF1 and/or
    MPNN profile scoring. Score extraction/merging is handled separately.
    """
    candidate_rows, _ = _candidate_rows_from_csv(candidates_csv, top_n=top_n, sort_by=sort_by, ascending=ascending)
    structure_rows, _ = _read_csv_rows(Path(structures_csv).expanduser())
    structure_by_variant = {_sanitize_name(str(row.get(variant_column) or row.get("variant") or "")): row for row in structure_rows}

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "05_inverse_folding_scoring"
    fasta_dir = outdir / "per_variant_fastas"
    score_dir = outdir / "scores"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    model_names = [item.strip().lower().replace("-", "_") for item in model_set.split(",") if item.strip()]
    if not candidate_rows:
        job_table = outdir / "inverse_folding_jobs.csv"
        _write_csv_rows(job_table, [], ["variant", "sequence", "fasta_path", "structure_path"])
        log_dir = outdir / "logs"
        status_csv = outdir / "inverse_folding_command_status.csv"
        _write_csv_rows(
            status_csv,
            [],
            ["variant", "model", "status", "returncode", "log_path", "output_path", "structure_path"],
        )
        payload = {
            "models": model_names,
            "candidate_count": 0,
            "job_table": str(job_table),
            "fasta_dir": str(fasta_dir),
            "score_dir": str(score_dir),
            "log_dir": str(log_dir),
            "command_status_csv": str(status_csv),
            "run_script": "",
            "esm_if1_repo": "",
            "esm_if1_conda_env": "",
            "esm_if1_execution_mode": "",
            "esm_if1_score_script": "",
            "esm_if1_pythonpath_mode": "",
            "chain_id": chain_id,
            "warnings": ["No candidates were supplied; inverse-folding scoring was not prepared."],
            "missing_structure_count": 0,
            "non_pdb_structure_count": 0,
        }
        manifest_path = _write_json(outdir / "inverse_folding_job_manifest.json", payload)
        job = _job_record(project_name, "inverse_folding_scoring", "blocked_no_candidates", payload)
        _update_project_manifest(
            project_name,
            {
                "stage": "inverse_folding_scoring_blocked_no_candidates",
                "inverse_folding_job_manifest_path": manifest_path,
                "inverse_folding_job_table": str(job_table),
                "inverse_folding_log_dir": str(log_dir),
                "inverse_folding_command_status_csv": str(status_csv),
            },
        )
        return {
            "status": "blocked_no_candidates",
            "project_name": project_name,
            "job_id": job["job_id"],
            "job_manifest_path": manifest_path,
            **payload,
            "next_tool": "none",
            "stop_after_this_tool": True,
            "agent_instruction": (
                "No candidates remain for inverse-folding scoring. Stop the workflow; do not run the empty job."
            ),
        }

    jobs = []
    warnings: List[str] = []
    missing_structure_count = 0
    non_pdb_structure_count = 0
    for row in candidate_rows:
        variant = _variant_id(row, variant_column)
        seq = str(row.get(sequence_column) or "").strip().upper()
        structure_row = structure_by_variant.get(variant)
        structure_path = str((structure_row or {}).get(structure_column) or "")
        if not structure_path:
            missing_structure_count += 1
        elif Path(structure_path).suffix.lower() != ".pdb":
            non_pdb_structure_count += 1
        fasta_path = fasta_dir / f"{variant}.fasta"
        if seq:
            fasta_path.write_text(f">{variant}\n{seq}\n", encoding="utf-8")
        jobs.append({"variant": variant, "sequence": seq, "fasta_path": str(fasta_path), "structure_path": structure_path})
    if missing_structure_count:
        warnings.append(f"{missing_structure_count} candidates do not have a matched predicted structure.")
    if non_pdb_structure_count:
        warnings.append(
            f"{non_pdb_structure_count} matched structures are not PDB files. "
            "Run standardize_structure_prediction_results before inverse-folding scoring if ESM-IF1/MPNN rejects mmCIF."
        )

    job_table = outdir / "inverse_folding_jobs.csv"
    _write_csv_rows(job_table, jobs, ["variant", "sequence", "fasta_path", "structure_path"])

    esm_cfg = _tool_config("esm_if1")
    esm_env = _expand_path_text(str(os.environ.get("ESM_IF1_ENV") or esm_cfg.get("conda_env") or "current")).strip()
    esm_repo = _expand_path_text(str(esm_cfg.get("repo") or "${ESM_IF1_REPO:-./external/esm}"))
    esm_script = _expand_path_text(
        str(
            esm_cfg.get("score_script")
            or "${ESM_IF1_SCORE_SCRIPT:-./external/esm/examples/inverse_folding/score_log_likelihoods.py}"
        )
    )
    esm_env_lower = esm_env.lower()
    esm_use_current_env = esm_env_lower in {"", "current", "active", "none"}
    if esm_use_current_env:
        esm_cmd_prefix = f"PYTHONNOUSERSITE=1 PYTHONPATH={_quote_bash(esm_repo)} python"
        esm_execution_mode = "current_python"
    else:
        esm_cmd_prefix = (
            f"PYTHONNOUSERSITE=1 PYTHONPATH={_quote_bash(esm_repo)} "
            f"conda run --no-capture-output -n {_quote_bash(esm_env)} python"
        )
        esm_execution_mode = f"conda_env:{esm_env}"
    mpnn_model_type = "ligand_mpnn" if "ligand_mpnn" in model_names else "protein_mpnn"
    mpnn_paths = _default_ligandmpnn_paths(mpnn_model_type)
    log_dir = outdir / "logs"
    status_csv = outdir / "inverse_folding_command_status.csv"

    script_lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "",
        f"JOB_TABLE={_quote_bash(str(job_table))}",
        f"OUT_DIR={_quote_bash(str(score_dir))}",
        f"LOG_DIR={_quote_bash(str(log_dir))}",
        f"STATUS_CSV={_quote_bash(str(status_csv))}",
        "mkdir -p \"$OUT_DIR\"",
        "mkdir -p \"$LOG_DIR\"",
        "printf 'variant,model,status,returncode,log_path,output_path,structure_path\\n' > \"$STATUS_CSV\"",
        "echo \"START inverse-folding scoring: $(date)\"",
        "",
        "tail -n +2 \"$JOB_TABLE\" | while IFS=, read -r variant sequence fasta_path structure_path; do",
        "  variant=${variant//$'\\r'/}",
        "  sequence=${sequence//$'\\r'/}",
        "  fasta_path=${fasta_path//$'\\r'/}",
        "  structure_path=${structure_path//$'\\r'/}",
        "  if [[ -z \"$variant\" || -z \"$structure_path\" || ! -s \"$structure_path\" ]]; then",
        "    echo \"Skipping ${variant}: missing structure: ${structure_path}\" >&2",
        "    printf '%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"input\" \"failed_missing_structure\" \"\" \"\" \"\" \"$structure_path\" >> \"$STATUS_CSV\"",
        "    continue",
        "  fi",
        "  if [[ -z \"$fasta_path\" || ! -s \"$fasta_path\" ]]; then",
        "    echo \"Skipping ${variant}: missing FASTA: ${fasta_path}\" >&2",
        "    printf '%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"input\" \"failed_missing_fasta\" \"\" \"\" \"\" \"$structure_path\" >> \"$STATUS_CSV\"",
        "    continue",
        "  fi",
        "  echo \"Processing ${variant}\"",
    ]
    if "esm_if1" in model_names:
        esm_out = "\"$OUT_DIR/${variant}_esm_if1.csv\""
        esm_log = "\"$LOG_DIR/${variant}_esm_if1.log\""
        script_lines.extend(
            [
                f"  esm_out={esm_out}",
                f"  esm_log={esm_log}",
                "  echo \"  - ESM-IF1 -> ${esm_out}\"",
                f"  if {esm_cmd_prefix} {_quote_bash(esm_script)} \"$structure_path\" \"$fasta_path\" --chain {_quote_bash(chain_id)} --outpath \"$esm_out\" > \"$esm_log\" 2>&1; then",
                "    printf '%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"esm_if1\" \"completed\" \"0\" \"$esm_log\" \"$esm_out\" \"$structure_path\" >> \"$STATUS_CSV\"",
                "  else",
                "    rc=$?",
                "    echo \"    ESM-IF1 failed for ${variant}; see ${esm_log}\" >&2",
                "    printf '%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"esm_if1\" \"failed\" \"$rc\" \"$esm_log\" \"$esm_out\" \"$structure_path\" >> \"$STATUS_CSV\"",
                "  fi",
            ]
        )
    if "protein_mpnn" in model_names or "ligand_mpnn" in model_names:
        mpnn_out = f"\"$OUT_DIR/${{variant}}_{mpnn_model_type}\""
        mpnn_log = f"\"$LOG_DIR/${{variant}}_{mpnn_model_type}.log\""
        script_lines.extend(
            [
                f"  mpnn_out={mpnn_out}",
                f"  mpnn_log={mpnn_log}",
                "  mkdir -p \"$mpnn_out\"",
                f"  echo \"  - {mpnn_model_type} -> ${{mpnn_out}}\"",
                f"  if conda run -n {_quote_bash(mpnn_paths['conda_env'])} python {_quote_bash(mpnn_paths['score_script'])} --model_type {_quote_bash(mpnn_model_type)} {mpnn_paths['checkpoint_arg']} {_quote_bash(mpnn_paths['checkpoint'])} --pdb_path \"$structure_path\" --out_folder \"$mpnn_out\" --use_sequence 1 --single_aa_score 1 --autoregressive_score 0 --batch_size 1 --number_of_batches 1 --seed 0 --verbose 1 > \"$mpnn_log\" 2>&1; then",
                f"    printf '%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"{mpnn_model_type}\" \"completed\" \"0\" \"$mpnn_log\" \"$mpnn_out\" \"$structure_path\" >> \"$STATUS_CSV\"",
                "  else",
                "    rc=$?",
                f"    echo \"    {mpnn_model_type} failed for ${{variant}}; see ${{mpnn_log}}\" >&2",
                f"    printf '%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"{mpnn_model_type}\" \"failed\" \"$rc\" \"$mpnn_log\" \"$mpnn_out\" \"$structure_path\" >> \"$STATUS_CSV\"",
                "  fi",
            ]
        )
    script_lines.extend(
        [
            "done",
            "",
            "echo \"DONE inverse-folding scoring: $(date)\"",
            "echo \"Status CSV: $STATUS_CSV\"",
            "cat \"$STATUS_CSV\"",
        ]
    )

    script_path = outdir / "run_inverse_folding_scoring.sh"
    _write_shell_script(script_path, script_lines)

    payload = {
        "models": model_names,
        "candidate_count": len(jobs),
        "job_table": str(job_table),
        "fasta_dir": str(fasta_dir),
        "score_dir": str(score_dir),
        "log_dir": str(log_dir),
        "command_status_csv": str(status_csv),
        "run_script": str(script_path),
        "esm_if1_repo": esm_repo,
        "esm_if1_conda_env": esm_env,
        "esm_if1_execution_mode": esm_execution_mode,
        "esm_if1_score_script": esm_script,
        "esm_if1_pythonpath_mode": "PYTHONNOUSERSITE=1; PYTHONPATH set to esm_if1_repo only",
        "chain_id": chain_id,
        "warnings": warnings,
        "missing_structure_count": missing_structure_count,
        "non_pdb_structure_count": non_pdb_structure_count,
    }
    manifest_path = _write_json(outdir / "inverse_folding_job_manifest.json", payload)
    job = _job_record(project_name, "inverse_folding_scoring", "prepared", payload)
    result: Dict[str, Any] = {
        "status": "prepared",
        "project_name": project_name,
        "job_id": job["job_id"],
        "job_manifest_path": manifest_path,
        **payload,
        "stop_after_this_tool": True,
        "agent_instruction": "Return a final <solution> now. Do not run inverse-folding scoring inside the agent call.",
        "next_step": "Run run_script, then collect/merge score CSVs with collect_inverse_folding_scores.",
    }
    if execute:
        completed = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(outdir),
            text=True,
            capture_output=True,
            timeout=None if int(timeout_seconds) <= 0 else int(timeout_seconds),
        )
        status_rows, _status_fields = _read_csv_rows(status_csv)
        failed_status_count = sum(
            1
            for status_row in status_rows
            if str(status_row.get("status") or "").strip().lower().startswith("failed")
        )
        if completed.returncode != 0:
            execution_status = "failed"
        elif failed_status_count:
            execution_status = "completed_with_failures"
        else:
            execution_status = "completed"
        result.update(
            {
                "status": execution_status,
                "returncode": completed.returncode,
                "command_status_count": len(status_rows),
                "command_failed_count": failed_status_count,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )

    _update_project_manifest(
        project_name,
        {
            "stage": "inverse_folding_scoring_prepared",
            "inverse_folding_job_manifest_path": manifest_path,
            "inverse_folding_run_script": str(script_path),
            "inverse_folding_job_table": str(job_table),
            "inverse_folding_log_dir": str(log_dir),
            "inverse_folding_command_status_csv": str(status_csv),
        },
    )
    return result


def _infer_variant_from_score_path(path: Path, known_variants: List[str]) -> str:
    text = str(path)
    for variant in sorted(set(known_variants), key=len, reverse=True):
        if not variant:
            continue
        if re.search(rf"(^|[/\\_.-]){re.escape(variant)}($|[/\\_.-])", text):
            return variant
        if variant in path.name:
            return variant
    name = path.name
    for marker in ("_esm_if1", "_protein_mpnn", "_ligand_mpnn", "_mpnn"):
        if marker in name:
            return _sanitize_name(name.split(marker, 1)[0])
    for part in reversed(path.parts):
        for marker in ("_protein_mpnn", "_ligand_mpnn"):
            if marker in part:
                return _sanitize_name(part.split(marker, 1)[0])
    return _sanitize_name(path.stem)


def _infer_inverse_folding_model(path: Path) -> str:
    text = str(path).lower()
    if "esm" in text or "if1" in text:
        return "esm_if1"
    if "ligand_mpnn" in text or "ligandmpnn" in text:
        return "ligand_mpnn"
    if "protein_mpnn" in text or "proteinmpnn" in text:
        return "protein_mpnn"
    if path.suffix.lower() == ".pt" or "mpnn" in text:
        return "mpnn"
    return "unknown"


def _summarize_numeric_csv(path: Path) -> Dict[str, Any]:
    try:
        rows, fieldnames = _read_csv_rows(path)
    except Exception:
        return {}
    metrics: Dict[str, Any] = {}
    for field in fieldnames:
        values = [_safe_float(row.get(field)) for row in rows]
        numeric = [value for value in values if value is not None and math.isfinite(value)]
        if not numeric:
            continue
        clean = _sanitize_name(field).lower()
        metrics[f"{clean}_mean"] = mean(numeric)
        metrics[f"{clean}_min"] = min(numeric)
        metrics[f"{clean}_max"] = max(numeric)
        if len(numeric) > 1:
            metrics[f"{clean}_sd"] = pstdev(numeric)
    return metrics


def _choose_primary_numeric_metric(
    metrics: Dict[str, Any],
    *,
    prefer_higher: Optional[List[str]] = None,
) -> Optional[Tuple[str, float]]:
    numeric_items = [(key, _safe_float(value)) for key, value in metrics.items()]
    numeric_items = [(key, value) for key, value in numeric_items if value is not None and math.isfinite(value)]
    if not numeric_items:
        return None
    preferred_terms = [term.lower() for term in (prefer_higher or [])]
    for term in preferred_terms:
        for key, value in numeric_items:
            key_lower = key.lower()
            if term in key_lower and key_lower.endswith("_mean"):
                return key, float(value)
    for key, value in numeric_items:
        if key.lower().endswith("_mean"):
            return key, float(value)
    key, value = numeric_items[0]
    return key, float(value)


@mcp.tool()
def collect_inverse_folding_scores(
    project_name: str,
    score_dir: str,
    output_dir: Optional[str] = None,
    candidates_csv: Optional[str] = None,
    job_manifest_path: Optional[str] = None,
    sequence_column: str = "sequence",
    variant_column: str = "variant",
) -> Dict[str, Any]:
    """Collect inverse-folding score files and extract numeric evidence when possible."""
    root = Path(score_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"score_dir not found: {score_dir}")
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "05_inverse_folding_scoring"
    outdir.mkdir(parents=True, exist_ok=True)

    if job_manifest_path:
        manifest = _read_json(Path(job_manifest_path).expanduser())
        candidates_csv = candidates_csv or manifest.get("candidates_csv")

    candidate_sequences: Dict[str, str] = {}
    known_variants: List[str] = []
    if candidates_csv:
        candidate_rows, _candidate_fields = _read_csv_rows(Path(candidates_csv).expanduser())
        for row in candidate_rows:
            variant = _variant_id(row, variant_column)
            known_variants.append(variant)
            seq = str(row.get(sequence_column) or "").strip().upper()
            if seq:
                candidate_sequences[variant] = seq

    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = []
    evidence_by_variant: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".pt", ".json", ".txt"}:
            continue
        name = path.name
        variant = _infer_variant_from_score_path(path, known_variants)
        model = _infer_inverse_folding_model(path)
        rows.append({"variant": variant, "model": model, "file_path": str(path), "file_type": suffix.lstrip(".")})
        evidence = evidence_by_variant.setdefault(variant, {"variant": variant})
        evidence["inverse_folding_file_count"] = int(evidence.get("inverse_folding_file_count") or 0) + 1

        if suffix == ".csv" and model == "esm_if1":
            metrics = _summarize_numeric_csv(path)
            for key, value in metrics.items():
                evidence[f"esm_if1_{key}"] = value
            primary = _choose_primary_numeric_metric(metrics, prefer_higher=["log_likelihood", "ll", "score"])
            if primary:
                metric_name, metric_value = primary
                evidence["esm_if1_primary_metric"] = metric_name
                evidence["esm_if1_primary_value"] = metric_value
                if "nll" in metric_name.lower() or "perplex" in metric_name.lower():
                    evidence["esm_if1_nll"] = metric_value
                    evidence["esm_if1_compatibility_score"] = -metric_value
                else:
                    evidence["esm_if1_log_likelihood"] = metric_value
                    evidence["esm_if1_nll"] = -metric_value
                    evidence["esm_if1_compatibility_score"] = metric_value
        elif suffix == ".pt" and model in {"protein_mpnn", "ligand_mpnn", "mpnn"}:
            seq = candidate_sequences.get(variant, "")
            if not seq:
                warnings.append(f"Cannot numerically score {path.name}: candidate sequence for {variant} is missing.")
                continue
            if not _torch_available():
                warnings.append(
                    f"Cannot numerically parse {path.name}: PyTorch is unavailable in the current MCP environment. "
                    "Run collection from ligandmpnn_env or use the inventory table only."
                )
                continue
            try:
                profile = _load_ligandmpnn_pt_profile(str(path))
                score = _score_full_sequence_delta(profile, {sequence_column: seq}, sequence_column=sequence_column)
                prefix = "ligand_mpnn" if model == "ligand_mpnn" else "protein_mpnn"
                evidence[f"{prefix}_score_status"] = score.get("score_status")
                evidence[f"{prefix}_log_prob"] = score.get("mut_log_prob")
                evidence[f"{prefix}_nll"] = -score["mut_log_prob"] if score.get("mut_log_prob") is not None else None
                evidence[f"{prefix}_delta_log_prob"] = score.get("delta_log_prob")
                evidence[f"{prefix}_delta_nll"] = score.get("delta_nll")
                evidence[f"{prefix}_compatibility_score"] = score.get("mut_log_prob")
            except Exception as exc:
                warnings.append(f"Failed to parse {path.name}: {exc}")

    output_csv = outdir / "inverse_folding_score_files.csv"
    _write_csv_rows(output_csv, rows, ["variant", "model", "file_path", "file_type"])
    evidence_rows = list(evidence_by_variant.values())
    for row in evidence_rows:
        scores = [
            _safe_float(row.get("esm_if1_compatibility_score")),
            _safe_float(row.get("protein_mpnn_compatibility_score")),
            _safe_float(row.get("ligand_mpnn_compatibility_score")),
        ]
        scores = [score for score in scores if score is not None and math.isfinite(score)]
        row["inverse_folding_model_count"] = len(scores)
        row["inverse_folding_compatibility_score"] = mean(scores) if scores else None
        row["inverse_folding_status"] = "scored" if scores else "inventory_only"
    evidence_csv = outdir / "inverse_folding_evidence.csv"
    _write_csv_rows(
        evidence_csv,
        evidence_rows,
        _csv_fieldnames_union(
            evidence_rows,
            [
                "variant",
                "inverse_folding_compatibility_score",
                "inverse_folding_model_count",
                "inverse_folding_status",
                "esm_if1_nll",
                "protein_mpnn_nll",
                "ligand_mpnn_nll",
            ],
        ),
    )
    summary = {
        "status": "completed",
        "project_name": project_name,
        "score_dir": str(root),
        "output_csv": str(output_csv),
        "evidence_csv": str(evidence_csv),
        "score_file_count": len(rows),
        "evidence_variant_count": len(evidence_rows),
        "scored_variant_count": sum(1 for row in evidence_rows if row.get("inverse_folding_status") == "scored"),
        "warnings": warnings[:50],
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "inverse_folding_score_collection_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "inverse_folding_scores_collected",
            "inverse_folding_score_files_csv": str(output_csv),
            "inverse_folding_evidence_csv": str(evidence_csv),
            "inverse_folding_score_collection_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


@mcp.tool()
def select_candidates_from_evidence(
    project_name: str,
    stage_name: str = "bioemu",
    candidate_csv: Optional[str] = None,
    cb_scored_csv: Optional[str] = None,
    structure_prediction_evidence_csv: Optional[str] = None,
    structure_state_similarity_evidence_csv: Optional[str] = None,
    structure_confidence_evidence_csv: Optional[str] = None,
    inverse_folding_evidence_csv: Optional[str] = None,
    bioemu_evidence_csv: Optional[str] = None,
    rosetta_evidence_csv: Optional[str] = None,
    extra_evidence_csvs: str = "",
    output_dir: Optional[str] = None,
    variant_column: str = "variant",
    metric_columns: str = "",
    top_n: Optional[int] = None,
    retain_all_candidates: bool = False,
    max_per_mutation_site: Optional[int] = None,
    site_diversity_policy: str = "site_round_robin_then_rank",
) -> Dict[str, Any]:
    """Select candidates for the next SwitchTune stage from available evidence.

    This is a lightweight stage-gate tool for agentic runs. It uses partial
    evidence, for example CB + structure + IF scores before BioEmu, without
    implying that the final full consensus ranking is complete.

    When ``retain_all_candidates`` is true, the tool still computes ranking
    metadata but does not use ``top_n`` to prune the current candidate set.
    This supports ranking-only stages such as Rosetta in the formal workflow.

    When ``max_per_mutation_site`` is set, selection uses a site-aware
    round-robin policy instead of a global top-N. This is used for BioEmu
    because a global evidence ranking can otherwise spend the entire
    downstream budget on substitutions at one residue.
    """
    if max_per_mutation_site is not None and int(max_per_mutation_site) <= 0:
        raise ValueError("max_per_mutation_site must be positive when provided.")
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    stage_clean = _sanitize_name(stage_name or "next_stage").lower()
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "03_state_specific_candidates"
    outdir.mkdir(parents=True, exist_ok=True)

    candidate_csv = candidate_csv or _first_nonempty_csv_path(
        manifest.get(f"{stage_clean}_candidate_csv"),
        manifest.get("bioemu_candidate_csv") if stage_clean == "rosetta" else None,
        manifest.get("structure_confidence_retained_candidates_csv"),
        manifest.get("structure_confidence_review_candidates_csv"),
        manifest.get("state_specific_candidates_adaptive_csv"),
        manifest.get("state_specific_candidates_csv"),
        manifest.get("library_csv"),
    )
    cb_scored_csv = cb_scored_csv or manifest.get("cb_scored_library_csv")
    structure_prediction_evidence_csv = (
        structure_prediction_evidence_csv or manifest.get("structure_prediction_evidence_csv")
    )
    structure_state_similarity_evidence_csv = (
        structure_state_similarity_evidence_csv or manifest.get("structure_state_similarity_evidence_csv")
    )
    structure_confidence_evidence_csv = (
        structure_confidence_evidence_csv or manifest.get("structure_confidence_evidence_csv")
    )
    inverse_folding_evidence_csv = inverse_folding_evidence_csv or manifest.get("inverse_folding_evidence_csv")
    bioemu_evidence_csv = bioemu_evidence_csv or manifest.get("bioemu_evidence_csv")
    rosetta_evidence_csv = rosetta_evidence_csv or manifest.get("rosetta_evidence_csv")

    sources: List[Tuple[str, str]] = []
    for source_name, path_text in [
        ("candidate", candidate_csv),
        ("cb", cb_scored_csv),
        ("structure_prediction", structure_prediction_evidence_csv),
        ("structure", structure_state_similarity_evidence_csv),
        ("structure_confidence", structure_confidence_evidence_csv),
        ("inverse_folding", inverse_folding_evidence_csv),
        ("bioemu", bioemu_evidence_csv),
        ("rosetta", rosetta_evidence_csv),
    ]:
        if path_text:
            sources.append((source_name, str(path_text)))
    for idx, path_text in enumerate([item.strip() for item in str(extra_evidence_csvs).split(",") if item.strip()], start=1):
        sources.append((f"extra{idx}", path_text))
    if not sources:
        raise ValueError("No candidate or evidence CSVs were provided and none were found in the project manifest.")

    allowed_variants: Optional[set[str]] = None
    if candidate_csv:
        candidate_path = _resolve_project_output_path(str(candidate_csv), project_name)
        candidate_rows, _candidate_fields = _read_csv_rows(candidate_path)
        allowed_variants = {
            _variant_id(row, variant_column)
            for row in candidate_rows
            if _variant_id(row, variant_column)
        }

    merged: Dict[str, Dict[str, Any]] = {}
    source_summaries: List[Dict[str, Any]] = []
    for source_name, path_text in sources:
        rows, _fields, source_summary = _read_optional_evidence_csv(path_text, source_name)
        source_summaries.append(source_summary)
        _merge_table_into_variants(
            merged,
            rows,
            source_name=source_name,
            variant_column=variant_column,
            allowed_variants=allowed_variants,
        )

    merged_rows = list(merged.values())
    fields = _csv_fieldnames_union(merged_rows, ["variant"])
    selected_metrics = [item.strip() for item in str(metric_columns).split(",") if item.strip()]
    if not selected_metrics:
        selected_metrics = _auto_consensus_metric_columns(fields)
    selected_metrics = [metric for metric in selected_metrics if metric in fields]

    component_values: Dict[str, List[Optional[float]]] = {}
    for metric in selected_metrics:
        direction = _infer_metric_direction(metric)
        raw_values = [_bool_or_numeric(row.get(metric)) for row in merged_rows]
        transformed = [
            (-value if direction == "lower" else value) if value is not None and math.isfinite(value) else None
            for value in raw_values
        ]
        component_values[metric] = _z_scale(transformed)

    for idx, row in enumerate(merged_rows):
        components: List[float] = []
        component_names: List[str] = []
        for metric in selected_metrics:
            value = component_values[metric][idx]
            if value is None or not math.isfinite(value):
                continue
            row[f"selection_component__{metric}"] = value
            components.append(value)
            component_names.append(metric)
        row["selection_stage"] = stage_clean
        row["selection_evidence_count"] = len(components)
        row["selection_metric_columns"] = ";".join(component_names)
        row["selection_score"] = mean(components) if components else None
        row["selection_status"] = "ranked" if components else "no_numeric_evidence"

    ranked_rows = sorted(
        merged_rows,
        key=lambda row: (
            0 if _safe_float(row.get("selection_score")) is not None else 1,
            -(_safe_float(row.get("selection_score")) or float("-inf")),
            str(row.get("variant")),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["selection_rank"] = rank if row.get("selection_status") == "ranked" else ""

    ranked_numeric_rows = [
        row for row in ranked_rows
        if row.get("selection_status") == "ranked"
    ]
    site_diversity_info: Dict[str, Any] = {
        "policy": "none",
        "max_per_mutation_site": None,
        "input_site_count": len(
            {
                _row_mutation_site(row, variant_column=variant_column) or "unknown"
                for row in merged_rows
            }
        ),
        "selected_site_count": 0,
        "site_counts_before": {},
        "site_counts_after": {},
        "maximum_selected_site_count": 0,
        "diversity_backfill_variants": [],
        "diversity_backfill_count": 0,
        "replaced_preferred_variants": [],
        "replaced_preferred_count": 0,
        "site_cap_limited": False,
    }
    adaptive_budget = None
    if not retain_all_candidates and (top_n is None or int(top_n) <= 0):
        adaptive_budget = adaptive_candidate_budget(len(ranked_numeric_rows))
        requested_count = adaptive_budget
    else:
        requested_count = int(top_n) if top_n is not None and int(top_n) > 0 else len(ranked_numeric_rows)

    if retain_all_candidates:
        # Preserve rows with unavailable downstream evidence as explicit
        # candidates. They can still carry valid upstream evidence and should
        # not silently disappear before the ranking-only Rosetta pass.
        selected_rows = ranked_rows
    else:
        if max_per_mutation_site is not None:
            selected_rows, site_diversity_info = select_rows_with_site_diversity(
                ranked_numeric_rows,
                target_count=requested_count,
                max_per_mutation_site=int(max_per_mutation_site),
                variant_column=variant_column,
            )
            site_diversity_info["policy"] = str(site_diversity_policy or "site_round_robin_then_rank")
        else:
            selected_rows = ranked_numeric_rows[:requested_count]
            selected_counts = Counter(
                _row_mutation_site(row, variant_column=variant_column) or "unknown"
                for row in selected_rows
            )
            site_diversity_info.update(
                {
                    "selected_site_count": len(selected_counts),
                    "site_counts_before": dict(
                        sorted(
                            Counter(
                                _row_mutation_site(row, variant_column=variant_column) or "unknown"
                                for row in merged_rows
                            ).items()
                        )
                    ),
                    "site_counts_after": dict(sorted(selected_counts.items())),
                    "maximum_selected_site_count": max(selected_counts.values()) if selected_counts else 0,
                }
            )
    merged_csv = outdir / f"{stage_clean}_candidate_evidence_merged.csv"
    output_csv = outdir / f"{stage_clean}_selected_candidates.csv"
    preferred = [
        "selection_rank",
        "variant",
        "selection_stage",
        "selection_score",
        "selection_evidence_count",
        "selection_status",
        "sequence",
    ]
    final_fields = _csv_fieldnames_union(ranked_rows, preferred + fields)
    _write_csv_rows(merged_csv, ranked_rows, final_fields)
    _write_csv_rows(output_csv, selected_rows, final_fields)

    selection_status = "completed" if selected_rows else "blocked_no_candidates"
    retention_policy = (
        "retain_all_candidates_ranking_only"
        if retain_all_candidates
        else (
            f"{site_diversity_policy}_max_{int(max_per_mutation_site)}_per_site"
            if max_per_mutation_site is not None
            else "top_n_ranked_candidates"
        )
    )
    summary = {
        "status": selection_status,
        "project_name": project_name,
        "stage_name": stage_clean,
        "merged_csv": str(merged_csv),
        "output_csv": str(output_csv),
        "source_summaries": source_summaries,
        "input_variant_count": len(merged_rows),
        "ranked_variant_count": len(ranked_numeric_rows),
        "selected_variant_count": len(selected_rows),
        "selected_metric_columns": selected_metrics,
        "metric_directions": {metric: _infer_metric_direction(metric) for metric in selected_metrics},
        "top_n": int(requested_count),
        "adaptive_budget": adaptive_budget,
        "retain_all_candidates": bool(retain_all_candidates),
        "retention_policy": retention_policy,
        "max_per_mutation_site": (
            int(max_per_mutation_site)
            if max_per_mutation_site is not None
            else None
        ),
        "site_diversity_policy": (
            str(site_diversity_policy)
            if max_per_mutation_site is not None
            else "none"
        ),
        "site_diversity": site_diversity_info,
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / f"{stage_clean}_candidate_selection_summary.json", summary)
    manifest_updates = {
        "stage": (
            f"{stage_clean}_candidates_selected"
            if selected_rows
            else f"{stage_clean}_candidates_blocked_no_candidates"
        ),
        "selected_candidates_csv": str(output_csv),
        "selected_candidates_stage": stage_clean,
        "selected_candidates_summary_path": summary_path,
    }
    if stage_clean:
        manifest_updates[f"{stage_clean}_candidate_csv"] = str(output_csv)
        manifest_updates[f"{stage_clean}_candidate_selection_summary_path"] = summary_path
    _update_project_manifest(project_name, manifest_updates)

    next_tool = "prepare_bioemu_sampling_job" if stage_clean == "bioemu" else "prepare_rosetta_energy_job"
    return {
        **summary,
        "summary_path": summary_path,
        "next_tool": next_tool if selected_rows else "none",
        "agent_instruction": (
            (
                f"Use output_csv as candidates_csv for {next_tool}. The output is "
                "ranking-only and intentionally retains the full current candidate set; "
                "do not prune it again before the downstream job."
            )
            if retain_all_candidates and selected_rows
            else
            (
                f"Use output_csv as candidates_csv for {next_tool}. Do not repeat "
                "candidate selection unless the downstream budget changes. "
                + (
                    f"The site-diversity guard enforced at most {int(max_per_mutation_site)} "
                    "variants per mutation site and used site coverage before global ranking."
                    if max_per_mutation_site is not None
                    else ""
                )
                if selected_rows
                else
                "No candidates remain after evidence selection. Stop the workflow and "
                "do not prepare a downstream job."
            )
        ),
    }


@mcp.tool()
def prepare_bioemu_sampling_job(
    project_name: str,
    candidates_csv: str,
    sequence_column: str = "sequence",
    variant_column: str = "variant",
    top_n: Optional[int] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    num_samples: int = 2000,
    batch_size: int = 10,
    output_dir: Optional[str] = None,
    extra_args: str = "",
    execute: bool = False,
    timeout_seconds: int = 0,
) -> Dict[str, Any]:
    """Prepare a BioEmu sampling batch job for selected variants."""
    rows, _ = _candidate_rows_from_csv(candidates_csv, top_n=top_n, sort_by=sort_by, ascending=ascending)
    cfg = _tool_config("bioemu")
    # The registry keeps a portable default, but servers often install BioEmu
    # into the main Biomni environment.  Allow an explicit per-run override so
    # a stale registry entry cannot silently invoke a different environment.
    conda_env = str(
        os.environ.get("SWITCHTUNE_BIOEMU_CONDA_ENV")
        or cfg.get("conda_env")
        or "bioemu"
    ).strip()
    module = str(cfg.get("module") or "bioemu.sample")
    requested_device = str(os.environ.get("SWITCHTUNE_BIOEMU_DEVICE") or "auto").strip().lower()
    if requested_device not in {"auto", "cuda", "cpu"}:
        requested_device = "auto"
    batch_size_100 = max(1, int(batch_size))
    extra_args_text = str(extra_args or "").strip()
    if not re.search(r"(^|\s)--batch_size_100(?:=|\s|$)", extra_args_text):
        extra_args_text = f"--batch_size_100 {batch_size_100} {extra_args_text}".strip()

    # Avoid an implicit ColabFold/MMseqs2 network call for every sequence.
    # The generated jobs use a minimal local A3M and explicit caches; the
    # model checkpoint is downloaded once (or supplied through the two path
    # environment variables below).
    bioemu_model_name = str(os.environ.get("SWITCHTUNE_BIOEMU_MODEL", "bioemu-v1.1")).strip()
    bioemu_ckpt_path = str(os.environ.get("SWITCHTUNE_BIOEMU_CKPT_PATH", "")).strip()
    bioemu_model_config_path = str(os.environ.get("SWITCHTUNE_BIOEMU_MODEL_CONFIG_PATH", "")).strip()
    # Expand user paths while generating the job.  Quoted shell variables do
    # not perform tilde expansion, so leaving ``~`` here would create literal
    # directories named ``~`` in the project working directory.
    bioemu_embeds_dir = str(
        Path(os.environ.get("SWITCHTUNE_BIOEMU_EMBEDS_DIR", "~/.bioemu_embeds_cache")).expanduser()
    )
    bioemu_so3_dir = str(
        Path(os.environ.get("SWITCHTUNE_BIOEMU_SO3_DIR", "~/bioemu_so3_cache")).expanduser()
    )
    bioemu_hf_home = str(
        Path(os.environ.get("SWITCHTUNE_BIOEMU_HF_HOME", "~/.cache/huggingface")).expanduser()
    )
    # BioEmu's inlined AlphaFold2 embedding runner uses the fixed path
    # ``~/.cache/colabfold``.  Point that path at a configurable persistent
    # cache by giving the subprocess a small runtime HOME and a symlink.
    bioemu_colabfold_dir = str(
        Path(os.environ.get("SWITCHTUNE_BIOEMU_COLABFOLD_DIR", "~/.cache/colabfold")).expanduser()
    )
    bioemu_runtime_home = str(
        Path(os.environ.get("SWITCHTUNE_BIOEMU_RUNTIME_HOME", "~/.switchtune_bioemu_home")).expanduser()
    )
    bioemu_offline = str(os.environ.get("SWITCHTUNE_BIOEMU_OFFLINE", "0")).strip().lower() in {"1", "true", "yes", "on"}

    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "06_bioemu_sampling"
    sample_root = outdir / "samples"
    msa_root = outdir / "msa_inputs"
    sample_root.mkdir(parents=True, exist_ok=True)
    msa_root.mkdir(parents=True, exist_ok=True)

    if not rows:
        job_table = outdir / "bioemu_sampling_jobs.csv"
        _write_csv_rows(job_table, [], ["variant", "sequence", "output_dir"])
        log_dir = outdir / "logs"
        status_csv = outdir / "bioemu_command_status.csv"
        _write_csv_rows(
            status_csv,
            [],
            [
                "variant",
                "status",
                "returncode",
                "log_path",
                "output_dir",
                "sequence_length",
                "execution_device",
                "fallback_used",
                "error_summary",
            ],
        )
        payload = {
            "candidate_count": 0,
            "num_samples": int(num_samples),
            "batch_size": int(batch_size),
            "batch_size_100": batch_size_100,
            "device": requested_device,
            "job_table": str(job_table),
            "sample_root": str(sample_root),
            "run_script": "",
            "log_dir": str(log_dir),
            "command_status_csv": str(status_csv),
            "conda_env": conda_env,
            "module": module,
            "warnings": ["No candidates were supplied; BioEmu sampling was not prepared."],
        }
        manifest_path = _write_json(outdir / "bioemu_sampling_job_manifest.json", payload)
        job = _job_record(project_name, "bioemu_sampling", "blocked_no_candidates", payload)
        _update_project_manifest(
            project_name,
            {
                "stage": "bioemu_sampling_blocked_no_candidates",
                "bioemu_sampling_job_manifest_path": manifest_path,
                "bioemu_sample_root": str(sample_root),
                "bioemu_command_status_csv": str(status_csv),
            },
        )
        return {
            "status": "blocked_no_candidates",
            "project_name": project_name,
            "job_id": job["job_id"],
            "job_manifest_path": manifest_path,
            **payload,
            "next_tool": "none",
            "stop_after_this_tool": True,
            "agent_instruction": "No candidates remain for BioEmu sampling. Stop the workflow; do not run an empty job.",
        }

    job_rows = []
    for row in rows:
        variant = _variant_id(row, variant_column)
        seq = str(row.get(sequence_column) or "").strip().upper()
        msa_path = msa_root / f"{variant}.a3m"
        msa_path.write_text(f">{variant}\n{seq}\n", encoding="utf-8")
        job_rows.append({"variant": variant, "sequence": seq, "msa_path": str(msa_path), "output_dir": str(sample_root / variant)})
    job_table = outdir / "bioemu_sampling_jobs.csv"
    _write_csv_rows(job_table, job_rows, ["variant", "sequence", "msa_path", "output_dir"])

    script_path = outdir / "run_bioemu_sampling.sh"
    log_dir = outdir / "logs"
    status_csv = outdir / "bioemu_command_status.csv"
    script_lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "",
        *_conda_run_prefix(conda_env),
        "",
        f"JOB_TABLE={_quote_bash(str(job_table))}",
        f"LOG_DIR={_quote_bash(str(log_dir))}",
        f"STATUS_CSV={_quote_bash(str(status_csv))}",
        f"BIOEMU_DEVICE={_quote_bash(requested_device)}",
        f"EMBEDS_DIR={_quote_bash(bioemu_embeds_dir)}",
        f"SO3_DIR={_quote_bash(bioemu_so3_dir)}",
        f"HF_HOME_DIR={_quote_bash(bioemu_hf_home)}",
        f"COLABFOLD_DIR={_quote_bash(bioemu_colabfold_dir)}",
        f"RUNTIME_HOME={_quote_bash(bioemu_runtime_home)}",
        f"MODEL_NAME={_quote_bash(bioemu_model_name)}",
        f"CKPT_PATH={_quote_bash(bioemu_ckpt_path)}",
        f"MODEL_CONFIG_PATH={_quote_bash(bioemu_model_config_path)}",
        f"BIOEMU_OFFLINE={_quote_bash('1' if bioemu_offline else '0')}",
        "mkdir -p \"$LOG_DIR\"",
        "mkdir -p \"$EMBEDS_DIR\" \"$SO3_DIR\" \"$HF_HOME_DIR\"",
        "mkdir -p \"$RUNTIME_HOME/.cache\"",
        "if [[ \"$COLABFOLD_DIR\" != \"$RUNTIME_HOME/.cache/colabfold\" ]]; then",
        "  mkdir -p \"$COLABFOLD_DIR\"",
        "  ln -sfn \"$COLABFOLD_DIR\" \"$RUNTIME_HOME/.cache/colabfold\"",
        "fi",
        "export HF_HOME=\"$HF_HOME_DIR\"",
        "export HF_HUB_CACHE=\"$HF_HOME_DIR/hub\"",
        "export HF_DATASETS_CACHE=\"$HF_HOME_DIR/datasets\"",
        "export TRANSFORMERS_CACHE=\"$HF_HOME_DIR/transformers\"",
        "if [[ \"$BIOEMU_OFFLINE\" == \"1\" ]]; then export HF_HUB_OFFLINE=1; fi",
        "if [[ ! -f \"$COLABFOLD_DIR/params/params_model_3.npz\" ]]; then",
        "  echo \"Missing AlphaFold2 params: $COLABFOLD_DIR/params/params_model_3.npz\" >&2",
        "  echo \"Pre-cache AlphaFold2 model_3 weights there before launching the 30-candidate batch.\" >&2",
        "  exit 66",
        "fi",
        "printf 'variant,status,returncode,log_path,output_dir,sequence_length,execution_device,fallback_used,error_summary\\n' > \"$STATUS_CSV\"",
        "echo \"START BioEmu sampling: $(date)\"",
        "tail -n +2 \"$JOB_TABLE\" | while IFS=, read -r variant sequence msa_path output_dir; do",
        "  variant=${variant//$'\\r'/}",
        "  sequence=${sequence//$'\\r'/}",
        "  msa_path=${msa_path//$'\\r'/}",
        "  output_dir=${output_dir//$'\\r'/}",
        "  seq_len=${#sequence}",
        "  bioemu_log=\"$LOG_DIR/${variant}_bioemu.log\"",
        "  if [[ -z \"$variant\" || -z \"$sequence\" ]]; then",
        "    echo \"Skipping ${variant}: missing sequence\" >&2",
        "    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"failed_missing_sequence\" \"\" \"$bioemu_log\" \"$output_dir\" \"$seq_len\" \"$BIOEMU_DEVICE\" \"false\" \"missing sequence\" >> \"$STATUS_CSV\"",
        "    continue",
        "  fi",
        "  mkdir -p \"$output_dir\"",
        "  echo \"Processing ${variant} (${seq_len} aa)\"",
        "  rc=0",
        "  model_args=()",
        "  if [[ -n \"$CKPT_PATH\" && -n \"$MODEL_CONFIG_PATH\" ]]; then",
        "    model_args+=(--model_name None --ckpt_path \"$CKPT_PATH\" --model_config_path \"$MODEL_CONFIG_PATH\")",
        "  else",
        "    model_args+=(--model_name \"$MODEL_NAME\")",
        "  fi",
        "  if [[ \"$BIOEMU_DEVICE\" == \"cpu\" ]]; then",
        f"    env HOME=\"$RUNTIME_HOME\" CUDA_VISIBLE_DEVICES='' ${{CONDA_RUN[@]}} python -m {_quote_bash(module)} --sequence \"$msa_path\" --num_samples {int(num_samples)} --output_dir \"$output_dir\" --cache_embeds_dir \"$EMBEDS_DIR\" --cache_so3_dir \"$SO3_DIR\" \"${{model_args[@]}}\" {extra_args_text} > \"$bioemu_log\" 2>&1 || rc=$?".rstrip(),
        "  else",
        f"    env HOME=\"$RUNTIME_HOME\" ${{CONDA_RUN[@]}} python -m {_quote_bash(module)} --sequence \"$msa_path\" --num_samples {int(num_samples)} --output_dir \"$output_dir\" --cache_embeds_dir \"$EMBEDS_DIR\" --cache_so3_dir \"$SO3_DIR\" \"${{model_args[@]}}\" {extra_args_text} > \"$bioemu_log\" 2>&1 || rc=$?".rstrip(),
        "  fi",
        "  npz_count=$(find \"$output_dir\" -type f -name '*.npz' -print 2>/dev/null | wc -l | tr -d ' ')",
        "  if [[ \"$rc\" -eq 0 && \"$npz_count\" -gt 0 ]]; then",
        "    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"completed\" \"0\" \"$bioemu_log\" \"$output_dir\" \"$seq_len\" \"$BIOEMU_DEVICE\" \"false\" \"\" >> \"$STATUS_CSV\"",
        "  elif [[ \"$rc\" -eq 0 ]]; then",
        "    error_summary='BioEmu exited successfully but produced no NPZ coordinate batches.'",
        "    echo \"BioEmu produced no NPZ coordinate batches for ${variant}; see ${bioemu_log}\" >&2",
        "    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"failed_missing_output\" \"0\" \"$bioemu_log\" \"$output_dir\" \"$seq_len\" \"$BIOEMU_DEVICE\" \"false\" \"$error_summary\" >> \"$STATUS_CSV\"",
        "  else",
        "    if grep -Eiq 'outofmemoryerror|cuda out of memory|cuda error' \"$bioemu_log\"; then",
        "      error_summary='CUDA OOM or CUDA error; set SWITCHTUNE_BIOEMU_DEVICE=cpu or use a larger GPU.'",
        "    elif grep -Eiq 'timeout|connection reset|huggingface|hf_hub_download|mmseqs|colabfold|connection refused|name or service not known' \"$bioemu_log\"; then",
        "      error_summary='blocked_external_bioemu_download_or_msa; use local A3M and cached checkpoint/embedding directories.'",
        "    else",
        "      error_summary='BioEmu command failed; inspect the per-variant log.'",
        "    fi",
        "    echo \"BioEmu failed for ${variant}; see ${bioemu_log}\" >&2",
        "    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$variant\" \"failed\" \"$rc\" \"$bioemu_log\" \"$output_dir\" \"$seq_len\" \"$BIOEMU_DEVICE\" \"false\" \"$error_summary\" >> \"$STATUS_CSV\"",
        "  fi",
        "done",
        "echo \"DONE BioEmu sampling: $(date)\"",
        "echo \"Status CSV: $STATUS_CSV\"",
        "cat \"$STATUS_CSV\"",
    ]
    _write_shell_script(script_path, script_lines)

    payload = {
        "candidate_count": len(job_rows),
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "batch_size_100": batch_size_100,
        "device": requested_device,
        "job_table": str(job_table),
        "sample_root": str(sample_root),
        "run_script": str(script_path),
        "log_dir": str(log_dir),
        "command_status_csv": str(status_csv),
        "conda_env": conda_env,
        "module": module,
        "extra_args": extra_args_text,
        "msa_dir": str(msa_root),
        "embedding_cache_dir": bioemu_embeds_dir,
        "so3_cache_dir": bioemu_so3_dir,
        "hf_home": bioemu_hf_home,
        "colabfold_dir": bioemu_colabfold_dir,
        "runtime_home": bioemu_runtime_home,
        "model_name": bioemu_model_name,
        "checkpoint_path": bioemu_ckpt_path,
        "model_config_path": bioemu_model_config_path,
        "offline_mode": bioemu_offline,
    }
    manifest_path = _write_json(outdir / "bioemu_sampling_job_manifest.json", payload)
    job = _job_record(project_name, "bioemu_sampling", "prepared", payload)
    result: Dict[str, Any] = {
        "status": "prepared",
        "project_name": project_name,
        "job_id": job["job_id"],
        "job_manifest_path": manifest_path,
        **payload,
        "stop_after_this_tool": True,
        "agent_instruction": "Return a final <solution> now. Do not run BioEmu inside the agent call.",
        "next_step": "Run run_script, then call collect_bioemu_outputs.",
    }
    if execute:
        completed = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(outdir),
            text=True,
            capture_output=True,
            timeout=None if int(timeout_seconds) <= 0 else int(timeout_seconds),
        )
        result.update(
            {
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )

    _update_project_manifest(
        project_name,
        {
            "stage": "bioemu_sampling_prepared",
            "bioemu_sampling_job_manifest_path": manifest_path,
            "bioemu_sampling_run_script": str(script_path),
            "bioemu_sample_root": str(sample_root),
            "bioemu_command_status_csv": str(status_csv),
            "bioemu_device": requested_device,
            "bioemu_batch_size_100": batch_size_100,
        },
    )
    return result


@mcp.tool()
def collect_bioemu_outputs(
    project_name: str,
    sample_root: str,
    output_dir: Optional[str] = None,
    target_state_structure: Optional[str] = None,
    reference_state_structure: Optional[str] = None,
    mutation_region: Optional[str] = None,
    first_residue_number: Optional[int] = None,
    chain_id: str = "A",
    compute_metrics: bool = True,
) -> Dict[str, Any]:
    """Collect BioEmu outputs and compute ensemble-level evidence.

    Inventory and command-status rows define the allowed variant set. Optional
    analysis CSVs are merged only for those variants, which prevents stale or
    unrelated BioEmu output directories from entering downstream selection.
    """
    root = Path(sample_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"sample_root not found: {sample_root}")
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "06_bioemu_sampling"
    outdir.mkdir(parents=True, exist_ok=True)

    project_manifest = _load_project_manifest(project_name)
    target_value = (
        target_state_structure
        if target_state_structure is not None
        else project_manifest.get("target_state_structure")
    )
    reference_value = (
        reference_state_structure
        if reference_state_structure is not None
        else project_manifest.get("reference_state_structure")
    )
    target_path = (
        str(_resolve_project_output_path(str(target_value), project_name))
        if str(target_value or "").strip()
        else ""
    )
    reference_path = (
        str(_resolve_project_output_path(str(reference_value), project_name))
        if str(reference_value or "").strip()
        else ""
    )
    resolved_mutation_region = (
        mutation_region
        if mutation_region is not None
        else project_manifest.get("mutation_region")
    )
    if first_residue_number is None:
        try:
            resolved_first_residue_number = int(project_manifest.get("first_residue_number") or 1)
        except (TypeError, ValueError):
            resolved_first_residue_number = 1
    else:
        resolved_first_residue_number = int(first_residue_number)
    resolved_chain_id = str(chain_id or "").strip()
    if not resolved_chain_id:
        metadata = project_manifest.get("metadata")
        if isinstance(metadata, dict):
            resolved_chain_id = str(metadata.get("parse_chains") or "").strip()
    resolved_chain_id = resolved_chain_id or "A"

    rows: List[Dict[str, Any]] = []
    inventory_by_variant: Dict[str, Dict[str, Any]] = {}
    variant_dirs: Dict[str, Path] = {}
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = list(variant_dir.rglob("*"))
        variant = _sanitize_name(variant_dir.name)
        row = {
            "variant": variant,
            "output_dir": str(variant_dir),
            "file_count": sum(1 for path in files if path.is_file()),
            "pdb_count": sum(1 for path in files if path.is_file() and path.suffix.lower() == ".pdb"),
            "xtc_count": sum(1 for path in files if path.is_file() and path.suffix.lower() == ".xtc"),
            "npy_count": sum(1 for path in files if path.is_file() and path.suffix.lower() == ".npy"),
            "npz_count": sum(1 for path in files if path.is_file() and path.suffix.lower() == ".npz"),
            "csv_count": sum(1 for path in files if path.is_file() and path.suffix.lower() == ".csv"),
        }
        rows.append(row)
        inventory_by_variant[variant] = row
        variant_dirs[variant] = variant_dir
    output_csv = outdir / "bioemu_output_inventory.csv"
    fields = ["variant", "output_dir", "file_count", "pdb_count", "xtc_count", "npy_count", "npz_count", "csv_count"]
    _write_csv_rows(output_csv, rows, fields)

    job_manifest_path = str(project_manifest.get("bioemu_sampling_job_manifest_path") or "").strip()
    job_manifest = (
        _read_json(Path(_resolve_project_output_path(job_manifest_path, project_name)))
        if job_manifest_path
        else {}
    )
    status_candidates = [
        outdir / "bioemu_command_status.csv",
        root.parent / "bioemu_command_status.csv",
        Path(str(project_manifest.get("bioemu_command_status_csv") or "")).expanduser(),
        Path(str(job_manifest.get("command_status_csv") or "")).expanduser(),
    ]
    command_status_path = next(
        (path for path in status_candidates if str(path) and path.exists() and path.is_file()),
        outdir / "bioemu_command_status.csv",
    )
    command_status_rows: List[Dict[str, str]] = []
    command_status_by_variant: Dict[str, Dict[str, str]] = {}
    if command_status_path.exists() and command_status_path.is_file():
        command_status_rows, _status_fields = _read_csv_rows(command_status_path)
        for status_row in command_status_rows:
            variant = _variant_id(status_row)
            if variant:
                command_status_by_variant[variant] = status_row

    recognized_names = [
        "bioemu_final_experimental_candidate_recommendation.csv",
        "bioemu_top30_priority_ranking.csv",
        "variant_level_bioemu_key_metrics.csv",
        "all_variant_metric_summary.csv",
    ]
    recognized_files: Dict[str, str] = {}
    for name in recognized_names:
        matches = sorted(root.rglob(name))
        if not matches and outdir.exists():
            matches = sorted(outdir.rglob(name))
        if matches:
            recognized_files[name] = str(matches[0])

    # A completed job-status table is the authoritative candidate boundary.
    # This excludes stale sample directories left by an earlier run. When no
    # status table exists, discovered sample directories are the fallback.
    allowed_variants = (
        set(command_status_by_variant)
        if command_status_by_variant
        else set(inventory_by_variant)
    )
    external_by_variant: Dict[str, Dict[str, Any]] = {}
    external_sources_by_variant: Dict[str, List[str]] = {}
    ignored_external_variant_count = 0
    for name in recognized_names:
        path_text = recognized_files.get(name)
        if not path_text:
            continue
        source_rows, _source_fields = _read_csv_rows(Path(path_text))
        for source_row in source_rows:
            variant = _variant_id(source_row)
            if variant not in allowed_variants:
                ignored_external_variant_count += 1
                continue
            out: Dict[str, Any] = {"variant": variant, "bioemu_evidence_source": name}
            for key, value in source_row.items():
                if key == "variant":
                    continue
                out[key if key.startswith("bioemu_") else f"bioemu_{key}"] = value
            existing = external_by_variant.setdefault(variant, {})
            for key, value in out.items():
                if key == "variant":
                    continue
                if key not in existing or existing.get(key) in {"", None}:
                    existing[key] = value
            external_sources_by_variant.setdefault(variant, []).append(name)

    evidence_rows: List[Dict[str, Any]] = []
    for variant in sorted(allowed_variants):
        inventory = inventory_by_variant.get(variant, {})
        status_row = command_status_by_variant.get(variant, {})
        variant_dir = variant_dirs.get(variant)
        file_count = int(inventory.get("file_count") or 0)
        pdb_count = int(inventory.get("pdb_count") or 0)
        xtc_count = int(inventory.get("xtc_count") or 0)
        npy_count = int(inventory.get("npy_count") or 0)
        npz_count = int(inventory.get("npz_count") or 0)
        csv_count = int(inventory.get("csv_count") or 0)
        # BioEmu metrics are computed from coordinate batches in NPZ files.
        # A log, PDB, or other side artifact alone is not a valid sampling
        # postcondition and must not be reported as completed evidence.
        has_sampling_artifact = npz_count > 0
        command_status = str(status_row.get("status") or "")
        sampling_complete = (
            command_status == "completed" and has_sampling_artifact
            if command_status
            else has_sampling_artifact
        )

        output_dir_text = str(inventory.get("output_dir") or status_row.get("output_dir") or "")
        log_path_text = str(status_row.get("log_path") or "")
        log_candidates = [
            Path(_resolve_project_output_path(log_path_text, project_name))
            if log_path_text
            else Path(""),
            outdir / "logs" / f"{variant}_bioemu.log",
            root.parent / "logs" / f"{variant}_bioemu.log",
        ]
        resolved_log_path = next(
            (str(path) for path in log_candidates if str(path) and path.exists() and path.is_file()),
            log_path_text,
        )

        evidence: Dict[str, Any] = {
            "variant": variant,
            "bioemu_evidence_source": "inventory",
            "bioemu_output_dir": output_dir_text,
            "bioemu_command_status": command_status,
            "bioemu_command_returncode": status_row.get("returncode", ""),
            "bioemu_command_log_path": resolved_log_path,
            "bioemu_execution_device": status_row.get("execution_device", ""),
            "bioemu_fallback_used": status_row.get("fallback_used", ""),
            "bioemu_error_summary": (
                status_row.get("error_summary", "")
                or _bioemu_log_error_summary(resolved_log_path)
            ),
            "bioemu_file_count": file_count,
            "bioemu_pdb_count": pdb_count,
            "bioemu_xtc_count": xtc_count,
            "bioemu_npy_count": npy_count,
            "bioemu_npz_count": npz_count,
            "bioemu_csv_count": csv_count,
            "bioemu_sampling_complete": sampling_complete,
        }
        external = external_by_variant.get(variant, {})
        for key, value in external.items():
            if key not in evidence or evidence.get(key) in {"", None}:
                evidence[key] = value

        if variant_dir is not None and compute_metrics:
            computed = _compute_bioemu_metrics(
                variant_dir,
                target_state_structure=target_path,
                reference_state_structure=reference_path,
                mutation_region=str(resolved_mutation_region or ""),
                first_residue_number=resolved_first_residue_number,
                chain_id=resolved_chain_id,
                log_path=resolved_log_path or None,
            )
        elif variant_dir is None:
            computed = _empty_bioemu_metrics(
                "sample directory not found; command status was used for inventory evidence"
            )
        else:
            computed = _empty_bioemu_metrics("ensemble metric computation disabled by caller")
        # Preserve a value from a trusted external BioEmu analysis table when
        # it uses the same field name; computed inventory/NPZ metrics fill
        # missing fields without silently replacing that analysis.
        for key, value in computed.items():
            if key not in evidence or evidence.get(key) in {"", None}:
                evidence[key] = value

        source_names = external_sources_by_variant.get(variant, [])
        if (
            variant_dir is not None
            and compute_metrics
            and str(computed.get("bioemu_metrics_status") or "").strip().lower() == "completed"
        ):
            source_names.insert(0, "npz_metrics")
        if source_names:
            evidence["bioemu_evidence_source"] = "inventory+" + "+".join(dict.fromkeys(source_names))
        evidence_rows.append(evidence)

    evidence_csv = outdir / "bioemu_evidence.csv"
    _write_csv_rows(
        evidence_csv,
        evidence_rows,
        _csv_fieldnames_union(
            evidence_rows,
            [
                "variant",
                "bioemu_evidence_source",
                "bioemu_output_dir",
                "bioemu_command_status",
                "bioemu_sampling_complete",
            ]
            + BIOEMU_COMPUTED_METRIC_FIELDS,
        ),
    )
    metrics_completed_count = sum(
        1 for row in evidence_rows if str(row.get("bioemu_metrics_status") or "") == "completed"
    )
    command_completed_count = sum(1 for row in command_status_rows if row.get("status") == "completed")
    command_failed_count = sum(
        1 for row in command_status_rows if str(row.get("status", "")).startswith("failed")
    )
    collection_has_failures = bool(
        command_failed_count
        or (compute_metrics and metrics_completed_count < len(evidence_rows))
    )
    summary = {
        "status": "completed_with_failures" if collection_has_failures else "completed",
        "project_name": project_name,
        "sample_root": str(root),
        "output_csv": str(output_csv),
        "evidence_csv": str(evidence_csv),
        "evidence_source": "inventory+computed_metrics" if compute_metrics else "inventory",
        "recognized_analysis_files": recognized_files,
        "command_status_csv": str(command_status_path) if command_status_path.exists() else "",
        "command_status_count": len(command_status_rows),
        "command_completed_count": command_completed_count,
        "command_failed_count": command_failed_count,
        "variant_count": len(rows),
        "evidence_variant_count": len(evidence_rows),
        "allowed_variant_count": len(allowed_variants),
        "metrics_enabled": bool(compute_metrics),
        "metrics_completed_variant_count": metrics_completed_count,
        "metrics_unavailable_variant_count": len(evidence_rows) - metrics_completed_count,
        "ignored_external_variant_count": ignored_external_variant_count,
        "metric_context": {
            "target_state_structure": target_path,
            "reference_state_structure": reference_path,
            "mutation_region": str(resolved_mutation_region or ""),
            "first_residue_number": resolved_first_residue_number,
            "chain_id": resolved_chain_id,
            "coordinate_source": "pos",
            "coordinate_semantics": "residue_node_coordinates_unverified",
        },
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "bioemu_output_inventory_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "bioemu_outputs_collected",
            "bioemu_output_inventory_csv": str(output_csv),
            "bioemu_evidence_csv": str(evidence_csv),
            "bioemu_output_inventory_summary_path": summary_path,
            "bioemu_metrics_context": summary["metric_context"],
        },
    )
    return {**summary, "summary_path": summary_path}


def _collect_pdb_inputs(
    input_pdb_dir: Optional[str],
    structures_csv: Optional[str],
    structure_column: str,
    allowed_variants: Optional[set[str]] = None,
    variant_column: str = "variant",
) -> List[str]:
    pdbs: List[str] = []
    seen_keys: set[str] = set()
    if input_pdb_dir:
        root = Path(input_pdb_dir).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"input_pdb_dir not found: {input_pdb_dir}")
        for path in sorted(root.glob("*.pdb")):
            variant_key = _sanitize_name(path.stem)
            if allowed_variants is not None and variant_key not in allowed_variants:
                continue
            if variant_key in seen_keys:
                continue
            seen_keys.add(variant_key)
            pdbs.append(str(path))
    if structures_csv:
        rows, _ = _read_csv_rows(Path(structures_csv).expanduser())
        for row in rows:
            value = row.get(structure_column)
            if not value:
                continue
            variant_key = _sanitize_name(_variant_id(row, variant_column) or Path(value).stem)
            if allowed_variants is not None and variant_key not in allowed_variants:
                continue
            if variant_key in seen_keys:
                continue
            seen_keys.add(variant_key)
            pdbs.append(str(Path(value).expanduser()))
    return pdbs


ROSETTA_ANALYSIS_FOCI = {
    "global_stability",
    "local_stability",
    "interface_or_binding",
    "state_switching",
    "fast_screen",
}
ROSETTA_ANALYSIS_MODES = {
    "score",
    "relax",
    "interface_analyzer",
    "rosetta_scripts",
}


def _normalize_rosetta_focus(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "global": "global_stability",
        "overall": "global_stability",
        "fold_stability": "global_stability",
        "local": "local_stability",
        "mutation_site": "local_stability",
        "mutation_window": "local_stability",
        "interface": "interface_or_binding",
        "binding": "interface_or_binding",
        "protein_ligand": "interface_or_binding",
        "state": "state_switching",
        "switching": "state_switching",
        "two_state": "state_switching",
        "fast": "fast_screen",
        "screen": "fast_screen",
    }
    return aliases.get(cleaned, cleaned)


def _normalize_rosetta_mode(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "interface": "interface_analyzer",
        "interfaceanalysis": "interface_analyzer",
        "interface_analysis": "interface_analyzer",
        "scripts": "rosetta_scripts",
        "rosetta_script": "rosetta_scripts",
    }
    return aliases.get(cleaned, cleaned)


def _rosetta_metrics_for_focus(focus: str, mode: str) -> List[str]:
    if mode == "interface_analyzer" or focus == "interface_or_binding":
        return [
            "dG_separated",
            "dSASA_int",
            "interface_hbonds",
            "interface_delta_unsat_hbonds",
            "packstat",
        ]
    if focus == "local_stability" or mode == "rosetta_scripts":
        return [
            "total_score",
            "mutation_window_energy",
            "mutation_residue_energy",
            "local_ddg",
            "local_sasa",
            "local_packing_score",
        ]
    if focus == "state_switching":
        return ["total_score", "fa_atr", "fa_rep", "rama", "cart_bonded"]
    if focus == "global_stability":
        return ["total_score", "fa_atr", "fa_rep", "rama", "omega", "cart_bonded"]
    return ["total_score"]


@mcp.tool()
def select_rosetta_analysis_strategy(
    project_name: str,
    design_objective: Optional[str] = None,
    mutation_region: Optional[str] = None,
    candidate_csv: Optional[str] = None,
    structure_prediction_evidence_csv: Optional[str] = None,
    bioemu_evidence_csv: Optional[str] = None,
    target_state_structure: Optional[str] = None,
    reference_state_structure: Optional[str] = None,
    requested_focus: str = "auto",
    requested_mode: str = "auto",
    interface: Optional[str] = None,
    protocol_xml: Optional[str] = None,
    requested_metrics: Optional[str] = None,
) -> Dict[str, Any]:
    """Choose a safe, structured Rosetta analysis strategy.

    The selector maps natural-language design goals to a small whitelist of
    analysis focuses and Rosetta execution modes. It never accepts arbitrary
    binaries or shell arguments. Missing inputs downgrade the strategy to the
    fast global score protocol and are recorded explicitly.
    """
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    registry_config = _tool_config("rosetta")
    switchtune_config = _load_default_config().get("rosetta", {})
    if not isinstance(switchtune_config, dict):
        switchtune_config = {}

    objective = str(
        design_objective
        if design_objective is not None
        else manifest.get("design_objective") or ""
    ).strip()
    resolved_region = str(
        mutation_region
        if mutation_region is not None
        else manifest.get("mutation_region") or ""
    ).strip()
    resolved_candidate_csv = str(
        candidate_csv
        if candidate_csv is not None
        else _first_nonempty_csv_path(
            manifest.get("rosetta_candidate_csv"),
            manifest.get("bioemu_candidate_csv"),
            manifest.get("structure_confidence_retained_candidates_csv"),
            manifest.get("structure_confidence_review_candidates_csv"),
            "",
        )
    ).strip()
    resolved_structure_evidence = str(
        structure_prediction_evidence_csv
        if structure_prediction_evidence_csv is not None
        else manifest.get("structure_prediction_evidence_csv") or ""
    ).strip()
    resolved_bioemu_evidence = str(
        bioemu_evidence_csv
        if bioemu_evidence_csv is not None
        else manifest.get("bioemu_evidence_csv") or ""
    ).strip()
    resolved_target = str(
        target_state_structure
        if target_state_structure is not None
        else manifest.get("target_state_structure") or ""
    ).strip()
    resolved_reference = str(
        reference_state_structure
        if reference_state_structure is not None
        else manifest.get("reference_state_structure") or ""
    ).strip()
    resolved_interface = str(interface if interface is not None else manifest.get("rosetta_interface") or "").strip()
    configured_protocol = str(
        protocol_xml
        if protocol_xml is not None
        else registry_config.get("local_protocol_xml")
        or switchtune_config.get("local_protocol_xml")
        or ""
    ).strip()

    warnings: List[str] = []
    requested_focus_clean = _normalize_rosetta_focus(requested_focus)
    if requested_focus_clean in {"", "auto"}:
        text = objective.lower()
        if any(term in text for term in ("interface", "binding", "ligand", "cofactor", "protein-protein")):
            focus = "interface_or_binding"
            focus_reason = "The design objective mentions an interface, binding partner, ligand, or cofactor."
        elif any(term in text for term in ("local", "mutation site", "mutation window", "packing", "sasa", "ddg")):
            focus = "local_stability"
            focus_reason = "The design objective emphasizes the mutation site or a local structural environment."
        elif any(term in text for term in ("switch", "two-state", "two state", "allosteric", "conformation", "target-like")):
            focus = "state_switching"
            focus_reason = "The design objective is a reversible conformational or allosteric switching problem."
        elif any(term in text for term in ("global", "overall", "fold", "thermostab", "stability")):
            focus = "global_stability"
            focus_reason = "The design objective emphasizes whole-protein fold or global stability."
        else:
            focus = "fast_screen"
            focus_reason = "No specific Rosetta focus was stated; use a conservative fast screen."
    elif requested_focus_clean in ROSETTA_ANALYSIS_FOCI:
        focus = requested_focus_clean
        focus_reason = f"The requested Rosetta focus was `{focus}`."
    else:
        warnings.append(
            f"Unknown requested_focus={requested_focus!r}; falling back to automatic focus detection."
        )
        focus = "fast_screen"
        focus_reason = "The requested focus was not in the allowed strategy list."

    requested_mode_clean = _normalize_rosetta_mode(requested_mode)
    if requested_mode_clean not in {"", "auto"} and requested_mode_clean not in ROSETTA_ANALYSIS_MODES:
        warnings.append(
            f"Unknown requested_mode={requested_mode!r}; falling back to the mode implied by the selected focus."
        )
        requested_mode_clean = "auto"

    if requested_mode_clean in {"", "auto"}:
        if focus == "interface_or_binding" and resolved_interface:
            mode = "interface_analyzer"
        elif focus == "local_stability" and configured_protocol:
            mode = "rosetta_scripts"
        else:
            mode = "score"
    else:
        mode = requested_mode_clean

    fallback_reason = ""
    if mode == "interface_analyzer" and not resolved_interface:
        warnings.append(
            "InterfaceAnalyzer requires an explicit interface definition; falling back to global score."
        )
        fallback_reason = "No interface definition was supplied."
        mode = "score"
    if (
        focus == "local_stability"
        and mode == "score"
        and not configured_protocol
        and not fallback_reason
    ):
        fallback_reason = "No protocol XML was supplied."
    if mode == "rosetta_scripts":
        if not configured_protocol:
            warnings.append(
                "Local per-residue Rosetta analysis requires a protocol XML; falling back to global score."
            )
            fallback_reason = "No protocol XML was supplied."
            mode = "score"
        else:
            protocol_path = Path(_expand_path_text(configured_protocol))
            if not protocol_path.exists():
                warnings.append(
                    f"Rosetta protocol XML was not found at {protocol_path}; falling back to global score."
                )
                fallback_reason = f"Protocol XML not found: {protocol_path}"
                mode = "score"
                configured_protocol = ""

    if focus == "interface_or_binding" and mode == "score" and not resolved_interface:
        if not fallback_reason:
            fallback_reason = "No interface definition was supplied."
        warnings.append("Interface/binding focus is being represented by a global score because no interface was defined.")
    if focus == "local_stability" and mode == "score":
        warnings.append(
            "Local stability metrics such as per-residue energy, local ddG, and local SASA are unavailable in score-only mode."
        )

    metric_values = [
        item.strip()
        for item in str(requested_metrics or "").split(",")
        if item.strip()
    ]
    metrics = metric_values or _rosetta_metrics_for_focus(focus, mode)
    if mode == "score" and focus not in {"global_stability", "state_switching", "fast_screen"}:
        metrics = ["total_score", "fa_atr", "fa_rep"]

    configured_nstruct = switchtune_config.get("nstruct_relax", 5)
    try:
        nstruct = int(configured_nstruct) if mode == "relax" else 1
    except (TypeError, ValueError):
        nstruct = 1
    nstruct = max(1, nstruct)

    required_inputs = ["candidate_csv", "structure_prediction_results"]
    if focus == "state_switching":
        required_inputs.extend(["target_state_structure", "reference_state_structure"])
    if mode == "interface_analyzer":
        required_inputs.append("interface")
    if mode == "rosetta_scripts":
        required_inputs.append("protocol_xml")
    required_inputs = list(dict.fromkeys(required_inputs))

    if focus == "state_switching":
        limitation = (
            "Rosetta currently supplies global energy support only; target/reference state "
            "comparison remains primarily supported by CB, structure similarity, and BioEmu."
        )
    elif focus == "local_stability" and mode == "score":
        limitation = (
            "The requested local analysis was downgraded because no safe RosettaScripts protocol "
            "was available; interpret this as global energy support, not local ddG."
        )
    elif focus == "interface_or_binding" and mode == "score":
        limitation = (
            "The interface focus was downgraded because no explicit interface definition was available."
        )
    else:
        limitation = ""

    reason_parts = [focus_reason]
    if resolved_region:
        reason_parts.append(f"Mutation region: {resolved_region}.")
    if fallback_reason:
        reason_parts.append(f"Fallback: {fallback_reason}")
    if limitation:
        reason_parts.append(limitation)
    reason = " ".join(reason_parts)

    strategy_name = {
        "global_stability": "global_energy_screen",
        "local_stability": "local_environment_analysis",
        "interface_or_binding": "interface_or_binding_analysis",
        "state_switching": "state_switching_energy_support",
        "fast_screen": "fast_energy_screen",
    }[focus]
    confidence = {
        "global_stability": 0.86,
        "local_stability": 0.82 if mode == "rosetta_scripts" else 0.62,
        "interface_or_binding": 0.88 if mode == "interface_analyzer" else 0.55,
        "state_switching": 0.88,
        "fast_screen": 0.70,
    }[focus]
    if warnings and fallback_reason:
        confidence = max(0.45, confidence - 0.15)

    strategy = {
        "status": "completed",
        "project_name": project_name,
        "strategy_name": strategy_name,
        "focus": focus,
        "mode": mode,
        "reason": reason,
        "required_inputs": required_inputs,
        "candidate_csv": resolved_candidate_csv,
        "structure_prediction_evidence_csv": resolved_structure_evidence,
        "bioemu_evidence_csv": resolved_bioemu_evidence,
        "target_state_structure": resolved_target,
        "reference_state_structure": resolved_reference,
        "mutation_region": resolved_region,
        "interface": resolved_interface if mode == "interface_analyzer" else "",
        "protocol_xml": configured_protocol if mode == "rosetta_scripts" else "",
        "nstruct": nstruct,
        "metrics_to_collect": metrics,
        "confidence": round(confidence, 3),
        "fallback_strategy": "fast_screen",
        "fallback_mode": "score",
        "fallback_reason": fallback_reason,
        "warnings": warnings,
        "limitations": [limitation] if limitation else [],
        "requested_focus": requested_focus,
        "requested_mode": requested_mode,
        "next_tool": "prepare_rosetta_energy_job",
    }
    decision = write_decision_log(
        project_name=project_name,
        step_name="select_rosetta_analysis_strategy",
        decision=f"{focus}:{mode}",
        rationale=reason,
        input_summary={
            "design_objective": objective,
            "mutation_region": resolved_region,
            "candidate_csv": resolved_candidate_csv,
            "target_state_structure": resolved_target,
            "reference_state_structure": resolved_reference,
            "interface": resolved_interface,
            "protocol_xml": configured_protocol,
        },
        output_summary=strategy,
    )
    strategy["decision_log_path"] = decision.get("decision_log_path", "")
    _update_project_manifest(
        project_name,
        {
            "rosetta_strategy": f"{focus}:{mode}",
            "rosetta_strategy_name": strategy_name,
            "rosetta_strategy_reason": reason,
            "rosetta_focus": focus,
            "rosetta_mode": mode,
            "rosetta_requested_metrics": metrics,
            "rosetta_fallback_strategy": "fast_screen",
            "rosetta_fallback_mode": "score",
            "rosetta_strategy_confidence": round(confidence, 3),
            "rosetta_strategy_decision_log_path": strategy["decision_log_path"],
        },
    )
    return strategy


@mcp.tool()
def prepare_rosetta_energy_job(
    project_name: str,
    mode: str = "rosetta_scripts",
    input_pdb_dir: Optional[str] = None,
    structures_csv: Optional[str] = None,
    candidates_csv: Optional[str] = None,
    structure_column: str = "structure_path",
    variant_column: str = "variant",
    protocol_xml: Optional[str] = None,
    extra_res_fa: Optional[str] = None,
    nstruct: int = 1,
    output_dir: Optional[str] = None,
    rosetta_extra_args: str = "",
    interface: Optional[str] = None,
    execute: bool = False,
    timeout_seconds: int = 0,
    strategy_name: Optional[str] = None,
    strategy_focus: Optional[str] = None,
    strategy_reason: Optional[str] = None,
    requested_metrics: Optional[str] = None,
    target_state_structure: Optional[str] = None,
    reference_state_structure: Optional[str] = None,
    fallback_strategy: Optional[str] = None,
    strategy_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Prepare a flexible Rosetta energy-evaluation job.

    `mode` can be `rosetta_scripts`, `interface_analyzer`, `relax`, or `score`.
    This intentionally does not assume InterfaceAnalyzer for every protein.
    """
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "07_rosetta_energy"
    outdir.mkdir(parents=True, exist_ok=True)
    mode_clean = mode.strip().lower().replace("-", "_")
    allowed_variants: Optional[set[str]] = None
    warnings: List[str] = []
    project_manifest = _load_project_manifest(project_name)
    resolved_strategy_name = str(
        strategy_name
        if strategy_name is not None
        else project_manifest.get("rosetta_strategy_name") or ""
    ).strip()
    resolved_strategy_focus = str(
        strategy_focus
        if strategy_focus is not None
        else project_manifest.get("rosetta_focus") or ""
    ).strip()
    resolved_strategy_reason = str(
        strategy_reason
        if strategy_reason is not None
        else project_manifest.get("rosetta_strategy_reason") or ""
    ).strip()
    resolved_requested_metrics = (
        requested_metrics
        if requested_metrics is not None
        else project_manifest.get("rosetta_requested_metrics") or ""
    )
    if isinstance(resolved_requested_metrics, (list, tuple)):
        resolved_requested_metrics = ",".join(str(item) for item in resolved_requested_metrics if str(item).strip())
    resolved_target_state_structure = str(
        target_state_structure
        if target_state_structure is not None
        else project_manifest.get("target_state_structure") or ""
    ).strip()
    resolved_reference_state_structure = str(
        reference_state_structure
        if reference_state_structure is not None
        else project_manifest.get("reference_state_structure") or ""
    ).strip()
    resolved_fallback_strategy = str(
        fallback_strategy
        if fallback_strategy is not None
        else project_manifest.get("rosetta_fallback_strategy") or ""
    ).strip()
    resolved_strategy_confidence = (
        strategy_confidence
        if strategy_confidence is not None
        else project_manifest.get("rosetta_strategy_confidence")
    )
    if candidates_csv:
        candidate_path = _resolve_project_output_path(str(candidates_csv), project_name)
        candidate_rows, _candidate_fields = _read_csv_rows(candidate_path)
        allowed_variants = {
            _variant_id(row, variant_column)
            for row in candidate_rows
            if _variant_id(row, variant_column)
        }
        if not candidate_rows:
            input_list = outdir / "rosetta_input_pdbs.txt"
            input_list.write_text("", encoding="utf-8")
            scorefile = outdir / f"rosetta_{mode_clean}.sc"
            log_dir = outdir / "logs"
            status_csv = outdir / "rosetta_command_status.csv"
            _write_csv_rows(
                status_csv,
                [],
                ["input_pdb", "variant", "mode", "status", "returncode", "log_path", "output_dir", "scorefile"],
            )
            payload = {
                "mode": mode_clean,
                "input_count": 0,
                "candidates_csv": str(candidate_path),
                "candidate_filter_count": 0,
                "input_list": str(input_list),
                "run_script": "",
                "log_dir": str(log_dir),
                "command_status_csv": str(status_csv),
                "scorefile": str(scorefile),
                "resolved_input_pdb_dir": None,
                "nstruct": int(nstruct),
                "protocol_xml": protocol_xml,
                "extra_res_fa": extra_res_fa,
                "interface": interface,
                "strategy_name": resolved_strategy_name,
                "strategy_focus": resolved_strategy_focus,
                "strategy_reason": resolved_strategy_reason,
                "requested_metrics": resolved_requested_metrics,
                "target_state_structure": resolved_target_state_structure,
                "reference_state_structure": resolved_reference_state_structure,
                "fallback_strategy": resolved_fallback_strategy,
                "strategy_confidence": resolved_strategy_confidence,
                "binary": "",
                "warnings": ["No candidates were supplied; Rosetta energy evaluation was not prepared."],
                "non_pdb_input_count": 0,
            }
            manifest_path = _write_json(outdir / "rosetta_energy_job_manifest.json", payload)
            job = _job_record(project_name, "rosetta_energy", "blocked_no_candidates", payload)
            _update_project_manifest(
                project_name,
                {
                    "stage": "rosetta_energy_blocked_no_candidates",
                    "rosetta_energy_job_manifest_path": manifest_path,
                    "rosetta_energy_command_status_csv": str(status_csv),
                },
            )
            return {
                "status": "blocked_no_candidates",
                "project_name": project_name,
                "job_id": job["job_id"],
                "job_manifest_path": manifest_path,
                **payload,
                "score_file": str(scorefile),
                "next_tool": "none",
                "stop_after_this_tool": True,
                "agent_instruction": "No candidates remain for Rosetta. Stop the workflow; do not run an empty job.",
            }
    pdbs = _collect_pdb_inputs(input_pdb_dir, structures_csv, structure_column, allowed_variants, variant_column)
    resolved_input_pdb_dir: Optional[str] = str(Path(input_pdb_dir).expanduser().resolve()) if input_pdb_dir else None
    if not pdbs and structures_csv:
        try:
            structures_csv_path = _resolve_project_output_path(str(structures_csv), project_name)
        except Exception:
            structures_csv_path = Path(structures_csv).expanduser()
        structures_parent = structures_csv_path.parent
        fallback_dirs = []
        if input_pdb_dir:
            fallback_dirs.append(Path(input_pdb_dir).expanduser())
        fallback_dirs.extend(
            [
                structures_parent / "standardized_pdbs",
                structures_parent / "pdbs",
                structures_parent,
            ]
        )
        for fallback_dir in fallback_dirs:
            if not fallback_dir.exists() or not fallback_dir.is_dir():
                continue
            fallback_pdbs = _collect_pdb_inputs(
                str(fallback_dir),
                None,
                structure_column,
                allowed_variants,
                variant_column,
            )
            if fallback_pdbs:
                pdbs = fallback_pdbs
                resolved_input_pdb_dir = str(fallback_dir.resolve())
                warnings.append(
                    f"Auto-discovered Rosetta input PDBs from {resolved_input_pdb_dir} because the primary input set was empty."
                )
                break
    if not pdbs:
        raise ValueError(
            "No input PDB files found. Provide input_pdb_dir or structures_csv, "
            f"or ensure standardized PDBs exist near {structures_csv!r}."
        )
    non_pdb_inputs = [path for path in pdbs if Path(path).suffix.lower() != ".pdb"]
    if non_pdb_inputs:
        warnings.append(
            "Some Rosetta inputs are not PDB files. Run standardize_structure_prediction_results first "
            "and pass its output CSV for downstream Rosetta jobs."
        )

    cfg = _tool_config("rosetta")
    rosetta_scripts = _expand_path_text(str(cfg.get("rosetta_scripts") or "${ROSETTA_HOME:-./external/rosetta/main}/source/bin/rosetta_scripts.static.linuxgccrelease"))
    interface_analyzer = _expand_path_text(str(cfg.get("interface_analyzer") or "${ROSETTA_HOME:-./external/rosetta/main}/source/bin/InterfaceAnalyzer.static.linuxgccrelease"))
    relax_binary = _expand_path_text(str(cfg.get("relax") or "${ROSETTA_HOME:-./external/rosetta/main}/source/bin/relax.static.linuxgccrelease"))
    score_jd2 = _expand_path_text(str(cfg.get("score_jd2") or cfg.get("score") or "${ROSETTA_HOME:-./external/rosetta/main}/source/bin/score_jd2.static.linuxgccrelease"))
    score_jd2_path = Path(score_jd2)
    if not score_jd2_path.exists() and "score_jd2.default.linuxgccrelease" in score_jd2:
        alt_score_jd2 = score_jd2.replace("score_jd2.default.linuxgccrelease", "score_jd2.static.linuxgccrelease")
        if Path(alt_score_jd2).exists():
            warnings.append(f"Rosetta score_jd2 path not found at {score_jd2}; using {alt_score_jd2} instead.")
            score_jd2 = alt_score_jd2
    input_list = outdir / "rosetta_input_pdbs.txt"
    input_list.write_text("\n".join(pdbs) + "\n", encoding="utf-8")

    script_path = outdir / f"run_rosetta_{mode_clean}.sh"
    scorefile = outdir / f"rosetta_{mode_clean}.sc"
    log_dir = outdir / "logs"
    status_csv = outdir / "rosetta_command_status.csv"
    binary = rosetta_scripts
    base_args: List[str] = []
    input_flag = "-s"
    if mode_clean == "interface_analyzer":
        binary = interface_analyzer
        if interface:
            base_args.extend(["-interface", interface])
    elif mode_clean == "rosetta_scripts":
        if not protocol_xml:
            raise ValueError("protocol_xml is required for mode='rosetta_scripts'.")
        base_args.extend(["-parser:protocol", protocol_xml])
    elif mode_clean == "relax":
        binary = relax_binary
        base_args.extend(["-relax:constrain_relax_to_start_coords"])
    elif mode_clean == "score":
        binary = score_jd2
        input_flag = "-in:file:s"
    else:
        raise ValueError("mode must be one of: rosetta_scripts, interface_analyzer, relax, score.")
    if extra_res_fa:
        base_args.extend(["-extra_res_fa", extra_res_fa])
    if rosetta_extra_args.strip():
        base_args.append(rosetta_extra_args.strip())

    quoted_args = " ".join(_quote_bash(arg) for arg in base_args)
    script_lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "",
        f"ROSETTA_BIN={_quote_bash(binary)}",
        f"INPUT_LIST={_quote_bash(str(input_list))}",
        f"OUT_DIR={_quote_bash(str(outdir))}",
        f"LOG_DIR={_quote_bash(str(log_dir))}",
        f"STATUS_CSV={_quote_bash(str(status_csv))}",
        "mkdir -p \"$OUT_DIR\"",
        "mkdir -p \"$LOG_DIR\"",
        "printf 'input_pdb,variant,mode,status,returncode,log_path,output_dir,scorefile\\n' > \"$STATUS_CSV\"",
        "echo \"START Rosetta energy evaluation: $(date)\"",
        "while IFS= read -r pdb; do",
        "  pdb=${pdb//$'\\r'/}",
        "  [[ -z \"$pdb\" ]] && continue",
        "  base=$(basename \"$pdb\" .pdb)",
        "  mkdir -p \"$OUT_DIR/$base\"",
        "  rosetta_log=\"$LOG_DIR/${base}_rosetta.log\"",
        "  echo \"Processing ${base}\"",
        "  if (cd \"$OUT_DIR/$base\" && \"$ROSETTA_BIN\" " + input_flag + " \"$pdb\" "
        + quoted_args
        + f" -nstruct {int(nstruct)} -out:file:scorefile {_quote_bash(str(scorefile))} -overwrite > \"$rosetta_log\" 2>&1); then",
        "    printf '%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$pdb\" \"$base\" "
        + _quote_bash(mode_clean)
        + " \"completed\" \"0\" \"$rosetta_log\" \"$OUT_DIR/$base\" "
        + _quote_bash(str(scorefile))
        + " >> \"$STATUS_CSV\"",
        "  else",
        "    rc=$?",
        "    echo \"Rosetta failed for ${base}; see ${rosetta_log}\" >&2",
        "    printf '%s,%s,%s,%s,%s,%s,%s,%s\\n' \"$pdb\" \"$base\" "
        + _quote_bash(mode_clean)
        + " \"failed\" \"$rc\" \"$rosetta_log\" \"$OUT_DIR/$base\" "
        + _quote_bash(str(scorefile))
        + " >> \"$STATUS_CSV\"",
        "  fi",
        "done < \"$INPUT_LIST\"",
        "echo \"DONE Rosetta energy evaluation: $(date)\"",
        "echo \"Status CSV: $STATUS_CSV\"",
        "cat \"$STATUS_CSV\"",
    ]
    _write_shell_script(script_path, script_lines)

    payload = {
        "mode": mode_clean,
        "input_count": len(pdbs),
        "candidates_csv": candidates_csv,
        "candidate_filter_count": len(allowed_variants) if allowed_variants is not None else None,
        "input_list": str(input_list),
        "run_script": str(script_path),
        "log_dir": str(log_dir),
        "command_status_csv": str(status_csv),
        "scorefile": str(scorefile),
        "resolved_input_pdb_dir": resolved_input_pdb_dir,
        "nstruct": int(nstruct),
        "protocol_xml": protocol_xml,
        "extra_res_fa": extra_res_fa,
        "interface": interface,
        "strategy_name": resolved_strategy_name,
        "strategy_focus": resolved_strategy_focus,
        "strategy_reason": resolved_strategy_reason,
        "requested_metrics": resolved_requested_metrics,
        "target_state_structure": resolved_target_state_structure,
        "reference_state_structure": resolved_reference_state_structure,
        "fallback_strategy": resolved_fallback_strategy,
        "strategy_confidence": resolved_strategy_confidence,
        "binary": binary,
        "warnings": warnings,
        "non_pdb_input_count": len(non_pdb_inputs),
    }
    manifest_path = _write_json(outdir / "rosetta_energy_job_manifest.json", payload)
    job = _job_record(project_name, "rosetta_energy", "prepared", payload)
    result: Dict[str, Any] = {
        "status": "prepared",
        "project_name": project_name,
        "job_id": job["job_id"],
        "job_manifest_path": manifest_path,
        **payload,
        "score_file": str(scorefile),
        "stop_after_this_tool": True,
        "agent_instruction": "Return a final <solution> now. Do not run Rosetta inside the agent call.",
        "next_step": "Run run_script, then call collect_rosetta_scores.",
    }
    if execute:
        completed = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(outdir),
            text=True,
            capture_output=True,
            timeout=None if int(timeout_seconds) <= 0 else int(timeout_seconds),
        )
        result.update(
            {
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )

    _update_project_manifest(
        project_name,
        {
            "stage": "rosetta_energy_prepared",
            "rosetta_energy_job_manifest_path": manifest_path,
            "rosetta_energy_run_script": str(script_path),
            "rosetta_energy_scorefile": str(scorefile),
            "rosetta_energy_score_file": str(scorefile),
            "rosetta_command_status_csv": str(status_csv),
            "rosetta_strategy_name": resolved_strategy_name,
            "rosetta_strategy_focus": resolved_strategy_focus,
            "rosetta_strategy_reason": resolved_strategy_reason,
            "rosetta_mode": mode_clean,
            "rosetta_requested_metrics": resolved_requested_metrics,
            "rosetta_fallback_strategy": resolved_fallback_strategy,
            "rosetta_strategy_confidence": resolved_strategy_confidence,
        },
    )
    return result


def _parse_rosetta_scorefile(score_file: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not score_file.exists():
        raise FileNotFoundError(f"Rosetta score file not found: {score_file}")
    header: Optional[List[str]] = None
    rows: List[Dict[str, Any]] = []
    with score_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.startswith("SCORE:"):
                continue
            parts = line.split()
            values = parts[1:]
            if not values:
                continue
            if values[0] == "score" or "description" in values:
                header = values
                continue
            if header is None:
                continue
            row = {name: values[idx] if idx < len(values) else "" for idx, name in enumerate(header)}
            rows.append(row)
    return rows, header or []


@mcp.tool()
def collect_rosetta_scores(
    project_name: str,
    score_file: str,
    output_dir: Optional[str] = None,
    group_column: str = "description",
    primary_score_column: str = "total_score",
    strip_nstruct_suffix: bool = True,
    job_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse Rosetta scores and expose strategy-aware numeric evidence."""
    score_path = Path(score_file).expanduser()
    rows, fieldnames = _parse_rosetta_scorefile(score_path)
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "07_rosetta_energy"
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_job_manifest_path = (
        Path(job_manifest_path).expanduser()
        if job_manifest_path
        else project / "07_rosetta_energy" / "rosetta_energy_job_manifest.json"
    )
    rosetta_job_manifest = _read_json(resolved_job_manifest_path)

    long_csv = outdir / "rosetta_scores_long.csv"
    _write_csv_rows(long_csv, rows, fieldnames)

    groups: Dict[str, List[float]] = {}
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get(group_column) or row.get("description") or "unknown")
        if strip_nstruct_suffix:
            group = re.sub(r"_[0-9]{4}$", "", group)
        grouped_rows.setdefault(group, []).append(row)
        value = _safe_float(row.get(primary_score_column))
        if value is None and primary_score_column != "score":
            value = _safe_float(row.get("score"))
        if value is None:
            continue
        groups.setdefault(group, []).append(value)
    summary_rows = []
    for group, values in sorted(groups.items()):
        summary_rows.append(
            {
                "variant": group,
                "n": len(values),
                f"{primary_score_column}_mean": mean(values),
                f"{primary_score_column}_median": median(values),
                f"{primary_score_column}_sd": pstdev(values) if len(values) > 1 else 0.0,
                f"{primary_score_column}_min": min(values),
                f"{primary_score_column}_max": max(values),
            }
        )
        group_rows = grouped_rows.get(group, [])
        for field in fieldnames:
            if field in {group_column, "description"}:
                continue
            numeric_values = [
                value
                for value in (_safe_float(item.get(field)) for item in group_rows)
                if value is not None and math.isfinite(value)
            ]
            if not numeric_values:
                continue
            metric_name = _sanitize_name(field)
            summary_rows[-1][f"{metric_name}_mean"] = mean(numeric_values)
            summary_rows[-1][f"{metric_name}_median"] = median(numeric_values)
            summary_rows[-1][f"{metric_name}_sd"] = (
                pstdev(numeric_values) if len(numeric_values) > 1 else 0.0
            )
            summary_rows[-1][f"{metric_name}_min"] = min(numeric_values)
            summary_rows[-1][f"{metric_name}_max"] = max(numeric_values)
    summary_csv = outdir / "rosetta_scores_summary.csv"
    summary_fields = _csv_fieldnames_union(
        summary_rows,
        [
            "variant",
            "n",
            f"{primary_score_column}_mean",
            f"{primary_score_column}_median",
            f"{primary_score_column}_sd",
            f"{primary_score_column}_min",
            f"{primary_score_column}_max",
        ],
    )
    _write_csv_rows(summary_csv, summary_rows, summary_fields)
    evidence_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        out: Dict[str, Any] = {
            "variant": row["variant"],
            "rosetta_n": row["n"],
            "rosetta_primary_score_column": primary_score_column,
            "rosetta_primary_score_direction": "lower_better",
            "rosetta_primary_score_mean": row.get(f"{primary_score_column}_mean"),
            "rosetta_primary_score_median": row.get(f"{primary_score_column}_median"),
            "rosetta_primary_score_sd": row.get(f"{primary_score_column}_sd"),
            "rosetta_primary_score_min": row.get(f"{primary_score_column}_min"),
            "rosetta_primary_score_max": row.get(f"{primary_score_column}_max"),
            "rosetta_strategy_name": rosetta_job_manifest.get("strategy_name", ""),
            "rosetta_strategy_focus": rosetta_job_manifest.get("strategy_focus", ""),
            "rosetta_strategy_reason": rosetta_job_manifest.get("strategy_reason", ""),
            "rosetta_requested_metrics": rosetta_job_manifest.get("requested_metrics", ""),
            "rosetta_fallback_strategy": rosetta_job_manifest.get("fallback_strategy", ""),
            "rosetta_strategy_confidence": rosetta_job_manifest.get("strategy_confidence", ""),
            "rosetta_mode": rosetta_job_manifest.get("mode", ""),
        }
        for key, value in row.items():
            if key in {"variant", "n"}:
                continue
            out[f"rosetta_{key}"] = value
        evidence_rows.append(out)
    evidence_csv = outdir / "rosetta_evidence.csv"
    _write_csv_rows(
        evidence_csv,
        evidence_rows,
        _csv_fieldnames_union(
            evidence_rows,
            [
                "variant",
                "rosetta_primary_score_column",
                "rosetta_primary_score_direction",
                "rosetta_primary_score_mean",
                "rosetta_primary_score_median",
                "rosetta_primary_score_sd",
            ],
        ),
    )
    summary = {
        "status": "completed",
        "project_name": project_name,
        "score_file": str(score_path),
        "long_csv": str(long_csv),
        "summary_csv": str(summary_csv),
        "evidence_csv": str(evidence_csv),
        "score_row_count": len(rows),
        "variant_count": len(summary_rows),
        "primary_score_column": primary_score_column,
        "strip_nstruct_suffix": strip_nstruct_suffix,
        "job_manifest_path": str(resolved_job_manifest_path) if resolved_job_manifest_path.exists() else "",
        "strategy_name": rosetta_job_manifest.get("strategy_name", ""),
        "strategy_focus": rosetta_job_manifest.get("strategy_focus", ""),
        "strategy_reason": rosetta_job_manifest.get("strategy_reason", ""),
        "mode": rosetta_job_manifest.get("mode", ""),
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "rosetta_score_collection_summary.json", summary)
    _update_project_manifest(
        project_name,
        {
            "stage": "rosetta_scores_collected",
            "rosetta_scores_long_csv": str(long_csv),
            "rosetta_scores_summary_csv": str(summary_csv),
            "rosetta_evidence_csv": str(evidence_csv),
            "rosetta_score_collection_summary_path": summary_path,
        },
    )
    return {**summary, "summary_path": summary_path}


def _read_optional_evidence_csv(path_text: str, source_name: str) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{source_name} evidence CSV not found: {path_text}")
    rows, fields = _read_csv_rows(path)
    return [dict(row) for row in rows], fields, {"source": source_name, "path": str(path), "row_count": len(rows)}


def _merge_table_into_variants(
    merged: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    *,
    source_name: str,
    variant_column: str,
    allowed_variants: Optional[set[str]] = None,
) -> None:
    for row in rows:
        variant = _variant_id(row, variant_column)
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        out = merged.setdefault(variant, {"variant": variant})
        out[f"has_{source_name}_evidence"] = True
        for key, value in row.items():
            if key == variant_column or key == "variant":
                continue
            if key not in out or out.get(key) in {"", None}:
                out[key] = value
            elif str(out.get(key)) != str(value):
                out[f"{source_name}_{key}"] = value


def _infer_metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    lower_terms = [
        "nll",
        "rmsd_to_target",
        "rmsd_to_reference",
        "pairwise_rmsd",
        "rmsf",
        "bias_score_abs",
        "pae",
        "pde",
        "rank",
        "total_score",
        "primary_score",
        "dg_separated",
        "dg_cross",
        "delta_unsathbonds",
        "failed",
        "error_count",
        "sd",
        "stdev",
    ]
    higher_terms = [
        "target_similarity",
        "delta_reference_minus_target",
        "structure_confidence_score",
        "structure_confidence_pass",
        "compatibility_score",
        "log_likelihood",
        "confidence",
        "plddt",
        "ptm",
        "iptm",
        "composite_score",
        "priority_score",
        "pass",
        "sampling_complete",
        "target_like_region_count",
    ]
    if any(term in name for term in lower_terms):
        return "lower"
    if any(term in name for term in higher_terms):
        return "higher"
    return "higher"


def _auto_consensus_metric_columns(fieldnames: List[str]) -> List[str]:
    selected: List[str] = []
    if "bias_score_abs" in fieldnames:
        selected.append("bias_score_abs")
    if "structure_prediction_confidence_score" in fieldnames:
        selected.append("structure_prediction_confidence_score")
    if "structure_confidence_score" in fieldnames:
        selected.append("structure_confidence_score")
    if "structure_target_similarity_score" in fieldnames:
        selected.append("structure_target_similarity_score")

    if "inverse_folding_compatibility_score" in fieldnames:
        selected.append("inverse_folding_compatibility_score")
    else:
        for name in ("esm_if1_compatibility_score", "protein_mpnn_compatibility_score", "ligand_mpnn_compatibility_score"):
            if name in fieldnames:
                selected.append(name)

    for name in (
        "bioemu_composite_score",
        "bioemu_priority_score",
        "bioemu_target_like_fraction",
        "bioemu_target_preference_margin_A",
        "bioemu_rmsd_to_target_mean_A",
        "bioemu_mutation_window_rmsf_mean_A",
        "bioemu_priority_rank",
        "bioemu_sampling_complete",
    ):
        if name in fieldnames:
            selected.append(name)
            break

    for name in ("rosetta_primary_score_mean", "rosetta_dG_separated_mean", "dG_separated_mean", "rosetta_total_score_mean"):
        if name in fieldnames:
            selected.append(name)
            break

    if "interface_composite_score" in fieldnames:
        selected.append("interface_composite_score")

    if selected:
        return selected
    fallback_terms = [
        "target_similarity",
        "compatibility_score",
        "priority_rank",
        "composite_score",
        "primary_score_mean",
        "total_score_mean",
        "dg_separated_mean",
        "log_likelihood",
        "nll",
    ]
    return [name for name in fieldnames if any(term in name.lower() for term in fallback_terms)]


def _bool_or_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "pass", "passed", "target", "target_like", "completed"}:
        return 1.0
    if text in {"false", "no", "n", "fail", "failed", "reference", "reference_like", "inventory_only"}:
        return 0.0
    return _safe_float(value)


@mcp.tool()
def merge_switchtune_evidence_tables(
    project_name: str,
    candidate_csv: Optional[str] = None,
    cb_scored_csv: Optional[str] = None,
    structure_prediction_evidence_csv: Optional[str] = None,
    structure_state_similarity_evidence_csv: Optional[str] = None,
    structure_confidence_evidence_csv: Optional[str] = None,
    inverse_folding_evidence_csv: Optional[str] = None,
    bioemu_evidence_csv: Optional[str] = None,
    rosetta_evidence_csv: Optional[str] = None,
    extra_evidence_csvs: str = "",
    output_dir: Optional[str] = None,
    variant_column: str = "variant",
    metric_columns: str = "",
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge SwitchTune evidence tables and compute an interpretable consensus score."""
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "08_final_recommendation"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = _load_project_manifest(project_name)
    candidate_csv = candidate_csv or _first_nonempty_csv_path(
        manifest.get("rosetta_candidate_csv"),
        manifest.get("bioemu_candidate_csv"),
        manifest.get("structure_confidence_retained_candidates_csv"),
        manifest.get("structure_confidence_review_candidates_csv"),
        manifest.get("state_specific_candidates_adaptive_csv"),
        manifest.get("state_specific_candidates_csv"),
        manifest.get("library_csv"),
    )
    cb_scored_csv = cb_scored_csv or manifest.get("cb_scored_library_csv")
    structure_prediction_evidence_csv = (
        structure_prediction_evidence_csv or manifest.get("structure_prediction_evidence_csv")
    )
    structure_state_similarity_evidence_csv = (
        structure_state_similarity_evidence_csv or manifest.get("structure_state_similarity_evidence_csv")
    )
    structure_confidence_evidence_csv = (
        structure_confidence_evidence_csv or manifest.get("structure_confidence_evidence_csv")
    )
    inverse_folding_evidence_csv = inverse_folding_evidence_csv or manifest.get("inverse_folding_evidence_csv")
    bioemu_evidence_csv = bioemu_evidence_csv or manifest.get("bioemu_evidence_csv")
    rosetta_evidence_csv = rosetta_evidence_csv or manifest.get("rosetta_evidence_csv")

    sources: List[Tuple[str, str]] = []
    for source_name, path_text in [
        ("candidate", candidate_csv),
        ("cb", cb_scored_csv),
        ("structure_prediction", structure_prediction_evidence_csv),
        ("structure", structure_state_similarity_evidence_csv),
        ("structure_confidence", structure_confidence_evidence_csv),
        ("inverse_folding", inverse_folding_evidence_csv),
        ("bioemu", bioemu_evidence_csv),
        ("rosetta", rosetta_evidence_csv),
    ]:
        if path_text:
            sources.append((source_name, str(path_text)))
    for idx, path_text in enumerate([item.strip() for item in str(extra_evidence_csvs).split(",") if item.strip()], start=1):
        sources.append((f"extra{idx}", path_text))
    if not sources:
        raise ValueError("No evidence CSVs were provided and none were found in the project manifest.")

    allowed_variants: Optional[set[str]] = None
    if candidate_csv:
        candidate_rows, _candidate_fields = _read_csv_rows(Path(candidate_csv).expanduser())
        allowed_variants = {
            _variant_id(row, variant_column)
            for row in candidate_rows
            if _variant_id(row, variant_column)
        }

    merged: Dict[str, Dict[str, Any]] = {}
    source_summaries: List[Dict[str, Any]] = []
    for source_name, path_text in sources:
        rows, _fields, source_summary = _read_optional_evidence_csv(path_text, source_name)
        source_summaries.append(source_summary)
        _merge_table_into_variants(
            merged,
            rows,
            source_name=source_name,
            variant_column=variant_column,
            allowed_variants=allowed_variants,
        )

    merged_rows = list(merged.values())
    fields = _csv_fieldnames_union(merged_rows, ["variant"])
    selected_metrics = [item.strip() for item in str(metric_columns).split(",") if item.strip()]
    if not selected_metrics:
        selected_metrics = _auto_consensus_metric_columns(fields)
    selected_metrics = [metric for metric in selected_metrics if metric in fields]

    component_values: Dict[str, List[Optional[float]]] = {}
    for metric in selected_metrics:
        direction = _infer_metric_direction(metric)
        raw_values = [_bool_or_numeric(row.get(metric)) for row in merged_rows]
        transformed = [
            (-value if direction == "lower" else value) if value is not None and math.isfinite(value) else None
            for value in raw_values
        ]
        component_values[metric] = _z_scale(transformed)

    for idx, row in enumerate(merged_rows):
        components: List[float] = []
        component_names: List[str] = []
        for metric in selected_metrics:
            value = component_values[metric][idx]
            if value is None or not math.isfinite(value):
                continue
            row[f"consensus_component__{metric}"] = value
            components.append(value)
            component_names.append(metric)
        row["consensus_evidence_count"] = len(components)
        row["consensus_metric_columns"] = ";".join(component_names)
        row["consensus_score"] = mean(components) if components else None
        row["consensus_status"] = "ranked" if components else "no_numeric_evidence"

    ranked_rows = sorted(
        merged_rows,
        key=lambda row: (
            0 if _safe_float(row.get("consensus_score")) is not None else 1,
            -(_safe_float(row.get("consensus_score")) or float("-inf")),
            str(row.get("variant")),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["consensus_rank"] = rank if row.get("consensus_status") == "ranked" else ""

    ranked_numeric_rows = [
        row for row in ranked_rows
        if row.get("consensus_status") == "ranked"
    ]
    merged_csv = outdir / "switchtune_evidence_merged.csv"
    ranked_csv = outdir / "switchtune_evidence_ranked.csv"
    preferred = ["consensus_rank", "variant", "consensus_score", "consensus_evidence_count", "consensus_status"]
    final_fields = _csv_fieldnames_union(ranked_rows, preferred + fields)
    _write_csv_rows(merged_csv, merged_rows, final_fields)
    ranked_budget = (
        int(top_n)
        if top_n is not None and int(top_n) > 0
        else adaptive_candidate_budget(len(ranked_numeric_rows))
    )
    _write_csv_rows(ranked_csv, ranked_numeric_rows[:ranked_budget], final_fields)

    summary = {
        "status": "completed" if ranked_numeric_rows else "blocked_no_candidates",
        "project_name": project_name,
        "merged_csv": str(merged_csv),
        "ranked_csv": str(ranked_csv),
        "source_summaries": source_summaries,
        "variant_count": len(merged_rows),
        "ranked_variant_count": len(ranked_numeric_rows),
        "selected_metric_columns": selected_metrics,
        "metric_directions": {metric: _infer_metric_direction(metric) for metric in selected_metrics},
        "top_n": ranked_budget,
        "adaptive_budget": None if top_n is not None and int(top_n) > 0 else ranked_budget,
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "switchtune_evidence_merge_summary.json", summary)
    merge_status = "evidence_merged" if ranked_numeric_rows else "evidence_merge_blocked_no_candidates"
    _update_project_manifest(
        project_name,
        {
            "stage": merge_status,
            "switchtune_evidence_merged_csv": str(merged_csv),
            "switchtune_evidence_ranked_csv": str(ranked_csv),
            "switchtune_evidence_merge_summary_path": summary_path,
        },
    )
    return {
        **summary,
        "summary_path": summary_path,
        "next_tool": "prepare_final_candidate_recommendation" if ranked_numeric_rows else "none",
        "stop_after_this_tool": not bool(ranked_numeric_rows),
    }


@mcp.tool()
def prepare_final_candidate_recommendation(
    project_name: str,
    evidence_csv: str,
    output_dir: Optional[str] = None,
    rank_column: str = "consensus_score",
    ascending: bool = False,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a final ranked candidate table from an evidence CSV."""
    rows, fieldnames = _candidate_rows_from_csv(evidence_csv, top_n=None, sort_by=rank_column, ascending=ascending)
    eligible = [
        row
        for row in rows
        if _safe_float(row.get(rank_column)) is not None
        and str(row.get("consensus_status") or "").strip().lower() not in {
            "no_numeric_evidence",
            "failed",
            "blocked_no_candidates",
        }
    ]
    final_budget = (
        int(top_n)
        if top_n is not None and int(top_n) > 0
        else adaptive_candidate_budget(len(eligible))
    )
    selected = eligible[:final_budget]
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "08_final_recommendation"
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / "final_candidate_recommendation.csv"
    if "final_rank" not in fieldnames:
        fieldnames = ["final_rank"] + fieldnames
    ranked_rows = []
    for idx, row in enumerate(selected, start=1):
        ranked = dict(row)
        ranked["final_rank"] = idx
        ranked_rows.append(ranked)
    _write_csv_rows(output_csv, ranked_rows, fieldnames)
    recommendation_status = "completed" if ranked_rows else "blocked_no_candidates"
    summary = {
        "status": recommendation_status,
        "project_name": project_name,
        "evidence_csv": str(Path(evidence_csv).expanduser()),
        "output_csv": str(output_csv),
        "rank_column": rank_column,
        "ascending": ascending,
        "top_n": final_budget,
        "adaptive_budget": None if top_n is not None and int(top_n) > 0 else final_budget,
        "candidate_count": len(ranked_rows),
        "updated_at": _utc_now(),
    }
    summary_path = _write_json(outdir / "final_candidate_recommendation_summary.json", summary)
    final_stage = "final_recommendation_prepared" if ranked_rows else "final_recommendation_blocked_no_candidates"
    _update_project_manifest(
        project_name,
        {
            "stage": final_stage,
            "final_candidate_recommendation_csv": str(output_csv),
            "final_candidate_recommendation_summary_path": summary_path,
        },
    )
    return {
        **summary,
        "summary_path": summary_path,
        "next_tool": "none",
        "stop_after_this_tool": not bool(ranked_rows),
    }


def _manifest_existing_path(manifest: Dict[str, Any], key: str) -> str:
    value = str(manifest.get(key) or "").strip()
    if not value:
        return ""
    return value if Path(value).expanduser().exists() else ""


def _first_existing_path(paths: List[Any]) -> str:
    for value in paths:
        text = str(value or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            return str(path)
    return ""


def _manifest_or_project_file(
    manifest: Dict[str, Any],
    key: str,
    project: Path,
    relative_candidates: List[str],
) -> str:
    return _manifest_existing_path(manifest, key) or _first_existing_path(
        [project / rel for rel in relative_candidates]
    )


def _candidate_csv_from_structure_map(project: Path, manifest: Dict[str, Any]) -> str:
    """Create a downstream candidate CSV from structure-prediction map if needed."""
    job_manifest = _read_json(project / "04_structure_prediction" / "structure_prediction_job_manifest.json")
    candidate_map_path = _first_existing_path(
        [
            manifest.get("candidate_map_path"),
            manifest.get("structure_prediction_candidate_map_path"),
            job_manifest.get("candidate_map_path"),
            project / "04_structure_prediction" / "input" / "candidate_sequence_map.json",
        ]
    )
    if not candidate_map_path:
        return ""
    data = _read_json(Path(candidate_map_path))
    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    if not candidates:
        return ""

    rows: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        source_row = item.get("source_row") if isinstance(item.get("source_row"), dict) else {}
        row: Dict[str, Any] = dict(source_row)
        row["variant"] = _sanitize_name(str(item.get("variant") or source_row.get("variant") or ""))
        row["sequence"] = str(item.get("sequence") or source_row.get("sequence") or "").strip().upper()
        if row["variant"] and row["sequence"]:
            rows.append(row)
    if not rows:
        return ""
    outdir = project / "04_structure_prediction"
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / "structure_prediction_candidates.csv"
    _write_csv_rows(output_csv, rows, _csv_fieldnames_union(rows, ["variant", "sequence"]))
    return str(output_csv)


def _normalize_review_stage_name(stage_name: str) -> str:
    stage = _sanitize_name(stage_name or "all").lower()
    aliases = {
        "all_stages": "all",
        "structure": "structure_state",
        "state": "structure_state",
        "state_similarity": "structure_state",
        "structure_similarity": "structure_state",
        "structure_state_similarity": "structure_state",
        "confidence": "structure_confidence",
        "if": "inverse_folding",
        "inverse": "inverse_folding",
        "esm_if1": "inverse_folding",
        "protein_mpnn": "inverse_folding",
        "proteinmpnn": "inverse_folding",
        "boltz": "structure_prediction",
        "boltz2": "structure_prediction",
        "boltz2_nim": "structure_prediction",
        "boltz2_local": "structure_prediction",
        "boltz_local": "structure_prediction",
        "final_recommendation": "final",
        "candidate_review": "final",
    }
    return aliases.get(stage, stage)


def _review_stage_path_map(project: Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    return {
        "cb": _manifest_or_project_file(
            manifest,
            "cb_scored_library_csv",
            project,
            ["02_conformation_bias_scoring/cb_scored_library.csv"],
        ),
        "adaptive_selection": _manifest_or_project_file(
            manifest,
            "state_specific_candidates_adaptive_csv",
            project,
            ["03_state_specific_candidates/state_specific_retained_candidates_adaptive.csv"],
        ),
        "structure_prediction": _manifest_or_project_file(
            manifest,
            "structure_prediction_evidence_csv",
            project,
            ["04_structure_prediction/structure_prediction_evidence.csv"],
        ),
        "structure_state": _manifest_or_project_file(
            manifest,
            "structure_state_similarity_evidence_csv",
            project,
            ["04_structure_prediction/structure_state_similarity_evidence.csv"],
        ),
        "structure_confidence": _manifest_or_project_file(
            manifest,
            "structure_confidence_evidence_csv",
            project,
            ["04b_structure_confidence_filter/structure_confidence_evidence.csv"],
        ),
        "inverse_folding": _manifest_or_project_file(
            manifest,
            "inverse_folding_evidence_csv",
            project,
            ["05_inverse_folding_scoring/inverse_folding_evidence.csv"],
        ),
        "bioemu": _manifest_or_project_file(
            manifest,
            "bioemu_evidence_csv",
            project,
            ["06_bioemu_sampling/bioemu_evidence.csv"],
        ),
        "rosetta": _manifest_or_project_file(
            manifest,
            "rosetta_evidence_csv",
            project,
            ["07_rosetta_energy/rosetta_evidence.csv"],
        ),
        "merged": _manifest_or_project_file(
            manifest,
            "switchtune_evidence_merged_csv",
            project,
            ["08_final_recommendation/switchtune_evidence_merged.csv"],
        ),
        "ranked": _manifest_or_project_file(
            manifest,
            "switchtune_evidence_ranked_csv",
            project,
            ["08_final_recommendation/switchtune_evidence_ranked.csv"],
        ),
        "final": _manifest_or_project_file(
            manifest,
            "final_candidate_recommendation_csv",
            project,
            ["08_final_recommendation/final_candidate_recommendation.csv"],
        ),
    }


def _review_default_metrics(stage_name: str, fieldnames: List[str]) -> List[str]:
    stage = _normalize_review_stage_name(stage_name)
    preferred_by_stage = {
        "cb": [
            "bias_score_abs",
            "bias_score",
            "target_state_scaled",
            "reference_state_scaled",
            "switchtune_cb_pass",
        ],
        "adaptive_selection": ["adaptive_abs_bias_from_center", "bias_score_abs", "bias_score"],
        "structure_prediction": [
            "structure_prediction_confidence_score",
            "structure_prediction_has_template",
            "structure_prediction_write_full_pae",
            "structure_prediction_ligand_count",
            "structure_prediction_metric_model_inference_time_seconds",
        ],
        "structure_state": [
            "structure_target_similarity_score",
            "structure_target_like_region_count",
            "structure_reference_like_region_count",
            "state_similarity_global_delta_reference_minus_target_A",
        ],
        "structure_confidence": [
            "structure_confidence_score",
            "structure_confidence_pass",
            "structure_confidence_gate",
            "structure_confidence_rescue_eligible",
            "structure_confidence_confidence_score",
            "structure_confidence_mutation_window_plddt_mean",
            "structure_confidence_pae_primary_metric",
        ],
        "inverse_folding": [
            "inverse_folding_compatibility_score",
            "inverse_folding_model_count",
            "esm_if1_compatibility_score",
            "protein_mpnn_compatibility_score",
        ],
        "bioemu": [
            "bioemu_composite_score",
            "bioemu_priority_score",
            "bioemu_target_like_fraction",
            "bioemu_target_preference_margin_A",
            "bioemu_rmsd_to_target_mean_A",
            "bioemu_rmsd_to_reference_mean_A",
            "bioemu_pairwise_rmsd_mean_A",
            "bioemu_rmsf_global_mean_A",
            "bioemu_mutation_window_rmsf_mean_A",
        "bioemu_radius_gyration_mean_A",
        "bioemu_metrics_status",
        "bioemu_error_summary",
        "bioemu_sampling_complete",
            "bioemu_file_count",
            "bioemu_pdb_count",
            "bioemu_xtc_count",
            "bioemu_npz_count",
        ],
        "rosetta": [
            "rosetta_primary_score_mean",
            "rosetta_total_score_mean",
            "rosetta_dG_separated_mean",
            "rosetta_n",
        ],
        "merged": [
            "consensus_score",
            "consensus_evidence_count",
            "bias_score_abs",
            "structure_confidence_score",
            "inverse_folding_compatibility_score",
            "rosetta_primary_score_mean",
        ],
        "ranked": [
            "consensus_score",
            "consensus_evidence_count",
            "bias_score_abs",
            "structure_confidence_score",
            "inverse_folding_compatibility_score",
            "rosetta_primary_score_mean",
        ],
        "final": [
            "consensus_score",
            "consensus_evidence_count",
            "bias_score_abs",
            "structure_confidence_score",
            "structure_target_similarity_score",
            "inverse_folding_compatibility_score",
            "rosetta_primary_score_mean",
        ],
    }
    preferred = [name for name in preferred_by_stage.get(stage, []) if name in fieldnames]
    if preferred:
        return preferred
    return _auto_consensus_metric_columns(fieldnames)


def _compact_review_value(value: Any) -> Any:
    if value in {"", None}:
        return ""
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    number = _safe_float(text)
    if number is not None and math.isfinite(number):
        return round(number, 6)
    return text[:500]


def _value_counts_for_review(rows: List[Dict[str, Any]], field: str, limit: int = 12) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value:
            counts[value] += 1
    return dict(counts.most_common(limit))


def _metric_summary_for_review(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    values = [
        _bool_or_numeric(row.get(metric))
        for row in rows
        if _bool_or_numeric(row.get(metric)) is not None
    ]
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "metric": metric,
        "direction": _infer_metric_direction(metric),
        "numeric_count": len(finite_values),
        "missing_count": len(rows) - len(finite_values),
        "distribution": _score_distribution_summary(finite_values),
    }


def _compact_row_for_review(
    row: Dict[str, Any],
    *,
    variant_column: str,
    metric_columns: List[str],
    extra_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    preferred = [
        variant_column,
        "variant",
        "final_rank",
        "consensus_rank",
        "selection_rank",
        "position",
        "sequence_index_1based",
        "wt_aa",
        "mut_aa",
        "structure_confidence_pass",
        "structure_state_qc",
        "inverse_folding_status",
        "bioemu_command_status",
        "rosetta_primary_score_column",
    ]
    fields = []
    for field in preferred + list(extra_fields or []) + metric_columns:
        if field not in fields and field in row:
            fields.append(field)
    compact: Dict[str, Any] = {}
    variant = _variant_id(row, variant_column)
    if variant:
        compact["variant"] = variant
    for field in fields:
        if field in {variant_column, "variant"}:
            continue
        value = row.get(field)
        if value not in {"", None}:
            compact[field] = _compact_review_value(value)
    sequence = str(row.get("sequence") or "").strip()
    if sequence:
        compact["sequence_length"] = len(sequence)
    return compact


def _rank_rows_by_metric_for_review(
    rows: List[Dict[str, Any]],
    *,
    metric: str,
    variant_column: str,
    limit: int,
) -> List[Dict[str, Any]]:
    direction = _infer_metric_direction(metric)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        value = _bool_or_numeric(row.get(metric))
        if value is None or not math.isfinite(value):
            continue
        score = -value if direction == "lower" else value
        scored.append((score, row))
    ranked = sorted(scored, key=lambda item: (-item[0], str(item[1].get(variant_column) or item[1].get("variant") or "")))
    return [
        _compact_row_for_review(row, variant_column=variant_column, metric_columns=[metric])
        for _score, row in ranked[: max(0, int(limit))]
    ]


def _stage_findings_for_review(
    stage_name: str,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> Tuple[List[str], List[str]]:
    stage = _normalize_review_stage_name(stage_name)
    findings: List[str] = []
    warnings: List[str] = []
    n = len(rows)
    if n == 0:
        return ["No evidence rows were found for this stage."], ["Stage evidence is empty."]

    def count_true(field: str) -> int:
        return sum(1 for row in rows if _bool_or_numeric(row.get(field)) == 1.0)

    if stage == "structure_prediction":
        completed = count_true("structure_prediction_complete")
        templated = count_true("structure_prediction_has_template")
        full_pae = count_true("structure_prediction_write_full_pae")
        findings.append(f"{completed}/{n} structure predictions are marked complete.")
        if "structure_prediction_has_template" in fieldnames:
            findings.append(f"{templated}/{n} predictions used a structural template.")
            if templated < n:
                warnings.append("Some predictions lack target-state template evidence.")
        if "structure_prediction_write_full_pae" in fieldnames:
            findings.append(f"{full_pae}/{n} payloads requested full PAE output.")
            if full_pae < n:
                warnings.append("Some Boltz-2 payloads do not show write_full_pae=true.")
        failed = n - completed
        if failed:
            warnings.append(f"{failed} structure predictions are incomplete or failed.")
    elif stage == "structure_state":
        target_like = sum(1 for row in rows if str(row.get("structure_state_qc", "")).strip() == "target_like")
        findings.append(f"{target_like}/{n} predicted structures are classified target_like.")
        if target_like < n:
            warnings.append("At least one structure is not confidently target-like.")
    elif stage == "structure_confidence":
        passed = count_true("structure_confidence_pass")
        findings.append(f"{passed}/{n} candidates passed structure confidence filtering.")
        fail_counts = _value_counts_for_review(rows, "structure_confidence_fail_reasons")
        if fail_counts:
            warnings.append("Observed structure-confidence fail reasons: " + json.dumps(fail_counts, ensure_ascii=True))
    elif stage == "inverse_folding":
        scored = sum(1 for row in rows if str(row.get("inverse_folding_status", "")).strip() == "scored")
        two_model = sum(1 for row in rows if (_safe_int(row.get("inverse_folding_model_count")) or 0) >= 2)
        findings.append(f"{scored}/{n} candidates have inverse-folding scores.")
        if "inverse_folding_model_count" in fieldnames:
            findings.append(f"{two_model}/{n} candidates have scores from at least two inverse-folding models.")
        if scored < n:
            warnings.append("Some candidates are missing inverse-folding evidence.")
    elif stage == "bioemu":
        completed = count_true("bioemu_sampling_complete")
        findings.append(f"{completed}/{n} candidates have BioEmu sampling outputs.")
        metrics_completed = sum(
            1 for row in rows if str(row.get("bioemu_metrics_status") or "").strip() == "completed"
        )
        if "bioemu_metrics_status" in fieldnames:
            findings.append(f"{metrics_completed}/{n} candidates have computed ensemble metrics.")
        target_like_values = [
            _safe_float(row.get("bioemu_target_like_fraction"))
            for row in rows
            if _safe_float(row.get("bioemu_target_like_fraction")) is not None
        ]
        if target_like_values:
            findings.append(
                f"Mean target-like sample fraction is {mean(target_like_values):.3f}."
            )
        rmsf_values = [
            _safe_float(row.get("bioemu_mutation_window_rmsf_mean_A"))
            for row in rows
            if _safe_float(row.get("bioemu_mutation_window_rmsf_mean_A")) is not None
        ]
        if rmsf_values:
            findings.append(
                f"Mean mutation-window ensemble RMSF is {mean(rmsf_values):.3f} A."
            )
        source_counts = _value_counts_for_review(rows, "bioemu_evidence_source")
        if source_counts:
            findings.append("BioEmu evidence sources: " + json.dumps(source_counts, ensure_ascii=True))
        if metrics_completed < n:
            warnings.append(
                "Some candidates lack computed BioEmu ensemble metrics; do not treat inventory completion as dynamic evidence."
            )
    elif stage == "rosetta":
        scored = sum(1 for row in rows if _safe_float(row.get("rosetta_primary_score_mean")) is not None)
        findings.append(f"{scored}/{n} candidates have a Rosetta primary score.")
        if "rosetta_n" in fieldnames:
            low_n = sum(1 for row in rows if (_safe_int(row.get("rosetta_n")) or 0) <= 1)
            if low_n:
                warnings.append(f"{low_n}/{n} Rosetta summaries are based on one structure/score row; use larger nstruct for final runs.")
    elif stage in {"merged", "ranked", "final"}:
        ranked = sum(1 for row in rows if _safe_float(row.get("consensus_score")) is not None)
        variants = [_variant_id(row) for row in rows[: min(5, n)] if _variant_id(row)]
        findings.append(f"{ranked}/{n} candidates have consensus scores.")
        if variants:
            findings.append("Top visible candidates: " + ", ".join(variants))
    elif stage == "cb":
        values = [_safe_float(row.get("bias_score_abs")) for row in rows if _safe_float(row.get("bias_score_abs")) is not None]
        findings.append(f"{len(values)}/{n} candidates have numeric CB bias scores.")
        if values:
            findings.append(f"Median absolute CB bias is {median(values):.4g}.")
    else:
        findings.append(f"{n} evidence rows are available.")
    return findings, warnings


def _build_stage_evidence_review(
    *,
    project_name: str,
    stage_name: str,
    evidence_csv: str,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
    variant_column: str,
    metric_columns: List[str],
    top_n: int,
) -> Dict[str, Any]:
    selected_metrics = metric_columns or _review_default_metrics(stage_name, fieldnames)
    selected_metrics = [metric for metric in selected_metrics if metric in fieldnames]
    status_fields = [
        field
        for field in fieldnames
        if field.endswith("_status")
        or field.endswith("_pass")
        or field.endswith("_complete")
        or field.endswith("_qc")
        or field.endswith("_target_like")
        or field in {"status", "score_status"}
    ]
    findings, warnings = _stage_findings_for_review(stage_name, rows, fieldnames)
    top_rows_by_metric = {
        metric: _rank_rows_by_metric_for_review(
            rows,
            metric=metric,
            variant_column=variant_column,
            limit=min(int(top_n), 10),
        )
        for metric in selected_metrics
    }
    compact_examples = [
        _compact_row_for_review(row, variant_column=variant_column, metric_columns=selected_metrics)
        for row in rows[: min(int(top_n), 10)]
    ]
    return {
        "status": "completed",
        "project_name": project_name,
        "stage_name": _normalize_review_stage_name(stage_name),
        "evidence_csv": evidence_csv,
        "row_count": len(rows),
        "field_count": len(fieldnames),
        "key_metric_columns": selected_metrics,
        "metric_directions": {metric: _infer_metric_direction(metric) for metric in selected_metrics},
        "metric_summaries": {
            metric: _metric_summary_for_review(rows, metric)
            for metric in selected_metrics
        },
        "status_counts": {
            field: _value_counts_for_review(rows, field)
            for field in status_fields[:20]
        },
        "top_rows_by_metric": top_rows_by_metric,
        "example_rows": compact_examples,
        "findings": findings,
        "warnings": warnings,
    }


def _write_stage_review_markdown(path: Path, review: Dict[str, Any]) -> str:
    lines = [
        f"# SwitchTune Stage Evidence Review: {review.get('project_name')}",
        "",
        f"- Status: {review.get('status')}",
        f"- Project: {review.get('project_name')}",
        f"- Updated: {review.get('updated_at')}",
    ]
    for item in review.get("stage_reviews", []):
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                "",
                f"## {item.get('stage_name')}",
                f"- Evidence CSV: {item.get('evidence_csv')}",
                f"- Rows: {item.get('row_count')}",
                f"- Key metrics: {', '.join(item.get('key_metric_columns') or [])}",
            ]
        )
        for finding in item.get("findings") or []:
            lines.append(f"- Finding: {finding}")
        for warning in item.get("warnings") or []:
            lines.append(f"- Caution: {warning}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _candidate_support_and_cautions(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    support: List[str] = []
    cautions: List[str] = []

    bias_abs = _safe_float(row.get("bias_score_abs"))
    if bias_abs is not None:
        if bias_abs <= 0.05:
            support.append(f"CB bias is near the neutral/switchable center (abs={bias_abs:.4g}).")
        elif bias_abs > 1.0:
            cautions.append(f"CB bias is large (abs={bias_abs:.4g}); may over-stabilize one state.")

    if _bool_or_numeric(row.get("structure_prediction_has_template")) == 1.0:
        support.append("Structure prediction records target-state template use.")
    elif "structure_prediction_has_template" in row:
        cautions.append("No structural template evidence is recorded.")

    if _bool_or_numeric(row.get("structure_prediction_write_full_pae")) == 1.0:
        support.append("Boltz-2 payload requested full PAE.")
    elif "structure_prediction_write_full_pae" in row:
        cautions.append("Full PAE request is absent or false.")

    if _bool_or_numeric(row.get("structure_confidence_pass")) == 1.0:
        support.append("Structure confidence QC passed.")
    elif "structure_confidence_pass" in row:
        reason = str(row.get("structure_confidence_fail_reasons") or "unspecified reason")
        cautions.append(f"Structure confidence QC did not pass: {reason}.")

    state_qc = str(row.get("structure_state_qc") or row.get("structure_confidence_state_qc") or "").strip()
    if state_qc == "target_like":
        support.append("Predicted structure is closer to the target conformation.")
    elif state_qc:
        cautions.append(f"State similarity QC is {state_qc}.")

    local_plddt = _safe_float(
        row.get("structure_confidence_mutation_window_plddt_mean")
        or row.get("structure_confidence_mutation_window_pm3_plddt_mean")
    )
    if local_plddt is not None:
        if local_plddt >= 90:
            support.append(f"Mutation-window pLDDT is high ({local_plddt:.2f}).")
        elif local_plddt < 70:
            cautions.append(f"Mutation-window pLDDT is low ({local_plddt:.2f}).")

    pae = _safe_float(row.get("structure_confidence_pae_primary_metric"))
    pae_cutoff = _safe_float(row.get("structure_confidence_adaptive_pae_cutoff"))
    if pae is not None:
        if pae_cutoff is not None and pae <= pae_cutoff:
            support.append(f"Mutation-region PAE is within adaptive cutoff ({pae:.3g} <= {pae_cutoff:.3g}).")
        elif pae_cutoff is not None:
            cautions.append(f"Mutation-region PAE exceeds adaptive cutoff ({pae:.3g} > {pae_cutoff:.3g}).")

    model_count = _safe_int(row.get("inverse_folding_model_count"))
    if str(row.get("inverse_folding_status") or "").strip() == "scored":
        support.append(f"Inverse-folding compatibility was scored by {model_count or 'available'} model(s).")
    elif "inverse_folding_status" in row:
        cautions.append("Inverse-folding evidence is missing or unscored.")

    if _bool_or_numeric(row.get("bioemu_sampling_complete")) == 1.0:
        support.append("BioEmu sampling completed for this candidate.")
    elif "bioemu_sampling_complete" in row:
        cautions.append("BioEmu sampling is incomplete or not available.")

    bioemu_status = str(row.get("bioemu_metrics_status") or "").strip().lower()
    if bioemu_status == "completed":
        target_like = _safe_float(row.get("bioemu_target_like_fraction"))
        target_margin = _safe_float(row.get("bioemu_target_preference_margin_A"))
        local_rmsf = _safe_float(row.get("bioemu_mutation_window_rmsf_mean_A"))
        pairwise_rmsd = _safe_float(row.get("bioemu_pairwise_rmsd_mean_A"))
        if target_like is not None:
            support.append(f"BioEmu target-like sample fraction is {target_like:.3f}.")
        if target_margin is not None and target_margin > 0:
            support.append(f"BioEmu favors the target state by {target_margin:.3f} A RMSD margin.")
        if local_rmsf is not None:
            support.append(f"BioEmu mutation-window RMSF is {local_rmsf:.3f} A.")
        if pairwise_rmsd is not None:
            support.append(f"BioEmu ensemble pairwise RMSD is {pairwise_rmsd:.3f} A.")
    elif "bioemu_metrics_status" in row:
        cautions.append("Computed BioEmu ensemble metrics are unavailable.")
    bioemu_error = str(row.get("bioemu_error_summary") or "").strip()
    if bioemu_error:
        cautions.append(f"BioEmu execution issue: {bioemu_error}.")

    if _safe_float(row.get("rosetta_primary_score_mean")) is not None:
        support.append(f"Rosetta primary score is available ({row.get('rosetta_primary_score_mean')}).")
    elif "rosetta_primary_score_mean" in row:
        cautions.append("Rosetta primary score is missing.")

    for key, value in row.items():
        if not key.startswith("consensus_component__"):
            continue
        score = _safe_float(value)
        metric = key.replace("consensus_component__", "")
        if score is None:
            continue
        if score >= 1.0:
            support.append(f"Strong relative support from {metric} (z={score:.2f}).")
        elif score <= -1.0:
            raw_metric_value = _bool_or_numeric(row.get(metric))
            if metric == "bias_score_abs" and raw_metric_value is not None and raw_metric_value <= 0.05:
                continue
            if "confidence" in metric and raw_metric_value is not None and raw_metric_value >= 0.90:
                continue
            if "plddt" in metric and raw_metric_value is not None and raw_metric_value >= 90.0:
                continue
            cautions.append(f"Relative tradeoff from {metric} within this shortlist (z={score:.2f}).")

    return support[:12], cautions[:12]


def _sort_candidate_review_rows(rows: List[Dict[str, Any]], rank_column: str, ascending: bool) -> List[Dict[str, Any]]:
    rank_fields = ["final_rank", "consensus_rank", "selection_rank"]
    for field in rank_fields:
        if any(_safe_int(row.get(field)) is not None for row in rows):
            return sorted(rows, key=lambda row: (_safe_int(row.get(field)) or 10**9, str(row.get("variant") or "")))
    if any(_bool_or_numeric(row.get(rank_column)) is not None for row in rows):
        return sorted(
            rows,
            key=lambda row: (
                _bool_or_numeric(row.get(rank_column)) is None,
                (_bool_or_numeric(row.get(rank_column)) or 0.0) * (1 if ascending else -1),
                str(row.get("variant") or ""),
            ),
        )
    return sorted(rows, key=lambda row: str(row.get("variant") or ""))


def _write_agentic_review_markdown(path: Path, packet: Dict[str, Any]) -> str:
    lines = [
        f"# SwitchTune Agentic Candidate Review: {packet.get('project_name')}",
        "",
        f"- Status: {packet.get('status')}",
        f"- Evidence CSV: {packet.get('evidence_csv')}",
        f"- Candidate count reviewed: {packet.get('reviewed_candidate_count')}",
        f"- Updated: {packet.get('updated_at')}",
        "",
        "## Stage Snapshot",
    ]
    for item in packet.get("stage_reviews", []):
        if not isinstance(item, dict):
            continue
        first_finding = (item.get("findings") or [""])[0]
        warning_count = len(item.get("warnings") or [])
        lines.append(
            f"- {item.get('stage_name')}: rows={item.get('row_count')}; "
            f"metrics={','.join(item.get('key_metric_columns') or [])}; "
            f"warnings={warning_count}; {first_finding}"
        )
    lines.extend(["", "## Candidate Notes"])
    for item in packet.get("candidate_reviews", []):
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                "",
                f"### {item.get('rank')}. {item.get('variant')}",
                f"- Consensus score: {item.get('consensus_score', '')}",
                f"- Evidence: {json.dumps(item.get('key_evidence') or {}, ensure_ascii=True)}",
            ]
        )
        for signal in item.get("supporting_evidence") or []:
            lines.append(f"- Support: {signal}")
        for caution in item.get("caution_flags") or []:
            lines.append(f"- Caution: {caution}")
    lines.extend(["", "## Agent Review Prompt", str(packet.get("agent_review_prompt") or "")])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@mcp.tool()
def review_switchtune_stage_evidence(
    project_name: str,
    stage_name: str = "all",
    evidence_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    variant_column: str = "variant",
    metric_columns: str = "",
    top_n: int = 5,
) -> Dict[str, Any]:
    """Create a compact, agent-readable review of SwitchTune evidence tables.

    This tool does not rank final candidates or run external jobs. It compresses
    large stage CSVs into distributions, status counts, top metric examples, and
    stage-specific cautions so an LLM agent can reason over evidence safely.
    """
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "08_final_recommendation"
    outdir.mkdir(parents=True, exist_ok=True)
    selected_metrics = [item.strip() for item in str(metric_columns).split(",") if item.strip()]

    path_map = _review_stage_path_map(project, manifest)
    normalized_stage = _normalize_review_stage_name(stage_name)
    stage_inputs: List[Tuple[str, str]] = []
    if evidence_csv:
        stage_inputs.append((normalized_stage if normalized_stage != "all" else "custom", str(evidence_csv)))
    elif normalized_stage == "all":
        for stage in (
            "cb",
            "adaptive_selection",
            "structure_prediction",
            "structure_state",
            "structure_confidence",
            "inverse_folding",
            "bioemu",
            "rosetta",
            "final",
        ):
            if path_map.get(stage):
                stage_inputs.append((stage, path_map[stage]))
    elif path_map.get(normalized_stage):
        stage_inputs.append((normalized_stage, path_map[normalized_stage]))

    missing_sources: List[str] = []
    stage_reviews: List[Dict[str, Any]] = []
    if not stage_inputs:
        missing_sources.append(normalized_stage)

    for stage, path_text in stage_inputs:
        path = Path(path_text).expanduser()
        if not path.exists():
            missing_sources.append(stage)
            continue
        rows, fields = _read_csv_rows(path)
        stage_reviews.append(
            _build_stage_evidence_review(
                project_name=project_name,
                stage_name=stage,
                evidence_csv=str(path),
                rows=[dict(row) for row in rows],
                fieldnames=fields,
                variant_column=variant_column,
                metric_columns=selected_metrics,
                top_n=top_n,
            )
        )

    summary = {
        "status": "completed" if stage_reviews else "missing",
        "project_name": project_name,
        "requested_stage_name": stage_name,
        "stage_count": len(stage_reviews),
        "missing_sources": missing_sources,
        "stage_reviews": stage_reviews,
        "agent_instruction": (
            "Use this evidence review to reason about stage quality and metric conflicts. "
            "Do not rerun completed workflow stages unless a required evidence source is missing or failed."
        ),
        "updated_at": _utc_now(),
    }
    suffix = "all" if normalized_stage == "all" and not evidence_csv else _sanitize_name(normalized_stage)
    json_path = _write_json(outdir / f"switchtune_stage_evidence_review_{suffix}.json", summary)
    markdown_path = _write_stage_review_markdown(
        outdir / f"switchtune_stage_evidence_review_{suffix}.md",
        {**summary, "review_json": json_path},
    )
    _update_project_manifest(
        project_name,
        {
            "stage": "stage_evidence_reviewed",
            "stage_evidence_review_json": json_path,
            "stage_evidence_review_markdown": markdown_path,
        },
    )
    return {**summary, "review_json": json_path, "review_markdown": markdown_path}


@mcp.tool()
def prepare_agentic_candidate_review(
    project_name: str,
    evidence_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    variant_column: str = "variant",
    rank_column: str = "consensus_score",
    ascending: bool = False,
    metric_columns: str = "",
    top_n: int = 10,
    include_stage_reviews: bool = True,
) -> Dict[str, Any]:
    """Prepare a compact evidence packet for LLM/agent candidate review.

    The packet is intentionally deterministic and auditable: it selects visible
    candidates from the existing final/merged evidence table, annotates supports
    and cautions, and returns a prompt that tells Biomni how to make an explicit
    evidence-based decision without rerunning tools.
    """
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "08_final_recommendation"
    outdir.mkdir(parents=True, exist_ok=True)

    resolved_evidence_csv = (
        str(evidence_csv or "").strip()
        or _manifest_or_project_file(
            manifest,
            "switchtune_evidence_merged_csv",
            project,
            ["08_final_recommendation/switchtune_evidence_merged.csv"],
        )
        or _manifest_or_project_file(
            manifest,
            "switchtune_evidence_ranked_csv",
            project,
            ["08_final_recommendation/switchtune_evidence_ranked.csv"],
        )
        or _manifest_or_project_file(
            manifest,
            "final_candidate_recommendation_csv",
            project,
            ["08_final_recommendation/final_candidate_recommendation.csv"],
        )
    )
    if not resolved_evidence_csv:
        raise FileNotFoundError(
            "No final/merged candidate evidence CSV was found. Run merge_switchtune_evidence_tables first."
        )
    evidence_path = Path(resolved_evidence_csv).expanduser()
    if not evidence_path.exists():
        raise FileNotFoundError(f"candidate evidence CSV not found: {resolved_evidence_csv}")

    rows, fields = _read_csv_rows(evidence_path)
    ranked_rows = _sort_candidate_review_rows([dict(row) for row in rows], rank_column, ascending)
    selected_rows = ranked_rows[: max(0, int(top_n))]
    selected_metrics = [item.strip() for item in str(metric_columns).split(",") if item.strip()]
    if not selected_metrics:
        selected_metrics = _auto_consensus_metric_columns(fields)
    selected_metrics = [metric for metric in selected_metrics if metric in fields]

    candidate_reviews: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected_rows, start=1):
        support, cautions = _candidate_support_and_cautions(row)
        key_evidence = _compact_row_for_review(
            row,
            variant_column=variant_column,
            metric_columns=selected_metrics,
            extra_fields=["consensus_score", "selection_score", "bias_score", "structure_path"],
        )
        variant = _variant_id(row, variant_column)
        candidate_reviews.append(
            {
                "rank": idx,
                "variant": variant,
                "position": _compact_review_value(row.get("position") or ""),
                "wt_aa": _compact_review_value(row.get("wt_aa") or ""),
                "mut_aa": _compact_review_value(row.get("mut_aa") or ""),
                "consensus_score": _compact_review_value(row.get("consensus_score") or ""),
                "selection_score": _compact_review_value(row.get("selection_score") or ""),
                "key_evidence": key_evidence,
                "supporting_evidence": support,
                "caution_flags": cautions,
                "agent_question": (
                    "Should this candidate remain in the experimental shortlist after weighing switchability, "
                    "target-like structure, sequence compatibility, dynamics evidence, and Rosetta energy?"
                ),
            }
        )

    stage_reviews: List[Dict[str, Any]] = []
    if include_stage_reviews:
        path_map = _review_stage_path_map(project, manifest)
        for stage in (
            "cb",
            "structure_prediction",
            "structure_state",
            "structure_confidence",
            "inverse_folding",
            "bioemu",
            "rosetta",
        ):
            path_text = path_map.get(stage, "")
            path = Path(path_text).expanduser() if path_text else None
            if path is None or not path.exists():
                continue
            stage_rows, stage_fields = _read_csv_rows(path)
            stage_reviews.append(
                _build_stage_evidence_review(
                    project_name=project_name,
                    stage_name=stage,
                    evidence_csv=str(path),
                    rows=[dict(row) for row in stage_rows],
                    fieldnames=stage_fields,
                    variant_column=variant_column,
                    metric_columns=[],
                    top_n=min(3, int(top_n)),
                )
            )

    agent_prompt = (
        "Review the SwitchTune candidate evidence packet. Do not rerun completed tools. "
        "Identify candidates with convergent support across CB neutrality, target-like templated structure, "
        "structure-confidence/PAE, inverse-folding compatibility, BioEmu dynamics evidence, and Rosetta energy. "
        "If a candidate is ranked highly by consensus but has a serious caution flag, explain the tradeoff. "
        "Return a shortlist with rationale and call record_agentic_candidate_decision to log the final judgement."
    )
    packet = {
        "status": "completed",
        "project_name": project_name,
        "evidence_csv": str(evidence_path),
        "reviewed_candidate_count": len(candidate_reviews),
        "available_candidate_count": len(rows),
        "rank_column": rank_column,
        "ascending": ascending,
        "key_metric_columns": selected_metrics,
        "candidate_reviews": candidate_reviews,
        "stage_reviews": stage_reviews,
        "agent_review_prompt": agent_prompt,
        "updated_at": _utc_now(),
    }
    packet_json = _write_json(outdir / "agentic_candidate_review_packet.json", packet)
    packet_md = _write_agentic_review_markdown(outdir / "agentic_candidate_review_packet.md", {**packet, "packet_json": packet_json})
    _update_project_manifest(
        project_name,
        {
            "stage": "agentic_candidate_review_prepared",
            "agentic_candidate_review_json": packet_json,
            "agentic_candidate_review_markdown": packet_md,
        },
    )
    return {**packet, "packet_json": packet_json, "packet_markdown": packet_md}


@mcp.tool()
def record_agentic_candidate_decision(
    project_name: str,
    selected_variants: Any,
    decision_text: str,
    evidence_packet_json: Optional[str] = None,
    reviewer: str = "biomni",
    output_dir: Optional[str] = None,
    decision_scope: str = "final_round",
) -> Dict[str, Any]:
    """Record an explicit agent/human decision based on an evidence packet.

    ``decision_scope`` distinguishes a stage checkpoint from the final-round
    review so resumable workflows cannot mistake an earlier checkpoint record
    for completion of the final review.
    """
    project = _project_dir(project_name)
    outdir = Path(output_dir).expanduser().resolve() if output_dir else project / "08_final_recommendation"
    outdir.mkdir(parents=True, exist_ok=True)
    variants = _normalize_selected_variant_input(selected_variants)
    decision_status = "completed" if variants else "blocked_no_candidates"
    requested_scope = str(decision_scope or "").strip().lower()
    reviewer_scope = str(reviewer or "").strip().lower()
    # If a checkpoint reviewer omits the optional metadata argument, infer the
    # safe scope from its explicit reviewer name rather than misclassifying the
    # checkpoint as the final-round decision.
    if "checkpoint" in reviewer_scope and requested_scope in {"", "final", "final_round", "final-round"}:
        requested_scope = "checkpoint"
    normalized_scope = re.sub(r"[^a-z0-9_-]+", "_", requested_scope or "final_round")
    if not normalized_scope:
        normalized_scope = "final_round"
    decision_log_path = project / "decision_log.jsonl"

    # Biomni may emit the same execute block more than once before the guard
    # sees the first tool observation. Make decision recording idempotent for a
    # reviewer/scope/evidence packet within one project.
    for existing in reversed(_read_jsonl_records(decision_log_path)):
        if existing.get("step_name") != "agentic_candidate_decision":
            continue
        if str(existing.get("reviewer") or "") != str(reviewer or ""):
            continue
        if str(existing.get("decision_scope") or "") != normalized_scope:
            continue
        existing_input = existing.get("input_summary") or {}
        if str(existing_input.get("evidence_packet_json") or "") != str(evidence_packet_json or ""):
            continue
        existing_output = existing.get("output_summary") or {}
        existing_decision_json = str(existing_output.get("decision_json") or "")
        existing_decision_md = str(existing_output.get("decision_markdown") or "")
        existing_selected = _normalize_selected_variant_input(existing.get("decision") or "")
        return {
            "status": existing_output.get("selection_status") or ("completed" if existing_selected else "blocked_no_candidates"),
            "project_name": project_name,
            "reviewer": reviewer,
            "decision_scope": normalized_scope,
            "selected_variants": existing_selected,
            "selected_variant_count": len(existing_selected),
            "decision_json": existing_decision_json,
            "decision_markdown": existing_decision_md,
            "decision_log_path": str(decision_log_path),
            "duplicate": True,
            "next_tool": "none" if not existing_selected else "",
            "stop_after_this_tool": not bool(existing_selected),
            "updated_at": _utc_now(),
        }

    safe_stamp = re.sub(r"[^0-9A-Za-z_.-]+", "_", _utc_now()).strip("_")
    record = {
        "timestamp": _utc_now(),
        "project_name": project_name,
        "reviewer": reviewer,
        "decision_scope": normalized_scope,
        "selected_variants": variants,
        "selection_status": decision_status,
        "decision_text": decision_text,
        "evidence_packet_json": str(evidence_packet_json or ""),
    }
    decision_json = _write_json(outdir / f"agentic_candidate_decision_{safe_stamp}.json", record)
    decision_md = outdir / f"agentic_candidate_decision_{safe_stamp}.md"
    decision_md.write_text(
        "\n".join(
            [
                f"# SwitchTune Agentic Candidate Decision: {project_name}",
                "",
                f"- Reviewer: {reviewer}",
                f"- Decision scope: {normalized_scope}",
                f"- Selected variants: {', '.join(variants)}",
                f"- Selection status: {decision_status}",
                f"- Evidence packet: {evidence_packet_json or ''}",
                "",
                "## Decision",
                decision_text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    log_path = _append_jsonl(
        decision_log_path,
        {
            "timestamp": record["timestamp"],
            "project_name": project_name,
            "step_name": "agentic_candidate_decision",
            "decision": ",".join(variants) if variants else "NONE",
            "rationale": decision_text,
            "reviewer": reviewer,
            "decision_scope": normalized_scope,
            "input_summary": {
                "evidence_packet_json": evidence_packet_json or "",
                "decision_scope": normalized_scope,
            },
            "output_summary": {
                "decision_json": decision_json,
                "decision_markdown": str(decision_md),
                "selection_status": decision_status,
                "selected_variant_count": len(variants),
            },
        },
    )
    _update_project_manifest(
        project_name,
        {
            "stage": "agentic_candidate_decision_recorded",
            "agentic_candidate_decision_json": decision_json,
            "agentic_candidate_decision_markdown": str(decision_md),
        },
    )
    return {
        "status": decision_status,
        "project_name": project_name,
        "reviewer": reviewer,
        "decision_scope": normalized_scope,
        "selected_variants": variants,
        "selected_variant_count": len(variants),
        "decision_json": decision_json,
        "decision_markdown": str(decision_md),
        "decision_log_path": log_path,
        "duplicate": False,
        "next_tool": "none" if not variants else "",
        "stop_after_this_tool": not bool(variants),
        "updated_at": _utc_now(),
    }


@mcp.tool()
def recommend_next_workflow_step(
    project_name: str,
    target_state_structure: Optional[str] = None,
    reference_state_structure: Optional[str] = None,
    structure_similarity_regions: str = "global",
    desired_structure_candidate_count: Optional[int] = None,
    desired_bioemu_candidate_count: Optional[int] = None,
    desired_rosetta_candidate_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Recommend the next SwitchTune MCP action from current project state.

    This is a lightweight controller for 0-code agent runs. It never launches
    expensive jobs; it reports whether the agent should prepare a job, collect
    outputs after an external run, or merge evidence.
    """
    project = _project_dir(project_name)
    manifest = _load_project_manifest(project_name)
    derived_structure_candidate_csv = _candidate_csv_from_structure_map(project, manifest)
    files = {
        "library_csv": _manifest_or_project_file(
            manifest,
            "library_csv",
            project,
            [
                "01_saturation_library/saturation_single_mutant_library.csv",
                "01_saturation_library/filtered_saturation_single_mutant_library.csv",
            ],
        ),
        "cb_scored_library_csv": _manifest_or_project_file(
            manifest,
            "cb_scored_library_csv",
            project,
            ["02_conformation_bias_scoring/cb_scored_library.csv"],
        ),
        "state_specific_candidates_adaptive_csv": _manifest_or_project_file(
            manifest,
            "state_specific_candidates_adaptive_csv",
            project,
            [
                "03_state_specific_candidates/state_specific_retained_candidates_adaptive.csv",
                "04_structure_prediction/structure_prediction_candidates.csv",
            ],
        ),
        "state_specific_candidates_csv": _manifest_or_project_file(
            manifest,
            "state_specific_candidates_csv",
            project,
            [
                "03_state_specific_candidates/state_specific_retained_candidates.csv",
                "04_structure_prediction/structure_prediction_candidates.csv",
            ],
        ),
        "structure_prediction_results_csv": _manifest_or_project_file(
            manifest,
            "structure_prediction_results_csv",
            project,
            ["04_structure_prediction/structure_prediction_results.csv"],
        ),
        "structure_prediction_pdb_results_csv": _manifest_or_project_file(
            manifest,
            "structure_prediction_pdb_results_csv",
            project,
            ["04_structure_prediction/structure_prediction_results_pdb.csv"],
        ),
        "structure_prediction_evidence_csv": _manifest_or_project_file(
            manifest,
            "structure_prediction_evidence_csv",
            project,
            ["04_structure_prediction/structure_prediction_evidence.csv"],
        ),
        "structure_state_similarity_evidence_csv": _manifest_or_project_file(
            manifest,
            "structure_state_similarity_evidence_csv",
            project,
            ["04_structure_prediction/structure_state_similarity_evidence.csv"],
        ),
        "structure_confidence_evidence_csv": _manifest_or_project_file(
            manifest,
            "structure_confidence_evidence_csv",
            project,
            ["04b_structure_confidence_filter/structure_confidence_evidence.csv"],
        ),
        "structure_confidence_retained_candidates_csv": _manifest_or_project_file(
            manifest,
            "structure_confidence_retained_candidates_csv",
            project,
            ["04b_structure_confidence_filter/structure_confidence_retained_candidates.csv"],
        ),
        "structure_confidence_review_candidates_csv": _manifest_or_project_file(
            manifest,
            "structure_confidence_review_candidates_csv",
            project,
            ["04b_structure_confidence_filter/structure_confidence_review_candidates.csv"],
        ),
        "inverse_folding_evidence_csv": _manifest_or_project_file(
            manifest,
            "inverse_folding_evidence_csv",
            project,
            ["05_inverse_folding_scoring/inverse_folding_evidence.csv"],
        ),
        "bioemu_candidate_csv": _manifest_or_project_file(
            manifest,
            "bioemu_candidate_csv",
            project,
            ["03_state_specific_candidates/bioemu_selected_candidates.csv"],
        ),
        "bioemu_evidence_csv": _manifest_or_project_file(
            manifest,
            "bioemu_evidence_csv",
            project,
            ["06_bioemu_sampling/bioemu_evidence.csv"],
        ),
        "rosetta_candidate_csv": _manifest_or_project_file(
            manifest,
            "rosetta_candidate_csv",
            project,
            ["03_state_specific_candidates/rosetta_selected_candidates.csv"],
        ),
        "rosetta_evidence_csv": _manifest_or_project_file(
            manifest,
            "rosetta_evidence_csv",
            project,
            ["07_rosetta_energy/rosetta_evidence.csv"],
        ),
        "switchtune_evidence_ranked_csv": _manifest_or_project_file(
            manifest,
            "switchtune_evidence_ranked_csv",
            project,
            ["08_final_recommendation/switchtune_evidence_ranked.csv"],
        ),
        "final_candidate_recommendation_csv": _manifest_or_project_file(
            manifest,
            "final_candidate_recommendation_csv",
            project,
            ["08_final_recommendation/final_candidate_recommendation.csv"],
        ),
        "agentic_candidate_review_json": _manifest_or_project_file(
            manifest,
            "agentic_candidate_review_json",
            project,
            ["08_final_recommendation/agentic_candidate_review_packet.json"],
        ),
        "agentic_candidate_review_markdown": _manifest_or_project_file(
            manifest,
            "agentic_candidate_review_markdown",
            project,
            ["08_final_recommendation/agentic_candidate_review_packet.md"],
        ),
        "agentic_candidate_decision_json": _manifest_or_project_file(
            manifest,
            "agentic_candidate_decision_json",
            project,
            ["08_final_recommendation/agentic_candidate_decision.json"],
        ),
    }
    if derived_structure_candidate_csv:
        files["derived_structure_candidate_csv"] = derived_structure_candidate_csv
    candidate_csv = files["state_specific_candidates_adaptive_csv"] or files["state_specific_candidates_csv"]
    if not candidate_csv and derived_structure_candidate_csv:
        candidate_csv = derived_structure_candidate_csv
    post_structure_candidate_csv = _first_nonempty_csv_path(
        files["structure_confidence_retained_candidates_csv"],
        files["structure_confidence_review_candidates_csv"],
        candidate_csv,
    )
    budget_pool_count = (
        _csv_data_row_count(post_structure_candidate_csv)
        or _csv_data_row_count(candidate_csv)
        or _csv_data_row_count(files["library_csv"])
        or 0
    )
    desired_structure_candidate_count = (
        int(desired_structure_candidate_count)
        if desired_structure_candidate_count is not None and int(desired_structure_candidate_count) > 0
        else budget_pool_count
    )
    desired_bioemu_candidate_count = (
        int(desired_bioemu_candidate_count)
        if desired_bioemu_candidate_count is not None and int(desired_bioemu_candidate_count) > 0
        else adaptive_candidate_budget(budget_pool_count)
    )
    desired_rosetta_candidate_count = (
        int(desired_rosetta_candidate_count)
        if desired_rosetta_candidate_count is not None and int(desired_rosetta_candidate_count) > 0
        else adaptive_candidate_budget(budget_pool_count)
    )
    structure_csv = files["structure_prediction_pdb_results_csv"] or files["structure_prediction_results_csv"]
    resolved_target = str(target_state_structure or manifest.get("target_state_structure") or "").strip()
    resolved_reference = str(reference_state_structure or manifest.get("reference_state_structure") or "").strip()

    structure_evidence_gate: Dict[str, Any] = {}
    structure_evidence_blocked: Optional[Tuple[str, str]] = None
    structure_evidence_path = files["structure_prediction_evidence_csv"]
    if structure_evidence_path:
        evidence_rows, _evidence_fields = _read_csv_rows(Path(structure_evidence_path).expanduser())
        structure_rows_by_variant: Dict[str, Dict[str, str]] = {}
        if structure_csv:
            try:
                structure_rows, _structure_fields = _read_csv_rows(Path(structure_csv).expanduser())
                structure_rows_by_variant = {
                    _variant_id(row): row for row in structure_rows
                }
            except Exception:
                structure_rows_by_variant = {}
        job_manifest = _read_json(project / "04_structure_prediction" / "structure_prediction_job_manifest.json")
        required_pae = _required_pae_from_structure_job_manifest(job_manifest)
        if any(
            _is_true_marker(
                row.get("structure_prediction_required_pae")
                or row.get("required_pae")
            )
            for row in evidence_rows
        ):
            required_pae = True
        valid_count = 0
        missing_pae_count = 0
        failed_count = 0
        for row in evidence_rows:
            structure_row = structure_rows_by_variant.get(_variant_id(row), {})
            structure_path_text = (
                row.get("structure_prediction_structure_path")
                or row.get("structure_path")
                or structure_row.get("structure_path")
                or ""
            )
            structure_path = _resolve_project_output_path(str(structure_path_text), project_name)
            mapped = {
                "structure_path": str(structure_path),
                "prediction_status": row.get("structure_prediction_status") or row.get("prediction_status"),
                "structure_prediction_complete": row.get("structure_prediction_complete"),
                "required_pae": (
                    row.get("structure_prediction_required_pae")
                    or row.get("required_pae")
                    or structure_row.get("required_pae")
                ),
                "pae_matrix_usable": (
                    row.get("structure_prediction_pae_matrix_usable")
                    or row.get("pae_matrix_usable")
                    or structure_row.get("pae_matrix_usable")
                ),
            }
            row_valid = _is_structure_prediction_row_valid(
                mapped,
                required_pae_override=required_pae,
            )
            if row_valid:
                valid_count += 1
            else:
                failed_count += 1
                pae_usable = _is_true_marker(mapped.get("pae_matrix_usable"))
                if required_pae and not pae_usable:
                    missing_pae_count += 1
        structure_evidence_gate = {
            "evidence_csv": structure_evidence_path,
            "row_count": len(evidence_rows),
            "valid_structure_count": valid_count,
            "failed_count": failed_count,
            "missing_required_pae_count": missing_pae_count,
            "required_pae": required_pae,
        }
        if evidence_rows and valid_count == 0:
            structure_evidence_blocked = (
                "structure prediction evidence",
                structure_evidence_path,
            )

    empty_artifact: Optional[Tuple[str, str]] = None
    for label, path_text in [
        ("candidate table", candidate_csv),
        ("structure prediction results", structure_csv),
        ("structure prediction evidence", files["structure_prediction_evidence_csv"]),
        ("structure confidence evidence", files["structure_confidence_evidence_csv"]),
        (
            "structure-confidence candidates",
            post_structure_candidate_csv
            if files["structure_confidence_evidence_csv"]
            else "",
        ),
        ("inverse-folding evidence", files["inverse_folding_evidence_csv"]),
        ("BioEmu candidates", files["bioemu_candidate_csv"]),
        ("BioEmu evidence", files["bioemu_evidence_csv"]),
        ("Rosetta candidates", files["rosetta_candidate_csv"]),
        ("Rosetta evidence", files["rosetta_evidence_csv"]),
        ("ranked evidence", files["switchtune_evidence_ranked_csv"]),
        ("final recommendation", files["final_candidate_recommendation_csv"]),
    ]:
        if path_text and _csv_data_row_count(path_text) == 0:
            empty_artifact = (label, path_text)
            break

    recommendation: Dict[str, Any]
    if structure_evidence_blocked is not None:
        label, path_text = structure_evidence_blocked
        recommendation = {
            "next_tool": "none",
            "status": "blocked_no_candidates",
            "candidate_count": 0,
            "reason": (
                "Structure prediction evidence contains no valid structure that satisfies the recorded "
                "completion and strict PAE requirements. Do not standardize or run downstream scoring."
            ),
            "blocked_artifact": path_text,
            "required_inputs": [],
        }
    elif empty_artifact is not None:
        label, path_text = empty_artifact
        recommendation = {
            "next_tool": "none",
            "status": "blocked_no_candidates",
            "candidate_count": 0,
            "reason": (
                f"The {label} exists but contains zero data rows. "
                "Stop instead of preparing or running downstream tools; inspect the corresponding "
                "evidence, rejection, or external-job logs."
            ),
            "blocked_artifact": path_text,
            "required_inputs": [],
        }
    elif structure_csv and not candidate_csv:
        recommendation = {
            "next_tool": "none",
            "reason": (
                "Structure-prediction outputs exist, but no candidate sequence table could be found. "
                "Provide the retained candidates CSV or restore 04_structure_prediction/input/candidate_sequence_map.json."
            ),
            "required_inputs": ["candidates_csv"],
        }
    elif not files["library_csv"] and not structure_csv:
        recommendation = {
            "next_tool": "generate_saturation_library",
            "reason": "No saturation library is registered in the project manifest.",
            "required_inputs": ["sequence", "mutation_region"],
        }
    elif not files["cb_scored_library_csv"] and not structure_csv:
        recommendation = {
            "next_tool": "run_conformation_bias_scoring",
            "reason": "The saturation library exists but no CB-scored library has been collected.",
            "required_inputs": ["library_csv", "target_state_structure", "reference_state_structure"],
            "suggested_arguments": {"library_csv": files["library_csv"]},
        }
    elif not candidate_csv:
        recommendation = {
            "next_tool": "select_state_specific_candidates_adaptive",
            "reason": "CB scores exist but no adaptive moderate-bias candidate table is registered.",
            "suggested_arguments": {
                "scored_library_csv": files["cb_scored_library_csv"],
                "target_count": int(desired_structure_candidate_count),
            },
        }
    elif not structure_csv:
        recommendation = {
            "next_tool": "prepare_structure_prediction_job",
            "reason": "State-specific candidates exist but predicted structures have not been collected.",
            "external_execution_required_after_tool": True,
            "suggested_arguments": {
                "candidates_csv": candidate_csv,
                "engine": "auto",
                "target_state_structure": resolved_target,
                "reference_state_structure": resolved_reference,
                "top_n": int(desired_structure_candidate_count),
                "template_mode": "target_state",
            },
        }
    elif not files["structure_prediction_evidence_csv"]:
        recommendation = {
            "next_tool": "collect_structure_prediction_results",
            "reason": "Predicted structures exist but structure-prediction evidence has not been extracted.",
            "suggested_arguments": {
                "prediction_dir": str(project / "04_structure_prediction" / "predictions"),
            },
        }
    elif not files["structure_prediction_pdb_results_csv"]:
        recommendation = {
            "next_tool": "standardize_structure_prediction_results",
            "reason": "Structure predictions were collected but standardized PDBs are missing for IF/Rosetta.",
            "suggested_arguments": {"structures_csv": files["structure_prediction_results_csv"]},
        }
    elif not files["structure_state_similarity_evidence_csv"] and resolved_target and resolved_reference:
        recommendation = {
            "next_tool": "compare_structure_predictions_to_states",
            "reason": "Standardized structures exist but target/reference conformational similarity evidence is missing.",
            "suggested_arguments": {
                "structures_csv": files["structure_prediction_pdb_results_csv"],
                "target_state_structure": resolved_target,
                "reference_state_structure": resolved_reference,
                "regions": structure_similarity_regions,
            },
        }
    elif not files["inverse_folding_evidence_csv"] and not files["structure_confidence_evidence_csv"]:
        recommendation = {
            "next_tool": "filter_structure_confidence",
            "reason": (
                "Structure predictions and optional state-similarity evidence are ready; "
                "filter by Boltz confidence, local pLDDT, PAE, and target-like state before IF scoring."
            ),
            "suggested_arguments": {
                "candidates_csv": candidate_csv,
                "structures_csv": files["structure_prediction_pdb_results_csv"] or files["structure_prediction_results_csv"],
                "structure_prediction_evidence_csv": files["structure_prediction_evidence_csv"],
                "structure_state_similarity_evidence_csv": files["structure_state_similarity_evidence_csv"],
                "require_target_like": bool(files["structure_state_similarity_evidence_csv"]),
                "require_pae": True,
            },
        }
    elif not files["inverse_folding_evidence_csv"]:
        recommendation = {
            "next_tool": "prepare_inverse_folding_scoring_job",
            "reason": "Structure confidence QC is complete; IF compatibility scoring is next.",
            "external_execution_required_after_tool": True,
            "suggested_arguments": {
                "structures_csv": files["structure_prediction_pdb_results_csv"],
                "candidates_csv": post_structure_candidate_csv,
            },
        }
    elif not files["bioemu_evidence_csv"] and not files["bioemu_candidate_csv"]:
        recommendation = {
            "next_tool": "select_candidates_from_evidence",
            "reason": "Inverse-folding evidence exists; select the best post-IF candidates before BioEmu sampling.",
            "suggested_arguments": {
                "stage_name": "bioemu",
                "candidate_csv": post_structure_candidate_csv,
                "cb_scored_csv": files["cb_scored_library_csv"],
                "structure_prediction_evidence_csv": files["structure_prediction_evidence_csv"],
                "structure_state_similarity_evidence_csv": files["structure_state_similarity_evidence_csv"],
                "structure_confidence_evidence_csv": files["structure_confidence_evidence_csv"],
                "inverse_folding_evidence_csv": files["inverse_folding_evidence_csv"],
                "top_n": int(desired_bioemu_candidate_count),
                "metric_columns": "bias_score_abs,structure_confidence_score,structure_target_similarity_score,inverse_folding_compatibility_score",
            },
        }
    elif not files["bioemu_evidence_csv"]:
        recommendation = {
            "next_tool": "prepare_bioemu_sampling_job",
            "reason": "Post-IF candidate selection exists; BioEmu sampling should be prepared for top dynamic candidates.",
            "external_execution_required_after_tool": True,
            "suggested_arguments": {
                "candidates_csv": files["bioemu_candidate_csv"],
                "top_n": int(desired_bioemu_candidate_count),
                "num_samples": 2000,
                "batch_size": 10,
            },
        }
    elif not files["rosetta_evidence_csv"] and not files["rosetta_candidate_csv"]:
        recommendation = {
            "next_tool": "select_candidates_from_evidence",
            "reason": "BioEmu evidence exists; select the candidates that should enter Rosetta energy evaluation.",
            "suggested_arguments": {
                "stage_name": "rosetta",
                "candidate_csv": files["bioemu_candidate_csv"] or post_structure_candidate_csv,
                "cb_scored_csv": files["cb_scored_library_csv"],
                "structure_prediction_evidence_csv": files["structure_prediction_evidence_csv"],
                "structure_state_similarity_evidence_csv": files["structure_state_similarity_evidence_csv"],
                "structure_confidence_evidence_csv": files["structure_confidence_evidence_csv"],
                "inverse_folding_evidence_csv": files["inverse_folding_evidence_csv"],
                "bioemu_evidence_csv": files["bioemu_evidence_csv"],
                "top_n": int(desired_rosetta_candidate_count),
            },
        }
    elif (
        not files["rosetta_evidence_csv"]
        and not manifest.get("rosetta_strategy_name")
        and manifest.get("design_objective")
    ):
        recommendation = {
            "next_tool": "select_rosetta_analysis_strategy",
            "reason": (
                "Rosetta-stage candidates are ready; select a protein-specific Rosetta "
                "analysis strategy from the design objective before preparing the job."
            ),
            "suggested_arguments": {
                "project_name": project_name,
                "candidate_csv": files["rosetta_candidate_csv"],
                "structure_prediction_evidence_csv": files["structure_prediction_evidence_csv"],
                "bioemu_evidence_csv": files["bioemu_evidence_csv"],
                "target_state_structure": resolved_target,
                "reference_state_structure": resolved_reference,
                "requested_focus": "auto",
                "requested_mode": "auto",
            },
        }
    elif not files["rosetta_evidence_csv"]:
        recommendation = {
            "next_tool": "prepare_rosetta_energy_job",
            "reason": "Rosetta-stage candidate selection exists; prepare the protein-specific Rosetta energy evaluation.",
            "external_execution_required_after_tool": True,
            "suggested_arguments": {
                "structures_csv": files["structure_prediction_pdb_results_csv"],
                "candidates_csv": files["rosetta_candidate_csv"],
                "mode": str(manifest.get("rosetta_mode") or "score"),
                "nstruct": 1,
            },
            "decision_note": (
                "Use the recorded Rosetta strategy and do not replace it with arbitrary shell arguments. "
                "If strategy selection fell back to score, inspect its warning and limitation fields."
            ),
        }
    elif not files["switchtune_evidence_ranked_csv"]:
        recommendation = {
            "next_tool": "merge_switchtune_evidence_tables",
            "reason": "All major evidence sources are present and ready for consensus ranking.",
            "suggested_arguments": {
                "candidate_csv": files["rosetta_candidate_csv"] or files["bioemu_candidate_csv"] or post_structure_candidate_csv,
                "top_n": int(max(desired_rosetta_candidate_count, 20)),
            },
        }
    elif not files["final_candidate_recommendation_csv"]:
        recommendation = {
            "next_tool": "prepare_final_candidate_recommendation",
            "reason": "Consensus ranking exists but the final experimental recommendation table is missing.",
            "suggested_arguments": {
                "evidence_csv": files["switchtune_evidence_ranked_csv"],
                "top_n": int(desired_rosetta_candidate_count),
            },
        }
    else:
        recommendation = {
            "next_tool": "none",
            "reason": "The project already has a final candidate recommendation table.",
            "suggested_arguments": {},
        }

    controller_status = (
        "blocked_no_candidates"
        if recommendation.get("status") == "blocked_no_candidates"
        else "completed"
    )
    return {
        "status": controller_status,
        "project_name": project_name,
        "project_dir": str(project),
        "manifest_stage": manifest.get("stage", ""),
        "known_files": files,
        "structure_evidence_gate": structure_evidence_gate,
        "candidate_count": (
            0
            if controller_status == "blocked_no_candidates"
            else int(_csv_data_row_count(candidate_csv) or 0)
            if candidate_csv
            else 0
        ),
        "recommendation": recommendation,
        "agent_instruction": (
            "Call only the recommended next_tool once. If external_execution_required_after_tool is true, "
            "return a final <solution> after preparing the job and do not run the heavy job inside agent.go()."
        ),
        "updated_at": _utc_now(),
    }


@mcp.tool()
def get_job_status(project_name: str, job_id: str) -> Dict[str, Any]:
    """Read a SwitchTune job status record."""
    path = _project_dir(project_name) / "jobs" / f"{job_id}.json"
    if not path.exists():
        return {"status": "not_found", "job_id": job_id, "job_record_path": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@mcp.tool()
def get_project_summary(project_name: str) -> Dict[str, Any]:
    """Summarize known outputs for a SwitchTune project."""
    project = _project_dir(project_name)
    files = []
    for path in sorted(project.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(project)))
    manifest_path = project / "project_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    return {
        "status": "completed",
        "project_name": project_name,
        "project_dir": str(project),
        "manifest": manifest,
        "file_count": len(files),
        "files": files,
    }


if __name__ == "__main__":
    mcp.run()
