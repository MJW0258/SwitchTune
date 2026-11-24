#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import numpy as np

STANDARD_RESNAMES = {
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


CoordMap = Dict[int, Tuple[float, float, float]]
ResidueRecord = Tuple[int, str, Tuple[float, float, float]]
Region = Optional[Union[Tuple[int, int], FrozenSet[int]]]

AA3_TO1 = {
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


def parse_region(text: str) -> Tuple[str, Region]:
    """Parse a single position, interval, or exact union of positions/intervals."""
    text = str(text or "").strip()
    if not text or text.lower() == "global":
        return "global", None
    if ":" in text:
        name, span = text.split(":", 1)
    else:
        span = text
        name = f"region_{span.replace('-', '_').replace(',', '_')}"
    intervals: List[Tuple[int, int]] = []
    for token in span.split(","):
        match = re.fullmatch(r"\s*(-?\d+)\s*(?:-\s*(-?\d+))?\s*", token)
        if match is None:
            raise ValueError(
                f"Invalid structure comparison region {text!r}: {token!r}. "
                "Use global or name:10,12-47,49-60."
            )
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        intervals.append((min(start, end), max(start, end)))
    if len(intervals) == 1:
        region: Region = intervals[0]
    else:
        region = frozenset(
            position for lo, hi in intervals for position in range(lo, hi + 1)
        )
    return name.strip() or f"region_{span.replace('-', '_').replace(',', '_')}", region


def region_contains(region: Region, position: int) -> bool:
    if region is None:
        return True
    if isinstance(region, frozenset):
        return position in region
    lo, hi = region
    return lo <= position <= hi


def parse_pdb_ca(path: Path, chain_id: Optional[str]) -> List[ResidueRecord]:
    records: List[ResidueRecord] = []
    seen: set[Tuple[str, int, str]] = set()
    chain_filter = str(chain_id or "").strip()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip() != "ATOM":
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resname = line[17:20].strip().upper()
            if resname not in STANDARD_RESNAMES:
                continue
            chain = line[21:22].strip()
            if chain_filter and chain != chain_filter:
                continue
            try:
                resseq = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            records.append((resseq, AA3_TO1.get(resname, "X"), xyz))
    return records


def _cif_tokens(line: str) -> List[str]:
    return shlex.split(line, posix=True)


def _idx(columns: List[str], *names: str) -> Optional[int]:
    by_name = {name: idx for idx, name in enumerate(columns)}
    for name in names:
        if name in by_name:
            return by_name[name]
    return None


def parse_cif_ca(path: Path, chain_id: Optional[str]) -> List[ResidueRecord]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: List[ResidueRecord] = []
    seen: set[Tuple[str, int]] = set()
    chain_filter = str(chain_id or "").strip()
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

        group_i = _idx(columns, "_atom_site.group_PDB")
        atom_i = _idx(columns, "_atom_site.label_atom_id", "_atom_site.auth_atom_id")
        comp_i = _idx(columns, "_atom_site.label_comp_id", "_atom_site.auth_comp_id")
        seq_i = _idx(columns, "_atom_site.label_seq_id", "_atom_site.auth_seq_id")
        chain_i = _idx(columns, "_atom_site.label_asym_id", "_atom_site.auth_asym_id")
        x_i = _idx(columns, "_atom_site.Cartn_x")
        y_i = _idx(columns, "_atom_site.Cartn_y")
        z_i = _idx(columns, "_atom_site.Cartn_z")

        required = [atom_i, comp_i, seq_i, chain_i, x_i, y_i, z_i]
        if any(index is None for index in required):
            raise ValueError(f"Missing required _atom_site columns in {path}")

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
            if group_i is not None and tokens[group_i] not in {
                "ATOM",
                "ATOM_site",
                "ATOM_SITE",
            }:
                i += 1
                continue
            atom = tokens[atom_i].strip()
            comp = tokens[comp_i].strip().upper()
            chain = tokens[chain_i].strip()
            if atom != "CA" or comp not in STANDARD_RESNAMES:
                i += 1
                continue
            if chain_filter and chain != chain_filter:
                i += 1
                continue
            try:
                seq = int(float(tokens[seq_i]))
                xyz = (float(tokens[x_i]), float(tokens[y_i]), float(tokens[z_i]))
            except ValueError:
                i += 1
                continue
            key = (chain, seq)
            if key in seen:
                i += 1
                continue
            seen.add(key)
            records.append((seq, AA3_TO1.get(comp, "X"), xyz))
            i += 1
    return records


def parse_structure_ca(path: Path, chain_id: Optional[str]) -> List[ResidueRecord]:
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return parse_cif_ca(path, chain_id)
    return parse_pdb_ca(path, chain_id)


def subset_keys(keys: Iterable[int], region: Region) -> List[int]:
    selected = sorted(keys)
    return [key for key in selected if region_contains(region, key)]


def kabsch_rmsd(mobile_xyz: np.ndarray, reference_xyz: np.ndarray) -> float:
    if mobile_xyz.shape != reference_xyz.shape:
        raise ValueError("mobile/reference coordinate arrays must have the same shape")
    if mobile_xyz.shape[0] == 0:
        return float("nan")
    mobile_center = mobile_xyz.mean(axis=0)
    reference_center = reference_xyz.mean(axis=0)
    mobile = mobile_xyz - mobile_center
    reference = reference_xyz - reference_center
    covariance = mobile.T @ reference
    u, _s, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = mobile @ rotation
    diff = aligned - reference
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def records_to_number_map(records: List[ResidueRecord]) -> CoordMap:
    return {seq_id: xyz for seq_id, _aa, xyz in records}


def records_to_sequence(records: List[ResidueRecord]) -> str:
    return "".join(aa for _seq_id, aa, _xyz in records)


def global_align_indices(seq_a: str, seq_b: str) -> List[Tuple[int, int]]:
    """Needleman-Wunsch alignment returning 0-based matched residue indices.

    Mismatches are allowed because the query contains point mutations.
    Equal-scoring paths prefer less residue-index displacement. This keeps
    a truncated state sequence in register instead of matching a repeated
    terminal residue to an unrelated residue in the unresolved tail.
    """
    n, m = len(seq_a), len(seq_b)
    gap = -2
    match = 2
    mismatch = -1
    score = [[0] * (m + 1) for _ in range(n + 1)]
    displacement = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
        trace[i][0] = "up"
    for j in range(1, m + 1):
        score[0][j] = j * gap
        trace[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (
                match if seq_a[i - 1] == seq_b[j - 1] else mismatch
            )
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            options = [
                (diag, -(displacement[i - 1][j - 1] + abs(i - j)), 2, "diag"),
                (up, -displacement[i - 1][j], 1, "up"),
                (left, -displacement[i][j - 1], 0, "left"),
            ]
            best_score, negative_displacement, _priority, step = max(options)
            score[i][j] = best_score
            displacement[i][j] = -negative_displacement
            trace[i][j] = step

    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = trace[i][j]
        if step == "diag":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == "up":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def aligned_xyz_by_sequence(
    query: List[ResidueRecord],
    state: List[ResidueRecord],
    region: Region,
) -> Tuple[np.ndarray, np.ndarray, int]:
    query_seq = records_to_sequence(query)
    state_seq = records_to_sequence(state)
    pairs = global_align_indices(query_seq, state_seq)
    selected: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    for query_i, state_i in pairs:
        state_resnum = state[state_i][0]
        if not region_contains(region, state_resnum):
            continue
        selected.append((query[query_i][2], state[state_i][2]))
    if not selected:
        return np.empty((0, 3)), np.empty((0, 3)), 0
    query_xyz = np.array([item[0] for item in selected], dtype=float)
    state_xyz = np.array([item[1] for item in selected], dtype=float)
    return query_xyz, state_xyz, len(selected)


def compare_coords(
    query: List[ResidueRecord],
    state: List[ResidueRecord],
    region: Region,
    mapping: str,
) -> Tuple[float, int, int, int]:
    if mapping == "residue_number":
        query_map = records_to_number_map(query)
        state_map = records_to_number_map(state)
        common = subset_keys(set(query_map).intersection(state_map), region)
        if len(common) < 3:
            return float("nan"), len(common), len(query), len(state)
        query_xyz = np.array([query_map[key] for key in common], dtype=float)
        state_xyz = np.array([state_map[key] for key in common], dtype=float)
        aligned_count = len(common)
    else:
        query_xyz, state_xyz, aligned_count = aligned_xyz_by_sequence(
            query, state, region
        )
        if aligned_count < 3:
            return float("nan"), aligned_count, len(query), len(state)
    return kabsch_rmsd(query_xyz, state_xyz), aligned_count, len(query), len(state)


def compare_state_pair(
    query: List[ResidueRecord],
    target: List[ResidueRecord],
    reference: List[ResidueRecord],
    region: Region,
    mapping: str,
) -> Tuple[float, float, List[int]]:
    """Compare both states on exactly the same homologous prediction residues.

    Regions use target-state numbering. Compose the WT state alignment with
    the prediction-to-target alignment so a missing state tail cannot change
    the residue set used for one side of the RMSD difference.
    """
    if mapping == "residue_number":
        target_by_number = {record[0]: i for i, record in enumerate(target)}
        reference_by_number = {record[0]: i for i, record in enumerate(reference)}
        query_to_target = {
            i: target_by_number[record[0]]
            for i, record in enumerate(query)
            if record[0] in target_by_number
        }
        target_to_reference = {
            i: reference_by_number[record[0]]
            for i, record in enumerate(target)
            if record[0] in reference_by_number
        }
    else:
        query_to_target = dict(
            global_align_indices(
                records_to_sequence(query), records_to_sequence(target)
            )
        )
        target_to_reference = dict(
            global_align_indices(
                records_to_sequence(target), records_to_sequence(reference)
            )
        )
    triples = [
        (query_i, target_i, target_to_reference[target_i])
        for query_i, target_i in query_to_target.items()
        if target_i in target_to_reference
        and region_contains(region, target[target_i][0])
    ]
    positions = [query[query_i][0] for query_i, _, _ in triples]
    if len(triples) < 3:
        return float("nan"), float("nan"), positions
    query_xyz = np.array([query[i][2] for i, _, _ in triples], dtype=float)
    target_xyz = np.array([target[j][2] for _, j, _ in triples], dtype=float)
    reference_xyz = np.array([reference[k][2] for _, _, k in triples], dtype=float)
    return (
        kabsch_rmsd(query_xyz, target_xyz),
        kabsch_rmsd(query_xyz, reference_xyz),
        positions,
    )


def read_structure_table(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Boltz/AF predicted structures to target and reference conformational states using CA RMSD."
    )
    parser.add_argument(
        "--structures-csv", required=True, help="structure_prediction_results.csv"
    )
    parser.add_argument("--target", required=True, help="Target-state PDB/mmCIF path")
    parser.add_argument(
        "--reference", required=True, help="Reference-state PDB/mmCIF path"
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--prediction-chain", default="A", help="Predicted structure chain ID"
    )
    parser.add_argument("--target-chain", default="A", help="Target structure chain ID")
    parser.add_argument(
        "--reference-chain", default="A", help="Reference structure chain ID"
    )
    parser.add_argument(
        "--mapping",
        default="sequence_alignment",
        choices=["sequence_alignment", "residue_number"],
        help="How to map predicted residues to state residues before CA superposition.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=None,
        help="Region to score, e.g. global, site:10 or switch_region:10,12-47,49-60. Can be repeated.",
    )
    args = parser.parse_args()

    structures_csv = Path(args.structures_csv).expanduser()
    target_path = Path(args.target).expanduser()
    reference_path = Path(args.reference).expanduser()

    target_ca = parse_structure_ca(target_path, args.target_chain)
    reference_ca = parse_structure_ca(reference_path, args.reference_chain)
    if len(target_ca) < 3:
        raise RuntimeError(
            f"Target CA atoms not enough: {target_path} chain {args.target_chain}"
        )
    if len(reference_ca) < 3:
        raise RuntimeError(
            f"Reference CA atoms not enough: {reference_path} chain {args.reference_chain}"
        )

    region_inputs = args.region or ["global", "switch_87_110:87-110"]
    regions = [parse_region(item) for item in region_inputs]
    rows: List[Dict[str, object]] = []
    for record in read_structure_table(structures_csv):
        variant = record.get("variant") or Path(record.get("structure_path", "")).stem
        structure_path = Path(record["structure_path"]).expanduser()
        pred_ca = parse_structure_ca(structure_path, args.prediction_chain)
        for region_name, region_span in regions:
            raw_target, raw_n_target, pred_ca_count, target_ca_count = compare_coords(
                pred_ca, target_ca, region_span, args.mapping
            )
            raw_ref, raw_n_ref, _pred_ca_count, ref_ca_count = compare_coords(
                pred_ca, reference_ca, region_span, args.mapping
            )
            rmsd_target, rmsd_ref, shared_positions = compare_state_pair(
                pred_ca, target_ca, reference_ca, region_span, args.mapping
            )
            delta_ref_minus_target = (
                rmsd_ref - rmsd_target
                if math.isfinite(rmsd_ref) and math.isfinite(rmsd_target)
                else float("nan")
            )
            if not math.isfinite(delta_ref_minus_target):
                closer_state = "undetermined"
            elif delta_ref_minus_target > 0:
                closer_state = "target"
            elif delta_ref_minus_target < 0:
                closer_state = "reference"
            else:
                closer_state = "tie"
            rows.append(
                {
                    "variant": variant,
                    "region": region_name,
                    "structure_path": str(structure_path),
                    "rmsd_to_target_A": fmt(rmsd_target),
                    "rmsd_to_reference_A": fmt(rmsd_ref),
                    "delta_reference_minus_target_A": fmt(delta_ref_minus_target),
                    "closer_state": closer_state,
                    "aligned_ca_to_target": len(shared_positions),
                    "aligned_ca_to_reference": len(shared_positions),
                    "comparison_residue_policy": "shared_across_prediction_target_reference",
                    "compared_prediction_residue_numbers": ",".join(
                        map(str, shared_positions)
                    ),
                    "pairwise_rmsd_to_target_A": fmt(raw_target),
                    "pairwise_rmsd_to_reference_A": fmt(raw_ref),
                    "pairwise_aligned_ca_to_target": raw_n_target,
                    "pairwise_aligned_ca_to_reference": raw_n_ref,
                    "prediction_ca_count": pred_ca_count,
                    "target_ca_count": target_ca_count,
                    "reference_ca_count": ref_ca_count,
                }
            )

    output_csv = Path(args.output_csv).expanduser()
    write_csv(output_csv, rows)

    print(f"Wrote: {output_csv}")
    print()
    print("Variant state similarity summary:")
    for row in rows:
        print(
            f"{row['variant']:>10s} {row['region']:<16s} "
            f"target={row['rmsd_to_target_A']} A  "
            f"reference={row['rmsd_to_reference_A']} A  "
            f"delta(ref-target)={row['delta_reference_minus_target_A']} A  "
            f"closer={row['closer_state']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
