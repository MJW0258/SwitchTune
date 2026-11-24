"""AlphaFold 3 input and output adapters for the official local implementation.

No AF3 code or weights are imported. Native outputs remain available alongside
the normalized response consumed by SwitchTune's existing confidence filters.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import io
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return str(path)


def _parser(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    return (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    ).get_structure("model", str(path))


def protein_template(path: Path, query: str, chain_id: str = "auto") -> dict:
    """Extract one resolved protein chain and align its coordinates to the query.

    The generated mmCIF has a contiguous entity sequence; original missing
    residues are represented by gaps in queryIndices, never guessed indices.
    """
    from Bio.Align import PairwiseAligner
    from Bio.PDB import MMCIFIO, Chain, Model, Structure
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.SeqUtils import seq1

    aligner = PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-5,
        extend_gap_score=-0.1,
    )
    candidates = []
    for chain in next(_parser(path).get_models()):
        if chain_id not in {"", "auto", "best"} and chain.id != chain_id:
            continue
        residues = [
            r
            for r in chain
            if r.id[0] == " "
            and seq1(r.resname) in "ACDEFGHIKLMNPQRSTVWY"
            and "CA" in r
        ]
        sequence = "".join(seq1(r.resname) for r in residues)
        if sequence:
            alignment = aligner.align(query, sequence)[0]
            candidates.append((alignment.score, sequence, residues, alignment))
    if not candidates:
        raise ValueError(
            f"No protein coordinates for template chain {chain_id!r} in {path}"
        )
    _, sequence, residues, alignment = max(candidates, key=lambda item: item[0])
    query_indices, template_indices = [], []
    for qblock, tblock in zip(*alignment.aligned):
        query_indices.extend(range(int(qblock[0]), int(qblock[1])))
        template_indices.extend(range(int(tblock[0]), int(tblock[1])))
    identical = sum(
        query[q] == sequence[t] for q, t in zip(query_indices, template_indices)
    )
    if not query_indices or identical / len(query_indices) < 0.5:
        raise ValueError(
            "Target-state template does not align to the candidate sequence"
        )
    structure = Structure.Structure("template")
    model, chain = Model.Model(0), Chain.Chain("A")
    structure.add(model)
    model.add(chain)
    for index, residue in enumerate(residues, 1):
        residue = copy.deepcopy(residue)
        residue.detach_parent()
        residue.id = (" ", index, " ")
        chain.add(residue)
    writer = MMCIFIO()
    writer.set_structure(structure)
    buffer = io.StringIO()
    writer.save(buffer)
    data = MMCIF2Dict(io.StringIO(buffer.getvalue()))
    data.update(
        {
            "_entity.id": ["1"],
            "_entity.type": ["polymer"],
            "_entity_poly.entity_id": ["1"],
            "_entity_poly.type": ["polypeptide(L)"],
            "_entity_poly.pdbx_strand_id": ["A"],
            "_entity_poly.pdbx_seq_one_letter_code_can": [sequence],
            "_entity_poly_seq.entity_id": ["1"] * len(residues),
            "_entity_poly_seq.num": [str(i) for i in range(1, len(residues) + 1)],
            "_entity_poly_seq.mon_id": [r.resname for r in residues],
            "_entity_poly_seq.hetero": ["n"] * len(residues),
            "_struct_asym.id": ["A"],
            "_struct_asym.entity_id": ["1"],
            "_atom_site.label_entity_id": ["1"] * len(data["_atom_site.id"]),
            # AF3 requires a release date for explicit templates. This is the
            # creation date of this derived coordinate-only template, not a claimed
            # PDB deposition date.
            "_pdbx_audit_revision_history.revision_date": [
                datetime.date.today().isoformat()
            ],
            "_pdbx_poly_seq_scheme.asym_id": ["A"] * len(residues),
            "_pdbx_poly_seq_scheme.entity_id": ["1"] * len(residues),
            "_pdbx_poly_seq_scheme.seq_id": [
                str(i) for i in range(1, len(residues) + 1)
            ],
            "_pdbx_poly_seq_scheme.mon_id": [r.resname for r in residues],
            "_pdbx_poly_seq_scheme.pdb_seq_num": [
                str(i) for i in range(1, len(residues) + 1)
            ],
            "_pdbx_poly_seq_scheme.pdb_strand_id": ["A"] * len(residues),
            "_pdbx_poly_seq_scheme.pdb_ins_code": ["."] * len(residues),
        }
    )
    writer.set_dict(data)
    buffer = io.StringIO()
    writer.save(buffer)
    return {
        "mmcif": buffer.getvalue(),
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }


def _inline_files(data: dict, directory: Path) -> None:
    """Keep prepared inputs portable into a container by embedding file contents."""
    for key in list(data):
        if key in {"unpairedMsaPath", "pairedMsaPath", "mmcifPath", "userCCDPath"}:
            import gzip
            import lzma

            path = Path(data.pop(key)).expanduser()
            path = path if path.is_absolute() else directory / path
            if path.suffix == ".gz":
                content = gzip.decompress(path.read_bytes()).decode("utf-8")
            elif path.suffix == ".xz":
                content = lzma.decompress(path.read_bytes()).decode("utf-8")
            else:
                content = path.read_text(encoding="utf-8")
            data[key.removesuffix("Path")] = content
        elif isinstance(data[key], dict):
            _inline_files(data[key], directory)
        elif isinstance(data[key], list):
            for value in data[key]:
                if isinstance(value, dict):
                    _inline_files(value, directory)


def _mutate_msa_query(msa: str, sequence: str) -> str:
    lines = msa.splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError("Provided MSA must be A3M/FASTA with a query header")
    end = next(
        (i for i in range(1, len(lines)) if lines[i].startswith(">")), len(lines)
    )
    old = "".join(lines[1:end])
    if len(old) != len(sequence) or not old.isupper() or not old.isalpha():
        raise ValueError(
            "Provided MSA query must be an ungapped protein sequence of candidate length"
        )
    return "\n".join([lines[0], sequence, *lines[end:]]) + "\n"


def prepare_inputs(
    candidates: list[dict],
    input_dir: Path,
    prediction_dir: Path,
    settings: dict,
    *,
    target_template: str = "",
    template_chain_id: str = "auto",
    ligands: list[dict] | None = None,
) -> dict:
    config = copy.deepcopy(settings)
    base = {}
    if config.get("input_template"):
        path = Path(config["input_template"])
        base = json.loads(path.read_text(encoding="utf-8"))
        _inline_files(base, path.parent)
    chain_id = str(config.get("protein_chain_id") or "A")
    if not re.fullmatch(r"[A-Z]", chain_id):
        raise ValueError(
            "Use a single uppercase AF3 protein_chain_id for downstream PDB compatibility"
        )
    records, names = [], set()
    for candidate in candidates:
        variant, sequence = (
            str(candidate["variant"]),
            str(candidate["sequence"]).strip().upper(),
        )
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"Invalid protein sequence for {variant}")
        name = "st_" + re.sub(r"[^a-z0-9_]", "_", variant.lower())
        if name in names:
            raise ValueError(f"Duplicate AlphaFold3 job name: {name}")
        names.add(name)
        data = copy.deepcopy(base)
        data.update(name=name, dialect="alphafold3", modelSeeds=config["seeds"])
        data["version"] = max(2, int(base.get("version", 2)))
        entities = data.setdefault("sequences", [{"protein": {"id": chain_id}}])
        proteins = [
            e["protein"]
            for e in entities
            if "protein" in e and e["protein"].get("id") == chain_id
        ]
        if len(proteins) != 1:
            raise ValueError(
                f"Input template must have exactly one designed protein with id={chain_id}"
            )
        protein = proteins[0]
        old_sequence = protein.get("sequence", "")
        if old_sequence and len(old_sequence) != len(sequence):
            raise ValueError(
                "Input template and candidate sequences must have equal length (single substitutions)"
            )
        protein["sequence"] = sequence
        if config["msa_mode"] == "single_sequence":
            protein.update(unpairedMsa="", pairedMsa="")
        else:
            for key in ("unpairedMsa", "pairedMsa"):
                if protein.get(key):
                    protein[key] = _mutate_msa_query(protein[key], sequence)
        if target_template:
            protein["templates"] = [
                protein_template(Path(target_template), sequence, template_chain_id)
            ]
        elif config["msa_mode"] != "search":
            protein.setdefault("templates", [])
        if config["msa_mode"] != "search":
            for entity in entities:
                if "protein" in entity:
                    p = entity["protein"]
                    if (
                        not isinstance(p.get("unpairedMsa"), str)
                        or not isinstance(p.get("pairedMsa"), str)
                        or not isinstance(p.get("templates"), list)
                    ):
                        raise ValueError(
                            "Inference-only AF3 needs explicit MSAs (possibly empty) and templates for every protein chain"
                        )
                elif "rna" in entity and not isinstance(
                    entity["rna"].get("unpairedMsa"), str
                ):
                    raise ValueError(
                        "Inference-only AF3 needs an explicit unpairedMsa for every RNA chain"
                    )
        # Prepared templates own their ligand entities/bonds; generic ligands are
        # appended only for sequence-only inputs to avoid duplicate cofactors.
        if not base:
            used_ids = {chain_id}
            for ligand in ligands or []:
                ligand_id = next(
                    (c for c in "LMNOPQRSTUVWXYZBCDEFGHIJK" if c not in used_ids), None
                )
                if ligand_id is None:
                    raise ValueError(
                        "Too many ligand entities for single-letter output chain IDs"
                    )
                used_ids.add(ligand_id)
                content = {"id": ligand_id}
                code = ligand.get("ccd") or ligand.get("ccdCode")
                if code:
                    content["ccdCodes"] = [str(code)]
                elif ligand.get("smiles"):
                    content["smiles"] = ligand["smiles"]
                else:
                    raise ValueError(
                        "AF3 generic ligands need ccd or smiles; use AF3_INPUT_TEMPLATE for complex chemistry"
                    )
                entities.append({"ligand": content})
        fingerprint = hashlib.sha256(
            json.dumps({"input": data, "settings": config}, sort_keys=True).encode()
        ).hexdigest()[:20]
        path = input_dir / f"{name}.json"
        write_json(path, data)
        records.append(
            {
                "variant": variant,
                "sequence": sequence,
                "job_name": name,
                "input_path": str(path),
                "fingerprint": fingerprint,
                "output_base": str(prediction_dir / name / fingerprint),
                "chain_id": chain_id,
                "has_template": bool(protein.get("templates")),
                "ligand_count": sum("ligand" in e for e in entities),
            }
        )
    return {
        **config,
        "backend": "alphafold3",
        "require_pae": True,
        "polymer_chain_id": chain_id,
        "input_records": records,
        "expected_sample_count": len(config["seeds"]) * int(config["num_samples"]),
    }


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a confidence value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite confidence value")
    return number


def read_sample(model_path: Path, record: dict) -> dict:
    stem = model_path.name.removesuffix("model.cif")
    confidences_path = model_path.with_name(stem + "confidences.json")
    summary_path = model_path.with_name(stem + "summary_confidences.json")
    full = json.loads(confidences_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    chain_id, expected = record["chain_id"], record["sequence"]
    from Bio.SeqUtils import seq1

    structure = _parser(model_path)
    residues = [r for r in structure[0][chain_id] if r.id[0] == " "]
    sequence = "".join(seq1(r.resname) for r in residues)
    if sequence != expected:
        raise ValueError(
            "Predicted designed-chain sequence does not match submitted candidate"
        )
    residue_plddts = [
        mean(_finite_number(a.bfactor) for a in r.get_atoms()) for r in residues
    ]
    if any(v < 0 or v > 100 for v in residue_plddts):
        raise ValueError("pLDDT is outside 0-100")
    atom_plddts = [_finite_number(x) for x in full["atom_plddts"]]
    if (
        len(atom_plddts) != len(list(structure.get_atoms()))
        or not atom_plddts
        or any(x < 0 or x > 100 for x in atom_plddts)
    ):
        raise ValueError("Missing or invalid per-atom pLDDT")
    token_chains, pae = full["token_chain_ids"], full["pae"]
    if len(pae) != len(token_chains) or any(
        not isinstance(row, list) or len(row) != len(token_chains) for row in pae
    ):
        raise ValueError("PAE dimensions do not match AF3 token_chain_ids")
    indices = [i for i, c in enumerate(token_chains) if c == chain_id]
    if len(indices) != len(expected):
        raise ValueError(
            "Designed-chain PAE token count does not match candidate length"
        )
    protein_pae = [[_finite_number(pae[i][j]) for j in indices] for i in indices]
    if any(v < 0 for row in protein_pae for v in row):
        raise ValueError("Negative PAE")
    ranking = _finite_number(summary["ranking_score"])
    ptm = _finite_number(summary["ptm"])
    iptm = summary.get("iptm")
    if iptm is not None:
        iptm = _finite_number(iptm)
    has_clash = summary.get("has_clash")
    if not isinstance(has_clash, bool):
        raise ValueError("AF3 summary is missing boolean has_clash")
    disorder = _finite_number(summary["fraction_disordered"])
    return {
        "structure_path": str(model_path),
        "native_confidences_json": str(confidences_path),
        "native_summary_json": str(summary_path),
        "ranking_score": ranking,
        "ptm": ptm,
        "iptm": iptm,
        "has_clash": has_clash,
        "fraction_disordered": disorder,
        "plddt_mean": mean(residue_plddts),
        "residue_plddts": residue_plddts,
        "pae_mean": mean(v for row in protein_pae for v in row),
        "pae": protein_pae,
    }


def collect_candidate(record: dict, config: dict, response_dir: Path) -> dict:
    """Validate every expected sample, then select a matched structure/metric pair."""
    root = Path(record["output_base"]) / record["job_name"]
    samples, errors = [], []
    for seed in config["seeds"]:
        for index in range(int(config["num_samples"])):
            directory = root / f"seed-{seed}_sample-{index}"
            paths = sorted(directory.glob("*_model.cif")) + sorted(
                directory.glob("model.cif")
            )
            try:
                if len(paths) != 1:
                    raise ValueError(
                        f"Expected one sample structure, found {len(paths)}"
                    )
                sample = read_sample(paths[0], record)
                sample.update(seed=seed, sample_index=index)
                samples.append(sample)
            except Exception as exc:
                # Third-party mmCIF parsers have their own exception classes.
                # A malformed candidate must not abort collection of the batch.
                errors.append(
                    f"seed={seed}, sample={index}: {type(exc).__name__}: {exc}"
                )
    row = {
        "variant": record["variant"],
        "structure_path": "",
        "status": "failed_incomplete_af3_outputs",
        "required_pae": True,
        "pae_matrix_usable": False,
        "structure_prediction_engine": "alphafold3",
        "structure_prediction_sample_count": len(samples),
        "structure_prediction_expected_sample_count": config["expected_sample_count"],
        "structure_prediction_has_template": record["has_template"],
        "structure_prediction_ligand_count": record["ligand_count"],
        "error": "; ".join(errors),
    }
    if errors or not samples:
        return row
    eligible = [s for s in samples if not s["has_clash"]]
    if not eligible:
        return {
            **row,
            "status": "failed_af3_clashes",
            "error": "Every sample has has_clash=true",
        }
    best = max(
        eligible, key=lambda s: (s["ranking_score"], -s["seed"], -s["sample_index"])
    )
    score = best["plddt_mean"] / 100.0
    response = {
        "engine": "alphafold3",
        "confidence_scores": [score],
        "ptm_scores": [best["ptm"]],
        "iptm_scores": [best["iptm"]] if best["iptm"] is not None else [],
        "complex_plddt_scores": [score],
        "pae": best["pae"],
        "pae_scope": "designed_protein_chain",
        "pae_chain_id": record["chain_id"],
        "confidence_definition": "mean designed-chain residue pLDDT / 100; not AF3 ranking_score",
        "selected_sample": {
            k: v for k, v in best.items() if k not in {"pae", "residue_plddts"}
        },
        "sample_metrics": [
            {k: v for k, v in s.items() if k not in {"pae", "residue_plddts"}}
            for s in samples
        ],
    }
    path = Path(
        write_json(response_dir / f"{record['job_name']}_response.json", response)
    )
    row.update(
        status="completed",
        structure_path=best["structure_path"],
        response_json=str(path),
        confidence_score=score,
        pae_matrix_usable=True,
        pae_matrix_shape=json.dumps([len(best["pae"])] * 2),
        structure_prediction_confidence_definition=response["confidence_definition"],
        structure_prediction_af3_ranking_score=best["ranking_score"],
        structure_prediction_af3_has_clash=best["has_clash"],
        structure_prediction_af3_fraction_disordered=best["fraction_disordered"],
        structure_prediction_af3_native_confidences_json=best[
            "native_confidences_json"
        ],
        structure_prediction_af3_native_summary_json=best["native_summary_json"],
        structure_prediction_plddt_sample_mean=mean(s["plddt_mean"] for s in samples),
        structure_prediction_plddt_sample_sd=pstdev(s["plddt_mean"] for s in samples),
        structure_prediction_pae_sample_mean=mean(s["pae_mean"] for s in samples),
        error="",
    )
    return row
