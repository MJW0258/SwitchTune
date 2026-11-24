# Test AlphaFold3 inputs, outputs and resume handling.
"""Adapter/integration tests with synthetic AF3 artifacts, without model inference."""
# ruff: noqa: E402 -- Add the repository root before importing the local package.

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from switchtune.alphafold3 import (
    collect_candidate,
    prepare_inputs,
    protein_template,
    write_json,
)
from switchtune.mcp_server import (
    collect_structure_prediction_results,
    compare_structure_predictions_to_states,
    filter_structure_confidence,
    prepare_inverse_folding_scoring_job,
    prepare_structure_prediction_job,
    select_structure_prediction_engine,
    standardize_structure_prediction_results,
)
from switchtune.structure_backend import af3_settings, configured_engine

spec = importlib.util.spec_from_file_location(
    "af3_runner", ROOT / "scripts/run_alphafold3_batch.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def pdb(sequence="ACD", chain="C", start=101, confidence=95):
    names = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "V": "VAL", "G": "GLY"}
    lines = []
    for i, aa in enumerate(sequence):
        for j, atom in enumerate(["N", "CA", "C", "O"]):
            serial = i * 4 + j + 1
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} {names[aa]:>3s} {chain}{start + i:4d}    {i * 3.8 + j * 0.4:8.3f}{j * 0.5:8.3f}{i * 0.2:8.3f}  1.00{confidence:6.2f}          {atom[0]:>2s}"
            )
    return "\n".join(lines) + "\nEND\n"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SWITCHTUNE_HOME", str(ROOT))
    monkeypatch.setenv("SWITCHTUNE_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv(
        "SWITCHTUNE_TOOL_REGISTRY", str(ROOT / "config/tool_registry.yaml")
    )
    monkeypatch.setenv("SWITCHTUNE_STRUCTURE_CONFIG", str(tmp_path / "selection.json"))
    monkeypatch.delenv("SWITCHTUNE_STRUCTURE_ENGINE", raising=False)
    for name in list(os.environ):
        if name.startswith("AF3_"):
            monkeypatch.delenv(name)
    target = tmp_path / "target.pdb"
    target.write_text(pdb(), encoding="utf-8")
    return target


def create_job(tmp_path, target, *, samples=2):
    os.environ["AF3_NUM_SAMPLES"] = str(samples)
    os.environ["AF3_MSA_MODE"] = "single_sequence"
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "variant,sequence,sequence_index_1based\nA101V,VCD,1\nD103E,ACE,3\n",
        encoding="utf-8",
    )
    result = prepare_structure_prediction_job(
        "test_af3",
        str(candidates),
        engine="alphafold3",
        target_state_structure=str(target),
    )
    manifest = json.loads(Path(result["job_manifest_path"]).read_text())
    return result, manifest


def sample(
    record,
    index,
    *,
    confidence=95,
    ranking=0.8,
    clash=False,
    wrong_sequence=False,
    extra_ligand=True,
    seed=1,
):
    root = (
        Path(record["output_base"]) / record["job_name"] / f"seed-{seed}_sample-{index}"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['job_name']}_seed-{seed}_sample-{index}_model.cif"
    source = root / "source.pdb"
    sequence = "GGG" if wrong_sequence else record["sequence"]
    source.write_text(pdb(sequence, "A", 1, confidence), encoding="utf-8")
    cif = protein_template(source, sequence)["mmcif"]
    path.write_text(cif, encoding="utf-8")
    source.unlink()
    prefix = path.name.removesuffix("model.cif")
    n = len(sequence) + int(extra_ligand)
    write_json(
        root / (prefix + "confidences.json"),
        {
            "atom_plddts": [confidence] * (len(sequence) * 4),
            "token_chain_ids": ["A"] * len(sequence) + (["L"] if extra_ligand else []),
            "pae": [
                [
                    1.5 if i < len(sequence) and j < len(sequence) else 20
                    for j in range(n)
                ]
                for i in range(n)
            ],
        },
    )
    write_json(
        root / (prefix + "summary_confidences.json"),
        {
            "ranking_score": ranking,
            "ptm": 0.9,
            "iptm": 0.8,
            "has_clash": clash,
            "fraction_disordered": 0.05,
        },
    )
    return path


def test_deployment_choice_before_dependencies(environment, tmp_path):
    selection = Path(os.environ["SWITCHTUNE_STRUCTURE_CONFIG"])
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/configure_structure_backend.py"),
            "--engine",
            "alphafold3",
            "--storage-root",
            str(tmp_path),
            "--output",
            str(selection),
            "--msa-mode",
            "single_sequence",
        ],
        check=True,
        capture_output=True,
    )
    assert configured_engine() == "alphafold3"
    result = select_structure_prediction_engine(
        "test",
        target_state_structure=str(environment),
        requested_engine="auto",
        ligand_codes="FMN",
    )
    assert result["recommended_engine"] == "alphafold3"
    with pytest.raises(ValueError, match="configured as"):
        select_structure_prediction_engine("test", requested_engine="boltz2_local")


def test_native_inputs_template_alignment_and_ligands(environment, tmp_path):
    result, manifest = create_job(tmp_path, environment)
    config = manifest["alphafold3"]
    assert config["expected_sample_count"] == 2
    assert "run_alphafold3_structure_prediction.sh" in result["run_script"]
    data = json.loads(Path(config["input_records"][0]["input_path"]).read_text())
    p = data["sequences"][0]["protein"]
    assert data["dialect"] == "alphafold3" and data["version"] == 2
    assert p["sequence"] == "VCD" and p["unpairedMsa"] == "" and p["pairedMsa"] == ""
    assert p["templates"][0]["queryIndices"] == [0, 1, 2]
    assert p["templates"][0]["templateIndices"] == [0, 1, 2]
    assert "_entity_poly_seq.mon_id" in p["templates"][0]["mmcif"]
    assert "_pdbx_audit_revision_history.revision_date" in p["templates"][0]["mmcif"]
    assert "_pdbx_poly_seq_scheme.asym_id" in p["templates"][0]["mmcif"]
    config = af3_settings({"msa_mode": "search"})
    config["msa_mode"] = "search"
    raw = prepare_inputs(
        [{"variant": "A1V", "sequence": "VCD"}],
        tmp_path / "in",
        tmp_path / "out",
        config,
        ligands=[{"ccd": "FMN"}, {"smiles": "CCO"}],
    )
    raw_data = json.loads(Path(raw["input_records"][0]["input_path"]).read_text())
    assert "unpairedMsa" not in raw_data["sequences"][0]["protein"]
    assert raw_data["sequences"][1]["ligand"] == {"id": "L", "ccdCodes": ["FMN"]}
    assert raw_data["sequences"][2]["ligand"]["smiles"] == "CCO"


def test_gapped_template_and_mmcif_input(environment, tmp_path):
    environment.write_text(pdb("ACD"), encoding="utf-8")
    t = protein_template(environment, "AVCD")
    assert t["queryIndices"] == [0, 2, 3] and t["templateIndices"] == [0, 1, 2]
    cif = tmp_path / "source.cif"
    cif.write_text(t["mmcif"], encoding="utf-8")
    assert protein_template(cif, "ACD")["queryIndices"] == [0, 1, 2]


def test_prepared_msa_and_covalent_cofactor_preserved(environment, tmp_path):
    base = tmp_path / "prepared.json"
    write_json(
        base,
        {
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "ACD",
                        "unpairedMsa": ">q\nACD\n>homolog\nACE\n",
                        "pairedMsa": "",
                        "templates": [],
                    }
                },
                {"ligand": {"id": "L", "ccdCodes": ["FMN"]}},
            ],
            "bondedAtomPairs": [[["A", 2, "SG"], ["L", 1, "C4A"]]],
        },
    )
    config = af3_settings()
    config.update(msa_mode="provided", input_template=str(base))
    prepared = prepare_inputs(
        [{"variant": "A1V", "sequence": "VCD"}],
        tmp_path / "in",
        tmp_path / "out",
        config,
    )
    data = json.loads(Path(prepared["input_records"][0]["input_path"]).read_text())
    assert data["sequences"][0]["protein"]["unpairedMsa"] == ">q\nVCD\n>homolog\nACE\n"
    assert data["bondedAtomPairs"] == [[["A", 2, "SG"], ["L", 1, "C4A"]]]
    assert len(data["sequences"]) == 2


def test_all_samples_selected_metrics_and_downstream_handoff(environment, tmp_path):
    result, manifest = create_job(tmp_path, environment)
    config = manifest["alphafold3"]
    for record in config["input_records"]:
        sample(record, 0, confidence=98, ranking=0.8)
        sample(record, 1, confidence=92, ranking=1.2)
    collected = collect_structure_prediction_results(
        "test_af3",
        result["prediction_dir"],
        job_manifest_path=result["job_manifest_path"],
    )
    assert (
        collected["valid_structure_count"] == 2 and collected["result_row_count"] == 2
    )
    with Path(collected["output_csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(float(r["structure_prediction_confidence_score"]) == 0.92 for r in rows)
    assert all("sample-1" in r["structure_path"] for r in rows)
    assert all(float(r["structure_prediction_plddt_sample_mean"]) == 95 for r in rows)
    response = json.loads(Path(rows[0]["prediction_response_json"]).read_text())
    assert len(response["pae"]) == 3 and response["pae"][0] == [1.5, 1.5, 1.5]
    assert (
        response["confidence_scores"] == [0.92]
        and response["selected_sample"]["ranking_score"] == 1.2
    )
    standardized = standardize_structure_prediction_results(
        "test_af3", collected["output_csv"], chain_id="A", require_pae=True
    )
    assert standardized["standardized_count"] == 2
    compare_structure_predictions_to_states(
        "test_af3",
        standardized["output_csv"],
        str(environment),
        str(environment),
        prediction_chain="A",
        target_chain="C",
        reference_chain="C",
        regions="global",
    )
    filtered = filter_structure_confidence(
        "test_af3",
        candidates_csv=manifest["candidates_csv"],
        structures_csv=standardized["output_csv"],
        structure_prediction_evidence_csv=collected["evidence_csv"],
        require_target_like=False,
        adaptive_pae_quantile=None,
    )
    assert filtered["retained_count"] == 2
    inverse = prepare_inverse_folding_scoring_job(
        "test_af3",
        structures_csv=standardized["output_csv"],
        candidates_csv=filtered["output_csv"],
        chain_id="A",
        execute=False,
    )
    assert inverse["candidate_count"] == 2
    # Rescanning does not need a summary CSV or any AF3 runtime installation.
    Path(config["summary_csv"]).unlink()
    assert (
        collect_structure_prediction_results(
            "test_af3",
            result["prediction_dir"],
            job_manifest_path=result["job_manifest_path"],
        )["valid_structure_count"]
        == 2
    )


@pytest.mark.parametrize(
    "failure", ["missing_sample", "bad_sequence", "bad_pae", "all_clashes", "nonfinite"]
)
def test_incomplete_or_invalid_outputs_never_pass(environment, tmp_path, failure):
    result, manifest = create_job(tmp_path, environment)
    config = manifest["alphafold3"]
    record = config["input_records"][0]
    sample(record, 0, clash=failure == "all_clashes")
    if failure != "missing_sample":
        second = sample(
            record,
            1,
            clash=failure == "all_clashes",
            wrong_sequence=failure == "bad_sequence",
        )
        if failure in {"bad_pae", "nonfinite"}:
            full_path = second.with_name(
                second.name.replace("model.cif", "confidences.json")
            )
            data = json.loads(full_path.read_text())
            data["pae"] = [[1]] if failure == "bad_pae" else [[float("nan")] * 4] * 4
            full_path.write_text(json.dumps(data), encoding="utf-8")
    row = collect_candidate(record, config, Path(config["response_dir"]))
    assert row["status"].startswith("failed") and row["structure_path"] == ""
    collected = collect_structure_prediction_results(
        "test_af3",
        result["prediction_dir"],
        job_manifest_path=result["job_manifest_path"],
    )
    assert collected["valid_structure_count"] == 0 and collected["failed_count"] == 2


def test_resume_does_not_invoke_external_model(environment, tmp_path, monkeypatch):
    result, manifest = create_job(tmp_path, environment)
    for record in manifest["alphafold3"]["input_records"]:
        for i in range(2):
            sample(record, i)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Complete artifact-based resume must not launch an external process"
        )

    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    rows = runner.run_batch(manifest)
    assert all(r["status"] == "skipped_existing" for r in rows)


def test_command_isolation_and_changed_input_invalidates_reuse(environment, tmp_path):
    _, manifest = create_job(tmp_path, environment)
    config = manifest["alphafold3"]
    model_dir = tmp_path / "weights with spaces"
    model_dir.mkdir()
    (model_dir / "dummy.bin").write_bytes(b"test fixture only")
    config.update(model_dir=str(model_dir), deployment="docker")
    command = runner.build_command(config["input_records"][0], config)
    assert command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert f"{model_dir}:/mnt/models:ro" in command
    assert (
        "--run_data_pipeline=false" in command
        and "--num_diffusion_samples=2" in command
    )
    old = config["input_records"][0]["fingerprint"]
    settings = af3_settings()
    settings["num_samples"] = 3
    changed = prepare_inputs(
        [{"variant": "A101V", "sequence": "VCD"}],
        tmp_path / "in",
        tmp_path / "out",
        settings,
    )
    assert changed["input_records"][0]["fingerprint"] != old
    config["extra_args"] = ["--num_diffusion_samples=1"]
    with pytest.raises(ValueError, match="managed"):
        runner.build_command(config["input_records"][0], config)


def test_batch_executes_native_cli_and_resumes_without_relaunch(environment, tmp_path):
    """Exercise the real subprocess path using an explicit synthetic CLI stub."""
    _, manifest = create_job(tmp_path, environment)
    config = manifest["alphafold3"]
    fixture_dir = tmp_path / "fixture samples"
    for record in config["input_records"]:
        for i in range(2):
            sample(record, i)
        root = Path(record["output_base"]) / record["job_name"]
        shutil.copytree(root, fixture_dir / record["job_name"])
        shutil.rmtree(Path(record["output_base"]))
    fake_repo = tmp_path / "fake_af3"
    fake_repo.mkdir()
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    (model_dir / "synthetic.bin").write_bytes(b"not model weights")
    fake_script = fake_repo / "run_alphafold.py"
    fake_script.write_text(
        "import argparse,json,shutil\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser()\np.add_argument('--json_path')\np.add_argument('--output_dir')\n"
        "a,_=p.parse_known_args()\nname=json.loads(Path(a.json_path).read_text())['name']\n"
        f"source=Path({str(fixture_dir)!r})/name\n"
        "shutil.copytree(source,Path(a.output_dir)/name,dirs_exist_ok=True)\n",
        encoding="utf-8",
    )
    config.update(
        deployment="native",
        repo=str(fake_repo),
        python=sys.executable,
        model_dir=str(model_dir),
    )
    rows = runner.run_batch(manifest)
    assert len(rows) == 2 and all(
        r["status"] == "completed" and r["return_code"] == 0 for r in rows
    )
    assert all(Path(r["log_path"]).exists() for r in rows)
    fake_script.unlink()
    assert all(r["status"] == "skipped_existing" for r in runner.run_batch(manifest))


def test_partial_batch_preserves_complete_candidates(environment, tmp_path):
    result, manifest = create_job(tmp_path, environment)
    config = manifest["alphafold3"]
    first, second = config["input_records"]
    for i in range(2):
        sample(first, i)
    sample(second, 0)
    rows = runner.run_batch(manifest, collect_only=True)
    assert rows[0]["status"] == "completed" and rows[1]["status"].startswith("failed")
    collected = collect_structure_prediction_results(
        "test_af3",
        result["prediction_dir"],
        job_manifest_path=result["job_manifest_path"],
    )
    assert collected["valid_structure_count"] == 1 and collected["failed_count"] == 1


def test_multiple_seeds_and_clashing_top_sample(environment, tmp_path):
    _, manifest = create_job(tmp_path, environment, samples=1)
    config = manifest["alphafold3"]
    config.update(seeds=[1, 2], expected_sample_count=2)
    record = config["input_records"][0]
    sample(record, 0, seed=1, ranking=1.4, clash=True)
    sample(record, 0, seed=2, ranking=0.7)
    row = collect_candidate(record, config, Path(config["response_dir"]))
    assert row["status"] == "completed" and "seed-2_sample-0" in row["structure_path"]


def test_local_boltz_backend_is_unchanged(environment, tmp_path):
    candidates = tmp_path / "boltz.csv"
    candidates.write_text("variant,sequence\nA101V,VCD\n", encoding="utf-8")
    result = prepare_structure_prediction_job(
        "test_boltz",
        str(candidates),
        engine="boltz2_local",
        target_state_structure=str(environment),
    )
    manifest = json.loads(Path(result["job_manifest_path"]).read_text())
    assert manifest["engine"] == "boltz2_local" and "alphafold3" not in manifest
    record = manifest["boltz2"]["input_yaml_records"][0]
    assert Path(record["yaml_path"]).is_file()
    assert manifest["boltz2"]["require_pae"] is True


def test_smoke_cli_reads_saved_af3_choice(environment, tmp_path):
    write_json(
        Path(os.environ["SWITCHTUNE_STRUCTURE_CONFIG"]),
        {
            "engine": "alphafold3",
            "alphafold3": {"msa_mode": "single_sequence"},
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/smoke_structure_backend.py"),
            "--structure",
            str(environment),
            "--chain",
            "C",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Backend: alphafold3" in result.stdout
    reports = list((tmp_path / "runs").glob("structure_smoke_*/smoke_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert (
        report["job"]["candidate_count"] == 1 and report["job"]["status"] == "prepared"
    )
