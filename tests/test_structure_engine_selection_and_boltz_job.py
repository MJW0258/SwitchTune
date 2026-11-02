#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from switchtune.mcp_server import (  # noqa: E402
    prepare_structure_prediction_job,
    select_structure_prediction_engine,
)


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def main() -> None:
    os.environ.setdefault("SWITCHTUNE_RUNS", str(PROJECT_ROOT / "SwitchTune_runs"))
    os.environ.setdefault("SWITCHTUNE_HOME", str(PROJECT_ROOT))
    os.environ.setdefault("SWITCHTUNE_TOOL_REGISTRY", str(PROJECT_ROOT / "config" / "tool_registry.yaml"))

    project_name = "toy_structure_engine_boltz_test"
    run_root = Path(os.environ["SWITCHTUNE_RUNS"])
    tmp = run_root / project_name / "00_test_inputs"
    candidates_csv = tmp / "candidates.csv"
    write_text(
        candidates_csv,
        "variant,sequence,bias_score\n"
        "V1,ACDEFGHIK,0.001\n"
        "V2,ACEEFGHIK,-0.002\n",
    )

    target_pdb = tmp / "target_with_fmn.pdb"
    reference_pdb = tmp / "reference_no_ligand.pdb"
    write_text(
        target_pdb,
        "ATOM      1  N   ALA A   1      11.104  13.207   7.947  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.207   7.947  1.00 20.00           C\n"
        "HETATM    3  N1  FMN A 201      13.000  12.000   8.000  1.00 20.00           N\n"
        "END\n",
    )
    write_text(
        reference_pdb,
        "ATOM      1  N   ALA A   1      11.104  13.207   7.947  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.207   7.947  1.00 20.00           C\n"
        "END\n",
    )

    decision = select_structure_prediction_engine(
        project_name=project_name,
        target_state_structure=str(target_pdb),
        reference_state_structure=str(reference_pdb),
        requested_engine="boltz2_nim",
    )
    assert decision["recommended_engine"] == "boltz2_nim", decision
    assert "FMN" in decision["detected_ligand_codes"], decision

    boltz_job = prepare_structure_prediction_job(
        project_name=project_name,
        candidates_csv=str(candidates_csv),
        engine="boltz2_nim",
        target_state_structure=str(target_pdb),
        reference_state_structure=str(reference_pdb),
        boltz_recycling_steps=1,
        boltz_sampling_steps=10,
        boltz_use_template=True,
    )
    assert boltz_job["engine"] == "boltz2_nim", boltz_job
    assert boltz_job["candidate_count"] == 2, boltz_job
    assert Path(boltz_job["run_script"]).exists(), boltz_job
    manifest = json.loads(Path(boltz_job["job_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["boltz2"]["payload_records"], manifest
    first_payload = json.loads(Path(manifest["boltz2"]["payload_records"][0]["payload_path"]).read_text(encoding="utf-8"))
    assert first_payload["ligands"][0]["ccd"] == "FMN", first_payload
    assert first_payload["constraints"] == [], first_payload
    assert first_payload["diffusion_samples"] == 1, first_payload
    assert first_payload["write_full_pae"] is True, first_payload
    assert "step_scale" in first_payload, first_payload
    assert first_payload["polymers"][0]["cyclic"] is False, first_payload
    assert first_payload["polymers"][0]["modifications"] == [], first_payload
    assert "output_format" not in first_payload, first_payload
    templates = first_payload["polymers"][0].get("structural_templates") or []
    assert len(templates) == 1, first_payload
    assert templates[0]["format"] == "pdb", templates
    assert "ATOM" in templates[0]["structure"], templates[0]
    assert "HETATM" in templates[0]["structure"], templates[0]
    assert "FMN" in templates[0]["structure"], templates[0]
    assert "chain_id" not in templates[0], templates[0]
    metadata = manifest["boltz2"]["structural_template_metadata"]
    assert metadata["cleaning_mode"] == "raw", metadata
    assert metadata["format_mode"] == "source", metadata

    default_project = "toy_structure_engine_boltz_default_template_test"
    default_boltz_job = prepare_structure_prediction_job(
        project_name=default_project,
        candidates_csv=str(candidates_csv),
        engine="boltz2_nim",
        target_state_structure=str(target_pdb),
        reference_state_structure=str(reference_pdb),
        boltz_recycling_steps=1,
        boltz_sampling_steps=10,
    )
    assert default_boltz_job["engine"] == "boltz2_nim", default_boltz_job
    default_manifest = json.loads(Path(default_boltz_job["job_manifest_path"]).read_text(encoding="utf-8"))
    default_payload = json.loads(Path(default_manifest["boltz2"]["payload_records"][0]["payload_path"]).read_text(encoding="utf-8"))
    assert "structural_templates" in default_payload["polymers"][0], default_payload
    assert default_manifest["boltz2"]["boltz_use_template"] is True, default_manifest["boltz2"]
    assert default_manifest["boltz2"]["write_full_pae"] is True, default_manifest["boltz2"]
    assert default_manifest["boltz2"]["structural_template_metadata"]["enabled"] is True, default_manifest["boltz2"]

    fallback_project = "toy_structure_engine_boltz_fallback_no_template_test"
    fallback_boltz_job = prepare_structure_prediction_job(
        project_name=fallback_project,
        candidates_csv=str(candidates_csv),
        engine="boltz2_nim",
        target_state_structure=str(target_pdb),
        reference_state_structure=str(reference_pdb),
        boltz_recycling_steps=1,
        boltz_sampling_steps=10,
        boltz_use_template=False,
    )
    fallback_manifest = json.loads(Path(fallback_boltz_job["job_manifest_path"]).read_text(encoding="utf-8"))
    fallback_payload = json.loads(Path(fallback_manifest["boltz2"]["payload_records"][0]["payload_path"]).read_text(encoding="utf-8"))
    assert "structural_templates" not in fallback_payload["polymers"][0], fallback_payload
    assert fallback_payload["write_full_pae"] is True, fallback_payload
    assert fallback_manifest["boltz2"]["boltz_use_template"] is False, fallback_manifest["boltz2"]

    no_ligand_project = "toy_structure_engine_colabfold_test"
    colab_job = prepare_structure_prediction_job(
        project_name=no_ligand_project,
        candidates_csv=str(candidates_csv),
        engine="colabfold",
        target_state_structure=str(reference_pdb),
    )
    assert colab_job["engine"] == "colabfold", colab_job
    assert Path(colab_job["run_script"]).name == "run_colabfold_structure_prediction.sh", colab_job
    assert "--msa-mode" in colab_job["command"], colab_job["command"]
    assert "single_sequence" in colab_job["command"], colab_job["command"]
    assert "--templates" in colab_job["command"], colab_job["command"]
    assert "--custom-template-path" in colab_job["command"], colab_job["command"]
    assert colab_job["template_info"]["enabled"] is True, colab_job["template_info"]

    print("STRUCTURE_ENGINE_SELECTION_AND_BOLTZ_JOB_TEST: PASSED")
    print(f"Boltz manifest: {boltz_job['job_manifest_path']}")
    print(f"Boltz run script: {boltz_job['run_script']}")
    print(f"ColabFold run script: {colab_job['run_script']}")


if __name__ == "__main__":
    main()
