"""Deployment-time structure backend choice, shared by CLI and MCP processes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def deployment_config() -> dict[str, Any]:
    root = Path(
        os.environ.get("SWITCHTUNE_HOME") or Path(__file__).resolve().parents[1]
    )
    path = Path(
        os.environ.get("SWITCHTUNE_STRUCTURE_CONFIG")
        or root / "config/structure_backend.local.json"
    )
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def configured_engine() -> str:
    selected = os.environ.get("SWITCHTUNE_STRUCTURE_ENGINE") or deployment_config().get(
        "engine"
    )
    if selected:
        return str(selected)
    root = Path(
        os.environ.get("SWITCHTUNE_HOME") or Path(__file__).resolve().parents[1]
    )
    config_path = Path(
        os.environ.get("SWITCHTUNE_CONFIG") or root / "config/switchtune.yaml"
    )
    if config_path.is_file():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return str(
            config.get("structure_prediction", {}).get("default_engine")
            or "boltz2_local"
        )
    return "boltz2_local"


def af3_settings(registry: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "deployment": "docker",
        "docker_image": "alphafold3",
        "python": "python",
        "repo": "",
        "model_dir": "",
        "db_dir": "",
        "msa_mode": "search",
        "num_samples": 5,
        "seeds": [1],
        "protein_chain_id": "A",
        "input_template": "",
        "extra_args": [],
        **(registry or {}),
        **deployment_config().get("alphafold3", {}),
    }
    for name in (
        "deployment",
        "docker_image",
        "python",
        "repo",
        "model_dir",
        "db_dir",
        "msa_mode",
        "input_template",
    ):
        if os.environ.get("AF3_" + name.upper()):
            config[name] = os.environ["AF3_" + name.upper()]
    if os.environ.get("AF3_NUM_SAMPLES"):
        config["num_samples"] = int(os.environ["AF3_NUM_SAMPLES"])
    if os.environ.get("AF3_SEEDS"):
        config["seeds"] = [int(x) for x in os.environ["AF3_SEEDS"].split(",")]
    for name in ("repo", "model_dir", "db_dir", "input_template"):
        if config[name]:
            config[name] = str(
                Path(os.path.expandvars(config[name])).expanduser().resolve()
            )
    if config["msa_mode"] not in {"search", "single_sequence", "provided"}:
        raise ValueError("AF3_MSA_MODE must be search, single_sequence, or provided")
    if (
        int(config["num_samples"]) < 1
        or not config["seeds"]
        or len(set(config["seeds"])) != len(config["seeds"])
        or any(int(x) < 0 for x in config["seeds"])
    ):
        raise ValueError(
            "AlphaFold3 requires positive num_samples and distinct non-negative seeds"
        )
    if config["deployment"] not in {"native", "docker"}:
        raise ValueError("AlphaFold3 deployment must be native or docker")
    if not isinstance(config["extra_args"], list):
        raise ValueError("AlphaFold3 extra_args must be a list of command arguments")
    return config
