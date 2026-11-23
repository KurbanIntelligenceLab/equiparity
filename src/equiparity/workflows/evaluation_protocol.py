"""Freeze scientific configurations and data identities before test evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from equiparity.domain.experiment import ExperimentConfig
from equiparity.io.config import load_experiment_config
from equiparity.workflows.prepare_data import sha256_file
from equiparity.workflows.run_experiment import _config_snapshot


def scientific_digest(config: ExperimentConfig) -> str:
    """Ignore storage location, but retain every model and optimization setting."""
    payload = json.dumps(_config_snapshot(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def lock_evaluation_protocol(root: Path, configs: list[Path], output: Path) -> dict[str, Any]:
    """Create an immutable declaration; never silently replace an existing lock."""
    rows = []
    for path in configs:
        cfg = load_experiment_config(path)
        if cfg.protocol != "validation_selected" or cfg.seed not in (0, 1, 2):
            raise ValueError("protocol requires validation_selected and seeds 0, 1, 2")
        if cfg.training.max_train_samples or cfg.training.max_eval_samples:
            raise ValueError("test protocol requires complete partitions")
        rows.append(
            {
                "config": str(path),
                "scientific_sha256": scientific_digest(cfg),
                "data_sha256": sha256_file(root / cfg.processed_npz),
                "split_sha256": sha256_file(root / cfg.split_npz),
            }
        )
    if not rows or len({r["scientific_sha256"] for r in rows}) != len(rows):
        raise ValueError("a nonempty list of distinct configurations is required")
    result = {
        "schema": 1,
        "selection": "lowest validation MAE checkpoint; test never selects models",
        "primary_piezoelectric_loss": "tensor_huber",
        "primary_elastic_loss": "mse",
        "reporting": "all declared runs; three-seed mean and sample standard deviation",
        "baselines": ["zero", "training_mean", "training_median"],
        "configs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def verify_test_authorization(config: ExperimentConfig, audit: dict[str, Any], lock: Path) -> str:
    """Reject unregistered settings or changed datasets before loading test targets."""
    protocol = json.loads(lock.read_text())
    digest = scientific_digest(config)
    matches = [r for r in protocol["configs"] if r["scientific_sha256"] == digest]
    if protocol.get("schema") != 1 or len(matches) != 1:
        raise ValueError("configuration is not uniquely registered in the protocol lock")
    for key in ("data_sha256", "split_sha256"):
        if audit[key] != matches[0][key]:
            raise ValueError(f"protocol lock {key} mismatch")
    return sha256_file(lock)
