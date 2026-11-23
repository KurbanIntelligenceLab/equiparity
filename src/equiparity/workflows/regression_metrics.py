"""Aggregate the complete protocol-locked regression, independently checking predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from equiparity.io.config import load_experiment_config
from equiparity.training.tensor_optimization import atomic_json
from equiparity.workflows.evaluation_protocol import scientific_digest, verify_test_authorization
from equiparity.workflows.prepare_data import sha256_file


def three_seed_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep training-seed variation distinct from crystal-resampling uncertainty."""
    if sorted(r["seed"] for r in rows) != [0, 1, 2]:
        raise ValueError("each comparison must contain exactly seeds 0, 1, 2")
    rows = sorted(rows, key=lambda r: r["seed"])
    result: dict[str, Any] = {"seeds": [0, 1, 2], "n_seeds": 3}
    for metric in ("mae", "rmse", "frobenius_rmse"):
        values = np.asarray([r["metrics"][metric] for r in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite evaluation metric")
        result[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "by_seed": values.tolist(),
        }
    result["baselines"] = {}
    for name in ("zero", "training_mean", "training_median"):
        values = np.array(
            [r["metrics"]["mae"] + r["comparisons"][name]["mean_mae_reduction"] for r in rows]
        )
        if not np.allclose(values, values[0], rtol=1e-10, atol=1e-10):
            raise ValueError("baseline changes across seeds with fixed training data")
        baseline = float(values.mean())
        result["baselines"][name] = {
            "mae": baseline,
            "mean_mae_reduction": baseline - result["mae"]["mean"],
            "relative_mae_reduction": (1 - result["mae"]["mean"] / baseline)
            if baseline > 0
            else None,
            "all_three_seeds_improve": all(r["metrics"]["mae"] < baseline for r in rows),
        }
    return result


def summarize_regressions(lock: Path, evaluations: list[Path], output: Path) -> dict[str, Any]:
    """Reject incomplete, duplicated, changed, or inconsistent test evidence."""
    protocol = json.loads(lock.read_text())
    expected = {r["scientific_sha256"] for r in protocol["configs"]}
    seen: set[str] = set()
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    populations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    lock_hash = sha256_file(lock)
    for directory in evaluations:
        cfg = load_experiment_config(directory / "config_snapshot.yaml")
        row = json.loads((directory / "test_diagnostics.json").read_text())
        digest = scientific_digest(cfg)
        if digest in seen:
            raise ValueError("duplicate configuration evaluation")
        if row["partition"] != "test" or row["protocol_lock_sha256"] != lock_hash:
            raise ValueError("evaluation is not bound to this held-out protocol")
        verify_test_authorization(cfg, row, lock)
        if sha256_file(directory / "checkpoint_selected.pt") != row["checkpoint_sha256"]:
            raise ValueError("evaluation checkpoint digest mismatch")
        with np.load(directory / "test_predictions.npz", allow_pickle=False) as data:
            ids, prediction, target = data["ids"], data["prediction"], data["target"]
        if prediction.shape != target.shape or len(ids) != len(target):
            raise ValueError("prediction/target alignment mismatch")
        if len(set(ids.tolist())) != len(ids) or not np.isfinite(prediction).all():
            raise ValueError("duplicate IDs or nonfinite predictions")
        if cfg.target in populations:
            old_ids, old_target = populations[cfg.target]
            if not np.array_equal(ids, old_ids) or target.shape != old_target.shape:
                raise ValueError("target population differs across model comparisons")
            # The two locked e3nn versions round the change-of-basis coefficients
            # differently. Bound the projection error for each crystal separately.
            delta = np.linalg.norm(target - old_target, axis=1)
            tolerance = np.finfo(np.float32).eps * np.linalg.norm(old_target, axis=1)
            if not np.all(delta <= tolerance):
                raise ValueError("target population differs across model comparisons")
        else:
            populations[cfg.target] = (ids.copy(), target.copy())
        error = prediction - target
        independent = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "frobenius_rmse": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        }
        for name, value in independent.items():
            if not np.isclose(value, row["metrics"][name], rtol=1e-12, atol=1e-12):
                raise ValueError(f"saved prediction disagrees with {name}")
        row.update(seed=cfg.seed, scientific_sha256=digest, evaluation_dir=str(directory))
        group = (
            cfg.target,
            cfg.dataset,
            cfg.core,
            cfg.parity.value,
            cfg.training.loss,
            cfg.training.zero_row_loss_weight,
        )
        groups[group].append(row)
        records.append(row)
        seen.add(digest)
    if seen != expected:
        raise ValueError(f"incomplete protocol: {len(expected - seen)} missing configurations")
    summary = []
    for group, rows in sorted(groups.items()):
        target, dataset, core, parity, loss, weight = group
        summary.append(
            dict(
                target=target,
                dataset=dataset,
                core=core,
                parity=parity,
                loss=loss,
                zero_row_weight=weight,
                **three_seed_statistics(rows),
            )
        )
    result = {
        "protocol_lock_sha256": lock_hash,
        "n_runs": len(records),
        "uncertainty": "sample standard deviation across three training seeds",
        "all_predictions_independently_recomputed": True,
        "comparisons": summary,
        "runs": records,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output)
    return result
