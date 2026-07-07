"""Flatten per-run outputs (opaque experiment_id dirs) into easy-to-track flat files + a CSV.

Reads every ``<raw>/box*/<experiment_id>/{metrics,config_snapshot,manifest}`` triple and writes:
  <dest>/metrics/<run_label>.json   -- flat copy, named by run_label (core_parity_target_seedN)
  <dest>/summary.csv                -- one row per run, sorted, easy to scan/track

run_label is unique per (core, parity, target, seed), so there are no cross-box collisions.
Idempotent: safe to re-run every sync; later (more-trained) copies of a run overwrite earlier ones
only if they have more epochs, so partial fetches never clobber a completed run.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

_COLS = [
    "run_label", "core", "parity", "target", "seed", "epochs_run",
    "val_mae", "test_mae", "ood_false_flag", "ood_median", "ood_max",
    "n_params", "git_sha", "git_dirty", "box", "experiment_id",
]


def _load(p: Path) -> dict:
    try:
        if p.suffix == ".json":
            return json.loads(p.read_text())
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def main() -> None:
    dest = Path(sys.argv[1]).expanduser()
    raw = dest / "raw"
    flat = dest / "metrics"
    flat.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    for run_dir in sorted(raw.glob("box*/*/")):
        metrics = _load(run_dir / "metrics.json")
        if not metrics:
            continue  # run still in progress (no metrics yet)
        cfg = _load(run_dir / "config_snapshot.yaml")
        man = _load(run_dir / "manifest.json")
        label = metrics.get("run_label") or run_dir.name
        box = run_dir.parent.name
        epochs = metrics.get("epochs_run", 0) or 0

        # Keep the most-trained copy if the same run appears twice (partial + complete fetch).
        prev = rows.get(label)
        if prev is not None and (prev.get("epochs_run") or 0) >= epochs:
            continue

        val = metrics.get("val", {})
        test = metrics.get("test", {})
        rows[label] = {
            "run_label": label,
            "core": cfg.get("core", ""),
            "parity": metrics.get("parity", cfg.get("parity", "")),
            "target": metrics.get("target", cfg.get("target", "")),
            "seed": cfg.get("seed", ""),
            "epochs_run": epochs,
            "val_mae": val.get("mae", ""),
            "test_mae": test.get("mae", ""),
            "ood_false_flag": metrics.get("ood_false_flag_fraction", ""),
            "ood_median": metrics.get("ood_violation_median", ""),
            "ood_max": metrics.get("ood_violation_max", ""),
            "n_params": metrics.get("n_params", ""),
            "git_sha": man.get("git_sha", ""),
            "git_dirty": man.get("git_dirty", ""),
            "box": box,
            "experiment_id": run_dir.name,
        }
        shutil.copyfile(run_dir / "metrics.json", flat / f"{label}.json")

    with (dest / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for label in sorted(rows):
            w.writerow(rows[label])

    n = len(rows)
    n_odd = sum(1 for r in rows.values() if r["target"] in ("piezoelectric", "dipole"))
    print(f">>> flattened {n} completed runs -> {flat} ({n_odd} odd-target)")
    print(f">>> summary -> {dest / 'summary.csv'}")


if __name__ == "__main__":
    main()
