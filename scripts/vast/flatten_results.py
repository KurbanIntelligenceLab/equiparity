"""Flatten per-run outputs (opaque experiment_id dirs) into easy-to-track flat files + a CSV.

Reads every ``<raw>/box*/<experiment_id>/{metrics,config_snapshot,manifest}`` triple and writes:
  <dest>/metrics/<run_label>.json   -- flat copy, named by run_label (core_parity_target_seedN)
  <dest>/summary.csv                -- one row per run, sorted, easy to scan/track

run_label is (core, parity, target, seed) and does NOT include the dataset, so two datasets sharing
a target collide. Files are therefore named by the dataset-qualified **run_key**: a run on a
non-canonical dataset (e.g. the E1 augmented piezoelectric set) gets a ``__<dataset>`` suffix and
can never overwrite the headline run it shadows. The dataset is read from config_snapshot.yaml,
which every run writes, because older metrics.json predate the `dataset` field.

Idempotent: safe to re-run every sync; later copies of a run supersede earlier ones by timestamp.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

# Datasets used by the headline grid; anything else is a side study and gets a suffixed key.
CANONICAL_DATASETS = frozenset({"qm9", "mp_elastic", "mp_piezoelectric"})

_COLS = [
    "run_key",
    "run_label",
    "dataset",
    "core",
    "parity",
    "target",
    "seed",
    "epochs_run",
    "val_mae",
    "test_mae",
    "ood_false_flag",
    "ood_median",
    "ood_max",
    "n_params",
    "git_sha",
    "git_dirty",
    "box",
    "experiment_id",
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
        # metrics.json only carries `dataset` for runs made after that field was added; the config
        # snapshot always has it.
        dataset = metrics.get("dataset") or cfg.get("dataset") or ""
        key = label if dataset in CANONICAL_DATASETS else f"{label}__{dataset}"
        box = run_dir.parent.name
        epochs = metrics.get("epochs_run", 0) or 0
        # experiment_id is <sha>_<confighash>_<utc_timestamp>; the trailing stamp sorts lexically.
        ts = run_dir.name.rsplit("_", 1)[-1]

        # Same run_label can appear twice (partial fetch, or a re-run like the idealized-OOD piezo
        # pass). Keep the LATEST by timestamp so re-runs supersede the originals.
        prev = rows.get(key)
        if prev is not None and prev["_ts"] >= ts:
            continue

        val = metrics.get("val", {})
        test = metrics.get("test", {})
        rows[key] = {
            "run_key": key,
            "run_label": label,
            "dataset": dataset,
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
            "_ts": ts,
        }
        # Inject `dataset` so downstream analysis can filter even on runs made before the field
        # existed, then write under the dataset-qualified key.
        payload = dict(metrics)
        payload.setdefault("dataset", dataset)
        (flat / f"{key}.json").write_text(json.dumps(payload, indent=1))

    with (dest / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        for label in sorted(rows):
            w.writerow(rows[label])

    n = len(rows)
    n_odd = sum(1 for r in rows.values() if r["target"] in ("piezoelectric", "dipole"))
    n_side = sum(1 for r in rows.values() if r["dataset"] not in CANONICAL_DATASETS)
    print(f">>> flattened {n} completed runs -> {flat} ({n_odd} odd-target, {n_side} side-study)")
    print(f">>> summary -> {dest / 'summary.csv'}")


if __name__ == "__main__":
    main()
