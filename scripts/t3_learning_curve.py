"""T3 -- the N_zero learning curve (Tier 3; decides the title).

For N_zero in {0, 250, 1000, 4000, 16000}: false-flag fraction and violation median on the
held-out 2,000 (global + SEEN-SG + UNSEEN-SG partitions from ``results/e1_eval_split.json``),
non-centrosymmetric test MAE, and (with ``--control``, GPU) the in-training control -- the
false-flag fraction on the very crystals trained on with explicit zero labels, via checkpoint
reload exactly as ``e1_augmentation.py::in_train_control``.

N = 0 reuses the headline NequIP SO(3) runs; N = 1000 the E1 augmented runs; the rest come
from the T3 grid (datasets ``mp_piezoelectric_augmented_n{N}``). All sets are nested.

Title rule (prespecified, applied neutrally): if the curve plateaus well above zero,
"Not all symmetry can be learned" survives; if it falls towards zero, revert to
"Rotation is not enough..." and reframe E1 as cost-of-data.

    uv run --extra nequip python scripts/t3_learning_curve.py [--control]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path(os.environ.get("PARITY_RUNS", Path.home() / "Desktop" / "parity_work"))
EVAL_SPLIT = REPO / "results" / "e1_eval_split.json"
OUT_JSON = REPO / "results" / "t3_learning_curve.json"
OUT_MD = REPO / "docs" / "results" / "t3_learning_curve.md"

THRESHOLD = 0.01
SEEDS = [0, 1, 2]
N_DATASET = {
    0: "mp_piezoelectric",
    250: "mp_piezoelectric_augmented_n250",
    1000: "mp_piezoelectric_augmented",
    4000: "mp_piezoelectric_augmented_n4000",
    16000: "mp_piezoelectric_augmented_n16000",
}


def _dataset_of(run_dir: Path, metrics: dict) -> str | None:
    """Older runs predate the `dataset` metrics field; config_snapshot.yaml always has it."""
    if "dataset" in metrics:
        return metrics["dataset"]
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return None
    for line in snapshot.read_text().splitlines():
        if line.startswith("dataset:"):
            return line.split(":", 1)[1].strip()
    return None


def find_runs() -> dict[tuple[int, int], Path]:
    """(n_zero, seed) -> latest NequIP SO(3) piezo run dir carrying that dataset."""
    wanted = {v: k for k, v in N_DATASET.items()}
    latest: dict[tuple[int, int], tuple[str, Path]] = {}
    for mfile in MIRROR.glob("raw/box*/*/metrics.json"):
        try:
            m = json.loads(mfile.read_text())
        except json.JSONDecodeError:
            continue
        label = m.get("run_label", "")
        dataset = _dataset_of(mfile.parent, m)
        if dataset is None and label.startswith("nequip_so3_piezoelectric"):
            dataset = "mp_piezoelectric"  # pre-field headline runs are canonical by definition
        if dataset not in wanted or not label.startswith("nequip_so3_piezoelectric"):
            continue
        if not (mfile.parent / "ood_violations_idealized.npy").exists():
            continue
        n = wanted[dataset]
        seed = int(label.rsplit("seed", 1)[1])
        ts = mfile.parent.name.split("_")[-1]
        if (n, seed) not in latest or ts > latest[(n, seed)][0]:
            latest[(n, seed)] = (ts, mfile.parent)
    return {k: d for k, (_, d) in latest.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--control",
        action="store_true",
        help="also reload checkpoints for the in-training control (GPU)",
    )
    args = ap.parse_args()

    split = json.loads(EVAL_SPLIT.read_text())
    seen = np.array(split["seen_indices"])
    unseen = np.array(split["unseen_indices"])

    runs = find_runs()
    curve: dict[str, dict] = {}
    for n in sorted(N_DATASET):
        per_seed = []
        for s in SEEDS:
            d = runs.get((n, s))
            if d is None:
                continue
            v = np.load(d / "ood_violations_idealized.npy")
            m = json.loads((d / "metrics.json").read_text())
            per_seed.append(
                {
                    "seed": s,
                    "ff": float((v > THRESHOLD).mean()),
                    "ff_seen": float((v[seen] > THRESHOLD).mean()),
                    "ff_unseen": float((v[unseen] > THRESHOLD).mean()),
                    "median": float(np.median(v)),
                    "test_mae": float(m["test"]["mae"]),
                }
            )
        if not per_seed:
            continue
        agg = {
            k: {
                "mean": float(np.mean([p[k] for p in per_seed])),
                "std": float(np.std([p[k] for p in per_seed], ddof=1))
                if len(per_seed) > 1
                else 0.0,
            }
            for k in ("ff", "ff_seen", "ff_unseen", "median", "test_mae")
        }
        curve[str(n)] = {"n_seeds": len(per_seed), "per_seed": per_seed, **agg}

    if args.control:
        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from e1_augmentation import in_train_control

        for n in sorted(N_DATASET):
            if n == 0 or str(n) not in curve:
                continue
            controls = []
            for s in SEEDS:
                d = runs.get((n, s))
                if d is not None:
                    controls.append(in_train_control(d))
            if controls:
                key = "false_flag_on_trained_zeros"
                curve[str(n)]["in_train_control"] = {
                    "mean": float(np.mean([c[key] for c in controls])),
                    "std": float(np.std([c[key] for c in controls], ddof=1))
                    if len(controls) > 1
                    else 0.0,
                    "median_violation": float(
                        np.mean([c["median_violation_on_trained_zeros"] for c in controls])
                    ),
                }

    OUT_JSON.write_text(json.dumps({"threshold": THRESHOLD, "curve": curve}, indent=1) + "\n")

    lines = [
        "# T3 — the N_zero learning curve",
        "",
        "NequIP SO(3), 3 seeds per point, nested augmentation sets; held-out 2,000 population.",
        "",
        "| N_zero | ff (global) | ff SEEN | ff UNSEEN | median | test MAE | ff trained zeros |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in sorted(N_DATASET):
        c = curve.get(str(n))
        if c is None:
            lines.append(f"| {n} | (pending) | | | | | |")
            continue
        itc = c.get("in_train_control")
        itc_s = f"{itc['mean']:.4f} ± {itc['std']:.4f}" if itc else "-"
        lines.append(
            f"| {n} | {c['ff']['mean']:.4f} ± {c['ff']['std']:.4f} "
            f"| {c['ff_seen']['mean']:.4f} | {c['ff_unseen']['mean']:.4f} "
            f"| {c['median']['mean']:.4f} | {c['test_mae']['mean']:.4f} | {itc_s} |"
        )
    lines += [
        "",
        "Reading: a plateau well above zero means the deficit does not close with data;",
        "a fall towards zero would mean it does.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    for n in sorted(N_DATASET):
        c = curve.get(str(n))
        if c:
            print(
                f"N={n:6d}: ff={c['ff']['mean']:.4f}±{c['ff']['std']:.4f} "
                f"(seen {c['ff_seen']['mean']:.4f} / unseen {c['ff_unseen']['mean']:.4f}) "
                f"n_seeds={c['n_seeds']}"
            )


if __name__ == "__main__":
    main()
