"""H3 : does up-weighting the zero-labelled rows force the SO(3) violation to zero?

Reads the loss-weight sweep (NequIP SO(3) on the augmented piezoelectric set): W=1 from the
committed E1 runs, W=10/100 from the H3 runs. For each weight, reports on the same three populations
E1 uses -- trained zero-labelled rows, SEEN-SG, UNSEEN-SG -- plus non-centrosymmetric test MAE.

    uv run --extra nequip --extra data python scripts/h3_loss_weight.py
    python scripts/h3_loss_weight.py --render
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
OUT_JSON = REPO / "results" / "h3_loss_weight.json"
OUT_MD = REPO / "docs" / "results" / "h3_loss_weight.md"
THRESHOLD = 0.01
SEEDS = [0, 1, 2]


def _dataset_of(run_dir: Path, metrics: dict) -> str | None:
    if "dataset" in metrics:
        return metrics["dataset"]
    snap = run_dir / "config_snapshot.yaml"
    if snap.exists():
        for line in snap.read_text().splitlines():
            if line.startswith("dataset:"):
                return line.split(":", 1)[1].strip()
    return None


def _runs_for_dataset(dataset: str) -> dict[str, Path]:
    """core_parity_target_seedN -> latest run dir whose dataset matches exactly."""
    latest: dict[str, tuple[str, Path]] = {}
    for mfile in MIRROR.glob("raw/box*/*/metrics.json"):
        run_dir = mfile.parent
        try:
            m = json.loads(mfile.read_text())
        except json.JSONDecodeError:
            continue
        if _dataset_of(run_dir, m) != dataset or "piezoelectric" not in m.get("run_label", ""):
            continue
        stamp = run_dir.name.split("_")[-1]
        label = m["run_label"]
        if label not in latest or stamp > latest[label][0]:
            latest[label] = (stamp, run_dir)
    return {k: v[1] for k, v in latest.items()}


def collect() -> dict:
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from e1_augmentation import in_train_control

    split = json.loads(EVAL_SPLIT.read_text())
    seen, unseen = np.array(split["seen_indices"]), np.array(split["unseen_indices"])
    datasets = {
        1: "mp_piezoelectric_augmented",
        10: "mp_piezoelectric_augmented_w10",
        100: "mp_piezoelectric_augmented_w100",
    }

    rows = []
    for weight, dataset in datasets.items():
        runs = _runs_for_dataset(dataset)
        per_seed = []
        for seed in SEEDS:
            label = f"nequip_so3_piezoelectric_seed{seed}"
            if label not in runs:
                continue
            run_dir = runs[label]
            v = np.load(run_dir / "ood_violations_idealized.npy")
            metrics = json.loads((run_dir / "metrics.json").read_text())
            ctrl = in_train_control(run_dir)
            per_seed.append(
                {
                    "seed": seed,
                    "ff_trained_zeros": ctrl["false_flag_on_trained_zeros"],
                    "median_trained_zeros": ctrl["median_violation_on_trained_zeros"],
                    "ff_seen": float((v[seen] > THRESHOLD).mean()),
                    "ff_unseen": float((v[unseen] > THRESHOLD).mean()),
                    "median_seen": float(np.median(v[seen])),
                    "median_unseen": float(np.median(v[unseen])),
                    "test_mae": float(metrics["test"]["mae"]),
                }
            )
        if per_seed:
            keys = [k for k in per_seed[0] if k != "seed"]
            rows.append(
                {
                    "weight": weight,
                    "dataset": dataset,
                    "n_seeds": len(per_seed),
                    **{f"{k}_mean": float(np.mean([s[k] for s in per_seed])) for k in keys},
                    **{
                        f"{k}_std": float(np.std([s[k] for s in per_seed], ddof=1))
                        if len(per_seed) > 1
                        else 0.0
                        for k in keys
                    },
                }
            )
    return {
        "rows": rows,
        "n_seen": len(seen),
        "n_unseen": len(unseen),
        "n_augmentation": split["n_augmentation"],
    }


def render() -> None:
    d = json.loads(OUT_JSON.read_text())
    lines = [
        "# H3 — the loss-weight sweep",
        "",
        "Can up-weighting the zero-labelled rows force an SO(3) model's centrosymmetric violation "
        "to",
        "zero? NequIP SO(3) on the augmented set (2,649 real tensors + 1,000 injected zeros), with "
        "the",
        "exactly-zero-target rows (1,016: 1,000 injected + 16 real) weighted by W in the MSE. W=1 "
        "is",
        "the committed E1 run; W=10, 100 are the sweep. Mean ± std over 3 seeds.",
        "",
        "| W | ff trained zeros | median trained zeros | ff SEEN-SG | ff UNSEEN-SG | test MAE |",
        "|---|---|---|---|---|---|",
    ]
    for r in d["rows"]:
        lines.append(
            f"| {r['weight']} "
            f"| {r['ff_trained_zeros_mean']:.4f} ± {r['ff_trained_zeros_std']:.4f} "
            f"| {r['median_trained_zeros_mean']:.4f} "
            f"| {r['ff_seen_mean']:.4f} ± {r['ff_seen_std']:.4f} "
            f"| {r['ff_unseen_mean']:.4f} ± {r['ff_unseen_std']:.4f} "
            f"| {r['test_mae_mean']:.4f} ± {r['test_mae_std']:.4f} |"
        )
    by_w = {r["weight"]: r for r in d["rows"]}
    lines += [
        "",
        "For reference, O(3) untrained: 0.0000 on every population, median ~3e-07 (structural).",
        "",
        "## Reading",
        "",
        "**Extreme reweighting does not buy the zero.** As W goes 1 -> 10 -> 100, the false-flag "
        "rate on the crystals the model was *trained to call zero* falls only "
        f"{by_w[1]['ff_trained_zeros_mean']:.3f} -> {by_w[10]['ff_trained_zeros_mean']:.3f} -> "
        f"{by_w[100]['ff_trained_zeros_mean']:.3f}, and on held-out SEEN-SG "
        f"{by_w[1]['ff_seen_mean']:.3f} -> {by_w[100]['ff_seen_mean']:.3f}. Even at 100x it still "
        "false-flags roughly two-thirds of its own trained-zero crystals and ~87% of held-out seen "
        "ones -- far above O(3)'s 0.0000.",
        "",
        "The median violation does shrink with weight "
        f"({by_w[1]['median_trained_zeros_mean']:.3f} -> "
        f"{by_w[100]['median_trained_zeros_mean']:.3f} on the trained zeros, ~9x), and "
        f"non-centrosymmetric test MAE is not harmed -- it slightly improves "
        f"({by_w[1]['test_mae_mean']:.4f} -> {by_w[100]['test_mae_mean']:.4f}). So the fix costs "
        "nothing in regression quality; it simply cannot reach zero. Gradient descent pushes the "
        "prediction towards zero and cannot arrive, because the physical answer is *exactly* zero "
        "-- which only the O(3) structure delivers, for free, at any weight.",
        "",
        "No weight setting achieved ~0.00 false-flag with intact MAE, so no off-cycle report is "
        "triggered. The E1 conclusion stands, sharpened: augmentation with even 100x loss "
        "weighting reduces but does not remove the impossible predictions.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        render()
        return
    result = collect()
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT_JSON} ({len(result['rows'])} weights)")
    for r in result["rows"]:
        print(
            f"  W={r['weight']:3d}  ff_trained={r['ff_trained_zeros_mean']:.4f}  "
            f"ff_seen={r['ff_seen_mean']:.4f}  ff_unseen={r['ff_unseen_mean']:.4f}  "
            f"testMAE={r['test_mae_mean']:.4f}"
        )
    render()


if __name__ == "__main__":
    main()
