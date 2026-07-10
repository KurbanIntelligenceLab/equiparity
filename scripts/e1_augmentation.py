"""E1 -- does training SO(3) on zero-labelled centrosymmetric crystals fix the false flags?

The reviewer's proposed fix is data augmentation. This measures what it buys, split by whether the
evaluation crystal's space group appeared in the augmentation set:

* **SEEN-SG** (1,232 of the OOD 2,000): space group in the augmented training list, but the crystals
  themselves never trained on.
* **UNSEEN-SG** (768): space groups that appear nowhere in training.

A fix that only works on SEEN-SG is not a fix -- it is curated data with no guarantee off-manifold.
O(3) arms are not retrained: their zeros are structural, so the headline runs already answer for
them, and they are quoted here as the reference row.

    uv run --extra nequip --extra data python scripts/e1_augmentation.py
    python scripts/e1_augmentation.py --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path.home() / "Desktop" / "parity_work"
EVAL_SPLIT = REPO / "results" / "e1_eval_split.json"
OUT_JSON = REPO / "results" / "e1_augmentation.json"
OUT_MD = REPO / "docs" / "results" / "e1_augmentation.md"

THRESHOLD = 0.01
_TABLE_HEADER = (
    "| core | arm | false-flag SEEN-SG | false-flag UNSEEN-SG "
    "| median SEEN | median UNSEEN | test MAE |"
)
SEEDS = [0, 1, 2]
_METRICS = ("ff_seen", "ff_unseen", "median_seen", "median_unseen", "test_mae")
CORES = ["nequip", "allegro", "mace", "equiformer_v2"]
CORE_LABEL = {
    "nequip": "NequIP",
    "allegro": "Allegro",
    "mace": "MACE",
    "equiformer_v2": "EquiformerV2",
}


def _mean(per_seed: list[dict], key: str) -> float:
    return float(np.mean([s[key] for s in per_seed]))


def _std(per_seed: list[dict], key: str) -> float:
    if len(per_seed) < 2:
        return 0.0
    return float(np.std([s[key] for s in per_seed], ddof=1))


def _dataset_of(run_dir: Path, metrics: dict) -> str | None:
    if "dataset" in metrics:
        return metrics["dataset"]
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return None
    for line in snapshot.read_text().splitlines():
        if line.startswith("dataset:"):
            return line.split(":", 1)[1].strip()
    return None


def _find_runs(pattern: str) -> dict[str, Path]:
    """Latest run directory per run_label whose dataset is ``pattern``."""
    latest: dict[str, tuple[str, Path]] = {}
    for metrics_file in MIRROR.glob("raw/box*/*/metrics.json"):
        run_dir = metrics_file.parent
        try:
            m = json.loads(metrics_file.read_text())
        except json.JSONDecodeError:
            continue
        if _dataset_of(run_dir, m) != pattern or "piezoelectric" not in m.get("run_label", ""):
            continue
        stamp = run_dir.name.split("_")[-1]
        label = m["run_label"]
        if label not in latest or stamp > latest[label][0]:
            latest[label] = (stamp, run_dir)
    return {k: v[1] for k, v in latest.items()}


def _vector(run_dir: Path, variant: str = "idealized") -> np.ndarray:
    return np.load(run_dir / f"ood_violations_{variant}.npy")


def collect() -> dict:
    split = json.loads(EVAL_SPLIT.read_text())
    seen = np.array(split["seen_indices"])
    unseen = np.array(split["unseen_indices"])

    augmented = _find_runs("mp_piezoelectric_augmented")
    baseline = _find_runs("mp_piezoelectric")
    if not augmented:
        raise SystemExit("no augmented runs found in the results mirror yet")

    out: dict[str, dict] = {
        "n_seen": len(seen),
        "n_unseen": len(unseen),
        "seen_spacegroups": split["seen_spacegroups"],
        "n_augmentation": split["n_augmentation"],
        "arms": {},
    }

    for core in CORES:
        for tag, runs, parity in (
            ("baseline_so3", baseline, "so3"),
            ("augmented_so3", augmented, "so3"),
            ("baseline_o3", baseline, "o3"),
        ):
            if core == "equiformer_v2" and parity == "o3":
                continue
            per_seed = []
            for seed in SEEDS:
                label = f"{core}_{parity}_piezoelectric_seed{seed}"
                if label not in runs:
                    continue
                run_dir = runs[label]
                v = _vector(run_dir)
                metrics = json.loads((run_dir / "metrics.json").read_text())
                per_seed.append(
                    {
                        "seed": seed,
                        "ff_seen": float((v[seen] > THRESHOLD).mean()),
                        "ff_unseen": float((v[unseen] > THRESHOLD).mean()),
                        "median_seen": float(np.median(v[seen])),
                        "median_unseen": float(np.median(v[unseen])),
                        "test_mae": float(metrics["test"]["mae"]),
                    }
                )
            if per_seed:
                out["arms"][f"{core}_{tag}"] = {
                    "core": core,
                    "arm": tag,
                    "n_seeds": len(per_seed),
                    "per_seed": per_seed,
                    **{f"{k}_mean": _mean(per_seed, k) for k in _METRICS},
                    **{f"{k}_std": _std(per_seed, k) for k in _METRICS},
                }
    return out


def render() -> None:
    d = json.loads(OUT_JSON.read_text())
    arms = d["arms"]
    lines = [
        "# E1 — the augmentation rebuttal",
        "",
        f"Training set: 2,649 real piezoelectric tensors + {d['n_augmentation']} centrosymmetric",
        f"crystals labelled exactly zero, drawn only from space groups {d['seen_spacegroups']}.",
        "The target normalisation scale is frozen at the un-augmented value (0.749134) so that",
        "violation magnitudes stay directly comparable to the headline table.",
        "",
        "Evaluation splits the untouched OOD 2,000 by space group: "
        f"**SEEN-SG** ({d['n_seen']}) and",
        f"**UNSEEN-SG** ({d['n_unseen']}). Neither overlaps the training ids. "
        "Mean ± std over 3 seeds.",
        "",
        _TABLE_HEADER,
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(arms):
        a = arms[key]
        lines.append(
            f"| {CORE_LABEL[a['core']]} | {a['arm']} "
            f"| {a['ff_seen_mean']:.4f} ± {a['ff_seen_std']:.4f} "
            f"| {a['ff_unseen_mean']:.4f} ± {a['ff_unseen_std']:.4f} "
            f"| {a['median_seen_mean']:.3e} | {a['median_unseen_mean']:.3e} "
            f"| {a['test_mae_mean']:.4f} ± {a['test_mae_std']:.4f} |"
        )
    lines += [
        "",
        "`baseline_o3` is quoted from the headline runs and was **not** retrained: an O(3) model's",
        "zeros are structural, holding for any weights and any training data. That is the point.",
        "",
        "## Reading",
        "",
        "See the generated numbers above. The decisive comparison is `augmented_so3` on SEEN-SG",
        "versus UNSEEN-SG. A large gap means learned zeros do not transfer across symmetry",
        "classes.",
        "No gap means the fix works only because the space groups were curated into training,",
        "and it still carries no guarantee for an unseen class. Either outcome leaves the O(3)",
        "guarantee as the only one that holds by construction rather than by data coverage.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        render()
        return
    result = collect()
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT_JSON} ({len(result['arms'])} arms)")
    render()


if __name__ == "__main__":
    main()
