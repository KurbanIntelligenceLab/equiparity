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
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path(os.environ.get("PARITY_RUNS", Path.home() / "Desktop" / "parity_work"))
EVAL_SPLIT = REPO / "results" / "e1_eval_split.json"
OUT_JSON = REPO / "results" / "e1_augmentation.json"
OUT_MD = REPO / "docs" / "results" / "e1_augmentation.md"

THRESHOLD = 0.01
FROZEN_SCALE = 0.749134
_CONTROL_HEADER = (
    "| core | median ‖T‖ on trained zeros | false-flag on trained zeros "
    "| train MAE (zero rows) | train MAE (real rows) | mean &#124;target&#124; (real rows) |"
)
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


def in_train_control(run_dir: Path) -> dict:
    """What does the model predict on the zero-labelled crystals it actually trained on?

    Without this, E1 cannot separate "learned zeros do not generalize" from "the model never
    learned the zeros at all". Reload the run, predict on its own training split, and split the
    rows by whether the label was an exact zero.
    """
    import torch
    import yaml

    from equiparity.inference import load_trained
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
    from equiparity.training.nequip_tensor import _irreps_targets

    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
    # The snapshot predates target_scale serialisation; the frozen value is verified empirically.
    trained = load_trained(run_dir, repo_root=REPO, scale=FROZEN_SCALE)
    data = load_crystal_dataset(REPO / cfg["processed_npz"], ("piezoelectric",))
    train = CrystalDataset(data, load_split(REPO / cfg["split_npz"], "train"))

    targets = _irreps_targets(train, "piezoelectric", "piezoelectric")
    zero_row = np.abs(targets).max(axis=1) == 0

    torch.manual_seed(0)
    preds = trained.predict([train[i].structure for i in range(len(train))])
    mags = np.sqrt((preds**2).sum(axis=1))
    errors = np.abs(preds - targets).mean(axis=1)

    return {
        "n_zero_labelled": int(zero_row.sum()),
        "n_real_tensor": int((~zero_row).sum()),
        "median_violation_on_trained_zeros": float(np.median(mags[zero_row])),
        "false_flag_on_trained_zeros": float((mags[zero_row] > THRESHOLD).mean()),
        "train_mae_zero_rows": float(errors[zero_row].mean()),
        "train_mae_real_rows": float(errors[~zero_row].mean()),
        "mean_abs_target_real_rows": float(np.abs(targets[~zero_row]).mean()),
    }


def collect(cores: list[str]) -> dict:
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

    for core in cores:
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
                if tag == "augmented_so3":
                    controls = []
                    for seed in SEEDS:
                        lbl = f"{core}_{parity}_piezoelectric_seed{seed}"
                        if lbl in runs:
                            controls.append(in_train_control(runs[lbl]))
                    if controls:
                        ctl: dict[str, object] = {
                            k: float(np.mean([c[k] for c in controls])) for k in controls[0]
                        }
                        if len(controls) > 1:
                            ctl.update(
                                {
                                    f"{k}_std": float(np.std([c[k] for c in controls], ddof=1))
                                    for k in controls[0]
                                }
                            )
                        ctl["per_seed"] = controls
                        out.setdefault("in_train_control", {})[core] = ctl
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
    control = d.get("in_train_control", {})

    def arm(core: str, tag: str) -> dict | None:
        return arms.get(f"{core}_{tag}")

    lines = [
        "# E1 — the augmentation rebuttal",
        "",
        "The obvious objection to the headline is: *just train the SO(3) model on centrosymmetric",
        "crystals labelled zero.* This measures what that buys.",
        "",
        f"Training set: 2,649 real piezoelectric tensors + {d['n_augmentation']} fresh",
        "centrosymmetric insulators labelled **exactly zero**, drawn only from space groups",
        f"{d['seen_spacegroups']}. Validation and test partitions are inherited unchanged from the",
        "headline split, and the target normalisation scale is frozen at the un-augmented value",
        "(0.749134), so every number below is directly comparable to the main table. ",
        "SO(3) arms only,",
        "3 seeds, hyperparameters identical to the headline runs.",
        "",
        "## The control that decides what this experiment means",
        "",
        "Before asking whether learned zeros *generalize*, ask whether they are learned at all.",
        "Below: what each model predicts on the very crystals it trained on with ",
        "exact-zero labels.",
        "",
        _CONTROL_HEADER,
        "|---|---|---|---|---|---|",
    ]
    for core, v in control.items():
        lines.append(
            f"| {CORE_LABEL[core]} | {v['median_violation_on_trained_zeros']:.4f} "
            f"| {v['false_flag_on_trained_zeros']:.4f} | {v['train_mae_zero_rows']:.4f} "
            f"| {v['train_mae_real_rows']:.4f} | {v['mean_abs_target_real_rows']:.4f} |"
        )
    lines += [
        "",
        "The zero rows *are* fit better than the real-tensor rows — train MAE ",
        "0.014–0.045 against",
        "0.086–0.128, on targets whose mean component magnitude is 0.145. The model is ",
        "not ignoring",
        "them. **But it still false-flags ~90% of the crystals it was explicitly trained to call",
        "zero.** Gradient descent drives the prediction towards zero; it cannot make it ",
        "zero. An O(3)",
        "model does not have to try: the zero is structural.",
        "",
        "## Transfer: does it help on space groups seen in training?",
        "",
        f"Evaluation splits the untouched OOD 2,000 by space group — **SEEN-SG** ({d['n_seen']}),",
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
        "### Change in false-flag rate, baseline → augmented",
        "",
        "| core | SEEN-SG | UNSEEN-SG |",
        "|---|---|---|",
    ]
    for core in CORES:
        b, g = arm(core, "baseline_so3"), arm(core, "augmented_so3")
        if not (b and g):
            continue
        lines.append(
            f"| {CORE_LABEL[core]} "
            f"| {b['ff_seen_mean']:.4f} → {g['ff_seen_mean']:.4f} "
            f"({g['ff_seen_mean'] - b['ff_seen_mean']:+.4f}) "
            f"| {b['ff_unseen_mean']:.4f} → {g['ff_unseen_mean']:.4f} "
            f"({g['ff_unseen_mean'] - b['ff_unseen_mean']:+.4f}) |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "**Augmentation does not buy the zero.** The false-flag rate on SEEN-SG — the ",
        "space groups",
        "the augmentation was drawn from — falls by between 0.0000 and 0.0206. On ",
        "UNSEEN-SG it falls",
        "by 0.013–0.032. There is no meaningful transfer advantage for the space groups ",
        "the model was",
        "trained on, because there is barely any improvement to transfer.",
        "",
        "What augmentation *does* do is shrink violation magnitudes roughly uniformly, ",
        "by about half",
        "(e.g. NequIP SEEN 0.826 → 0.449, UNSEEN 0.422 → 0.254). The predictions move ",
        "towards zero",
        "everywhere and reach it nowhere. Since the physical answer is exactly zero, a ",
        "factor-of-two",
        "reduction in an impossible quantity is not a fix.",
        "",
        "**A caveat on SEEN vs UNSEEN.** The two subsets differ in composition, not only in space",
        "group: median ‖T‖ is higher on SEEN-SG than on UNSEEN-SG in the *baseline* ",
        "runs too, before",
        "any augmentation exists. The interpretable quantity is therefore the change relative to",
        "baseline within each subset, which is what the table above reports — not the ",
        "SEEN/UNSEEN",
        "difference itself.",
        "",
        "**The O(3) rows were not retrained.** They are quoted from the headline runs, ",
        "and they are",
        "0.0000 on both subsets at machine-zero medians (3e-07 – 3e-06). Their zeros ",
        "hold for any",
        "weights and any training data; that is the entire point, and it is why ",
        "retraining them would",
        "answer nothing.",
        "",
        "**Augmentation is not free of benefits.** Non-centrosymmetric test MAE ",
        "*improves* for every",
        "core (NequIP 0.2405 → 0.2278; EquiformerV2 0.2157 → 0.1774): a thousand extra ",
        "crystals is a",
        "thousand extra crystals. The augmented models are better regressors that still predict",
        "physically impossible values.",
        "",
        "## Off-cycle note",
        "",
        "The design anticipated two outcomes — SEEN-SG false-flags drop substantially ",
        "(fix requires",
        "curated data), or they drop and UNSEEN-SG does not (learned zeros do not generalize). The",
        "measured outcome is the third one the plan flagged: **no drop even on SEEN-SG**, which",
        "triggers the standing rule. See the Supplementary Information's augmentation note.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        render()
        return

    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    fresh = collect(args.cores)
    if merged:
        merged["arms"].update(fresh["arms"])
        merged.setdefault("in_train_control", {}).update(fresh.get("in_train_control", {}))
        for k, v in fresh.items():
            if k not in ("arms", "in_train_control"):
                merged[k] = v
        fresh = merged
    OUT_JSON.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"wrote {OUT_JSON} ({len(fresh['arms'])} arms)")


if __name__ == "__main__":
    main()
