"""Aggregate the 84-run instrumented grid into paper tables, curves, and figures.

Reads the flattened per-run metrics mirror plus the per-structure OOD violation vectors,
and emits:
  results/threshold_curves.csv        false-flag fraction vs threshold, per arm x OOD variant
  results/tables.md                   mean+-std tables over seeds (accuracy + OOD)
  results/stats.json                  paired significance tests
  results/fig_threshold_curves.png    item-3 figure (curves)
  results/fig_violation_hist.png      item-3 figure (violation distributions)

Statistical note: seed-level paired tests have n=3, so a two-sided Wilcoxon signed-rank
cannot fall below p=0.25 regardless of effect size. Seed-level p-values are therefore
reported as descriptive only; the headline OOD claim is tested at the structure level
(n=2000 centrosymmetric crystals, paired by structure), where the test has real power.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

MIRROR = Path.home() / "Desktop" / "parity_work"
OUT = Path(__file__).resolve().parent.parent / "results"

CORES = ["nequip", "allegro", "mace", "equiformer_v2"]
CORE_LABEL = {
    "nequip": "NequIP",
    "allegro": "Allegro",
    "mace": "MACE",
    "equiformer_v2": "EquiformerV2",
}
TARGETS = ["U0", "dipole", "elastic", "piezoelectric"]
TARGET_LABEL = {
    "U0": "U0 (scalar, even)",
    "dipole": "dipole (vector, odd)",
    "elastic": "elastic (rank-4, even)",
    "piezoelectric": "piezoelectric (rank-3, odd)",
}
VARIANTS = ["idealized", "raw"]
SEEDS = [0, 1, 2]

C_O3, C_SO3 = "#2a78d6", "#e34948"  # validated categorical slots 1 and 6


def load_metrics() -> dict[tuple[str, str, str, int], dict]:
    """label -> metrics dict, for the four-core scope (clifford_stf excluded)."""
    out = {}
    for f in sorted((MIRROR / "metrics").glob("*.json")):
        if f.name.startswith("clifford"):
            continue
        m = json.loads(f.read_text())
        # run_label = <core>_<parity>_<target>_seed<N>; core may contain '_'
        label = m["run_label"]
        parity, target = m["parity"], m["target"]
        seed = int(label.rsplit("seed", 1)[1])
        core = label.split(f"_{parity}_{target}_")[0]
        out[(core, parity, target, seed)] = m
    return out


def load_vectors() -> dict[tuple[str, str, int, str], np.ndarray]:
    """(core,parity,seed,variant) -> per-structure violation magnitudes (piezo runs only)."""
    latest: dict[tuple, tuple[str, Path]] = {}
    for mfile in MIRROR.glob("raw/box*/*/metrics.json"):
        d = mfile.parent
        if not (d / "ood_violations_idealized.npy").exists():
            continue
        try:
            m = json.loads(mfile.read_text())
        except json.JSONDecodeError:
            continue
        label = m.get("run_label", "")
        if not label or label.startswith("clifford"):
            continue
        ts = d.name.split("_")[-1]
        if label not in latest or ts > latest[label][0]:
            latest[label] = (ts, d)

    vecs = {}
    for label, (_, d) in latest.items():
        parity = "o3" if "_o3_" in label else "so3"
        seed = int(label.rsplit("seed", 1)[1])
        core = label.split(f"_{parity}_")[0]
        for v in VARIANTS:
            p = d / f"ood_violations_{v}.npy"
            if p.exists():
                vecs[(core, parity, seed, v)] = np.load(p)
    return vecs


def ms(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def fmt(mean: float, std: float, sig: int = 4) -> str:
    if mean != 0 and abs(mean) < 1e-3:
        return f"{mean:.2e} ± {std:.1e}"
    return f"{mean:.{sig}f} ± {std:.{sig}f}"


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def accuracy_table(runs_by_key) -> tuple[str, dict]:
    lines = [
        "| Core | Target | O(3) test MAE | SO(3) test MAE | Δ (SO3−O3) | Δ / seed-σ |",  # noqa: RUF001
        "|---|---|---|---|---|---|",
    ]
    stats = {}
    for core in CORES:
        for t in TARGETS:
            o3 = [
                runs_by_key[(core, "o3", t, s)]["test"]["mae"]
                for s in SEEDS
                if (core, "o3", t, s) in runs_by_key
            ]
            so3 = [
                runs_by_key[(core, "so3", t, s)]["test"]["mae"]
                for s in SEEDS
                if (core, "so3", t, s) in runs_by_key
            ]
            if not so3:
                continue
            so3_m, so3_s = ms(so3)
            if not o3:
                lines.append(
                    f"| {CORE_LABEL[core]} | {t} | - (SO(3)-only) | {fmt(so3_m, so3_s)} | - | - |"
                )
                continue
            o3_m, o3_s = ms(o3)
            delta = so3_m - o3_m
            pooled = float(np.sqrt((np.var(o3, ddof=1) + np.var(so3, ddof=1)) / 2))
            ratio = delta / pooled if pooled > 0 else float("nan")
            # seed-level paired Wilcoxon: n=3, floor p=0.25 -> descriptive only
            try:
                p = float(wilcoxon(o3, so3).pvalue)
            except ValueError:
                p = float("nan")
            stats[f"{core}/{t}"] = dict(
                o3_mean=o3_m,
                o3_std=o3_s,
                so3_mean=so3_m,
                so3_std=so3_s,
                delta=delta,
                delta_over_seed_sigma=ratio,
                wilcoxon_seed_p=p,
                n_seeds=len(o3),
            )
            lines.append(
                f"| {CORE_LABEL[core]} | {t} | {fmt(o3_m, o3_s)} | {fmt(so3_m, so3_s)} "
                f"| {delta:+.4f} | {ratio:+.2f} |"
            )
    return "\n".join(lines), stats


def ood_table(runs_by_key) -> tuple[str, dict]:
    lines = [
        "| Core | Parity | Variant | false-flag @1e-2 | median violation | max violation |",
        "|---|---|---|---|---|---|",
    ]
    stats = {}
    for core in CORES:
        for parity in ["o3", "so3"]:
            runs = [
                runs_by_key[(core, parity, "piezoelectric", s)]
                for s in SEEDS
                if (core, parity, "piezoelectric", s) in runs_by_key
            ]
            if not runs:
                continue
            for v in VARIANTS:
                ff = [r["ood_variants"][v]["false_flag_at_0.01"] for r in runs]
                med = [r["ood_variants"][v]["median"] for r in runs]
                mx = [r["ood_variants"][v]["max"] for r in runs]
                ff_m, ff_s = ms(ff)
                md_m, md_s = ms(med)
                mx_m, _ = ms(mx)
                stats[f"{core}/{parity}/{v}"] = dict(
                    false_flag_mean=ff_m,
                    false_flag_std=ff_s,
                    median_mean=md_m,
                    median_std=md_s,
                    max_mean=mx_m,
                )
                lines.append(
                    f"| {CORE_LABEL[core]} | {parity.upper()} | {v} | {fmt(ff_m, ff_s)} "
                    f"| {fmt(md_m, md_s)} | {mx_m:.3g} |"
                )
    return "\n".join(lines), stats


def variant_shift(runs_by_key) -> tuple[str, dict]:
    """Does either parity mode degrade going from idealized to raw DFT-relaxed OOD?"""
    lines = ["| Core | Parity | ff(idealized) | ff(raw) | Δ |", "|---|---|---|---|---|"]
    stats = {}
    for core in CORES:
        for parity in ["o3", "so3"]:
            runs = [
                runs_by_key[(core, parity, "piezoelectric", s)]
                for s in SEEDS
                if (core, parity, "piezoelectric", s) in runs_by_key
            ]
            if not runs:
                continue
            i = [r["ood_variants"]["idealized"]["false_flag_at_0.01"] for r in runs]
            r_ = [r["ood_variants"]["raw"]["false_flag_at_0.01"] for r in runs]
            im, isd = ms(i)
            rm, rsd = ms(r_)
            stats[f"{core}/{parity}"] = dict(idealized=im, raw=rm, delta=rm - im)
            lines.append(
                f"| {CORE_LABEL[core]} | {parity.upper()} | {fmt(im, isd)} "
                f"| {fmt(rm, rsd)} | {rm - im:+.5f} |"
            )
    return "\n".join(lines), stats


def structure_level_tests(vecs) -> dict:
    """Paired Wilcoxon over the 2000 OOD structures: O(3) vs SO(3) violation magnitude."""
    out = {}
    for core in ["nequip", "allegro", "mace"]:  # cores with both arms
        for v in VARIANTS:
            ps, stat_desc = [], []
            for s in SEEDS:
                a = vecs.get((core, "o3", s, v))
                b = vecs.get((core, "so3", s, v))
                if a is None or b is None:
                    continue
                res = wilcoxon(a, b, alternative="less")  # O(3) violations < SO(3)
                ps.append(float(res.pvalue))
                stat_desc.append(
                    dict(
                        seed=s,
                        n=int(a.size),
                        o3_median=float(np.median(a)),
                        so3_median=float(np.median(b)),
                        frac_o3_lt_so3=float(np.mean(a < b)),
                    )
                )
            if ps:
                out[f"{core}/{v}"] = dict(
                    per_seed_p=ps,
                    max_p=max(ps),
                    detail=stat_desc,
                    alternative="o3 violation < so3 violation",
                )
    return out


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def curves_csv_and_fig(runs_by_key) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["core,parity,variant,threshold,false_flag_mean,false_flag_std"]
    curve = {}
    for core in CORES:
        for parity in ["o3", "so3"]:
            runs = [
                runs_by_key[(core, parity, "piezoelectric", s)]
                for s in SEEDS
                if (core, parity, "piezoelectric", s) in runs_by_key
            ]
            if not runs:
                continue
            for v in VARIANTS:
                thr = np.asarray(runs[0]["ood_variants"][v]["thresholds"])
                ffs = np.stack(
                    [np.asarray(r["ood_variants"][v]["false_flag_fraction"]) for r in runs]
                )
                mean, std = ffs.mean(0), ffs.std(0, ddof=1)
                curve[(core, parity, v)] = (thr, mean, std)
                for t, m_, s_ in zip(thr, mean, std, strict=True):
                    rows.append(f"{core},{parity},{v},{t:.6g},{m_:.6f},{s_:.6f}")
    (OUT / "threshold_curves.csv").write_text("\n".join(rows) + "\n")

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), sharey=True)
    for ax, core in zip(axes, CORES, strict=True):
        for parity, color in (("o3", C_O3), ("so3", C_SO3)):
            for v, ls in (("idealized", "-"), ("raw", "--")):
                if (core, parity, v) not in curve:
                    continue
                thr, mean, std = curve[(core, parity, v)]
                ax.plot(thr, mean, ls, color=color, lw=2, zorder=3, label=f"{parity.upper()} · {v}")
                ax.fill_between(thr, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.axvline(1e-2, color="#9a9a94", lw=1, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(CORE_LABEL[core], fontsize=11)
        ax.set_xlabel("violation threshold  (C/m$^2$)")
        ax.grid(alpha=0.18, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("false-flag fraction")
    # direct labels so identity is not color-alone
    axes[0].annotate("SO(3)", (1.2e-4, 0.86), color=C_SO3, fontsize=10, fontweight="bold")
    axes[0].annotate("O(3)", (1.2e-4, 0.06), color=C_O3, fontsize=10, fontweight="bold")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center right", frameon=False, fontsize=9)
    fig.suptitle(
        "False-flag fraction vs violation threshold — 2000 centrosymmetric crystals "
        "(mean +/- s.d. over 3 seeds)",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(OUT / "fig_threshold_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_threshold_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def hist_fig(vecs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.logspace(-14, 2, 60)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), sharey=True)
    for ax, core in zip(axes, CORES, strict=True):
        for parity, color in (("o3", C_O3), ("so3", C_SO3)):
            pooled = [
                vecs[(core, parity, s, "idealized")]
                for s in SEEDS
                if (core, parity, s, "idealized") in vecs
            ]
            if not pooled:
                continue
            a = np.clip(np.concatenate(pooled), 1e-14, None)
            ax.hist(a, bins=bins, color=color, alpha=0.6, label=parity.upper(), edgecolor="none")
        ax.axvline(1e-2, color="#9a9a94", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_title(CORE_LABEL[core], fontsize=11)
        ax.set_xlabel("|predicted piezo tensor|  (C/m$^2$)")
        ax.grid(alpha=0.18, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("crystals")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Predicted piezoelectric magnitude on centrosymmetric crystals "
        "(true value = 0 exactly; 3 seeds pooled, idealized OOD set)",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_violation_hist.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_violation_hist.pdf", bbox_inches="tight")
    plt.close(fig)


def timing_table(runs_by_key) -> str:
    lines = [
        "| Core | train s/epoch | throughput (struct/s) | peak GPU (MB) | OOD eval (s) |",
        "|---|---|---|---|---|",
    ]
    for core in CORES:
        rows = [
            m
            for (c, _, t, _), m in runs_by_key.items()
            if c == core and t == "piezoelectric" and m.get("timing")
        ]
        if not rows:
            continue
        spe = ms([r["timing"]["train_seconds_per_epoch"] for r in rows])
        thr = ms([r["timing"]["train_throughput_structs_per_s"] for r in rows])
        mem = ms([r["timing"]["peak_gpu_mem_mb"] for r in rows])
        ood = ms([r["timing"].get("ood_seconds", 0.0) for r in rows])
        lines.append(
            f"| {CORE_LABEL[core]} | {spe[0]:.1f} ± {spe[1]:.1f} "
            f"| {thr[0]:.1f} ± {thr[1]:.1f} | {mem[0]:.0f} | {ood[0]:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = load_metrics()
    vecs = load_vectors()
    print(f"loaded {len(runs)} runs, {len(vecs)} violation vectors")
    assert len(runs) == 84, f"expected 84 runs, got {len(runs)}"

    acc_md, acc_stats = accuracy_table(runs)
    ood_md, ood_stats = ood_table(runs)
    shift_md, shift_stats = variant_shift(runs)
    struct = structure_level_tests(vecs)
    tim_md = timing_table(runs)

    curves_csv_and_fig(runs)
    hist_fig(vecs)

    (OUT / "tables.md").write_text(
        "# Aggregated tables (mean +/- s.d. over 3 seeds)\n\n"
        "## Accuracy (test MAE)\n\n"
        + acc_md
        + "\n\n## OOD piezoelectric — both variants\n\n"
        + ood_md
        + "\n\n## Idealized vs raw OOD shift\n\n"
        + shift_md
        + "\n\n## Timing (piezoelectric runs)\n\n"
        + tim_md
        + "\n"
    )
    (OUT / "stats.json").write_text(
        json.dumps(
            dict(
                accuracy=acc_stats,
                ood=ood_stats,
                variant_shift=shift_stats,
                structure_level_wilcoxon=struct,
                note=(
                    "seed-level Wilcoxon has n=3 -> two-sided p floor = 0.25; descriptive only. "
                    "Headline OOD claim tested at structure level (n=2000, paired)."
                ),
            ),
            indent=2,
        )
    )
    print(acc_md)
    print()
    print(ood_md)
    print()
    print(shift_md)
    print()
    print("structure-level paired Wilcoxon (O(3) < SO(3) violation):")
    for k, v in struct.items():
        print(
            f"  {k}: max_p={v['max_p']:.3e}  frac(o3<so3)="
            f"{np.mean([d['frac_o3_lt_so3'] for d in v['detail']]):.4f}"
        )


if __name__ == "__main__":
    main()
