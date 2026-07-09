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


# --------------------------------------------------------------------------------------
# appendix analyses (added for the paper write-up)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "results"
PIEZO_NPZ = ROOT / "data" / "raw" / "mp" / "mp_piezoelectric_processed.npz"
PIEZO_SPLIT = ROOT / "data" / "splits" / "mp_piezoelectric_split.npz"
OOD_NPZ = ROOT / "data" / "raw" / "mp" / "mp_ood_centrosymmetric_processed.npz"

# Which GPU ran which core (from the per-run manifests); timing is not comparable across classes.
GPU_OF_CORE = {
    "nequip": "RTX 5090",
    "allegro": "RTX 5090",
    "equiformer_v2": "RTX 5090",
    "mace": "RTX PRO 6000 Blackwell WS",
}
# vast.ai offer-selection ceiling; the price actually paid was never persisted.
MAX_DPH = 1.00


def capacity_table(runs_by_key) -> tuple[str, dict]:
    """Parameter counts per arm. SO(3) is never smaller than O(3), so capacity is not the cause."""
    lines = ["| Core | Target | O(3) params | SO(3) params | SO(3)/O(3) |", "|---|---|---|---|---|"]
    stats = {}
    for core in CORES:
        for t in TARGETS:
            o3 = [
                runs_by_key[(core, "o3", t, s)]["n_params"]
                for s in SEEDS
                if (core, "o3", t, s) in runs_by_key
            ]
            so3 = [
                runs_by_key[(core, "so3", t, s)]["n_params"]
                for s in SEEDS
                if (core, "so3", t, s) in runs_by_key
            ]
            if not so3:
                continue
            if not o3:
                lines.append(f"| {CORE_LABEL[core]} | {t} | - (SO(3)-only) | {so3[0]:,} | - |")
                continue
            ratio = so3[0] / o3[0]
            stats[f"{core}/{t}"] = dict(o3=o3[0], so3=so3[0], ratio=ratio)
            lines.append(f"| {CORE_LABEL[core]} | {t} | {o3[0]:,} | {so3[0]:,} | {ratio:.3f} |")
    return "\n".join(lines), stats


def accuracy_full_table(runs_by_key) -> str:
    """Validation and test, MAE and RMSE, mean +/- s.d. over seeds."""
    lines = [
        "| Core | Parity | Target | val MAE | val RMSE | test MAE | test RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for core in CORES:
        for parity in ("o3", "so3"):
            for t in TARGETS:
                rows = [
                    runs_by_key[(core, parity, t, s)]
                    for s in SEEDS
                    if (core, parity, t, s) in runs_by_key
                ]
                if not rows:
                    continue
                cells = [
                    fmt(*ms([r[split][metric] for r in rows]))
                    for split in ("val", "test")
                    for metric in ("mae", "rmse")
                ]
                vm, vr, tm, tr = cells[0], cells[1], cells[2], cells[3]
                lines.append(
                    f"| {CORE_LABEL[core]} | {parity.upper()} | {t} | {vm} | {vr} | {tm} | {tr} |"
                )
    return "\n".join(lines)


def target_calibration() -> tuple[str, dict]:
    """The real piezoelectric tensor magnitudes, for scale reference against OOD violations."""
    data = np.load(PIEZO_NPZ, allow_pickle=True)
    mags = np.sqrt((np.asarray(data["piezoelectric"], dtype=float) ** 2).sum(axis=1))
    split = np.load(PIEZO_SPLIT, allow_pickle=True)
    ids = np.asarray([str(x) for x in data["ids"]])
    train_ids = np.asarray([str(x) for x in split["train"]])
    mask = np.isin(ids, train_ids)
    if not mask.any():  # split stored as positional indices
        mask = np.zeros(ids.size, dtype=bool)
        mask[np.asarray(split["train"], dtype=int)] = True

    stats, lines = (
        {},
        [
            "| Subset | n | p5 | median | p95 | max | fraction > 0.01 C/m² |",
            "|---|---|---|---|---|---|---|",
        ],
    )
    for label, sel in (("train split", mask), ("all labelled", np.ones(ids.size, dtype=bool))):
        m = mags[sel]
        p5, p50, p95 = (float(x) for x in np.percentile(m, [5, 50, 95]))
        frac = float((m > 0.01).mean())
        stats[label] = dict(
            n=int(sel.sum()),
            p5=p5,
            median=p50,
            p95=p95,
            max=float(m.max()),
            fraction_above_0p01=frac,
        )
        lines.append(
            f"| {label} | {int(sel.sum())} | {p5:.4f} | {p50:.4f} | {p95:.4f} "
            f"| {m.max():.3f} | {frac:.4f} |"
        )
    return "\n".join(lines), stats


def violation_agreement(vecs) -> tuple[str, dict]:
    """Is the SO(3) violation a property of the structure, or of the model/seed?"""
    from scipy.stats import spearmanr

    stats: dict[str, object] = {}
    seed_lines = ["| Core | mean seed-pair Spearman rho | min |", "|---|---|---|"]
    for core in CORES:
        vs = [
            vecs[(core, "so3", s, "idealized")]
            for s in SEEDS
            if (core, "so3", s, "idealized") in vecs
        ]
        if len(vs) < 2:
            continue
        rs = [
            float(spearmanr(vs[i], vs[j]).statistic)
            for i in range(len(vs))
            for j in range(i + 1, len(vs))
        ]
        stats[f"seed_rho/{core}"] = dict(mean=float(np.mean(rs)), min=float(min(rs)))
        seed_lines.append(f"| {CORE_LABEL[core]} | {np.mean(rs):.3f} | {min(rs):.3f} |")

    cross_lines = [
        "| Core A | Core B | Spearman rho (mean over seeds) | flag-set Jaccard (mean) | (min) |",
        "|---|---|---|---|---|",
    ]
    for i in range(len(CORES)):
        for j in range(i + 1, len(CORES)):
            a_core, b_core = CORES[i], CORES[j]
            rhos, jacs = [], []
            for s in SEEDS:
                a = vecs.get((a_core, "so3", s, "idealized"))
                b = vecs.get((b_core, "so3", s, "idealized"))
                if a is None or b is None:
                    continue
                rhos.append(float(spearmanr(a, b).statistic))
                fa, fb = a > 0.01, b > 0.01
                jacs.append(float((fa & fb).sum() / (fa | fb).sum()))
            if not rhos:
                continue
            stats[f"cross/{a_core}-{b_core}"] = dict(
                rho_mean=float(np.mean(rhos)),
                jaccard_mean=float(np.mean(jacs)),
                jaccard_min=float(min(jacs)),
            )
            cross_lines.append(
                f"| {CORE_LABEL[a_core]} | {CORE_LABEL[b_core]} | {np.mean(rhos):+.3f} "
                f"| {np.mean(jacs):.3f} | {min(jacs):.3f} |"
            )
    md = (
        "### Seed-to-seed reproducibility of the SO(3) violation vector\n\n"
        + "\n".join(seed_lines)
        + "\n\n### Agreement between independent SO(3) architectures\n\n"
        + "\n".join(cross_lines)
    )
    return md, stats


def o3_floor_table(vecs) -> tuple[str, dict]:
    """The O(3) residual, pooled over seeds -- a numerical-precision floor, not a symmetry error."""
    lines = ["| Core | median | p95 | max | (idealized, 3 seeds pooled) |", "|---|---|---|---|---|"]
    stats = {}
    for core in ("nequip", "allegro", "mace"):
        pooled = [
            vecs[(core, "o3", s, "idealized")]
            for s in SEEDS
            if (core, "o3", s, "idealized") in vecs
        ]
        if not pooled:
            continue
        a = np.concatenate(pooled)
        med, p95, mx = float(np.median(a)), float(np.percentile(a, 95)), float(a.max())
        stats[core] = dict(median=med, p95=p95, max=mx)
        lines.append(f"| {CORE_LABEL[core]} | {med:.3e} | {p95:.3e} | {mx:.3e} | |")
    return "\n".join(lines), stats


def size_dependence(vecs) -> tuple[str, dict]:
    """||T|| sums over atoms, so it is extensive: a fixed absolute threshold is size-dependent."""
    from scipy.stats import spearmanr

    n_atoms = np.asarray(np.load(OOD_NPZ, allow_pickle=True)["n_atoms"], dtype=float)
    lines = ["| Core | Parity | Spearman rho(violation, n_atoms) |", "|---|---|---|"]
    stats = {}
    for core in CORES:
        for parity in ("o3", "so3"):
            vs = [
                vecs[(core, parity, s, "idealized")]
                for s in SEEDS
                if (core, parity, s, "idealized") in vecs
            ]
            if not vs:
                continue
            rho = float(spearmanr(np.mean(vs, axis=0), n_atoms).statistic)
            stats[f"{core}/{parity}"] = rho
            lines.append(f"| {CORE_LABEL[core]} | {parity.upper()} | {rho:+.3f} |")
    return "\n".join(lines), stats


def compute_table(runs_by_key) -> tuple[str, dict]:
    """Measured compute. Reported per GPU class; the two classes are not interchangeable."""
    total_train = sum(m["timing"]["train_seconds"] for m in runs_by_key.values())
    total_eval = sum(m["timing"]["eval_seconds"] for m in runs_by_key.values())
    total_ood = sum(m["timing"].get("ood_seconds", 0.0) for m in runs_by_key.values())

    per_core, lines = (
        {},
        [
            "| Core | GPU | runs | train (h) | s/epoch | throughput (struct/s) "
            "| peak GPU (MB) | eval (s) | OOD (s) |",
            "|---|---|---|---|---|---|---|---|---|",
        ],
    )
    for core in CORES:
        rows = [m for (c, _, _, _), m in runs_by_key.items() if c == core]
        t = [r["timing"] for r in rows]
        train_h = sum(x["train_seconds"] for x in t) / 3600
        spe = ms([x["train_seconds_per_epoch"] for x in t])
        thr = ms([x["train_throughput_structs_per_s"] for x in t])
        mem = ms([x["peak_gpu_mem_mb"] for x in t])
        ev = ms([x["eval_seconds"] for x in t])
        od = ms([x.get("ood_seconds", 0.0) for x in t])
        per_core[core] = dict(
            gpu=GPU_OF_CORE[core],
            runs=len(rows),
            train_hours=train_h,
            s_per_epoch=spe[0],
            throughput=thr[0],
            peak_gpu_mb=mem[0],
            eval_seconds=ev[0],
            ood_seconds=od[0],
        )
        lines.append(
            f"| {CORE_LABEL[core]} | {GPU_OF_CORE[core]} | {len(rows)} | {train_h:.2f} "
            f"| {spe[0]:.1f} ± {spe[1]:.1f} | {thr[0]:.1f} ± {thr[1]:.1f} | {mem[0]:.0f} "
            f"| {ev[0]:.1f} | {od[0]:.1f} |"
        )

    by_gpu: dict[str, float] = {}
    for rec in per_core.values():
        by_gpu[rec["gpu"]] = by_gpu.get(rec["gpu"], 0.0) + rec["train_hours"]
    gpu_lines = ["| GPU | runs | train (h) |", "|---|---|---|"]
    for gpu in sorted(by_gpu):
        n = sum(r["runs"] for r in per_core.values() if r["gpu"] == gpu)
        gpu_lines.append(f"| {gpu} | {n} | {by_gpu[gpu]:.2f} |")

    stats = dict(
        total_train_seconds=total_train,
        total_train_hours=total_train / 3600,
        total_eval_seconds=total_eval,
        total_ood_seconds=total_ood,
        per_core=per_core,
        train_hours_by_gpu=by_gpu,
        cost_upper_bound_usd=(total_train / 3600) * MAX_DPH,
        cost_note=(
            "price actually paid was not recorded; launch_grid.sh selects the cheapest "
            f"offer with dph < ${MAX_DPH:.2f}/hr, so this is an upper bound, not an estimate"
        ),
    )
    md = (
        f"Total training: **{total_train:,.1f} s = {total_train / 3600:.2f} GPU-hours** "
        "across 84 runs.\n"
        f"Total evaluation: {total_eval:,.1f} s. Total OOD evaluation: {total_ood:,.1f} s.\n\n"
        "### By GPU class\n\n" + "\n".join(gpu_lines) + "\n\n"
        "The two GPU classes are not interchangeable; per-core wall-clock is not a like-for-like\n"
        "architecture comparison.\n\n### Per core\n\n" + "\n".join(lines)
    )
    return md, stats


def threshold_tables(runs_by_key) -> str:
    """The full 25-point false-flag curve as markdown, per arm and variant."""
    out = []
    for variant in VARIANTS:
        rows, thr = {}, None
        for core in CORES:
            for parity in ("o3", "so3"):
                runs = [
                    runs_by_key[(core, parity, "piezoelectric", s)]
                    for s in SEEDS
                    if (core, parity, "piezoelectric", s) in runs_by_key
                ]
                if not runs:
                    continue
                thr = np.asarray(runs[0]["ood_variants"][variant]["thresholds"])
                ff = np.stack(
                    [np.asarray(r["ood_variants"][variant]["false_flag_fraction"]) for r in runs]
                )
                rows[f"{CORE_LABEL[core]} {parity.upper()}"] = ff.mean(0)
        header = "| threshold (C/m²) | " + " | ".join(rows) + " |"
        sep = "|---" * (len(rows) + 1) + "|"
        body = [
            f"| {t:.3e} | " + " | ".join(f"{v[i]:.4f}" for v in rows.values()) + " |"
            for i, t in enumerate(thr)
        ]
        out.append(f"## {variant}\n\n" + "\n".join([header, sep, *body]))
    return "\n\n".join(out)


def distribution_tables(runs_by_key) -> str:
    """Percentiles / IQR / max of the violation magnitude, per arm and variant."""
    lines = [
        "| Core | Parity | Variant | n | p5 | p25 | median | p75 | p95 | max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for core in CORES:
        for parity in ("o3", "so3"):
            for variant in VARIANTS:
                runs = [
                    runs_by_key[(core, parity, "piezoelectric", s)]
                    for s in SEEDS
                    if (core, parity, "piezoelectric", s) in runs_by_key
                ]
                if not runs:
                    continue
                pcts = np.mean(
                    [r["ood_variants"][variant]["percentiles_5_25_50_75_95"] for r in runs], axis=0
                )
                mx = np.mean([r["ood_variants"][variant]["max"] for r in runs])
                n = runs[0]["ood_variants"][variant]["n"]
                cells = " | ".join(f"{p:.3e}" for p in pcts)
                lines.append(
                    f"| {CORE_LABEL[core]} | {parity.upper()} | {variant} | {n} | {cells} "
                    f"| {mx:.3e} |"
                )
    return "\n".join(lines)


def per_seed_table(runs_by_key) -> str:
    """Every run, every headline metric -- the reproducibility dump."""
    lines = [
        "| Core | Parity | Target | Seed | params | epochs | val MAE | test MAE | test RMSE "
        "| ff(idealized) | ff(raw) | train (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for core in CORES:
        for parity in ("o3", "so3"):
            for t in TARGETS:
                for s in SEEDS:
                    m = runs_by_key.get((core, parity, t, s))
                    if m is None:
                        continue
                    ov = m.get("ood_variants") or {}
                    ffi = ov.get("idealized", {}).get("false_flag_at_0.01")
                    ffr = ov.get("raw", {}).get("false_flag_at_0.01")
                    f = lambda x: "-" if x is None else f"{x:.5f}"  # noqa: E731
                    lines.append(
                        f"| {CORE_LABEL[core]} | {parity.upper()} | {t} | {s} | {m['n_params']:,} "
                        f"| {m['epochs_run']} | {m['val']['mae']:.5g} | {m['test']['mae']:.5g} "
                        f"| {m['test']['rmse']:.5g} | {f(ffi)} | {f(ffr)} "
                        f"| {m['timing']['train_seconds']:.0f} |"
                    )
    return "\n".join(lines)


def write_appendices(runs, vecs, extra) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    gen = "Generated by `scripts/analyze_results.py`. Do not edit by hand.\n"
    (DOCS / "a1_per_seed.md").write_text(
        f"# A1 — Per-run results (all 84 runs)\n\n{gen}\n" + per_seed_table(runs) + "\n"
    )
    (DOCS / "a2_threshold_curves.md").write_text(
        f"# A2 — False-flag fraction vs threshold\n\n{gen}\n"
        "Mean over 3 seeds. Machine-readable form: `results/threshold_curves.csv`.\n\n"
        + threshold_tables(runs)
        + "\n"
    )
    (DOCS / "a3_distributions.md").write_text(
        f"# A3 — Violation-magnitude distributions\n\n{gen}\n"
        "Per-structure ||T||_F on the 2000 centrosymmetric crystals, whose true tensor is zero.\n\n"
        "## Percentiles (mean over seeds)\n\n"
        + distribution_tables(runs)
        + "\n\n## O(3) residual floor\n\n"
        + extra["o3_floor_md"]
        + "\n\n## Scale reference: the real piezoelectric tensors\n\n"
        + extra["calibration_md"]
        + "\n\n## Structure- vs model-dependence\n\n"
        + extra["agreement_md"]
        + "\n\n## Size dependence of the metric\n\n"
        + extra["size_md"]
        + "\n"
    )
    (DOCS / "a4_compute.md").write_text(
        f"# A4 — Compute resources\n\n{gen}\n" + extra["compute_md"] + "\n"
    )


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
    # Appendix analyses. Written to a separate json so results/stats.json stays byte-identical.
    cap_md, cap_stats = capacity_table(runs)
    calib_md, calib_stats = target_calibration()
    agree_md, agree_stats = violation_agreement(vecs)
    floor_md, floor_stats = o3_floor_table(vecs)
    size_md, size_stats = size_dependence(vecs)
    comp_md, comp_stats = compute_table(runs)

    write_appendices(
        runs,
        vecs,
        dict(
            o3_floor_md=floor_md,
            calibration_md=calib_md,
            agreement_md=agree_md,
            size_md=size_md,
            compute_md=comp_md,
        ),
    )
    (OUT / "appendix_stats.json").write_text(
        json.dumps(
            dict(
                capacity=cap_stats,
                target_calibration=calib_stats,
                violation_agreement=agree_stats,
                o3_floor=floor_stats,
                size_dependence=size_stats,
                compute=comp_stats,
            ),
            indent=2,
        )
    )
    (OUT / "tables_extra.md").write_text(
        "# Supplementary tables\n\n## Parameter counts (capacity control)\n\n"
        + cap_md
        + "\n\n## Accuracy: val and test, MAE and RMSE\n\n"
        + accuracy_full_table(runs)
        + "\n"
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
    print()
    print(comp_md.splitlines()[0])


if __name__ == "__main__":
    main()
