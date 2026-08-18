"""H-1 -- the false-flag-vs-epoch curves from the re-instrumented retrains.

Collects ``ood_false_flag_history`` (idealized variant, every epoch) from the 12 SO(3)
epoch-curve retrains (``dataset: mp_piezoelectric_epochcurve``) synced into the MIRROR, plus,
as a bonus, the T3 learning-curve runs which carry the same instrumentation. Aggregates per
core (mean and min-max band over seeds) and cross-checks every endpoint against the released
headline false-flag fractions.

    uv run python scripts/h1_epoch_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path.home() / "Desktop" / "parity_work"
OUT_JSON = REPO / "results" / "h1_epoch_curve.json"
OUT_MD = REPO / "docs" / "results" / "h1_epoch_curve.md"

# Released headline idealized false-flag fractions, per core per seed (from stats/e4 records).
HEADLINE = {
    ("nequip", 0): 0.8960,
    ("nequip", 1): 0.8955,
    ("nequip", 2): 0.8945,
    ("allegro", 0): 0.9095,
    ("allegro", 1): 0.9110,
    ("allegro", 2): 0.9080,
    ("mace", 0): 0.9085,
    ("mace", 1): 0.9070,
    ("mace", 2): 0.9075,
    ("equiformer_v2", 0): 0.9525,
    ("equiformer_v2", 1): 0.9600,
    ("equiformer_v2", 2): 0.9525,
}


def collect(dataset: str) -> dict[tuple[str, int], list[dict]]:
    latest: dict[tuple[str, int], tuple[str, Path]] = {}
    for mfile in MIRROR.glob("raw/box*/*/metrics.json"):
        try:
            m = json.loads(mfile.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("dataset") != dataset or not m.get("ood_false_flag_history"):
            continue
        core = m["run_label"].split("_so3_")[0]
        seed = int(m["run_label"].rsplit("seed", 1)[1])
        ts = mfile.parent.name.split("_")[-1]
        key = (core, seed)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, mfile.parent)
    return {
        k: json.loads((d / "metrics.json").read_text())["ood_false_flag_history"]
        for k, (_, d) in latest.items()
    }


def main() -> None:
    runs = collect("mp_piezoelectric_epochcurve")
    out: dict = {"per_run": {}, "per_core": {}, "endpoint_check": {}}
    cores = sorted({c for c, _ in runs})
    for (core, seed), hist in sorted(runs.items()):
        ff = [e["false_flag_at_0.01"] for e in hist]
        med = [e["median"] for e in hist]
        out["per_run"][f"{core}_seed{seed}"] = {"false_flag": ff, "median": med}
        head = HEADLINE.get((core, seed))
        if head is not None:
            out["endpoint_check"][f"{core}_seed{seed}"] = {
                "endpoint": ff[-1],
                "headline": head,
                "diff": round(ff[-1] - head, 4),
            }
    for core in cores:
        seeds = [s for c, s in runs if c == core]
        curves = np.array([[e["false_flag_at_0.01"] for e in runs[(core, s)]] for s in seeds])
        out["per_core"][core] = {
            "n_seeds": len(seeds),
            "mean": curves.mean(axis=0).tolist(),
            "min": curves.min(axis=0).tolist(),
            "max": curves.max(axis=0).tolist(),
            "epoch_first_above_0p85": int(np.argmax(curves.mean(axis=0) >= 0.85) + 1),
        }
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")

    lines = [
        "# H-1 — false-flag fraction vs training epoch (re-instrumented retrains)",
        "",
        f"{len(runs)} runs collected. Idealized variant, every epoch, threshold 0.01 C/m².",
        "",
        "| core | seeds | ff@e1 | ff@e10 | ff@e50 | ff@e150 | first epoch ≥0.85 |",
        "|---|---|---|---|---|---|---|",
    ]
    for core in cores:
        m = out["per_core"][core]["mean"]
        lines.append(
            f"| {core} | {out['per_core'][core]['n_seeds']} | {m[0]:.3f} | {m[9]:.3f} "
            f"| {m[49]:.3f} | {m[149]:.3f} | {out['per_core'][core]['epoch_first_above_0p85']} |"
        )
    lines += [
        "",
        "## Endpoint vs released headline (per run)",
        "",
        "| run | endpoint | headline | diff |",
        "|---|---|---|---|",
    ]
    for k, v in sorted(out["endpoint_check"].items()):
        lines.append(f"| {k} | {v['endpoint']:.4f} | {v['headline']:.4f} | {v['diff']:+.4f} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
