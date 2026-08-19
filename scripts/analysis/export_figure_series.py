"""Export the machine-readable series behind the published figures.

Each series is derived from a record in ``results/`` and written to ``data/figure_series/``, so a
figure panel can be redrawn, or its underlying numbers inspected, without retraining anything:

- ``fig2b_thresholds.csv``            -- false-flag fraction across all 25 thresholds,
                                         idealized coordinate variant (Figure 2b)
- ``fig4b_jacobian_points.csv``       -- per-structure Jacobian even-subspace fractions,
                                         360 points (Figure 4b)
- ``figS1_epoch_curves.csv``          -- false-flag fraction against epoch
                                         (Supplementary Figure 1)
- ``figS2_rutile_sweep.csv``          -- the 33-amplitude rutile polar-distortion sweep
                                         (Supplementary Figure 2)
- ``raw_coordinate_thresholds.csv``   -- false-flag fraction across all 25 thresholds,
                                         raw coordinate variant

    uv run python scripts/analysis/export_figure_series.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIGDATA = REPO / "docs" / "draft" / "figdata"
CORE_LABEL = {
    "nequip": "NequIP",
    "allegro": "Allegro",
    "mace": "MACE",
    "equiformer_v2": "EquiformerV2",
}
ARM = {"o3": "O(3)", "so3": "SO(3)"}


def _write(name: str, header: list[str], rows: list[list]) -> None:
    path = FIGDATA / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def rutile_sweep() -> None:
    """Supplementary Figure 2: mean ||T|| over the 3 seeds per (delta, arm) for the TiO2 sweep."""
    with (REPO / "results" / "symmetry_breaking.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["material"] == "TiO2"]
    acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        acc[(r["delta"], r["core"], r["parity"])].append(float(r["norm_T"]))
    out = [
        [delta, ARM[parity], CORE_LABEL[core], float(np.mean(v))]
        for (delta, core, parity), v in sorted(
            acc.items(), key=lambda kv: (float(kv[0][0]), kv[0][1], kv[0][2])
        )
    ]
    _write("figS2_rutile_sweep.csv", ["delta", "arm", "core", "magnitude"], out)


def jacobian_points() -> None:
    """Figure 4b: every (structure, seed, arm) Jacobian even-subspace fraction."""
    rows = json.loads((REPO / "results" / "jacobian.json").read_text())["rows"]
    out = [
        [
            CORE_LABEL[r["core"]],
            ARM[r["parity"]],
            r["material_id"],
            int(r["run"].rsplit("seed", 1)[1]),
            r["even_energy_fraction"],
        ]
        for r in rows
    ]
    _write("fig4b_jacobian_points.csv", ["core", "arm", "structure", "seed", "value"], out)


def threshold_curves() -> None:
    """False-flag fraction against threshold, in both coordinate variants.

    The idealized variant is Figure 2b.
    """
    with (REPO / "results" / "threshold_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    variants = (("raw", "raw_coordinate_thresholds.csv"), ("idealized", "fig2b_thresholds.csv"))
    for variant, name in variants:
        out = [
            [
                r["threshold"],
                CORE_LABEL[r["core"]],
                ARM[r["parity"]],
                r["false_flag_mean"],
                r["false_flag_std"],
            ]
            for r in rows
            if r["variant"] == variant
        ]
        out.sort(key=lambda x: (float(x[0]), x[1], x[2]))
        _write(name, ["tau", "core", "arm", "false_flag_fraction", "sd"], out)


if __name__ == "__main__":
    FIGDATA.mkdir(parents=True, exist_ok=True)
    rutile_sweep()
    jacobian_points()
    threshold_curves()
