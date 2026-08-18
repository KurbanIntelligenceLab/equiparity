"""Export the three figure series that are not tabulated in the manuscript.

The figure generators rebuild every panel from the Supplementary Tables; three
series exist only in the committed ``results/`` artifacts and must be exported to
``data/figure_series/`` before the corresponding panels can be drawn:

- ``fig5a_rutile_sweep.csv``   -- the 33-amplitude rutile polar-distortion sweep (from E2)
- ``fig5b_jacobian_points.csv`` -- per-structure Jacobian even-fractions, 360 points (from E3)
- ``figS1_raw_thresholds.csv``  -- false-flag vs threshold, raw coordinate variant

Plus the optional ``fig2b_thresholds.csv`` (all 25 idealized thresholds; the manuscript
tabulates only five).

    uv run python scripts/export_figdata.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
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
    """fig5a: mean ||T|| over the 3 seeds per (delta, arm) for the TiO2 sweep."""
    with (REPO / "results" / "e2_symmetry_breaking.csv").open() as f:
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
    _write("fig5a_rutile_sweep.csv", ["delta", "arm", "core", "magnitude"], out)


def jacobian_points() -> None:
    """fig5b: every (structure, seed, arm) Jacobian even-subspace fraction from E3."""
    rows = json.loads((REPO / "results" / "e3_jacobian.json").read_text())["rows"]
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
    _write("fig5b_jacobian_points.csv", ["core", "arm", "structure", "seed", "value"], out)


def threshold_curves() -> None:
    """figS1 (raw variant) and fig2b (idealized variant): false-flag fraction vs threshold."""
    with (REPO / "results" / "threshold_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    variants = (("raw", "figS1_raw_thresholds.csv"), ("idealized", "fig2b_thresholds.csv"))
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
