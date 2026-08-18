"""T4-3 -- two more polar distortion paths, so rutile TiO2 is not load-bearing alone.

Both parents were chosen for the 432 lesson from E2: their point group is 4/mmm, whose
proper-rotation subgroup 422 admits a rank-3 invariant, so at delta = 0 only parity forbids a
piezoelectric response and the O(3)/SO(3) arms must separate exactly as they do on rutile.

* **SnO2 (cassiterite)** -- the same rutile structure type (P4_2/mnm, 136) with its own cell;
  the polar mode carries it to P4_2nm (102), as for TiO2.
* **Anatase TiO2** (I4_1/amd, 141) -- a different structure type of the same chemistry; the
  [001] polar mode carries it to I4_1md (109).

Every frame is spglib-verified at symprec 1e-8 before prediction, as in E2. Same sweep grid,
same models, same draws.

Run once per install profile:

    uv run --extra nequip python scripts/t4_distortion_materials.py \
        --cores nequip allegro equiformer_v2
    uv run --extra mace   python scripts/t4_distortion_materials.py --cores mace
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import spglib
import torch

from equiparity.inference import find_piezo_runs, load_trained, seeded_predict
from equiparity.inference.structures import (
    max_displacement_angstrom,
    parent,
    tetragonal_distortion,
)

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path(os.environ.get("PARITY_RUNS", Path.home() / "Desktop" / "parity_work"))
OUT_CSV = REPO / "results" / "t4_distortion_materials.csv"
OUT_JSON = REPO / "results" / "t4_distortion_materials.json"
OUT_MD = REPO / "docs" / "results" / "t4_distortion_materials.md"

MATERIALS = ["SnO2", "TiO2_anatase"]
PARENT_SG = {"SnO2": 136, "TiO2_anatase": 141}
POLAR_SG = {"SnO2": 102, "TiO2_anatase": 109}
LABEL = {"SnO2": "SnO2 (rutile-type)", "TiO2_anatase": "TiO2 (anatase)"}
VERIFY_SYMPREC = 1e-8
DRAWS = 5

# Same grid as E2: log-spaced tail + linear body.
DELTAS = np.unique(
    np.concatenate([[0.0], np.logspace(-3, -1.3, 8), np.linspace(0.05, 1.2, 24)])
).tolist()


def _spacegroup(structure, symprec: float) -> int:
    cell = (
        structure.cell,
        structure.positions @ np.linalg.inv(structure.cell),
        structure.atomic_numbers,
    )
    return int(spglib.get_symmetry_dataset(cell, symprec=symprec).number)


def _verify(material: str) -> list[dict]:
    rows = []
    for delta in DELTAS:
        structure = parent(material) if delta == 0 else tetragonal_distortion(material, delta)
        sg = _spacegroup(structure, VERIFY_SYMPREC)
        expected = PARENT_SG[material] if delta == 0 else POLAR_SG[material]
        if sg != expected:
            raise SystemExit(f"{material} delta={delta}: spglib gives {sg}, expected {expected}")
        rows.append(
            {
                "delta": delta,
                "spacegroup": sg,
                "centrosymmetric": delta == 0,
                "max_disp_angstrom": max_displacement_angstrom(material, delta),
            }
        )
    print(f"{material}: all {len(rows)} frames verified at symprec {VERIFY_SYMPREC}")
    return rows


def sweep(cores: list[str]) -> tuple[dict, list[dict]]:
    runs = find_piezo_runs(MIRROR)
    verification = {m: _verify(m) for m in MATERIALS}

    rows: list[dict] = []
    for material in MATERIALS:
        structures = [
            parent(material) if d == 0 else tetragonal_distortion(material, d) for d in DELTAS
        ]
        for label in sorted(runs):
            core = label.split("_o3_")[0].split("_so3_")[0]
            if core not in cores:
                continue
            trained = load_trained(runs[label], repo_root=REPO)
            seed = int(label.rsplit("seed", 1)[1])

            if trained.is_stochastic:
                draws = seeded_predict(trained, structures, draws=DRAWS)
                mags = np.sqrt((draws**2).sum(axis=2))
                norm, norm_std = mags.mean(axis=0), mags.std(axis=0, ddof=1)
            else:
                torch.manual_seed(0)
                preds = trained.predict(structures)
                norm = np.sqrt((preds**2).sum(axis=1))
                norm_std = np.zeros_like(norm)

            for i, delta in enumerate(DELTAS):
                rows.append(
                    {
                        "material": material,
                        "core": core,
                        "parity": trained.parity,
                        "seed": seed,
                        "delta": delta,
                        "norm_T": float(norm[i]),
                        "norm_T_std": float(norm_std[i]),
                        "spacegroup": verification[material][i]["spacegroup"],
                    }
                )
            print(
                f"{material:14s} {label:38s} |T|(0)={norm[0]:.3e}  "
                f"|T|(1)={norm[DELTAS.index(1.0)] if 1.0 in DELTAS else float('nan'):.3e}"
            )
    return verification, rows


def render() -> None:
    data = json.loads(OUT_JSON.read_text())
    rows = data["rows"]

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def agg(material: str, parity: str, delta: float) -> tuple[float, float]:
        vals = [
            r["norm_T"]
            for r in rows
            if r["material"] == material and r["parity"] == parity and r["delta"] == delta
        ]
        if not vals:
            return float("nan"), float("nan")
        return float(np.min(vals)), float(np.max(vals))

    lines = [
        "# T4-3 — additional polar distortion paths (SnO2 rutile-type, anatase TiO2)",
        "",
        "Both parents have point group 4/mmm (rotation subgroup 422 admits a rank-3 invariant),",
        "so only parity forbids a response at δ = 0: the rutile TiO2 separation must reproduce.",
        "Frames verified with spglib at symprec 1e-8: SnO2 136 → 102, anatase 141 → 109.",
        "",
        "| material | arm | ‖T‖ range at δ=0 | ‖T‖ range at δ=1 |",
        "|---|---|---|---|",
    ]
    for m in MATERIALS:
        for parity in ("o3", "so3"):
            lo0, hi0 = agg(m, parity, 0.0)
            lo1, hi1 = agg(m, parity, 1.0)
            lines.append(
                f"| {LABEL[m]} | {parity.upper()} | {lo0:.2e} – {hi0:.2e} | {lo1:.2e} – {hi1:.2e} |"
            )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV}\nwrote {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        render()
        return

    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    verification, rows = sweep(args.cores)
    done = {(r["material"], r["core"], r["parity"], r["seed"], r["delta"]) for r in rows}
    kept = [
        r
        for r in merged.get("rows", [])
        if (r["material"], r["core"], r["parity"], r["seed"], r["delta"]) not in done
    ]
    merged.update(
        {"verification": verification, "rows": kept + rows, "deltas": DELTAS, "draws": DRAWS}
    )
    OUT_JSON.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"wrote {OUT_JSON} ({len(merged['rows'])} rows)")
    render()


if __name__ == "__main__":
    main()
