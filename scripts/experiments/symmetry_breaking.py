"""Symmetry-breaking curves: the guarantee turns off exactly when the symmetry does.

Parameterise ``x(delta) = x_parent + delta * dx`` along a [001] polar mode. At ``delta = 0`` the
crystal is centrosymmetric and the piezoelectric tensor is exactly zero by Neumann's principle;
for every ``delta > 0`` it is polar and a response is allowed.

**The choice of parent material decides what this experiment can show.** Running it first on the
textbook perovskites revealed the trap:

* **TiO2 (rutile, P4_2/mnm 136 -> P4_2nm 102).** Rotation subgroup **422**, which admits a rank-3
  invariant. Only *parity* forbids a response at delta = 0, so O(3) and SO(3) separate here.
* **BaTiO3 / PbTiO3 (Pm-3m 221 -> P4mm 99).** Rotation subgroup **432**, which admits *no* rank-3
  invariant. An exactly SO(3)-equivariant model is forced to zero at delta = 0 by rotation alone
  (rotation-subgroup analysis), so both arms start at machine zero and the parity
  effect is invisible. Reported as the
  textbook reference, with that caveat stated -- not as evidence of the parity guarantee.

**Tolerance amendment (V0.2).** spglib's ``symprec`` is a *distance* tolerance. At symprec 1e-3
frames below delta ~ 0.006 are wrongly reported as the centrosymmetric parent, because the maximum
atomic displacement falls under it. Every frame here is verified at symprec 1e-8.

Run per profile, then render:

    uv run --extra nequip --extra data python scripts/experiments/symmetry_breaking.py \\
        --cores nequip allegro equiformer_v2
    uv run --extra mace --extra data python scripts/experiments/symmetry_breaking.py --cores mace
    python scripts/experiments/symmetry_breaking.py --render
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from equiparity.inference import find_piezo_runs, load_trained, seeded_predict
from equiparity.inference.structures import (
    max_displacement_angstrom,
    parent,
    tetragonal_distortion,
)

REPO = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("PARITY_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT_CSV = REPO / "results" / "symmetry_breaking.csv"
OUT_JSON = REPO / "results" / "symmetry_breaking.json"
OUT_MD = REPO / "docs" / "results" / "symmetry_breaking.md"

# TiO2 carries the comparison: its rotation subgroup 422 permits a rank-3 invariant, so only
# parity forbids a response at delta = 0. The perovskites' 432 forbids one by rotation alone, which
# makes both arms start at machine zero there and masks the parity effect entirely
# (rotation-subgroup analysis).
MATERIALS = ["TiO2", "BaTiO3", "PbTiO3"]
PARENT_SG = {"TiO2": 136, "BaTiO3": 221, "PbTiO3": 221}
POLAR_SG = {"TiO2": 102, "BaTiO3": 99, "PbTiO3": 99}
ROTATION_SUBGROUP = {
    "TiO2": "422 (permits rank-3)",
    "BaTiO3": "432 (forbids rank-3)",
    "PbTiO3": "432 (forbids rank-3)",
}
VERIFY_SYMPREC = 1e-8
DRAWS = 5

# Log-spaced tail down to 1e-3 (where SO(3)'s offset dwarfs the physical signal) + linear body.
DELTAS = np.unique(
    np.concatenate([[0.0], np.logspace(-3, -1.3, 8), np.linspace(0.05, 1.2, 24)])
).tolist()


def _spacegroup(structure, symprec: float) -> int:
    import spglib

    cell = (
        structure.cell,
        structure.positions @ np.linalg.inv(structure.cell),
        structure.atomic_numbers,
    )
    return int(spglib.get_symmetry_dataset(cell, symprec=symprec).number)


def _verify(material: str) -> list[dict]:
    """spglib-verify each frame: delta=0 -> centrosymmetric parent; delta>0 -> polar group."""
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


def sweep(cores: list[str]) -> dict:
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
                mags = np.sqrt((draws**2).sum(axis=2))  # (draws, n_delta)
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
                f"{material:8s} {label:38s} |T|(0)={norm[0]:.3e}  "
                f"|T|(1)={norm[DELTAS.index(1.0)] if 1.0 in DELTAS else float('nan'):.3e}"
            )
    return {"verification": verification, "rows": rows, "deltas": DELTAS, "draws": DRAWS}


def render() -> None:
    data = json.loads(OUT_JSON.read_text())
    rows = data["rows"]

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def agg(material: str, core: str, parity: str, delta: float) -> tuple[float, float]:
        vals = [
            r["norm_T"]
            for r in rows
            if r["material"] == material
            and r["core"] == core
            and r["parity"] == parity
            and r["delta"] == delta
        ]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    arms = sorted({(r["core"], r["parity"]) for r in rows})
    key_deltas = [d for d in data["deltas"] if d in (0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0)]

    lines = [
        "# Symmetry-breaking curves",
        "",
        "`x(δ) = x_parent + δ·Δx` along a [001] polar mode. δ = 0 is the ",
        "centrosymmetric parent,",
        "where the piezoelectric tensor is exactly zero; every δ > 0 is polar, where a ",
        "response is",
        "allowed. δ = 1 is the physical distortion amplitude.",
        "",
        f"All {len(data['deltas'])} frames per material were spglib-verified at symprec"
        f" {VERIFY_SYMPREC}.",
        "At the selection tolerance 1e-3 the frames below δ ≈ 0.006 are *wrongly* reported",
        "centrosymmetric — the maximum displacement falls below the tolerance. Same ",
        "phenomenon as",
        "the mp-1227949 raw-coordinate artifact (appendix A5).",
        "",
        "## Which material can show the parity effect",
        "",
        "The parent's **proper-rotation subgroup** decides whether SO(3) equivariance ",
        "alone already",
        "forbids a rank-3 tensor at δ = 0. Where it does, both arms start at machine zero and the",
        "curve says nothing about parity.",
        "",
        "| material | parent | polar | rotation subgroup | separates the arms at δ=0? |",
        "|---|---|---|---|---|",
        "| TiO₂ (rutile) | P4₂/mnm (136) | P4₂nm (102) | 422 — permits rank-3 | **yes** |",
        "| BaTiO₃ | Pm-3̄m (221) | P4mm (99) | 432 — forbids rank-3 | no |",
        "| PbTiO₃ | Pm-3̄m (221) | P4mm (99) | 432 — forbids rank-3 | no |",
        "",
        "Values are ‖T‖_F, mean over 3 seeds. Full curves: `results/symmetry_breaking.csv`.",
        "",
    ]
    for material in MATERIALS:
        lines += [
            f"## {material}  ({ROTATION_SUBGROUP[material]})",
            "",
            "| core | arm | " + " | ".join(f"δ={d:g}" for d in key_deltas) + " |",
            "|---" * (len(key_deltas) + 2) + "|",
        ]
        for core, parity in arms:
            cells = [f"{agg(material, core, parity, d)[0]:.3e}" for d in key_deltas]
            lines.append(f"| {core} | {parity} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Reading",
        "",
        "**TiO₂ is the panel that carries the argument.** The O(3) arms start at ",
        "machine zero and",
        "rise smoothly as inversion breaks: the guarantee switches off exactly when the symmetry",
        "does. The SO(3) arms start from an O(1) offset — a physically impossible response for a",
        "centrosymmetric crystal — and that spurious floor dominates the true signal through the",
        "small-δ regime, which is where displacive ferroelectrics actually live.",
        "",
        "**The perovskite panels do not show this, and that is itself the finding.** Cubic Pm-3̄m",
        "has",
        "rotation subgroup 432, under which no rank-3 tensor is invariant, so the SO(3) arms are",
        "forced to zero at δ = 0 without any parity label. Both arms therefore start ",
        "at machine zero.",
        "Had we run only BaTiO₃ — the material the experiment was originally designed ",
        "around — we",
        "would have concluded that SO(3) tracks the symmetry breaking correctly. It does not; a",
        "cubic parent simply hides the defect (the rotation-subgroup analysis).",
        "",
        "EquiformerV2's values are means over 5 seeded draws; its forward pass is stochastic (the "
        "output-parity audit).",
    ]
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

    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"rows": []}
    fresh = sweep(args.cores)
    keep = [r for r in merged["rows"] if r["core"] not in args.cores]
    fresh["rows"] = keep + fresh["rows"]
    OUT_JSON.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"\nwrote {OUT_JSON} ({len(fresh['rows'])} rows)")
    render()


if __name__ == "__main__":
    main()
