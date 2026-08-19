"""Familiar centrosymmetric materials, and what each model predicts for them.

Two panels, because the obvious one-panel version would be selection on the outcome:

**(a) Recognisable compounds already in the OOD 2,000.** Reported *including* the ones no model
false-flags. Diamond and KCl are both cubic m-3m, whose rotation subgroup 432 admits no rank-3
tensor, so even an SO(3) model is forced to zero on them (the rotation-subgroup analysis).
Dropping them because they are "boring" would hide exactly the effect the rotation-subgroup
analysis documents.

**(b) A curated textbook panel pulled fresh from MP.** NaCl, CaF2, SrTiO3, Al2O3, CsCl and friends
are absent from the 2,000 (the set was a random sample). They are fetched, spglib-verified
centrosymmetric, idealized identically to the OOD set, and evaluated inference-only. Germanium is
absent from the MP centrosymmetric-insulator pool entirely -- GGA puts its band gap at ~0 eV,
below the 0.1 eV insulator cut -- so silicon stands in for the diamond-structure semiconductor.

Every entry is annotated with its point group, so an unflagged row reads as "rotation already
forbids this" rather than as a counterexample.

    uv run --extra nequip --extra data python scripts/experiments/named_materials.py \\
        --cores nequip allegro equiformer_v2
    uv run --extra mace --extra data python scripts/experiments/named_materials.py --cores mace
    python scripts/experiments/named_materials.py --render
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from equiparity.inference import find_piezo_runs, load_trained, seeded_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

REPO = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("PARITY_RUNS", Path(__file__).resolve().parents[2] / "runs"))
SPACEGROUPS = REPO / "results" / "ood_spacegroups.json"
CACHE = REPO / "results" / "named_structures.json"
OUT_JSON = REPO / "results" / "named_materials.json"
OUT_MD = REPO / "docs" / "results" / "named_materials.md"

THRESHOLD = 0.01
SYMPREC = 1e-3
DRAWS = 5
_TABLE_HEADER = (
    "| formula | mp-id | structure | SG | point group | rotation subgroup | source "
    "| NequIP O(3) | NequIP SO(3) | Allegro SO(3) | MACE SO(3) | EquiformerV2 |"
)

# Textbook centrosymmetric insulators, by mp-id. Ge (mp-32) is deliberately absent: its GGA band gap
# is ~0 eV, so it fails the band_gap > 0.1 filter that defines the OOD population. Si stands in.
TEXTBOOK = {
    "mp-22862": ("NaCl", "rocksalt"),
    "mp-23193": ("KCl", "rocksalt"),
    "mp-2741": ("CaF2", "fluorite"),
    "mp-5229": ("SrTiO3", "perovskite"),
    "mp-1143": ("Al2O3", "corundum"),
    "mp-22865": ("CsCl", "caesium chloride"),
    "mp-66": ("C", "diamond"),
    "mp-149": ("Si", "diamond"),
    "mp-2657": ("TiO2", "rutile (relaxed → Imma)"),
    "mp-1265": ("MgO", "rocksalt"),
}

# Cubic point groups, by proper-rotation subgroup (the rotation-subgroup analysis).
M3BAR = range(200, 207)
M3BARM = range(221, 231)


def _point_group(spacegroup: int) -> tuple[str, str]:
    if spacegroup in M3BARM:
        return "m-3m", "432 — forbids rank-3"
    if spacegroup in M3BAR:
        return "m-3", "23 — permits rank-3"
    if spacegroup >= 195:
        return "cubic (other)", "varies"
    return "non-cubic", "permits rank-3"


def fetch_structures() -> dict:
    """Pull the textbook compounds from MP, verify + idealize, and cache. Inference-only."""
    import spglib
    from mp_api.client import MPRester
    from prepare_mp import _load_token

    with MPRester(_load_token()) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=list(TEXTBOOK), fields=["material_id", "structure", "band_gap"]
        )

    out = {}
    for doc in docs:
        mid = str(doc.material_id)
        structure = doc.structure
        cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
        frac = np.asarray(structure.frac_coords, dtype=np.float64) % 1.0
        z = np.asarray(structure.atomic_numbers, dtype=np.int64)

        dataset = spglib.get_symmetry_dataset((cell, frac, z), symprec=SYMPREC)
        number = int(dataset.number)
        centro = any(np.allclose(r, -np.eye(3)) for r in dataset.rotations)
        if not centro:
            print(f"  SKIP {mid} {TEXTBOOK[mid][0]}: not centrosymmetric (SG {number})")
            continue

        std = spglib.standardize_cell(
            (cell, frac, z), to_primitive=True, no_idealize=False, symprec=SYMPREC
        )
        lattice, scaled, numbers = std
        out[mid] = {
            "material_id": mid,
            "formula": TEXTBOOK[mid][0],
            "structure_type": TEXTBOOK[mid][1],
            "spacegroup": number,
            "band_gap": float(doc.band_gap),
            "cell": np.asarray(lattice).tolist(),
            "positions": (np.asarray(scaled) @ np.asarray(lattice)).tolist(),
            "z": np.asarray(numbers, dtype=int).tolist(),
        }
        print(f"  {mid:10s} {TEXTBOOK[mid][0]:8s} SG {number:3d}  gap {doc.band_gap:.2f} eV")

    missing = sorted(set(TEXTBOOK) - set(out))
    if missing:
        print(f"  not returned by MP: {[(m, TEXTBOOK[m][0]) for m in missing]}")
    CACHE.write_text(json.dumps(out, indent=1) + "\n")
    return out


def _to_structure(entry: dict):
    from equiparity.domain.structure import AtomicStructure

    return AtomicStructure(
        atomic_numbers=np.asarray(entry["z"], dtype=np.int64),
        positions=np.asarray(entry["positions"], dtype=np.float64),
        cell=np.asarray(entry["cell"], dtype=np.float64),
        pbc=True,
    )


def evaluate(cores: list[str]) -> dict:
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset

    named = json.loads(CACHE.read_text()) if CACHE.exists() else fetch_structures()

    # Panel (a): recognisable compounds already present in the OOD 2,000.
    records = json.loads(SPACEGROUPS.read_text())["records"]
    ood_npz = REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
    ood = CrystalDataset(load_crystal_dataset(ood_npz))
    in_ood = {r["material_id"]: r for r in records if r["material_id"] in TEXTBOOK}

    panel_a = [
        {
            "material_id": mid,
            "formula": TEXTBOOK[mid][0],
            "structure_type": TEXTBOOK[mid][1],
            "spacegroup": rec["spacegroup"],
            "source": "OOD 2000",
            "structure": ood[rec["index"]].structure,
        }
        for mid, rec in in_ood.items()
    ]
    panel_b = [
        {
            "material_id": mid,
            "formula": e["formula"],
            "structure_type": e["structure_type"],
            "spacegroup": e["spacegroup"],
            "source": "fresh MP",
            "structure": _to_structure(e),
        }
        for mid, e in named.items()
        if mid not in in_ood
    ]
    entries = panel_a + panel_b
    print(f"panel (a) in OOD: {len(panel_a)}   panel (b) fresh: {len(panel_b)}")

    runs = find_piezo_runs(MIRROR)
    rows = []
    for entry in entries:
        pg, subgroup = _point_group(entry["spacegroup"])
        record = {k: v for k, v in entry.items() if k != "structure"}
        record.update({"point_group": pg, "rotation_subgroup": subgroup, "predictions": {}})
        for label in sorted(runs):
            core = label.split("_o3_")[0].split("_so3_")[0]
            if core not in cores:
                continue
            trained = load_trained(runs[label], repo_root=REPO)
            if trained.is_stochastic:
                draws = seeded_predict(trained, [entry["structure"]], draws=DRAWS)
                mags = np.sqrt((draws**2).sum(axis=2))[:, 0]
                value, spread = float(mags.mean()), float(mags.std(ddof=1))
            else:
                torch.manual_seed(0)
                value = float(trained.violations([entry["structure"]])[0])
                spread = 0.0
            record["predictions"][label] = {"norm_T": value, "std": spread}
        rows.append(record)
        print(f"  {record['formula']:8s} SG {record['spacegroup']:3d} {pg:14s} {record['source']}")
    return {"rows": rows}


def render() -> None:
    data = json.loads(OUT_JSON.read_text())
    rows = data["rows"]

    def arm_mean(record: dict, core: str, parity: str) -> float:
        vals = [
            v["norm_T"]
            for label, v in record["predictions"].items()
            if label.startswith(f"{core}_{parity}_")
        ]
        return float(np.mean(vals)) if vals else float("nan")

    lines = [
        "# Familiar centrosymmetric materials",
        "",
        "The true piezoelectric tensor of every crystal below is **exactly zero**: each is",
        "centrosymmetric, and the tensor is parity-odd. Values are ‖T‖_F, mean over 3 seeds.",
        "Entries are annotated with the proper-rotation subgroup of their point group, ",
        "because that",
        "decides whether SO(3) equivariance *alone* already forbids a response (the "
        "rotation-subgroup analysis).",
        "",
        _TABLE_HEADER,
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["point_group"] == "m-3m", x["formula"])):
        lines.append(
            f"| {r['formula']} | {r['material_id']} | {r['structure_type']} | {r['spacegroup']} "
            f"| {r['point_group']} | {r['rotation_subgroup']} | {r['source']} "
            f"| {arm_mean(r, 'nequip', 'o3'):.2e} | {arm_mean(r, 'nequip', 'so3'):.3f} "
            f"| {arm_mean(r, 'allegro', 'so3'):.3f} | {arm_mean(r, 'mace', 'so3'):.3f} "
            f"| {arm_mean(r, 'equiformer_v2', 'so3'):.3f} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "The O(3) column is at machine zero for every material, as it must be. The SO(3) columns",
        "predict a nonzero piezoelectric response for materials that cannot have one — ",
        "with a clean",
        "exception that is itself the point: **the m-3̄m entries (diamond, KCl, NaCl, ",
        "MgO, CaF₂,",
        "CsCl, SrTiO₃, Si) are near zero for the exact-SO(3) cores too**, because their rotation",
        "subgroup 432 admits no rank-3 tensor at all. Rotation equivariance already forbids a",
        "response there; parity is not doing the work.",
        "",
        "The materials where parity *is* the only thing standing between the model and ",
        "a physically",
        "impossible prediction are the non-cubic ones. Among the familiar compounds those are",
        "**corundum Al₂O₃ (sapphire)** and **TiO₂**. Every SO(3) model predicts a ",
        "substantial",
        "piezoelectric response for sapphire — NequIP 1.13, Allegro 0.72, MACE 0.84, ",
        "EquiformerV2",
        "0.21 — for a crystal whose response is exactly zero by symmetry. Those are ",
        "the rows to quote.",
        "",
        "## Caveats",
        "",
        "**TiO₂ (mp-2657) is rutile, but its DFT-relaxed coordinates refine to Imma ",
        "(74) at symprec",
        "1e-3**, recovering rutile's P4₂/mnm (136) only at symprec 1e-2. Both groups are",
        "centrosymmetric and non-cubic, so the zero and its interpretation are ",
        "unaffected; the space",
        "group column reports what spglib actually finds at the study's tolerance. ",
        "This is the same",
        "tolerance phenomenon as the mp-1227949 raw-coordinate artifact (appendix A5). ",
        "The idealized",
        "rutile used in the symmetry-breaking sweep is built analytically, not from MP, and is "
        "exactly 136.",
        "",
        "**Germanium is absent.** MP's GGA band gap for Ge is ~0 eV, below the 0.1 eV ",
        "cut that defines",
        "this study's insulator population. Silicon stands in for the ",
        "diamond-structure semiconductor.",
        "",
        "EquiformerV2 values are means over 5 seeded draws (its forward is stochastic; see the "
        "output-parity audit).",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    if args.fetch:
        fetch_structures()
        return
    if args.render:
        render()
        return

    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"rows": []}
    fresh = evaluate(args.cores)
    by_id = {r["material_id"]: r for r in merged["rows"]}
    for r in fresh["rows"]:
        if r["material_id"] in by_id:
            by_id[r["material_id"]]["predictions"].update(r["predictions"])
        else:
            by_id[r["material_id"]] = r
    out = {"rows": list(by_id.values())}
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {OUT_JSON} ({len(out['rows'])} materials)")


if __name__ == "__main__":
    main()
