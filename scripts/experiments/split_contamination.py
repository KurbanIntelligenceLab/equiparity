"""Train/eval contamination check for the centrosymmetric evaluation population.

Methods verifies disjointness for the 1,000 augmentation crystals but never for the 2,000
evaluation crystals, and the piezoelectric training split contains 16 exactly-zero rows, so it
does contain centrosymmetric crystals. This intersects the identifier sets and, for the record,
identifies the point groups of the zero-norm training rows with spglib.

    uv run --extra nequip python scripts/experiments/split_contamination.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import spglib

REPO = Path(__file__).resolve().parents[2]
SPACEGROUPS = REPO / "results" / "ood_spacegroups.json"
SPLIT = REPO / "data" / "splits" / "mp_piezoelectric_split.npz"
PROCESSED = REPO / "data" / "raw" / "mp" / "mp_piezoelectric_processed.npz"
OUT_JSON = REPO / "results" / "split_contamination.json"
OUT_MD = REPO / "docs" / "results" / "split_contamination.md"

CENTROSYMMETRIC = {
    "-1",
    "2/m",
    "mmm",
    "4/m",
    "4/mmm",
    "-3",
    "-3m",
    "6/m",
    "6/mmm",
    "m-3",
    "m-3m",
}  # the 11 Laue classes


def main() -> None:
    eval_ids = {r["material_id"] for r in json.loads(SPACEGROUPS.read_text())["records"]}
    split = np.load(SPLIT, allow_pickle=True)
    splits = {k: [str(x) for x in split[k]] for k in ("train", "val", "test")}

    overlap = {k: sorted(eval_ids & set(v)) for k, v in splits.items()}

    proc = np.load(PROCESSED, allow_pickle=True)
    ids = np.array([str(x) for x in proc["ids"]])
    norms = np.linalg.norm(proc["piezoelectric"], axis=1)
    offsets = np.concatenate([[0], np.cumsum(proc["n_atoms"])])
    id_to_row = {ids[i]: i for i in range(len(ids))}

    zero_rows = []
    for k, members in splits.items():
        for mid in members:
            i = id_to_row[mid]
            if norms[i] != 0.0:
                continue
            lattice = proc["cells"][i]
            cart = proc["positions"][offsets[i] : offsets[i + 1]]
            frac = cart @ np.linalg.inv(lattice)
            numbers = proc["z"][offsets[i] : offsets[i + 1]]
            ds = spglib.get_symmetry_dataset((lattice, frac, numbers), symprec=1e-3)
            zero_rows.append(
                {
                    "material_id": mid,
                    "split": k,
                    "spacegroup": int(ds.number),
                    "international": ds.international,
                    "point_group": ds.pointgroup,
                    "centrosymmetric": ds.pointgroup in CENTROSYMMETRIC,
                    "in_eval_population": mid in eval_ids,
                }
            )

    result = {
        "n_eval": len(eval_ids),
        "n_split": {k: len(v) for k, v in splits.items()},
        "overlap_count": {k: len(v) for k, v in overlap.items()},
        "overlap_any": sorted(set().union(*[set(v) for v in overlap.values()])),
        "n_zero_norm_rows": len(zero_rows),
        "n_zero_norm_train": sum(1 for r in zero_rows if r["split"] == "train"),
        "n_zero_norm_centrosymmetric": sum(1 for r in zero_rows if r["centrosymmetric"]),
        "zero_norm_rows": zero_rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    _render(result)


def _render(r: dict) -> None:
    lines = [
        "# Train/eval contamination check",
        "",
        f"Evaluation population: **{r['n_eval']}** centrosymmetric crystals. Piezoelectric "
        f"splits: train {r['n_split']['train']}, val {r['n_split']['val']}, "
        f"test {r['n_split']['test']}.",
        "",
        f"Identifier overlap with the evaluation population: train "
        f"**{r['overlap_count']['train']}**, val **{r['overlap_count']['val']}**, "
        f"test **{r['overlap_count']['test']}**.",
        "",
        f"Exactly-zero tensor rows across the splits: **{r['n_zero_norm_rows']}** "
        f"({r['n_zero_norm_train']} in train), of which "
        f"**{r['n_zero_norm_centrosymmetric']}** are centrosymmetric by spglib "
        "(symprec 1e-3).",
        "",
        "| material | split | space group | point group | centrosymmetric | in eval set |",
        "|---|---|---|---|---|---|",
    ]
    for z in r["zero_norm_rows"]:
        lines.append(
            f"| {z['material_id']} | {z['split']} | {z['spacegroup']} ({z['international']}) "
            f"| {z['point_group']} | {'yes' if z['centrosymmetric'] else 'no'} "
            f"| {'yes' if z['in_eval_population'] else 'no'} |"
        )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
