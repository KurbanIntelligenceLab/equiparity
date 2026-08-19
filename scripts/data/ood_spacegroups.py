"""Cache the space group of every crystal in the centrosymmetric OOD set.

Space groups are not stored in the processed npz, and both the named-materials check (named
materials) and the rotation-subgroup analysis (the rotation-subgroup analysis) need them per
structure. Computed once with spglib at the same tolerance used to build the set (``symprec =
1e-3``) and cached.

    uv run --extra nequip --extra data python scripts/data/ood_spacegroups.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import spglib

REPO = Path(__file__).resolve().parents[2]
OOD_NPZ = REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
OUT = REPO / "results" / "ood_spacegroups.json"
SELECTION_SYMPREC = 1e-3

# Cubic point groups, by the proper-rotation subgroup that decides whether a rank-3 tensor exists.
M3BAR = range(200, 207)  # point group m-3, rotation subgroup 23 (order 12) -- piezoelectric
M3BARM = range(221, 231)  # point group m-3m, rotation subgroup 432 (order 24) -- forbids rank 3


def _cell(data: np.lib.npyio.NpzFile, index: int) -> tuple:
    offsets = np.concatenate([[0], np.cumsum(data["n_atoms"])])
    start, stop = int(offsets[index]), int(offsets[index + 1])
    lattice = data["cells"][index]
    positions = data["positions"][start:stop] @ np.linalg.inv(lattice)
    return (lattice, positions, data["z"][start:stop])


def main() -> None:
    data = np.load(OOD_NPZ, allow_pickle=False)
    ids = data["ids"]
    records = []
    for i in range(len(ids)):
        dataset = spglib.get_symmetry_dataset(_cell(data, i), symprec=SELECTION_SYMPREC)
        number = int(dataset.number)
        if number in M3BARM:
            family = "m-3m"
        elif number in M3BAR:
            family = "m-3"
        elif number >= 195:
            family = "cubic-other"
        else:
            family = "non-cubic"
        records.append(
            {
                "index": i,
                "material_id": str(ids[i]),
                "spacegroup": number,
                "international": str(dataset.international),
                "family": family,
                "n_atoms": int(data["n_atoms"][i]),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"symprec": SELECTION_SYMPREC, "records": records}, indent=1) + "\n")

    counts: dict[str, int] = {}
    for r in records:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    n_groups = len({r["spacegroup"] for r in records})
    print(f"wrote {OUT}  ({len(records)} structures, {n_groups} space groups)")
    for family, n in sorted(counts.items()):
        print(f"  {family:12s} {n:5d}")


if __name__ == "__main__":
    main()
