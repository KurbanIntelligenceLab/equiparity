"""Idealize the centrosymmetric OOD structures to their exact detected space group.

The OOD set is spglib-*verified* centrosymmetric (symprec=1e-3) but stores the raw MP-relaxed
coordinates, which deviate from perfect inversion symmetry by up to ~symprec. The piezoelectric
tensor is exactly zero for the IDEAL centrosymmetric crystal, so the OOD test must use exactly
symmetric geometries — otherwise an O(3)-exact model with a sensitive (tensor-product) head reports
a spurious nonzero response to that coordinate noise (observed: clifford false-flag 0.43 on raw
vs ~0 on idealized). Idealizing snaps coordinates onto the space group; linear-head models
(nequip etc.) were already ~0 on raw, so this only tightens every core's O(3) result and makes the
test measure the physical claim. Writes the npz in place (raw backed up to *_raw.npz).

Run: uv run --extra data python scripts/idealize_ood.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import spglib

OOD_SYMPREC = 1e-3
NPZ = Path("data/raw/mp/mp_ood_centrosymmetric_processed.npz")


def idealize_one(cell: np.ndarray, pos: np.ndarray, z: np.ndarray):
    """Return (cell, positions, numbers) snapped to the space group, or None if refine fails."""
    frac = (pos @ np.linalg.inv(cell)) % 1.0
    std = spglib.standardize_cell(
        (cell, frac, z), to_primitive=True, no_idealize=False, symprec=OOD_SYMPREC
    )
    if std is None:
        return None
    lat, scaled, nums = std
    return lat, scaled @ lat, np.asarray(nums, dtype=np.int64)


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    ids, natoms, z, pos, cells = d["ids"], d["n_atoms"], d["z"], d["positions"], d["cells"]
    off = np.concatenate([[0], np.cumsum(natoms)])

    out_ids, out_n, out_z, out_pos, out_cells = [], [], [], [], []
    failed = 0
    for i in range(len(ids)):
        zi = z[off[i] : off[i + 1]]
        pi = pos[off[i] : off[i + 1]]
        cell = cells[i]
        res = idealize_one(cell, pi, zi)
        if res is None:
            res = (cell, pi, zi)  # keep raw if spglib can't refine (rare)
            failed += 1
        lat, newpos, newz = res
        out_ids.append(ids[i])
        out_n.append(len(newz))
        out_z.append(newz)
        out_pos.append(newpos)
        out_cells.append(lat)

    if not (NPZ.parent / "mp_ood_centrosymmetric_processed_raw.npz").exists():
        shutil.copyfile(NPZ, NPZ.parent / "mp_ood_centrosymmetric_processed_raw.npz")
    np.savez_compressed(
        NPZ,
        ids=np.array(out_ids),
        n_atoms=np.array(out_n, dtype=np.int64),
        z=np.concatenate(out_z).astype(np.int64),
        positions=np.concatenate(out_pos).astype(np.float64),
        cells=np.stack(out_cells).astype(np.float64),
    )
    print(
        f"idealized {len(out_ids)} OOD structures (symprec={OOD_SYMPREC}); "
        f"{failed} kept raw (spglib refine failed). Backup: *_raw.npz"
    )


if __name__ == "__main__":
    main()
