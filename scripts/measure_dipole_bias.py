"""Measure the point-charge dipole's underestimate of the DFT dipole magnitude on QM9.

The dipole label is constructed as ``sum_i q_i r_i`` from the reference Mulliken charges
(scripts/prepare_qm9.py). Methods states this underestimates the density-functional magnitude;
this script measures that bias directly, since every raw .xyz carries both the reference DFT
dipole magnitude mu (comment line) and the per-atom Mulliken charges (last atom column).

Restricted to the 130,831 kept molecules (uncharacterized excluded), and cross-checked against
the stored ``dipole`` vectors in qm9_processed.npz.

    uv run python scripts/measure_dipole_bias.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
XYZ_DIR = REPO / "data" / "raw" / "qm9" / "xyz"
PROCESSED = REPO / "data" / "raw" / "qm9" / "qm9_processed.npz"
OUT_JSON = REPO / "results" / "dipole_bias.json"

EA_TO_DEBYE = 4.80320425  # 1 e*Angstrom in Debye
CUTOFFS = [0.1, 0.5, 1.0]  # Debye; exclude near-zero reference dipoles from the ratio


def _f(tok: str) -> float:
    """QM9 files write some floats in Mathematica's 1.23*^-4 notation."""
    return float(tok.replace("*^", "e"))


def read_molecule(path: Path) -> tuple[float, float]:
    """(reference DFT |mu| in Debye, point-charge |sum q_i r_i| in Debye)."""
    lines = path.read_text().splitlines()
    n = int(lines[0])
    mu_ref = _f(lines[1].split()[5])  # properties: gdb id, A, B, C, mu, ...
    q, r = [], []
    for line in lines[2 : 2 + n]:
        tok = line.split()
        r.append([_f(t) for t in tok[1:4]])
        q.append(_f(tok[4]))
    p = np.asarray(q) @ np.asarray(r)
    return mu_ref, float(np.linalg.norm(p)) * EA_TO_DEBYE


def main() -> None:
    proc = np.load(PROCESSED, allow_pickle=True)
    kept = np.asarray(proc["ids"], dtype=int)
    stored = np.sqrt((np.asarray(proc["dipole"]) ** 2).sum(axis=1))
    stored_of = dict(zip(kept.tolist(), stored.tolist(), strict=True))

    mu_ref = np.empty(kept.size)
    pc = np.empty(kept.size)
    for i, gdb_id in enumerate(kept):
        mu_ref[i], pc[i] = read_molecule(XYZ_DIR / f"dsgdb9nsd_{gdb_id:06d}.xyz")

    cross = np.abs(pc - np.array([stored_of[g] for g in kept.tolist()]))
    result: dict[str, object] = {
        "n_molecules": int(kept.size),
        "crosscheck_max_abs_diff_debye": float(cross.max()),
        "note": (
            "underestimate = (mu_ref - |sum q_i r_i|) / mu_ref, per molecule; "
            "mu_ref from the .xyz comment line, charges are the reference Mulliken charges"
        ),
    }
    for cut in CUTOFFS:
        sel = mu_ref >= cut
        frac = (mu_ref[sel] - pc[sel]) / mu_ref[sel]
        q25, q50, q75 = (float(x) for x in np.percentile(frac, [25, 50, 75]))
        result[f"cutoff_{cut}"] = {
            "n": int(sel.sum()),
            "median": q50,
            "iqr": [q25, q75],
            "p5_p95": [float(x) for x in np.percentile(frac, [5, 95])],
            "fraction_overestimated": float((frac < 0).mean()),
        }
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
