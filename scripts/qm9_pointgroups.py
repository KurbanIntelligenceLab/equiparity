"""Point-group census of the QM9 dipole evaluation population.

Reviewer question: the manuscript claimed no QM9 molecule has a point group forcing
its dipole to vanish. QM9 contains high-symmetry molecules (methane, T_d), so the
claim needs checking against the actual evaluation split rather than asserted.

Criterion. The dipole is a polar vector, so the symmetry-allowed dipoles of a
molecule with point group G are the vectors fixed by every operation of G. That
subspace is the image of the Reynolds projector

    P = (1/|G|) sum_{R in G} R,

whose rank is the number of independent dipole components symmetry permits. Rank 0
means the dipole is forced to vanish identically. This is Neumann's principle applied
to a rank-1 tensor, and it is the same argument the manuscript uses for the rank-3
piezoelectric tensor. Inversion symmetry is sufficient for a forced zero but not
necessary: T_d contains no inversion yet still gives rank 0.

The projector is built from operations that are verified individually, by applying
each to the molecule and requiring every atom to land on an atom of the same species
within ``RESID_TOL``. This matters: the Schoenflies label alone is not reliable here.
On this dataset the analyser labels several molecules ``D2`` while returning only the
two operations of ``C2``; those molecules carry reference dipoles of 1.6-3.0 D, which
a genuine D2 would forbid. Verifying the operations and computing the rank from them
resolves the contradiction, and reproduces the reference dipole in every case.

Reads the same raw archive, exclusion list and split as scripts/prepare_qm9.py, so
the census is over exactly the molecules the manuscript evaluates.

Writes results/qm9_pointgroups.json.

Run: ``uv run python scripts/qm9_pointgroups.py``
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

RAW_DIR = Path("data/raw/qm9")
XYZ_DIR = RAW_DIR / "xyz"
UNCHARACTERIZED = RAW_DIR / "uncharacterized.txt"
SPLIT_NPZ = Path("data/splits/qm9_split.npz")
OUT = Path("results/qm9_pointgroups.json")

# Tolerance for the symmetry search, in Angstrom. QM9 geometries are relaxed, so
# an exact-symmetry search at machine precision finds nothing; the census is
# reported at several tolerances to show the count is not an artefact of one choice.
TOLERANCES = (0.01, 0.1, 0.3)

# An operation counts as a symmetry of the molecule only if it maps every atom onto
# an atom of the same species within this distance (Angstrom).
RESID_TOL = 0.05

# Singular values of the Reynolds projector are 0 or 1 in exact arithmetic; this
# separates them. On T_d the "zero" block comes back at ~1e-5, so a machine-epsilon
# threshold would wrongly report a permitted dipole.
SV_TOL = 0.5


def dipole_rank(species: list[str], coords: np.ndarray, tol: float) -> tuple[str, int, int, int]:
    """Return (Schoenflies symbol, verified ops, total ops, symmetry-allowed dipole rank).

    Rank 0 means every symmetry-allowed dipole is the zero vector.
    """
    pga = PointGroupAnalyzer(Molecule(species, coords), tolerance=tol)
    centered = pga.centered_mol
    pos = np.array([site.coords for site in centered])
    sym = [site.specie.symbol for site in centered]

    ops = pga.get_symmetry_operations()
    verified = []
    for op in ops:
        moved = np.array([op.operate(x) for x in pos])
        worst = max(
            min(np.linalg.norm(y - pos[j]) for j in range(len(sym)) if sym[j] == sym[i])
            for i, y in enumerate(moved)
        )
        if worst <= RESID_TOL:
            verified.append(op.rotation_matrix)

    projector = sum(verified) / len(verified)
    rank = int((np.linalg.svd(projector, compute_uv=False) > SV_TOL).sum())
    return pga.sch_symbol, len(verified), len(ops), rank


def _excluded_indices() -> set[int]:
    text = UNCHARACTERIZED.read_text()
    return {int(m.group(1)) for line in text.splitlines() if (m := re.match(r"\s*(\d+)", line))}


def _parse_xyz(path: Path) -> tuple[list[str], np.ndarray, float]:
    """Return (species, coords, reference dipole in debye) from one QM9 .xyz file."""
    lines = path.read_text().splitlines()
    n = int(lines[0])
    # Comment line: gdb <id> <A> <B> <C> <mu> ... ; mu is field index 5.
    mu = float(lines[1].split()[5])
    species, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        species.append(parts[0])
        # QM9 writes Fortran-style exponents (e.g. 1.234*^-6) in some columns.
        coords.append([float(x.replace("*^", "e")) for x in parts[1:4]])
    return species, np.asarray(coords, dtype=float), mu


def main() -> None:
    excluded = _excluded_indices()
    test_ids = set(np.load(SPLIT_NPZ)["test"].tolist())

    per_tol: dict[float, dict[str, object]] = {}
    for tol in TOLERANCES:
        counts: Counter[str] = Counter()
        forced: list[dict[str, object]] = []
        n_seen = 0
        max_mu_forced = 0.0
        for path in sorted(XYZ_DIR.glob("dsgdb9nsd_*.xyz")):
            index = int(path.stem.split("_")[1])
            if index in excluded or index not in test_ids:
                continue
            species, coords, mu = _parse_xyz(path)
            pg, n_ok, n_ops, rank = dipole_rank(species, coords, tol)
            counts[pg] += 1
            n_seen += 1
            if rank == 0:
                max_mu_forced = max(max_mu_forced, abs(mu))
                forced.append(
                    {
                        "qm9_index": index,
                        "point_group": pg,
                        "verified_ops": n_ok,
                        "analyser_ops": n_ops,
                        "reference_dipole_debye": mu,
                    }
                )
        per_tol[tol] = {
            "n_molecules": n_seen,
            "n_dipole_forcing": len(forced),
            "fraction_dipole_forcing": len(forced) / n_seen,
            # Consistency check: a forced-zero molecule must carry a ~zero reference
            # dipole. A large value here would mean the criterion is wrong.
            "max_reference_dipole_among_forced_debye": max_mu_forced,
            "point_group_counts": dict(counts.most_common()),
            "dipole_forcing_molecules": sorted(forced, key=lambda r: r["qm9_index"]),
        }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "population": "QM9 test split (seed 42), the dipole evaluation population",
                "split_file": str(SPLIT_NPZ),
                "criterion": (
                    "Rank of the Reynolds projector (1/|G|) sum_R R over individually verified "
                    "symmetry operations; rank 0 means symmetry forces the dipole to vanish. "
                    "Inversion is sufficient but not necessary: Td gives rank 0 without it."
                ),
                "residual_tolerance_angstrom": RESID_TOL,
                "tolerances_angstrom": list(TOLERANCES),
                "by_tolerance": {str(k): v for k, v in per_tol.items()},
            },
            indent=2,
        )
    )
    for tol, res in per_tol.items():
        print(
            f"tol={tol}: {res['n_dipole_forcing']}/{res['n_molecules']} "
            f"({100 * res['fraction_dipole_forcing']:.3f}%) dipole-forcing"
        )


if __name__ == "__main__":
    main()
