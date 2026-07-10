"""Build perovskite structures in memory (no npz), for the E2 symmetry-breaking sweep.

The cubic aristotype is centrosymmetric (Pm-3m, 221); displacing the B-site cation along [001]
against the oxygen cage breaks inversion and yields the polar tetragonal phase (P4mm, 99). The
piezoelectric tensor is zero at delta = 0 by Neumann's principle and turns on for delta > 0.

Tolerance warning: spglib's ``symprec`` is a *distance* tolerance. At symprec 1e-3 a small-delta
frame is reported as the centrosymmetric parent (the crossover is near delta ~ 0.006, where the
maximum displacement ~ 7e-4 A falls below symprec). Verify sweep frames at symprec 1e-8.
"""

from __future__ import annotations

import numpy as np

from equiparity.domain.structure import AtomicStructure

# Cubic lattice constants (A) of the aristotype, and the [001] polar displacement pattern in
# fractional units at the physical amplitude (delta = 1).
PEROVSKITES: dict[str, dict[str, object]] = {
    "BaTiO3": {"a": 4.00, "A": 56, "B": 22, "X": 8},
    "PbTiO3": {"a": 3.97, "A": 82, "B": 22, "X": 8},
}

# Fractional displacements along +z at delta = 1: B-site up, oxygens down (apical O moves most).
DISPLACEMENT_Z = {"B": 0.030, "O_apical": -0.024, "O_equatorial": -0.010}


def perovskite(name: str) -> AtomicStructure:
    """The cubic (Pm-3m) 5-atom perovskite cell: A(0,0,0), B(1/2,1/2,1/2), O on the face centres."""
    spec = PEROVSKITES[name]
    a = float(spec["a"])  # type: ignore[arg-type]
    frac = np.array(
        [
            [0.0, 0.0, 0.0],  # A site
            [0.5, 0.5, 0.5],  # B site
            [0.5, 0.5, 0.0],  # O apical: shares (x, y) with B, so it lies on the polar [001] axis
            [0.5, 0.0, 0.5],  # O equatorial
            [0.0, 0.5, 0.5],  # O equatorial
        ]
    )
    z = np.array([spec["A"], spec["B"], spec["X"], spec["X"], spec["X"]], dtype=np.int64)
    cell = np.eye(3) * a
    return AtomicStructure(atomic_numbers=z, positions=frac @ cell, cell=cell, pbc=True)


def tetragonal_distortion(name: str, delta: float) -> AtomicStructure:
    """The cubic cell displaced by ``delta`` times the physical [001] polar pattern."""
    dz = np.zeros((5, 3))
    dz[1, 2] = DISPLACEMENT_Z["B"]
    dz[2, 2] = DISPLACEMENT_Z["O_apical"]
    dz[3, 2] = DISPLACEMENT_Z["O_equatorial"]
    dz[4, 2] = DISPLACEMENT_Z["O_equatorial"]

    base = perovskite(name)
    cell = base.cell
    assert cell is not None
    frac = base.positions @ np.linalg.inv(cell) + delta * dz
    return AtomicStructure(
        atomic_numbers=base.atomic_numbers, positions=frac @ cell, cell=cell, pbc=True
    )


def max_displacement_angstrom(name: str, delta: float) -> float:
    """Largest Cartesian atomic displacement (A) at ``delta``; compare against symprec."""
    a = float(PEROVSKITES[name]["a"])  # type: ignore[arg-type]
    return delta * max(abs(v) for v in DISPLACEMENT_Z.values()) * a
