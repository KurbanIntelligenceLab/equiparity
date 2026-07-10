"""Build reference crystal structures in memory (no npz), for the E2 symmetry-breaking sweep.

Two families, and the difference between them is the whole point.

**Perovskites (BaTiO3, PbTiO3).** The cubic aristotype is Pm-3m (221); displacing the B-site cation
along [001] against the oxygen cage breaks inversion and gives the polar tetragonal phase P4mm (99).

    Caveat, found when the sweep was first run: Pm-3m's *proper-rotation* subgroup is **432**, and
    no rank-3 tensor is invariant under 432. At delta = 0 an exactly SO(3)-equivariant model is
    therefore forced to predict zero as well -- by rotation alone, with no parity label involved
    (E7). On a perovskite the O(3) and SO(3) curves both start at machine zero, so the sweep cannot
    exhibit the parity effect it was designed to exhibit. Keep it as the textbook reference; do not
    draw the parity conclusion from it.

**Rutile (TiO2).** P4_2/mnm (136) is centrosymmetric, so the piezoelectric tensor is still exactly
zero -- but its rotation subgroup is **422**, which *does* admit a rank-3 invariant (dimension 1).
Here only parity forbids a response, and this is where the arms separate at delta = 0. The polar
mode displaces Ti against the O cage along [001], breaking inversion.

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

# Rutile TiO2, P4_2/mnm (136). a = 4.5937, c = 2.9587 A; internal parameter u = 0.3053.
RUTILE = {"a": 4.5937, "c": 2.9587, "u": 0.3053, "Ti": 22, "O": 8}

# Rutile polar mode at delta = 1: Ti sublattice against the O sublattice along [001], in fractional
# units of c. The two Ti move opposite to the four O, which breaks the inversion centre.
RUTILE_DISPLACEMENT_Z = 0.040

MATERIALS = (*PEROVSKITES, "TiO2")


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


def rutile() -> AtomicStructure:
    """The rutile TiO2 cell (P4_2/mnm, 136): 2 Ti + 4 O, centrosymmetric, rotation subgroup 422."""
    a, c, u = RUTILE["a"], RUTILE["c"], RUTILE["u"]
    frac = np.array(
        [
            [0.0, 0.0, 0.0],  # Ti
            [0.5, 0.5, 0.5],  # Ti
            [u, u, 0.0],  # O
            [1.0 - u, 1.0 - u, 0.0],  # O
            [0.5 + u, 0.5 - u, 0.5],  # O
            [0.5 - u, 0.5 + u, 0.5],  # O
        ]
    )
    z = np.array([RUTILE["Ti"]] * 2 + [RUTILE["O"]] * 4, dtype=np.int64)
    cell = np.diag([a, a, c]).astype(np.float64)
    return AtomicStructure(atomic_numbers=z, positions=frac @ cell, cell=cell, pbc=True)


def _displacement(name: str) -> np.ndarray:
    """Fractional [001] polar displacement pattern at amplitude delta = 1."""
    if name == "TiO2":
        dz = np.zeros((6, 3))
        dz[0:2, 2] = RUTILE_DISPLACEMENT_Z  # Ti up
        dz[2:6, 2] = -RUTILE_DISPLACEMENT_Z / 2.0  # O down (keeps the centre of mass fixed)
        return dz
    dz = np.zeros((5, 3))
    dz[1, 2] = DISPLACEMENT_Z["B"]
    dz[2, 2] = DISPLACEMENT_Z["O_apical"]
    dz[3, 2] = DISPLACEMENT_Z["O_equatorial"]
    dz[4, 2] = DISPLACEMENT_Z["O_equatorial"]
    return dz


def parent(name: str) -> AtomicStructure:
    """The centrosymmetric parent structure (delta = 0) for any E2 material."""
    return rutile() if name == "TiO2" else perovskite(name)


def tetragonal_distortion(name: str, delta: float) -> AtomicStructure:
    """The parent cell displaced by ``delta`` times the physical [001] polar pattern."""
    base = parent(name)
    cell = base.cell
    assert cell is not None
    frac = base.positions @ np.linalg.inv(cell) + delta * _displacement(name)
    return AtomicStructure(
        atomic_numbers=base.atomic_numbers, positions=frac @ cell, cell=cell, pbc=True
    )


def max_displacement_angstrom(name: str, delta: float) -> float:
    """Largest Cartesian atomic displacement (A) at ``delta``; compare against symprec."""
    if name == "TiO2":
        return delta * RUTILE_DISPLACEMENT_Z * float(RUTILE["c"])
    a = float(PEROVSKITES[name]["a"])  # type: ignore[arg-type]
    return delta * max(abs(v) for v in DISPLACEMENT_Z.values()) * a
