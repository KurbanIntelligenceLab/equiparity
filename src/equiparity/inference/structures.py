"""Reference crystal structures for the polar-distortion path, built in memory.

**Perovskites (BaTiO3, PbTiO3).** The cubic aristotype is Pm-3m (221); displacing the B-site
cation along [001] against the oxygen cage breaks inversion and gives the polar tetragonal phase
P4mm (99). Pm-3m's proper-rotation subgroup is 432, which admits no rank-3 invariant, so at
delta = 0 both O(3) and SO(3) models are forced to zero by rotation alone.

**Rutile (TiO2).** P4_2/mnm (136) is centrosymmetric, so the piezoelectric tensor is exactly zero,
but its rotation subgroup 422 admits a rank-3 invariant (dimension 1). Only parity forbids a
response, so the arms separate at delta = 0. The polar mode displaces Ti against the O cage
along [001], breaking inversion.

spglib's ``symprec`` is a distance tolerance: at symprec 1e-3 a small-delta frame is reported as
the centrosymmetric parent (crossover near delta ~ 0.006). Verify path frames at symprec 1e-8.
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

RUTILE_TYPE: dict[str, dict[str, float | int]] = {"TiO2": RUTILE}

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


def rutile(name: str = "TiO2") -> AtomicStructure:
    """A rutile-type cell (P4_2/mnm, 136): 2 cations + 4 O, centrosymmetric, subgroup 422."""
    spec = RUTILE_TYPE[name]
    a, c, u = float(spec["a"]), float(spec["c"]), float(spec["u"])
    cation = int(spec.get("cation", spec.get("Ti", 22)))
    frac = np.array(
        [
            [0.0, 0.0, 0.0],  # cation
            [0.5, 0.5, 0.5],  # cation
            [u, u, 0.0],  # O
            [1.0 - u, 1.0 - u, 0.0],  # O
            [0.5 + u, 0.5 - u, 0.5],  # O
            [0.5 - u, 0.5 + u, 0.5],  # O
        ]
    )
    z = np.array([cation] * 2 + [int(spec["O"])] * 4, dtype=np.int64)
    cell = np.diag([a, a, c]).astype(np.float64)
    return AtomicStructure(atomic_numbers=z, positions=frac @ cell, cell=cell, pbc=True)


def _displacement(name: str) -> np.ndarray:
    """Fractional [001] polar displacement pattern at amplitude delta = 1."""
    if name in RUTILE_TYPE:
        dz = np.zeros((6, 3))
        dz[0:2, 2] = RUTILE_DISPLACEMENT_Z  # cations up
        dz[2:6, 2] = -RUTILE_DISPLACEMENT_Z / 2.0  # O down (keeps the centre of mass fixed)
        return dz
    dz = np.zeros((5, 3))
    dz[1, 2] = DISPLACEMENT_Z["B"]
    dz[2, 2] = DISPLACEMENT_Z["O_apical"]
    dz[3, 2] = DISPLACEMENT_Z["O_equatorial"]
    dz[4, 2] = DISPLACEMENT_Z["O_equatorial"]
    return dz


def parent(name: str) -> AtomicStructure:
    """The centrosymmetric parent structure (delta = 0) for any sweep material."""
    if name in RUTILE_TYPE:
        return rutile(name)
    return perovskite(name)


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
    if name in RUTILE_TYPE:
        return delta * RUTILE_DISPLACEMENT_Z * float(RUTILE_TYPE[name]["c"])
    a = float(PEROVSKITES[name]["a"])  # type: ignore[arg-type]
    return delta * max(abs(v) for v in DISPLACEMENT_Z.values()) * a
