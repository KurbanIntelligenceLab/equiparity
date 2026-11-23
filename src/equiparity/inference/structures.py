"""Rutile TiO2 and its polar [001] displacement path."""

from __future__ import annotations

import numpy as np

from equiparity.domain.structure import AtomicStructure

# Rutile TiO2, P4_2/mnm (136). a = 4.5937, c = 2.9587 A; internal parameter u = 0.3053.
RUTILE = {"a": 4.5937, "c": 2.9587, "u": 0.3053, "Ti": 22, "O": 8}

# Rutile polar mode at delta = 1: Ti sublattice against the O sublattice along [001], in fractional
# units of c. The two Ti move opposite to the four O, which breaks the inversion centre.
RUTILE_DISPLACEMENT_Z = 0.040

RUTILE_TYPE: dict[str, dict[str, float | int]] = {"TiO2": RUTILE}


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
    if name != "TiO2":
        raise ValueError("the displacement path is defined for rutile TiO2")
    dz = np.zeros((6, 3))
    dz[:2, 2] = RUTILE_DISPLACEMENT_Z
    dz[2:, 2] = -RUTILE_DISPLACEMENT_Z / 2
    return dz


def tetragonal_distortion(name: str, delta: float) -> AtomicStructure:
    """The parent cell displaced by ``delta`` times the physical [001] polar pattern."""
    base = rutile(name)
    cell = base.cell
    assert cell is not None
    frac = base.positions @ np.linalg.inv(cell) + delta * _displacement(name)
    return AtomicStructure(
        atomic_numbers=base.atomic_numbers, positions=frac @ cell, cell=cell, pbc=True
    )


def max_displacement_angstrom(name: str, delta: float) -> float:
    """Largest Cartesian atomic displacement (A) at ``delta``; compare against symprec."""
    return abs(delta) * RUTILE_DISPLACEMENT_Z * float(RUTILE_TYPE[name]["c"])
