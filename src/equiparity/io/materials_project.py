"""Materials Project boundary: convert pymatgen structures/tensors into typed domain objects.

Fetching (network, MPRester queries) lives in ``scripts/prepare_mp.py``; this module holds the
pure conversions that turn untyped pymatgen objects into validated :class:`AtomicStructure` /
:class:`LabeledStructure`, and the spglib space-group check that gates the centrosymmetric OOD set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from equiparity.domain.sample import LabeledStructure
from equiparity.domain.structure import AtomicStructure, validate_structure


def pymatgen_to_structure(structure: Any) -> AtomicStructure:  # noqa: ANN401 (pymatgen boundary)
    """Convert a pymatgen ``Structure`` into a validated periodic :class:`AtomicStructure`."""
    numbers = np.asarray(structure.atomic_numbers, dtype=np.int64)
    positions = np.asarray(structure.cart_coords, dtype=np.float64)
    cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
    return validate_structure(
        AtomicStructure(atomic_numbers=numbers, positions=positions, cell=cell, pbc=True)
    )


def tensor_sample(
    structure: Any,  # noqa: ANN401 (pymatgen boundary)
    tensor: npt.NDArray[np.float64],
    target: str,
    mid: str,
) -> LabeledStructure:
    """Build a labeled crystal sample with a flattened tensor target.

    The tensor is stored flat (Voigt order); the equivariant output head projects it onto its
    irreps decomposition downstream. Elastic tensors are 6x6 (21 unique, parity-even); piezo
    tensors are 3x6 (18 components, parity-odd).
    """
    return LabeledStructure(
        structure=pymatgen_to_structure(structure),
        targets={target: np.asarray(tensor, dtype=np.float64).reshape(-1)},
        identifier=mid,
    )


def space_group_number(structure: Any, symprec: float = 1e-3) -> int:  # noqa: ANN401
    """Return the spglib space-group number of a pymatgen structure.

    Uses pymatgen's :class:`SpacegroupAnalyzer` (spglib-backed). This is the load-bearing check
    for the OOD set: only structures whose number is centrosymmetric are kept.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    return int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
