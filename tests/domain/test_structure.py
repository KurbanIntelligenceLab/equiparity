"""Tests for AtomicStructure validation."""

from __future__ import annotations

import numpy as np
import pytest

from equiparity.domain.structure import (
    AtomicStructure,
    StructureValidationError,
    validate_structure,
)


def _water() -> AtomicStructure:
    return AtomicStructure(
        atomic_numbers=np.array([8, 1, 1], dtype=np.int64),
        positions=np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]),
        cell=None,
        pbc=False,
    )


def test_valid_molecule_passes() -> None:
    structure = _water()
    assert validate_structure(structure) is structure
    assert structure.n_atoms == 3


def test_valid_crystal_passes() -> None:
    structure = AtomicStructure(
        atomic_numbers=np.array([11, 17], dtype=np.int64),
        positions=np.array([[0.0, 0.0, 0.0], [2.8, 2.8, 2.8]]),
        cell=np.eye(3) * 5.6,
        pbc=True,
    )
    assert validate_structure(structure) is structure


def test_position_shape_mismatch_raises() -> None:
    bad = AtomicStructure(
        atomic_numbers=np.array([1, 1], dtype=np.int64),
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=None,
        pbc=False,
    )
    with pytest.raises(StructureValidationError, match="positions must be"):
        validate_structure(bad)


def test_nan_positions_raise() -> None:
    bad = AtomicStructure(
        atomic_numbers=np.array([1], dtype=np.int64),
        positions=np.array([[np.nan, 0.0, 0.0]]),
        cell=None,
        pbc=False,
    )
    with pytest.raises(StructureValidationError, match="NaN or infinity"):
        validate_structure(bad)


def test_pbc_without_cell_raises() -> None:
    bad = AtomicStructure(
        atomic_numbers=np.array([1], dtype=np.int64),
        positions=np.zeros((1, 3)),
        cell=None,
        pbc=True,
    )
    with pytest.raises(StructureValidationError, match="pbc=True requires a cell"):
        validate_structure(bad)


def test_out_of_range_atomic_number_raises() -> None:
    bad = AtomicStructure(
        atomic_numbers=np.array([0], dtype=np.int64),
        positions=np.zeros((1, 3)),
        cell=None,
        pbc=False,
    )
    with pytest.raises(StructureValidationError, match=r"\[1, 118\]"):
        validate_structure(bad)
