"""Tests for the Materials Project converters and crystal dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equiparity.io.mp_dataset import CrystalData, CrystalDataset, load_crystal_dataset, load_split


def test_pymatgen_to_structure() -> None:
    pmg = pytest.importorskip("pymatgen.core")
    from equiparity.io.materials_project import pymatgen_to_structure

    struct = pmg.Structure(
        lattice=pmg.Lattice.cubic(5.64),
        species=["Na", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    out = pymatgen_to_structure(struct)
    assert out.pbc is True
    assert out.cell is not None
    np.testing.assert_array_equal(out.atomic_numbers, [11, 17])
    assert out.n_atoms == 2


def _synthetic_crystals() -> CrystalData:
    n_atoms = np.array([2, 1], dtype=np.int64)
    offsets = np.array([0, 2, 3], dtype=np.int64)
    return CrystalData(
        ids=np.array(["mp-1", "mp-2"], dtype=np.str_),
        n_atoms=n_atoms,
        z=np.array([11, 17, 8], dtype=np.int64),
        positions=np.arange(9, dtype=np.float64).reshape(3, 3),
        cells=np.stack([np.eye(3) * 4.0, np.eye(3) * 5.0]),
        offsets=offsets,
        targets={"elastic": np.array([[1.0, 2.0], [3.0, 4.0]])},
    )


def test_crystal_dataset_reconstructs_with_cell_and_target() -> None:
    ds = CrystalDataset(_synthetic_crystals())
    assert len(ds) == 2
    first = ds[0]
    assert first.identifier == "mp-1"
    assert first.structure.pbc is True
    np.testing.assert_array_equal(first.structure.atomic_numbers, [11, 17])
    np.testing.assert_allclose(first.structure.cell, np.eye(3) * 4.0)
    np.testing.assert_allclose(first.targets["elastic"], [1.0, 2.0])


def test_ood_dataset_has_no_targets() -> None:
    data = _synthetic_crystals()
    ood = CrystalData(
        ids=data.ids,
        n_atoms=data.n_atoms,
        z=data.z,
        positions=data.positions,
        cells=data.cells,
        offsets=data.offsets,
    )
    ds = CrystalDataset(ood)
    assert ds[0].targets == {}


def test_load_roundtrip(tmp_path: Path) -> None:
    data = _synthetic_crystals()
    npz = tmp_path / "proc.npz"
    np.savez(
        npz,
        ids=data.ids,
        n_atoms=data.n_atoms,
        z=data.z,
        positions=data.positions,
        cells=data.cells,
        elastic=data.targets["elastic"],
    )
    loaded = load_crystal_dataset(npz, target_keys=("elastic",))
    assert "elastic" in loaded.targets
    np.testing.assert_array_equal(loaded.offsets, data.offsets)

    split = tmp_path / "split.npz"
    np.savez(split, train=np.array(["mp-2"], dtype=np.str_))
    ids = load_split(split, "train")
    ds = CrystalDataset(loaded, ids)
    assert len(ds) == 1 and ds[0].identifier == "mp-2"
