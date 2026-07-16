"""Tests for the QM9 parser and dataset loader (no external data required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equiparity.io.qm9 import E_ANGSTROM_TO_DEBYE, HARTREE_TO_EV, parse_qm9_xyz
from equiparity.io.qm9_dataset import QM9Data, QM9Dataset, load_qm9, load_split

# A synthetic 2-atom QM9 record: 15 property values, U0 at index 10 (= -40.0 Hartree),
# mu at index 3 (= 2.0). Atom lines: symbol x y z mulliken_charge.
_SYNTH_XYZ = (
    "2\n"
    "gdb 42\t1.0\t1.0\t1.0\t2.0\t0\t0\t0\t0\t0\t0\t-40.0\t0\t0\t0\t0\n"
    "H\t0.0\t0.0\t0.0\t0.5\n"
    "O\t1.0\t0.0\t0.0\t-0.5\n"
    "extra frequency line\n"
)


def test_parse_u0_conversion() -> None:
    record = parse_qm9_xyz(_SYNTH_XYZ)
    assert record.sample.identifier == "42"
    np.testing.assert_allclose(record.sample.targets["U0"][0], -40.0 * HARTREE_TO_EV)
    assert record.dipole_magnitude == 2.0


def test_parse_dipole_vector_from_charges() -> None:
    record = parse_qm9_xyz(_SYNTH_XYZ)
    # sum q*r = 0.5*(0,0,0) + (-0.5)*(1,0,0) = (-0.5, 0, 0) e.Angstrom, then to Debye.
    expected = np.array([-0.5 * E_ANGSTROM_TO_DEBYE, 0.0, 0.0])
    np.testing.assert_allclose(record.sample.targets["dipole"], expected)


def test_fortran_exponent_notation() -> None:
    text = _SYNTH_XYZ.replace("0.5\n", "2.0*^-1\n", 1)  # 2.0*^-1 == 0.2
    record = parse_qm9_xyz(text)
    assert record.sample.structure.n_atoms == 2


def _synthetic_data() -> QM9Data:
    # Three molecules with 2, 3, 2 atoms.
    n_atoms = np.array([2, 3, 2], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_atoms)]).astype(np.int64)
    return QM9Data(
        ids=np.array([10, 20, 30], dtype=np.int64),
        n_atoms=n_atoms,
        u0=np.array([-1.0, -2.0, -3.0]),
        dipole=np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]),
        z=np.array([1, 8, 6, 1, 1, 7, 8], dtype=np.int64),
        positions=np.arange(7 * 3, dtype=np.float64).reshape(7, 3),
        offsets=offsets,
    )


def test_dataset_reconstructs_by_id_order() -> None:
    data = _synthetic_data()
    ds = QM9Dataset(data, np.array([30, 10], dtype=np.int64))
    assert len(ds) == 2
    first = ds[0]
    assert first.identifier == "30"
    assert first.structure.n_atoms == 2
    np.testing.assert_array_equal(first.structure.atomic_numbers, [7, 8])
    np.testing.assert_allclose(first.targets["dipole"], [0.0, 0.0, 0.3])
    assert ds[1].identifier == "10"


def test_load_roundtrip(tmp_path: Path) -> None:
    data = _synthetic_data()
    npz = tmp_path / "proc.npz"
    np.savez(
        npz,
        ids=data.ids,
        n_atoms=data.n_atoms,
        U0=data.u0,
        dipole=data.dipole,
        z=data.z,
        positions=data.positions,
    )
    loaded = load_qm9(npz)
    np.testing.assert_array_equal(loaded.offsets, data.offsets)

    split = tmp_path / "split.npz"
    np.savez(split, train=np.array([10, 20]), test=np.array([30]))
    train_ids = load_split(split, "train")
    assert list(train_ids) == [10, 20]
    assert len(QM9Dataset(loaded, train_ids)) == 2
