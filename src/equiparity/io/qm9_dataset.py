"""Load the processed QM9 dataset and its split into typed samples.

Reads the concatenated archive written by ``scripts/prepare_qm9.py`` and reconstructs
:class:`LabeledStructure` records on demand, restricted to a named split partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from equiparity.domain.sample import LabeledStructure
from equiparity.domain.structure import AtomicStructure


@dataclass(frozen=True, slots=True)
class QM9Data:
    """The processed QM9 arrays in concatenated form, indexed by molecule offsets."""

    ids: npt.NDArray[np.int64]
    n_atoms: npt.NDArray[np.int64]
    u0: npt.NDArray[np.float64]
    dipole: npt.NDArray[np.float64]
    z: npt.NDArray[np.int64]
    positions: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.int64]


def load_qm9(processed_npz: Path) -> QM9Data:
    """Load the processed QM9 archive and precompute per-molecule atom offsets."""
    with np.load(processed_npz) as raw:
        n_atoms = raw["n_atoms"].astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(n_atoms)]).astype(np.int64)
        return QM9Data(
            ids=raw["ids"].astype(np.int64),
            n_atoms=n_atoms,
            u0=raw["U0"].astype(np.float64),
            dipole=raw["dipole"].astype(np.float64),
            z=raw["z"].astype(np.int64),
            positions=raw["positions"].astype(np.float64),
            offsets=offsets,
        )


def load_split(split_npz: Path, partition: str) -> npt.NDArray[np.int64]:
    """Return the QM9 molecule ids for a split partition (``train``/``val``/``test``)."""
    with np.load(split_npz) as raw:
        if partition not in raw:
            raise KeyError(f"partition {partition!r} not in split; have {list(raw)}")
        ids: npt.NDArray[np.int64] = raw[partition].astype(np.int64)
        return ids


class QM9Dataset:
    """A QM9 split partition as a sequence of :class:`LabeledStructure` records."""

    def __init__(self, data: QM9Data, partition_ids: npt.NDArray[np.int64]) -> None:
        """Restrict ``data`` to the molecules in ``partition_ids``, preserving that order."""
        id_to_row = {int(mol_id): row for row, mol_id in enumerate(data.ids)}
        missing = [int(i) for i in partition_ids if int(i) not in id_to_row]
        if missing:
            raise KeyError(
                f"{len(missing)} split ids absent from processed data, e.g. {missing[:3]}"
            )
        self._data = data
        self._rows = np.array([id_to_row[int(i)] for i in partition_ids], dtype=np.int64)

    def __len__(self) -> int:
        return int(self._rows.shape[0])

    def __getitem__(self, index: int) -> LabeledStructure:
        row = int(self._rows[index])
        start = int(self._data.offsets[row])
        stop = start + int(self._data.n_atoms[row])
        structure = AtomicStructure(
            atomic_numbers=self._data.z[start:stop].copy(),
            positions=self._data.positions[start:stop].copy(),
            cell=None,
            pbc=False,
        )
        return LabeledStructure(
            structure=structure,
            targets={
                "U0": self._data.u0[row : row + 1].copy(),
                "dipole": self._data.dipole[row].copy(),
            },
            identifier=str(int(self._data.ids[row])),
        )
