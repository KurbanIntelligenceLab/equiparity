"""Load processed Materials Project crystal datasets (elastic, piezoelectric, OOD).

Reads the concatenated archives written by ``scripts/prepare_mp.py`` and reconstructs periodic
:class:`LabeledStructure` records. Handles an optional tensor target (absent for the OOD set,
whose target is exactly zero by symmetry).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from equiparity.domain.sample import LabeledStructure
from equiparity.domain.structure import AtomicStructure


@dataclass(frozen=True, slots=True)
class CrystalData:
    """Concatenated crystal geometry with per-structure offsets and optional flat targets."""

    ids: npt.NDArray[np.str_]
    n_atoms: npt.NDArray[np.int64]
    z: npt.NDArray[np.int64]
    positions: npt.NDArray[np.float64]
    cells: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.int64]
    targets: Mapping[str, npt.NDArray[np.float64]] = field(default_factory=dict)


def load_crystal_dataset(processed_npz: Path, target_keys: tuple[str, ...] = ()) -> CrystalData:
    """Load a processed crystal archive, precomputing per-structure atom offsets."""
    with np.load(processed_npz, allow_pickle=False) as raw:
        n_atoms = raw["n_atoms"].astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(n_atoms)]).astype(np.int64)
        targets = {key: raw[key].astype(np.float64) for key in target_keys}
        return CrystalData(
            ids=raw["ids"].astype(np.str_),
            n_atoms=n_atoms,
            z=raw["z"].astype(np.int64),
            positions=raw["positions"].astype(np.float64),
            cells=raw["cells"].astype(np.float64),
            offsets=offsets,
            targets=targets,
        )


def load_split(split_npz: Path, partition: str) -> npt.NDArray[np.str_]:
    """Return the material ids for a split partition (``train``/``val``/``test``)."""
    with np.load(split_npz, allow_pickle=False) as raw:
        if partition not in raw:
            raise KeyError(f"partition {partition!r} not in split; have {list(raw)}")
        ids: npt.NDArray[np.str_] = raw[partition].astype(np.str_)
        return ids


class CrystalDataset:
    """A crystal dataset (optionally a split partition) as :class:`LabeledStructure` records."""

    def __init__(
        self, data: CrystalData, partition_ids: npt.NDArray[np.str_] | None = None
    ) -> None:
        """Restrict to ``partition_ids`` (or use all structures when ``None``)."""
        id_to_row = {str(mid): row for row, mid in enumerate(data.ids)}
        if partition_ids is None:
            rows = np.arange(len(data.ids), dtype=np.int64)
        else:
            missing = [str(i) for i in partition_ids if str(i) not in id_to_row]
            if missing:
                raise KeyError(f"{len(missing)} split ids absent from data, e.g. {missing[:3]}")
            rows = np.array([id_to_row[str(i)] for i in partition_ids], dtype=np.int64)
        self._data = data
        self._rows = rows

    def __len__(self) -> int:
        return int(self._rows.shape[0])

    def __getitem__(self, index: int) -> LabeledStructure:
        row = int(self._rows[index])
        start = int(self._data.offsets[row])
        stop = start + int(self._data.n_atoms[row])
        structure = AtomicStructure(
            atomic_numbers=self._data.z[start:stop].copy(),
            positions=self._data.positions[start:stop].copy(),
            cell=self._data.cells[row].copy(),
            pbc=True,
        )
        targets = {key: values[row].copy() for key, values in self._data.targets.items()}
        return LabeledStructure(
            structure=structure, targets=targets, identifier=str(self._data.ids[row])
        )
