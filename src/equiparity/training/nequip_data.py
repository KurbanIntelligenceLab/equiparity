"""Shared NequIP data helpers: structure -> AtomicDataDict, batching, neighbor stats.

Used by both the scalar (energy/U0) and vector (dipole) trainers.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.data import chemical_symbols


def element_type_map(z_values: np.ndarray) -> tuple[tuple[str, ...], dict[int, int]]:
    """Return (type_names, z->type-index) for the elements present in the data."""
    unique = sorted(int(z) for z in np.unique(z_values))
    type_names = tuple(chemical_symbols[z] for z in unique)
    return type_names, {z: i for i, z in enumerate(unique)}


def to_atomic_data(structure, symbol_to_type, r_max, dtype):  # noqa: ANN001, ANN201
    """Convert an AtomicStructure into a NequIP AtomicDataDict with atom types set.

    Handles periodic crystals (cell + PBC) and aperiodic molecules alike.
    """
    from ase import Atoms
    from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase

    if structure.pbc and structure.cell is not None:
        atoms = Atoms(
            numbers=structure.atomic_numbers,
            positions=structure.positions,
            cell=structure.cell,
            pbc=True,
        )
    else:
        atoms = Atoms(numbers=structure.atomic_numbers, positions=structure.positions)
    data = compute_neighborlist_(from_ase(atoms), r_max=r_max)
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
        [[symbol_to_type[int(z)]] for z in structure.atomic_numbers], dtype=torch.long
    )
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype)
    return data


def move_batch(batch, device, dtype):  # noqa: ANN001, ANN201
    """Move a batched dict to device, casting floating tensors to the model dtype."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.is_floating_point():
            out[key] = value.to(device=device, dtype=dtype)
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def avg_num_neighbors(graphs) -> float:  # noqa: ANN001
    """Mean number of neighbors per atom over a list of AtomicDataDicts."""
    from nequip.data import AtomicDataDict

    edges = sum(int(g[AtomicDataDict.EDGE_INDEX_KEY].shape[1]) for g in graphs)
    atoms = sum(int(g[AtomicDataDict.POSITIONS_KEY].shape[0]) for g in graphs)
    return max(edges / max(atoms, 1), 1.0)
