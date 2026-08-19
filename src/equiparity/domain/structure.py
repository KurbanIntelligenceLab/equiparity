"""Atomic structure: the typed geometry representation shared across datasets and cores.

Raw files and APIs are untyped; they are converted into :class:`AtomicStructure` at the I/O
boundary, where invariants are checked once. Molecular data (QM9) is non-periodic (``pbc=False``,
``cell=None``); crystal data (Materials Project) is periodic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class StructureValidationError(ValueError):
    """Raised when an atomic structure violates a domain invariant."""


@dataclass(frozen=True, slots=True)
class AtomicStructure:
    """A validated atomic structure.

    Attributes:
        atomic_numbers: Integer atomic numbers, shape ``(n_atoms,)``.
        positions: Cartesian coordinates in angstrom, shape ``(n_atoms, 3)``.
        cell: Lattice vectors in angstrom as rows, shape ``(3, 3)``, or ``None`` if aperiodic.
        pbc: Whether periodic boundary conditions apply.
    """

    atomic_numbers: npt.NDArray[np.int64]
    positions: npt.NDArray[np.float64]
    cell: npt.NDArray[np.float64] | None
    pbc: bool

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.atomic_numbers.shape[0])


def validate_structure(structure: AtomicStructure) -> AtomicStructure:
    """Check structure invariants and return it unchanged, or raise.

    Validates array shapes, atomic-number range, coordinate finiteness, and the cell/pbc
    contract. Never silently repairs or drops data.

    Raises:
        StructureValidationError: If any invariant is violated.
    """
    numbers = structure.atomic_numbers
    positions = structure.positions
    if numbers.ndim != 1:
        raise StructureValidationError(f"atomic_numbers must be 1-D, got shape {numbers.shape}")
    if positions.shape != (numbers.shape[0], 3):
        raise StructureValidationError(
            f"positions must be (n_atoms, 3)=({numbers.shape[0]}, 3), got {positions.shape}"
        )
    if numbers.shape[0] == 0:
        raise StructureValidationError("structure has zero atoms")
    if not np.all((numbers >= 1) & (numbers <= 118)):
        raise StructureValidationError("atomic_numbers outside the valid range [1, 118]")
    if not np.all(np.isfinite(positions)):
        raise StructureValidationError("positions contain NaN or infinity")
    if structure.pbc:
        if structure.cell is None:
            raise StructureValidationError("pbc=True requires a cell")
        if structure.cell.shape != (3, 3):
            raise StructureValidationError(f"cell must be (3, 3), got {structure.cell.shape}")
        if not np.all(np.isfinite(structure.cell)):
            raise StructureValidationError("cell contains NaN or infinity")
        if abs(float(np.linalg.det(structure.cell))) < 1e-8:
            raise StructureValidationError("cell is singular (near-zero volume)")
    elif structure.cell is not None:
        raise StructureValidationError("pbc=False must not carry a cell")
    return structure
