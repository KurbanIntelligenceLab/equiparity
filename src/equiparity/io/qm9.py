"""QM9 raw parser (dsgdb9nsd .xyz files).

Converts each QM9 molecule file into a validated :class:`LabeledStructure` at the I/O boundary.
Two targets are produced: ``U0`` (parity-even scalar, converted Hartree -> eV) and ``dipole``
(parity-odd vector, formed as sum q_i r_i from the reference Mulliken charges, converted to
Debye). The model predicts the dipole via a direct L=1 head; the charge-based construction is used
only to form the training target.

QM9 uses Fortran ``*^`` exponent notation (e.g. ``2.1938*^-6``); it is normalized before parsing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from equiparity.domain.sample import LabeledStructure
from equiparity.domain.structure import AtomicStructure, validate_structure

HARTREE_TO_EV = 27.211386245988
E_ANGSTROM_TO_DEBYE = 4.803204544

_SYMBOL_TO_Z: dict[str, int] = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
# Index of U0 within the 15 property values on the QM9 comment line (0-based).
_U0_PROPERTY_INDEX = 10
_MU_PROPERTY_INDEX = 3


class QM9ParseError(ValueError):
    """Raised when a QM9 record cannot be parsed."""


def _to_float(token: str) -> float:
    """Parse a QM9 float, normalizing Fortran ``*^`` exponent notation."""
    return float(token.replace("*^", "e"))


@dataclass(frozen=True, slots=True)
class QM9Record:
    """A parsed QM9 molecule with its geometry, charges, and reference dipole magnitude."""

    sample: LabeledStructure
    dipole_magnitude: float  # reference |mu| in Debye, for cross-checking the vector target


def parse_qm9_xyz(text: str) -> QM9Record:
    """Parse one QM9 ``.xyz`` file's contents into a :class:`QM9Record`.

    Args:
        text: Full contents of a ``dsgdb9nsd_*.xyz`` file.

    Returns:
        The parsed and validated record.

    Raises:
        QM9ParseError: If the record is malformed or contains unknown elements.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        raise QM9ParseError("file too short")
    try:
        n_atoms = int(lines[0])
    except ValueError as exc:
        raise QM9ParseError(f"bad atom count: {lines[0]!r}") from exc

    props = lines[1].split()
    # Layout: 'gdb' <index> then 15 property values.
    if len(props) < 2 + 15:
        raise QM9ParseError(f"comment line has too few properties: {lines[1]!r}")
    identifier = props[1]
    property_values = props[2 : 2 + 15]
    u0_ev = _to_float(property_values[_U0_PROPERTY_INDEX]) * HARTREE_TO_EV
    dipole_magnitude = _to_float(property_values[_MU_PROPERTY_INDEX])

    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise QM9ParseError(f"expected {n_atoms} atom lines, got {len(atom_lines)}")

    numbers = np.empty(n_atoms, dtype=np.int64)
    positions = np.empty((n_atoms, 3), dtype=np.float64)
    charges = np.empty(n_atoms, dtype=np.float64)
    for i, line in enumerate(atom_lines):
        fields = line.split()
        if len(fields) != 5:
            raise QM9ParseError(f"atom line {i} malformed: {line!r}")
        symbol = fields[0]
        if symbol not in _SYMBOL_TO_Z:
            raise QM9ParseError(f"unknown element {symbol!r}")
        numbers[i] = _SYMBOL_TO_Z[symbol]
        positions[i] = [_to_float(fields[1]), _to_float(fields[2]), _to_float(fields[3])]
        charges[i] = _to_float(fields[4])

    structure = validate_structure(
        AtomicStructure(atomic_numbers=numbers, positions=positions, cell=None, pbc=False)
    )
    dipole_vector = (charges[:, None] * positions).sum(axis=0) * E_ANGSTROM_TO_DEBYE
    sample = LabeledStructure(
        structure=structure,
        targets={
            "U0": np.array([u0_ev], dtype=np.float64),
            "dipole": dipole_vector.astype(np.float64),
        },
        identifier=identifier,
    )
    return QM9Record(sample=sample, dipole_magnitude=dipole_magnitude)
