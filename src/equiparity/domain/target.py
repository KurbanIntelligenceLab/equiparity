"""Prediction targets and their parity character — the scientific spine of the study.

Each target has a physical parity (how it transforms under spatial inversion) and an e3nn
irreps decomposition for the model's output head. The paper's thesis is a spectrum ordered by
parity character, not tensor rank: even scalar (U0) -> odd vector (dipole) -> even tensor
(elastic) -> odd tensor (piezoelectric). O(3) models reproduce the symmetry-forced zeros of the
odd targets; SO(3) models do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TargetParity(StrEnum):
    """Physical parity of a target under spatial inversion."""

    EVEN = "even"  # unchanged under inversion (e.g. energy, elastic tensor)
    ODD = "odd"  # flips sign under inversion (e.g. dipole, piezoelectric tensor)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Definition of a prediction target.

    Attributes:
        name: Short identifier used in configs and result tables.
        irreps: e3nn irreps of the output head (its parity labels encode ``parity``).
        parity: Physical parity under inversion.
        n_components: Number of scalar components in the target.
        unit: Physical unit, for provenance and reporting.
        description: Human-readable description.
    """

    name: str
    irreps: str
    parity: TargetParity
    n_components: int
    unit: str
    description: str


# The four study targets, in parity-spectrum order (work plan Task 1, Fig 2).
U0 = TargetSpec(
    name="U0",
    irreps="1x0e",
    parity=TargetParity.EVEN,
    n_components=1,
    unit="eV",
    description="Internal energy at 0 K — parity-even scalar control.",
)
DIPOLE = TargetSpec(
    name="dipole",
    irreps="1x1o",
    parity=TargetParity.ODD,
    n_components=3,
    unit="Debye",
    description="Molecular dipole vector — parity-odd; direct L=1 head, not charge x position.",
)
ELASTIC = TargetSpec(
    name="elastic",
    irreps="2x0e + 2x2e + 1x4e",
    parity=TargetParity.EVEN,
    n_components=21,
    unit="GPa",
    description="Rank-4 elasticity tensor (Voigt) — parity-even tensor control.",
)
PIEZOELECTRIC = TargetSpec(
    name="piezoelectric",
    irreps="2x1o + 1x2o + 1x3o",
    parity=TargetParity.ODD,
    n_components=18,
    unit="C/m^2",
    description="Rank-3 piezoelectric tensor (DFPT 3x6) — parity-odd tensor, the headline target.",
)

TARGETS: dict[str, TargetSpec] = {t.name: t for t in (U0, DIPOLE, ELASTIC, PIEZOELECTRIC)}
