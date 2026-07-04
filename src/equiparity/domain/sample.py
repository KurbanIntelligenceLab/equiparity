"""A structure paired with its prediction targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from equiparity.domain.structure import AtomicStructure


@dataclass(frozen=True, slots=True)
class LabeledStructure:
    """An atomic structure with named target values.

    Attributes:
        structure: The validated geometry.
        targets: Target values keyed by :class:`TargetSpec` name; each array's shape matches
            that target's component layout (scalar ``()``/``(1,)``, vector ``(3,)``, etc.).
        identifier: Stable source identifier (e.g. QM9 index, MP material id).
    """

    structure: AtomicStructure
    targets: Mapping[str, npt.NDArray[np.float64]]
    identifier: str
