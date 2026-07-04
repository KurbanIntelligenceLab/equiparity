"""Parity verification gate (work-plan Task 0.3).

The gate probes whether a model is genuinely O(3)-equivariant (parity-respecting) or
SO(3)-only (parity-violating) by transforming an input structure and checking how the
model's internal equivariant features transform. A true SO(3) arm must break reflections
only, never rotations.
"""

from __future__ import annotations
