"""Centrosymmetric space groups — the load-bearing filter for the piezoelectric OOD set.

A crystal is centrosymmetric iff its space group contains an inversion center. By Neumann's
principle every odd-parity property (piezoelectric tensor) is then exactly zero. The OOD
evaluation set is built only from these space groups; a single non-centrosymmetric leak
invalidates the headline figure, so membership is verified per structure with spglib.
"""

from __future__ import annotations

# The 92 centrosymmetric space-group numbers (the 11 centrosymmetric Laue crystal classes).
CENTROSYMMETRIC_SPACE_GROUPS: frozenset[int] = frozenset(
    (
        2,  # -1 (triclinic)
        *range(10, 16),  # 2/m (monoclinic): 10-15
        *range(47, 75),  # mmm (orthorhombic): 47-74
        *range(83, 89),  # 4/m (tetragonal): 83-88
        *range(123, 143),  # 4/mmm: 123-142
        147,
        148,  # -3 (trigonal)
        *range(162, 168),  # -3m: 162-167
        175,
        176,  # 6/m (hexagonal)
        *range(191, 195),  # 6/mmm: 191-194
        *range(200, 207),  # m-3 (cubic): 200-206
        *range(221, 231),  # m-3m: 221-230
    )
)


def is_centrosymmetric(space_group_number: int) -> bool:
    """Whether a space-group number is centrosymmetric (contains an inversion center)."""
    return space_group_number in CENTROSYMMETRIC_SPACE_GROUPS
