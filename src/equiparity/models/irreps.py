"""Parity-labelled irreps strings for the O(3)/SO(3) matched pair.

The matched pair differs only in how degrees are parity-labelled: O(3) uses natural
spherical-harmonic parity ``(-1)**l`` (``0e,1o,2e,...``); SO(3) labels every degree even
(``0e,1e,2e,...``), the same geometric content with parity stripped out so e3nn stops enforcing
it. Both the edge spherical harmonics and hidden features use these.
"""

from __future__ import annotations

from equiparity.domain.parity import ParityMode


def degree_irreps(l_max: int, mult: int, mode: ParityMode) -> str:
    """Return an irreps string over degrees ``0..l_max`` for a parity mode.

    Args:
        l_max: Maximum spherical-harmonic degree.
        mult: Multiplicity (channels) per degree.
        mode: O(3) uses natural parity; SO(3) labels every degree even.
    """
    terms = []
    for degree in range(l_max + 1):
        even = not mode.has_parity or degree % 2 == 0
        parity = "e" if even else "o"
        terms.append(f"{mult}x{degree}{parity}")
    return " + ".join(terms)


def output_irreps(o3_irreps: str, mode: ParityMode) -> str:
    """Return the output-head irreps for a parity mode.

    The O(3) arm uses the target's true irreps (with its physical parity labels). The SO(3) arm
    relabels every term even (``o`` -> ``e``), which strips parity: the head can then produce a
    nonzero value for a symmetry-forbidden odd tensor. This is what makes O(3) give exact zeros
    on centrosymmetric crystals while SO(3) does not.
    """
    if mode.has_parity:
        return o3_irreps
    return o3_irreps.replace("o", "e")
