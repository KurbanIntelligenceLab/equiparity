"""Convert materials tensors between Voigt/Cartesian form and the e3nn irreps basis.

Training a tensor head needs the target in the same irreps basis the model outputs. The change
of basis comes from :class:`e3nn.o3.ReducedTensorProducts`, which encodes the index symmetries:

- Piezoelectric ``d`` (rank 3, symmetric in the strain pair) -> ``2x1o+1x2o+1x3o`` (18, odd).
- Elastic ``C`` (rank 4, Voigt symmetries) -> ``2x0e+2x2e+1x4e`` (21, even).

The change of basis is orthonormal, so the Frobenius norm is preserved (``||irreps||==||Cart||``).
MP stores these as direct tensor components (no engineering factor-of-2), so Voigt off-diagonals
expand to the symmetric Cartesian tensor directly.
"""

from __future__ import annotations

from functools import cache
from itertools import product

import numpy as np
import numpy.typing as npt

# Voigt index -> Cartesian pair (standard order: xx, yy, zz, yz, xz, xy).
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
_PAIR_TO_VOIGT = {}
for _v, (_i, _j) in enumerate(_VOIGT_PAIRS):
    _PAIR_TO_VOIGT[(_i, _j)] = _v
    _PAIR_TO_VOIGT[(_j, _i)] = _v


@cache
def _change_of_basis(kind: str) -> npt.NDArray[np.float64]:
    """Return the orthonormal change of basis Q mapping irreps coeffs -> Cartesian tensor."""
    from e3nn import o3

    if kind == "piezoelectric":
        rtp = o3.ReducedTensorProducts("ijk=ikj", i="1o", j="1o", k="1o")
    elif kind == "elastic":
        rtp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1o", j="1o", k="1o", l="1o")
    else:
        raise ValueError(f"unknown tensor kind {kind!r}")
    return rtp.change_of_basis.numpy().astype(np.float64)


def piezo_voigt_to_cartesian(voigt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Expand a piezoelectric Voigt tensor (3, 6) into the Cartesian (3, 3, 3)."""
    out = np.zeros((3, 3, 3), dtype=np.float64)
    for i, j, k in product(range(3), repeat=3):
        out[i, j, k] = voigt[i, _PAIR_TO_VOIGT[(j, k)]]
    return out


def elastic_voigt_to_cartesian(voigt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Expand an elastic Voigt tensor (6, 6) into the Cartesian (3, 3, 3, 3)."""
    out = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for i, j, k, m in product(range(3), repeat=4):
        out[i, j, k, m] = voigt[_PAIR_TO_VOIGT[(i, j)], _PAIR_TO_VOIGT[(k, m)]]
    return out


def voigt_to_irreps(voigt: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
    """Convert a Voigt tensor into its irreps coefficients (the training target basis).

    Args:
        voigt: ``(3, 6)`` piezoelectric or ``(6, 6)`` elastic Voigt tensor.
        kind: ``"piezoelectric"`` or ``"elastic"``.

    Returns:
        Irreps coefficients, shape ``(18,)`` or ``(21,)``.
    """
    q = _change_of_basis(kind)  # (n_irreps, 3, ..., 3)
    if kind == "piezoelectric":
        cartesian = piezo_voigt_to_cartesian(voigt)
        return np.einsum("aijk,ijk->a", q, cartesian)
    cartesian = elastic_voigt_to_cartesian(voigt)
    return np.einsum("aijkl,ijkl->a", q, cartesian)
