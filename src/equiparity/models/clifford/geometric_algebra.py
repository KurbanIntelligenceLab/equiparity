"""Signature-parameterized geometric algebra Cl(p, q).

One `GeometricAlgebra` instance owns everything signature-dependent:
- Multivector dimension (2^(p+q))
- Grade ranges (contiguous index blocks per grade)
- Cayley table (dim, dim, dim) of ±1 structure constants
- Reversion signs per grade
- Grade projection, geometric product, reverse, sandwich, rotor-from-bivector
- L=2 feature formula (branches by n_basis)
- Direction embedding (branches by signature and mode)

Cl(3,0) retains the existing unrolled fast-path in models/clifford/clifford.py via a dispatch in
`geometric_product` — used for bit-exact regression with pre-refactor code. Other signatures use
einsum on the Cayley table.
"""

from __future__ import annotations

from math import comb
from typing import List, Literal, Tuple

import torch
import torch.nn as nn


def _blade_bitmasks_by_grade(n: int) -> List[int]:
    """Return all 2^n bitmasks sorted by (grade, bitmask) so grade ranges are contiguous."""
    masks = list(range(1 << n))
    return sorted(masks, key=lambda m: (bin(m).count("1"), m))


def _blade_product(bits_a: int, bits_b: int, p: int) -> Tuple[int, int]:
    """Multiply two basis blades. Returns (result_bits, sign in {-1, +1}).

    Sign = permutation_parity * product(metric[i] for i in A ∩ B),
    where metric[i] = +1 if i < p else -1.
    """
    # Generator lists
    gens_a = [i for i in range(32) if bits_a & (1 << i)]
    gens_b = [i for i in range(32) if bits_b & (1 << i)]
    gens = gens_a + gens_b

    # Bubble-sort parity
    sign = 1
    n_g = len(gens)
    for i in range(n_g):
        for j in range(n_g - 1 - i):
            if gens[j] > gens[j + 1]:
                gens[j], gens[j + 1] = gens[j + 1], gens[j]
                sign = -sign

    # Eliminate adjacent duplicates, multiplying by metric
    result = []
    i = 0
    while i < len(gens):
        if i + 1 < len(gens) and gens[i] == gens[i + 1]:
            # e_i * e_i = +1 if i<p else -1
            if gens[i] >= p:
                sign = -sign
            i += 2
        else:
            result.append(gens[i])
            i += 1

    result_bits = 0
    for g in result:
        result_bits |= 1 << g
    return result_bits, sign


class GeometricAlgebra(nn.Module):
    """Signature-parameterized Clifford algebra Cl(p, q)."""

    def __init__(self, signature: Tuple[int, int]):
        super().__init__()
        p, q = signature
        if p < 0 or q < 0 or p + q == 0:
            raise ValueError(f"invalid signature {signature}")
        self.signature = (p, q)
        self.p = p
        self.q = q
        self.n_basis = p + q
        self.dim = 1 << self.n_basis
        self.max_grade = self.n_basis

        # Blade ordering: contiguous by grade, lexicographic within grade by bitmask
        self._sorted_bitmasks = _blade_bitmasks_by_grade(self.n_basis)
        self._bits_to_idx = {m: i for i, m in enumerate(self._sorted_bitmasks)}

        # Grade dims via binomial; grade ranges contiguous
        self.grade_dims = {k: comb(self.n_basis, k) for k in range(self.n_basis + 1)}
        self.grade_ranges: dict[int, tuple[int, int]] = {}
        cursor = 0
        for k in range(self.n_basis + 1):
            d = self.grade_dims[k]
            self.grade_ranges[k] = (cursor, cursor + d)
            cursor += d
        assert cursor == self.dim

        # Precompute Cayley table and reverse signs as buffers
        self.register_buffer("cayley", self._build_cayley_table())
        self.register_buffer("reverse_signs", self._build_reverse_signs())

    def _build_cayley_table(self) -> torch.Tensor:
        cayley = torch.zeros(self.dim, self.dim, self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                bits_i = self._sorted_bitmasks[i]
                bits_j = self._sorted_bitmasks[j]
                result_bits, sign = _blade_product(bits_i, bits_j, self.p)
                k = self._bits_to_idx[result_bits]
                cayley[i, j, k] = float(sign)
        return cayley

    def _build_reverse_signs(self) -> torch.Tensor:
        signs = torch.zeros(self.dim)
        for k, (s, e) in self.grade_ranges.items():
            val = (-1.0) ** ((k * (k - 1)) // 2)
            signs[s:e] = val
        return signs

    def slice_grade(self, mv: torch.Tensor, k: int) -> torch.Tensor:
        s, e = self.grade_ranges[k]
        return mv[..., s:e]

    def grade_decompose(self, mv: torch.Tensor) -> List[torch.Tensor]:
        return [self.slice_grade(mv, k) for k in range(self.max_grade + 1)]

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """General GP via einsum. Cl(3,0) call sites use the unrolled fast path
        in clifford.CliffordAlgebra directly (bit-exact regression)."""
        return torch.einsum("...i,...j,ijk->...k", a, b, self.cayley)

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        return mv * self.reverse_signs

    def sandwich_product(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.geometric_product(self.geometric_product(r, x), self.reverse(r))

    def rotor_from_bivector(self, bv: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """bv: (..., grade_dims[2]); angle: (..., 1). Returns (..., dim)."""
        half = angle * 0.5
        rotor = torch.zeros(*bv.shape[:-1], self.dim, device=bv.device, dtype=bv.dtype)
        s0, e0 = self.grade_ranges[0]
        rotor[..., s0:e0] = torch.cos(half)
        s2, e2 = self.grade_ranges[2]
        rotor[..., s2:e2] = torch.sin(half) * bv
        return rotor

    def embed_direction(self, r_3d: torch.Tensor, mode: Literal["spatial", "spatial_plus_distance"]) -> torch.Tensor:
        """r_3d: (..., 3). Returns (..., dim) with only grade-1 populated.

        Cl(2,0) uses (x, y)          (caller must project beforehand).
        Cl(3,0) uses (x, y, z).
        Cl(3,1) spatial            → (x, y, z, 0).
        Cl(3,1) spatial_plus_distance → (x, y, z, |r|).
        """
        out = r_3d.new_zeros(*r_3d.shape[:-1], self.dim)
        s, _ = self.grade_ranges[1]
        g1_dim = self.grade_dims[1]
        if self.signature == (2, 0):
            out[..., s] = r_3d[..., 0]
            out[..., s + 1] = r_3d[..., 1]
        elif self.signature == (3, 0):
            out[..., s : s + 3] = r_3d
        elif self.signature == (3, 1):
            out[..., s : s + 3] = r_3d
            if mode == "spatial_plus_distance":
                out[..., s + 3] = torch.linalg.norm(r_3d, dim=-1)
            # else: 4th slot stays 0
        else:
            raise NotImplementedError(f"embed_direction for {self.signature}")
        assert g1_dim == (2 if self.signature == (2, 0) else (3 if self.signature == (3, 0) else 4))
        return out

    def compute_l2_features(self, direction: torch.Tensor) -> torch.Tensor:
        """L=2 symmetric-traceless components of d⊗d. Output dim depends on n_basis.

        Cl(2,0):  3 components [d_x^2 − 1/2, d_xd_y, d_y^2 − 1/2]. Frobenius
                  norm of this is 1/2 for any unit direction (degenerate signal).
        Cl(3,0)/Cl(3,1): 5 components [d_x^2 − 1/3, d_xd_y, d_xd_z, d_y^2 − 1/3, d_yd_z].
        """
        if self.signature == (2, 0):
            dx, dy = direction[..., 0], direction[..., 1]
            half = 0.5
            return torch.stack([dx * dx - half, dx * dy, dy * dy - half], dim=-1)
        # 3D spatial formula for both Cl(3,0) and Cl(3,1)
        dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
        third = 1.0 / 3.0
        return torch.stack(
            [dx * dx - third, dx * dy, dx * dz, dy * dy - third, dy * dz],
            dim=-1,
        )

    def l2_frobenius_norm_sq(self, l2_feat: torch.Tensor) -> torch.Tensor:
        """||S||² for the symmetric-traceless tensor corresponding to `l2_feat`.

        Cl(2,0): 2 * (a² + b²) where feat = [a, b, -a]  (trace-free 2D → S_yy = -S_xx)
        Cl(3,0)/(3,1): S_zz = -S_xx - S_yy; ||S||² = S_xx²+S_yy²+S_zz²+2(S_xy²+S_xz²+S_yz²).
        """
        if self.signature == (2, 0):
            s_xx, s_xy, s_yy = l2_feat.unbind(-1)
            return 2.0 * (s_xx**2 + s_xy**2)
        s_xx, s_xy, s_xz, s_yy, s_yz = l2_feat.unbind(-1)
        s_zz = -s_xx - s_yy
        return s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)
