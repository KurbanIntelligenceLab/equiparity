"""
Clifford Algebra for Equivariant Neural Networks — Signature-aware refactor.

v3: Signature-parameterized via `GeometricAlgebra`.

Behaviour:
  - `CliffordAlgebra(GeometricAlgebra((3,0)))` keeps the hand-unrolled Cl(3,0)
    geometric product for bit-exact regression with the pre-refactor code.
  - Other signatures dispatch to einsum on `algebra.cayley`.
  - `CliffordLinear`, `CliffordNorm`, `CliffordGateActivation` read grade ranges
    / max_grade / grade_dims from the algebra — no module-level constants.

Memory layout (Cl(3,0) default): [s, e1, e2, e3, e12, e13, e23, e123]
        index:                     0   1   2   3    4    5    6     7
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from equiparity.models.clifford.geometric_algebra import GeometricAlgebra

# ============================================================
# Backward-compat module-level constants for Cl(3,0)-only downstream
# callers (clifford_stf, jetgnn, tests/test_equivariance.py).
# New code should derive these from a GeometricAlgebra instance instead.
# ============================================================

DIM = 8
N_GRADES = 4
GRADE_RANGES = {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}
GRADE_DIMS = {0: 1, 1: 3, 2: 3, 3: 1}
ALL_GRADES = (0, 1, 2, 3)

# Basis-blade indices for Cl(3,0): [s, e1, e2, e3, e12, e13, e23, e123]
S, E1, E2, E3, E12, E13, E23, E123 = range(8)


# Cached default algebra for backward-compat callers that don't pass one.
# Lazily constructed to avoid any import-time ordering issues.
_DEFAULT_ALGEBRA: Optional["GeometricAlgebra"] = None


def _get_default_algebra() -> "GeometricAlgebra":
    global _DEFAULT_ALGEBRA
    if _DEFAULT_ALGEBRA is None:
        _DEFAULT_ALGEBRA = GeometricAlgebra((3, 0))
    return _DEFAULT_ALGEBRA


# ============================================================
# Grade bookkeeping helpers (signature-independent)
# ============================================================


# Lookup: GP(grade_a, grade_b) → set of output grades (Cl(3,0) conventions;
# matches general (p,q) signatures because |a-b| ≤ out ≤ a+b with step 2,
# capped at p+q).
GP_GRADE_TABLE = {
    (0, 0): {0},
    (0, 1): {1},
    (0, 2): {2},
    (0, 3): {3},
    (1, 0): {1},
    (1, 1): {0, 2},
    (1, 2): {1, 3},
    (1, 3): {2},
    (2, 0): {2},
    (2, 1): {1, 3},
    (2, 2): {0, 2},
    (2, 3): {1},
    (3, 0): {3},
    (3, 1): {2},
    (3, 2): {1},
    (3, 3): {0},
}


def compute_gp_output_grades(grades_a: Tuple[int, ...], grades_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute which grades are produced by GP(a, b) given input grades."""
    result: set = set()
    for ga in grades_a:
        for gb in grades_b:
            result |= GP_GRADE_TABLE.get((ga, gb), set())
    return tuple(sorted(result))


def compute_layer_grades(
    n_layers: int,
    edge_grades: Tuple[int, ...] = (0, 1),
    max_grade: int = 3,
) -> List[Tuple[int, ...]]:
    """Progressive grade activation schedule.

    Args:
        max_grade: Cap on highest grade to activate. 1 = scalar+vector only (L=1),
                   3 = full Cl(3,0) (default).
    """
    node_grades: Tuple[int, ...] = (0,)
    layer_grades = []
    for _ in range(n_layers):
        gp_grades = compute_gp_output_grades(node_grades, edge_grades)
        node_grades = tuple(sorted(set(node_grades) | set(gp_grades)))
        node_grades = tuple(g for g in node_grades if g <= max_grade)
        layer_grades.append(node_grades)
    return layer_grades


# ============================================================
# Core Algebra — wraps GeometricAlgebra, Cl(3,0) fast-path
# ============================================================


class CliffordAlgebra(nn.Module):
    """Clifford algebra wrapper holding a GeometricAlgebra instance.

    For signature=(3,0), `geometric_product` uses the existing hand-unrolled
    implementation for bit-exact regression with pre-refactor code. All other
    signatures dispatch to the einsum path on `algebra.cayley`.
    """

    def __init__(self, algebra: Optional["GeometricAlgebra"] = None):
        super().__init__()
        if algebra is None:
            # Backward-compat: no-args → Cl(3,0) default.
            algebra = _get_default_algebra()
        self.algebra = algebra
        self.dim = algebra.dim
        self._is_cl30 = algebra.signature == (3, 0)

    # ---- Full GP ----

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Full GP: a * b. (..., dim) × (..., dim) → (..., dim).

        Single einsum with the algebra's precomputed Cayley tensor. In eager
        mode this is one kernel launch vs 33 (32 element-wise ops +
        torch.stack) for the hand-unrolled Cl(3,0) form. Use
        ``_geometric_product_unrolled`` for benchmarking the old form.
        """
        return torch.einsum("...i,...j,ijk->...k", a, b, self.algebra.cayley)

    @staticmethod
    def _geometric_product_unrolled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hand-unrolled Cl(3,0) GP — kept for benchmarking against einsum.

        Only valid for Cl(3,0) signature; ~33 kernel launches in eager mode.
        Replaced as the default in main; retained here as a legacy reference.
        """
        a0, a1, a2, a3, a4, a5, a6, a7 = a.unbind(-1)
        b0, b1, b2, b3, b4, b5, b6, b7 = b.unbind(-1)

        o0 = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7
        o1 = a0 * b1 + a1 * b0 + a4 * b2 - a2 * b4 + a5 * b3 - a3 * b5 - a6 * b7 - a7 * b6
        o2 = a0 * b2 + a2 * b0 + a1 * b4 - a4 * b1 + a6 * b3 - a3 * b6 + a5 * b7 + a7 * b5
        o3 = a0 * b3 + a3 * b0 + a1 * b5 - a5 * b1 + a2 * b6 - a6 * b2 - a4 * b7 - a7 * b4
        o4 = a0 * b4 + a4 * b0 + a1 * b2 - a2 * b1 + a6 * b5 - a5 * b6 + a3 * b7 + a7 * b3
        o5 = a0 * b5 + a5 * b0 + a1 * b3 - a3 * b1 + a4 * b6 - a6 * b4 - a2 * b7 - a7 * b2
        o6 = a0 * b6 + a6 * b0 + a2 * b3 - a3 * b2 + a5 * b4 - a4 * b5 + a1 * b7 + a7 * b1
        o7 = a0 * b7 + a7 * b0 + a1 * b6 + a6 * b1 - a2 * b5 - a5 * b2 + a3 * b4 + a4 * b3

        return torch.stack([o0, o1, o2, o3, o4, o5, o6, o7], dim=-1)

    # ---- Grade-sparse GP fast paths ----
    # The hand-unrolled grades01 / grades012 variants were Cl(3,0)-only and
    # exhibited the same multi-kernel-launch pathology that motivated the
    # einsum swap of the main GP. With the einsum form, dispatch_gp now
    # routes those grade combinations directly to geometric_product(), which
    # is one kernel launch on any signature. The scalar fast path is kept
    # because it's a single multiply, not a Cayley contraction.

    def gp_scalar_times_mv(self, scalar_mv: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
        s0, e0 = self.algebra.grade_ranges[0]
        return scalar_mv[..., s0:e0] * mv

    def dispatch_gp(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        grades_a: Tuple[int, ...],
        grades_b: Tuple[int, ...],
    ) -> torch.Tensor:
        if grades_a == (0,):
            return self.gp_scalar_times_mv(a, b)
        elif grades_b == (0,):
            return self.gp_scalar_times_mv(b, a)
        else:
            return self.geometric_product(a, b)

    # ---- Other algebra ops ----

    def geometric_product_reference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,...j,ijk->...k", a, b, self.algebra.cayley)

    def sandwich_product(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.geometric_product(self.geometric_product(r, x), self.reverse(r))

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        return mv * self.algebra.reverse_signs

    def grade_select(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        return self.algebra.slice_grade(mv, grade)

    def grade_decompose(self, mv: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(self.algebra.grade_decompose(mv))

    def norm_squared(self, mv: torch.Tensor) -> torch.Tensor:
        prod = self.geometric_product(mv, self.reverse(mv))
        s0, e0 = self.algebra.grade_ranges[0]
        return prod[..., s0:e0]

    def grade_norms(self, mv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        parts = self.algebra.grade_decompose(mv)
        out = []
        for k, part in enumerate(parts):
            # grades 0 and max are scalars (1 component); take abs
            if self.algebra.grade_dims[k] == 1:
                out.append(torch.abs(part))
            else:
                out.append(torch.sqrt(torch.sum(part**2, dim=-1, keepdim=True) + eps))
        return torch.cat(out, dim=-1)

    def rotor_from_bivector(self, bv: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        return self.algebra.rotor_from_bivector(bv, angle)


# ============================================================
# L=2 Feature Augmentation (deprecated shim for Cl(3,0) callers)
# ============================================================


def compute_l2_features(direction: torch.Tensor) -> torch.Tensor:
    """DEPRECATED shim — use GeometricAlgebra.compute_l2_features.

    Keeps Cl(3,0) behavior for any call site that has not yet moved to the algebra
    method directly.
    """
    dx, dy, dz = direction.unbind(-1)
    third = 1.0 / 3.0
    return torch.stack(
        [dx * dx - third, dx * dy, dx * dz, dy * dy - third, dy * dz],
        dim=-1,
    )


# ============================================================
# Multivector Construction (signature-aware)
# ============================================================


def make_scalar_mv(s: torch.Tensor, algebra: Optional["GeometricAlgebra"] = None) -> torch.Tensor:
    """(*, C) scalar → (*, C, algebra.dim) with only grade-0 populated."""
    alg = algebra if algebra is not None else _get_default_algebra()
    out = s.new_zeros(*s.shape, alg.dim)
    s0, e0 = alg.grade_ranges[0]
    out[..., s0:e0] = s.unsqueeze(-1)
    return out


def make_vec_mv(v: torch.Tensor, algebra: Optional["GeometricAlgebra"] = None) -> torch.Tensor:
    """(*, C, grade_dims[1]) vector → (*, C, algebra.dim) grade-1 only."""
    alg = algebra if algebra is not None else _get_default_algebra()
    out = v.new_zeros(*v.shape[:-1], alg.dim)
    s1, e1 = alg.grade_ranges[1]
    out[..., s1:e1] = v
    return out


def make_grades01_mv(
    g0: torch.Tensor,
    g1: torch.Tensor,
    algebra: Optional["GeometricAlgebra"] = None,
) -> torch.Tensor:
    """(*, C, 1) scalar + (*, C, grade_dims[1]) vector → (*, C, algebra.dim)."""
    alg = algebra if algebra is not None else _get_default_algebra()
    out = g0.new_zeros(*g0.shape[:-1], alg.dim)
    s0, e0 = alg.grade_ranges[0]
    s1, e1 = alg.grade_ranges[1]
    out[..., s0:e0] = g0
    out[..., s1:e1] = g1
    return out


# ============================================================
# Grade-Aware Neural Network Layers — signature-aware
# ============================================================


class CliffordLinear(nn.Module):
    """Grade-preserving linear map. One weight matrix per active grade."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        bias: bool = True,
        active_grades: Optional[Tuple[int, ...]] = None,
        algebra: Optional["GeometricAlgebra"] = None,
    ):
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.c_in = c_in
        self.c_out = c_out
        self.algebra = algebra
        if active_grades is None:
            active_grades = tuple(range(algebra.max_grade + 1))
        self.active = tuple(sorted(active_grades))

        for g in self.active:
            w = nn.Parameter(torch.empty(c_out, c_in))
            nn.init.normal_(w, std=c_in**-0.5)
            setattr(self, f"w{g}", w)

        # Biases on scalar (grade 0) and top-grade pseudoscalar
        top_grade = algebra.max_grade
        self.has_b0 = bias and 0 in self.active
        self.has_b_top = bias and top_grade in self.active and top_grade != 0
        if self.has_b0:
            self.b0 = nn.Parameter(torch.zeros(c_out, 1))
        if self.has_b_top:
            self.b_top = nn.Parameter(torch.zeros(c_out, 1))
        self._top_grade = top_grade

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        out = mv.new_zeros(*mv.shape[:-2], self.c_out, self.algebra.dim)
        for g in self.active:
            s, e = self.algebra.grade_ranges[g]
            w = getattr(self, f"w{g}")
            out[..., s:e] = torch.einsum("oc,...cd->...od", w, mv[..., s:e])
        if self.has_b0:
            s0, e0 = self.algebra.grade_ranges[0]
            out[..., s0:e0] = out[..., s0:e0] + self.b0
        if self.has_b_top:
            st, et = self.algebra.grade_ranges[self._top_grade]
            out[..., st:et] = out[..., st:et] + self.b_top
        return out


class CliffordNorm(nn.Module):
    """Grade-wise normalization."""

    def __init__(
        self,
        n_channels: int,
        active_grades: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-8,
        algebra: Optional["GeometricAlgebra"] = None,
    ):
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.eps = eps
        self.algebra = algebra
        if active_grades is None:
            active_grades = tuple(range(algebra.max_grade + 1))
        self.active = set(active_grades)
        self.n_channels = n_channels
        self._top_grade = algebra.max_grade

        for g in range(algebra.max_grade + 1):
            if g in self.active:
                setattr(self, f"s{g}", nn.Parameter(torch.ones(n_channels, 1)))

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        slices = []
        for g in range(self.algebra.max_grade + 1):
            s, e = self.algebra.grade_ranges[g]
            x = mv[..., s:e]
            if g not in self.active:
                slices.append(x)
                continue
            scale = getattr(self, f"s{g}")
            # grades with dim==1 (scalar, pseudoscalar) use mean/std normalization
            if self.algebra.grade_dims[g] == 1:
                mean = x.mean(dim=-2, keepdim=True)
                std = x.std(dim=-2, keepdim=True) + self.eps
                slices.append(scale * (x - mean) / std)
            else:
                ch_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
                rms = torch.sqrt(torch.mean(ch_norm_sq, dim=-2, keepdim=True) + self.eps)
                slices.append(scale * x / rms)
        return torch.cat(slices, dim=-1)


class CliffordGateActivation(nn.Module):
    """Equivariant norm-gated activation."""

    def __init__(
        self,
        n_channels: int,
        active_grades: Optional[Tuple[int, ...]] = None,
        algebra: Optional["GeometricAlgebra"] = None,
    ):
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.algebra = algebra
        if active_grades is None:
            active_grades = tuple(range(algebra.max_grade + 1))
        self.active = set(active_grades)
        self.scalar_act = nn.SiLU()

        # Gate any grade with dim > 1 (not scalar or pseudoscalar)
        for g in range(1, algebra.max_grade):  # skip 0 (scalar) and top grade if it's 1-dim
            if g in self.active and algebra.grade_dims[g] > 1:
                gate = nn.Sequential(
                    nn.Linear(n_channels, n_channels),
                    nn.SiLU(),
                    nn.Linear(n_channels, n_channels),
                    nn.Sigmoid(),
                )
                setattr(self, f"gate_g{g}", gate)

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        slices = []
        for g in range(self.algebra.max_grade + 1):
            s, e = self.algebra.grade_ranges[g]
            x = mv[..., s:e]
            if g not in self.active:
                slices.append(x)
                continue
            if self.algebra.grade_dims[g] == 1:
                slices.append(self.scalar_act(x))
            else:
                gate_fn = getattr(self, f"gate_g{g}")
                x_norm = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-8)
                gate = gate_fn(x_norm.squeeze(-1)).unsqueeze(-1)
                slices.append(x * gate)
        return torch.cat(slices, dim=-1)


# ============================================================
# Backward-compat: Cl(3,0) equivariance smoke check
# ============================================================


def test_equivariance(model_fn=None, n_atoms=5, n_channels=16, atol=1e-5, seed=42):
    """Numerically verify O(3) equivariance (Cl(3,0) backward-compat shim)."""
    torch.manual_seed(seed)
    alg = CliffordAlgebra()

    a = torch.randn(n_atoms, n_channels, 8)
    b = torch.randn(n_atoms, n_channels, 8)

    bv = torch.randn(3)
    bv = bv / (bv.norm() + 1e-8)
    angle = torch.tensor([torch.pi * torch.rand(1).item()])
    rotor = alg.rotor_from_bivector(bv.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)

    def rotate(mv):
        return alg.sandwich_product(mv, rotor.expand_as(mv))

    results = {}

    gp_then_rot = rotate(alg.geometric_product(a, b))
    rot_then_gp = alg.geometric_product(rotate(a), rotate(b))
    gp_err = (gp_then_rot - rot_then_gp).abs().max().item()
    results["gp_equivariance_error"] = gp_err
    results["gp_equivariant"] = gp_err < atol

    norms_orig = alg.grade_norms(a)
    norms_rot = alg.grade_norms(rotate(a))
    norm_err = (norms_orig - norms_rot).abs().max().item()
    results["grade_norm_invariance_error"] = norm_err
    results["grade_norms_invariant"] = norm_err < atol

    if model_fn is not None:
        with torch.no_grad():
            out_then_rot = rotate(model_fn(a, b))
            rot_then_out = model_fn(rotate(a), rotate(b))
        model_err = (out_then_rot - rot_then_out).abs().max().item()
        results["model_equivariance_error"] = model_err
        results["model_equivariant"] = model_err < atol

    return results
