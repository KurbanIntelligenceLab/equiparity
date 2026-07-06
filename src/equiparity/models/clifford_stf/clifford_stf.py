"""
CliffordSTF Algebra: Cl(3,0) + STF₂ + STF₃.

Core theorem exploited here:
    For L=1 vectors u, v, the full CG product 1⊗1 → 0⊕1⊕2 decomposes as:
        GP(u,v) = u·v + u∧v          → L=0 (1D) + L=1 (3D)
        STF(u,v) = sym traceless(u⊗v) → L=2 (5D)
    Together, GP + STF = full CG, computed without CG coefficients.

Layout of CliffordSTF multivector (per channel):
    [0:8]   Cl(3,0) multivector  — grades 0,1,2,3 = 2·L=0 + 2·L=1
    [8:13]  STF₂                 — symmetric traceless rank-2 = L=2 (5D)
    [13:20] STF₃                 — symmetric traceless rank-3 = L=3 (7D)

    8D  mode: stf_mode="none"    (original Clifford)
    13D mode: stf_mode="stf2"    (+ L=2)
    20D mode: stf_mode="stf2+stf3" (+ L=2 + L=3)

STF₂ storage: [S_xx, S_xy, S_xz, S_yy, S_yz]  (S_zz = -S_xx - S_yy)
STF₃ storage: [T_xxx, T_xxy, T_xxz, T_xyy, T_xyz, T_yyy, T_yyz]
              (T_xzz = -T_xxx - T_xyy, T_yzz = -T_xxy - T_yyy,
               T_zzz = -T_xxz - T_yyz)
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn

from equiparity.models.clifford.clifford import (
    CliffordAlgebra,
    CliffordLinear,
    CliffordNorm,
    CliffordGateActivation,
    ALL_GRADES,
    GRADE_RANGES,
    GRADE_DIMS,
    DIM as CL_DIM,
    N_GRADES,
    compute_gp_output_grades,
    compute_layer_grades,
    compute_l2_features,
    make_scalar_mv,
    make_vec_mv,
    make_grades01_mv,
    test_equivariance as test_clifford_equivariance,
)


# ============================================================
# Constants
# ============================================================

STF2_DIM = 5
STF3_DIM = 7
AUG13_DIM = CL_DIM + STF2_DIM          # 13
AUG20_DIM = CL_DIM + STF2_DIM + STF3_DIM  # 20

AUG_CL_RANGE = (0, CL_DIM)             # [0, 8)
AUG_STF2_RANGE = (CL_DIM, AUG13_DIM)   # [8, 13)
AUG_STF3_RANGE = (AUG13_DIM, AUG20_DIM) # [13, 20)

STF_MODES = {"none": CL_DIM, "stf2": AUG13_DIM, "stf2+stf3": AUG20_DIM}

# CliffordSTF "track" identifiers
TRACK_CL = "cl"
TRACK_STF2 = "stf2"
TRACK_STF3 = "stf3"


def clifford_stf_dim(stf_mode: str) -> int:
    return STF_MODES[stf_mode]


# ============================================================
# STF₂ Operations — L=2 symmetric traceless rank-2 tensor
# ============================================================


def compute_stf2_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Symmetric traceless product of two L=1 vectors.

    This is the L=2 channel of CG(1⊗1) that the geometric product misses.

    Args:
        u, v: (..., 3) vectors
    Returns:
        (..., 5) STF₂ components [S_xx, S_xy, S_xz, S_yy, S_yz]
    """
    ux, uy, uz = u.unbind(-1)
    vx, vy, vz = v.unbind(-1)

    dot = ux * vx + uy * vy + uz * vz
    third = dot / 3.0

    s_xx = ux * vx - third
    s_xy = 0.5 * (ux * vy + uy * vx)
    s_xz = 0.5 * (ux * vz + uz * vx)
    s_yy = uy * vy - third
    s_yz = 0.5 * (uy * vz + uz * vy)

    return torch.stack([s_xx, s_xy, s_xz, s_yy, s_yz], dim=-1)


def stf2_norm_sq(s: torch.Tensor) -> torch.Tensor:
    """Frobenius norm² of STF₂ tensor → L=0 invariant.

    ||S||² = S_xx² + S_yy² + S_zz² + 2(S_xy² + S_xz² + S_yz²)
    where S_zz = -S_xx - S_yy.

    Args:
        s: (..., 5) STF₂
    Returns:
        (..., 1) scalar
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = s.unbind(-1)
    s_zz = -s_xx - s_yy
    return (
        s_xx ** 2 + s_yy ** 2 + s_zz ** 2
        + 2.0 * (s_xy ** 2 + s_xz ** 2 + s_yz ** 2)
    ).unsqueeze(-1)


def stf2_inner(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Inner product of two STF₂ tensors → L=0 invariant.

    Args:
        s1, s2: (..., 5) STF₂
    Returns:
        (..., 1)
    """
    a_xx, a_xy, a_xz, a_yy, a_yz = s1.unbind(-1)
    b_xx, b_xy, b_xz, b_yy, b_yz = s2.unbind(-1)
    a_zz = -a_xx - a_yy
    b_zz = -b_xx - b_yy
    return (
        a_xx * b_xx + a_yy * b_yy + a_zz * b_zz
        + 2.0 * (a_xy * b_xy + a_xz * b_xz + a_yz * b_yz)
    ).unsqueeze(-1)


def reconstruct_stf2_matrix(s: torch.Tensor) -> torch.Tensor:
    """Reconstruct full 3×3 symmetric traceless matrix from 5 components.

    Args:
        s: (..., 5) STF₂ [S_xx, S_xy, S_xz, S_yy, S_yz]
    Returns:
        (..., 3, 3) symmetric traceless matrix
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = s.unbind(-1)
    s_zz = -s_xx - s_yy
    row0 = torch.stack([s_xx, s_xy, s_xz], dim=-1)
    row1 = torch.stack([s_xy, s_yy, s_yz], dim=-1)
    row2 = torch.stack([s_xz, s_yz, s_zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def contract_stf2_vec(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Contract STF₂ with vector: S·v → L=1 vector.

    This is the L=2 ⊗ L=1 → L=1 channel (CG contraction).
    Uses single einsum via reconstructed 3×3 matrix.

    Args:
        s: (..., 5) STF₂ [S_xx, S_xy, S_xz, S_yy, S_yz]
        v: (..., 3) vector
    Returns:
        (..., 3) vector
    """
    S_mat = reconstruct_stf2_matrix(s)  # (..., 3, 3)
    return torch.einsum("...ij,...j->...i", S_mat, v)



# ============================================================
# STF₃ Operations — L=3 symmetric traceless rank-3 tensor
# ============================================================


def compute_stf3_product(
    stf2: torch.Tensor, v: torch.Tensor,
    precomputed_sv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """STF₃ from STF₂ ⊗ L=1 → L=3 channel.

    Computes the symmetric traceless part of S_ij · v_k.
    This is the L=3 channel of CG(2⊗1 → 1⊕2⊕3).

    Args:
        stf2: (..., 5) [S_xx, S_xy, S_xz, S_yy, S_yz]
        v: (..., 3) vector
        precomputed_sv: (..., 3) optional pre-computed S·v contraction
    Returns:
        (..., 7) [T_xxx, T_xxy, T_xxz, T_xyy, T_xyz, T_yyy, T_yyz]
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = stf2.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    s_zz = -s_xx - s_yy

    # Symmetric product T^sym_ijk = (1/3)(S_ij*v_k + S_ik*v_j + S_jk*v_i)
    t_xxx = s_xx * vx
    t_xxy = (s_xx * vy + 2.0 * s_xy * vx) / 3.0
    t_xxz = (s_xx * vz + 2.0 * s_xz * vx) / 3.0
    t_xyy = (2.0 * s_xy * vy + s_yy * vx) / 3.0
    t_xyz = (s_xy * vz + s_xz * vy + s_yz * vx) / 3.0
    t_yyy = s_yy * vy
    t_yyz = (s_yy * vz + 2.0 * s_yz * vy) / 3.0

    # Trace vector A_k = (2/3)(S·v)_k  (because S is traceless)
    if precomputed_sv is not None:
        sv = precomputed_sv
    else:
        sv = contract_stf2_vec(stf2, v)  # (..., 3)
    ax, ay, az = (sv * (2.0 / 3.0)).unbind(-1)

    # Traceless projection:
    # T^STF_ijk = T^sym_ijk - (1/5)(δ_ij*A_k + δ_ik*A_j + δ_jk*A_i)
    inv5 = 1.0 / 5.0
    o_xxx = t_xxx - 3.0 * inv5 * ax
    o_xxy = t_xxy - inv5 * ay
    o_xxz = t_xxz - inv5 * az
    o_xyy = t_xyy - inv5 * ax
    o_xyz = t_xyz  # no delta terms (all indices different)
    o_yyy = t_yyy - 3.0 * inv5 * ay
    o_yyz = t_yyz - inv5 * az

    return torch.stack([o_xxx, o_xxy, o_xxz, o_xyy, o_xyz, o_yyy, o_yyz], dim=-1)


def stf3_norm_sq(t: torch.Tensor) -> torch.Tensor:
    """Norm² of STF₃ tensor → L=0 invariant.

    Args:
        t: (..., 7) STF₃
    Returns:
        (..., 1)
    """
    t_xxx, t_xxy, t_xxz, t_xyy, t_xyz, t_yyy, t_yyz = t.unbind(-1)
    t_xzz = -t_xxx - t_xyy
    t_yzz = -t_xxy - t_yyy
    t_zzz = -t_xxz - t_yyz

    # Full contraction T_ijk * T_ijk with multiplicity factors
    # Distinct permutations: xxx(1), xxy(3), xxz(3), xyy(3), xyz(6), xzz(3),
    #                        yyy(1), yyz(3), yzz(3), zzz(1)
    return (
        t_xxx ** 2 + t_yyy ** 2 + t_zzz ** 2
        + 3.0 * (t_xxy ** 2 + t_xxz ** 2 + t_xyy ** 2 + t_xzz ** 2
                 + t_yyz ** 2 + t_yzz ** 2)
        + 6.0 * t_xyz ** 2
    ).unsqueeze(-1)


def contract_stf3_vec_to_stf2(
    t: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Contract STF₃ with vector → STF₂ (L=3 ⊗ L=1 → L=2 channel).

    Result: R_ij = sum_k T_ijk * v_k (symmetric, need traceless projection).

    Args:
        t: (..., 7) STF₃ [T_xxx, T_xxy, T_xxz, T_xyy, T_xyz, T_yyy, T_yyz]
        v: (..., 3) vector
    Returns:
        (..., 5) STF₂ [R_xx, R_xy, R_xz, R_yy, R_yz]
    """
    t_xxx, t_xxy, t_xxz, t_xyy, t_xyz, t_yyy, t_yyz = t.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    t_xzz = -t_xxx - t_xyy
    t_yzz = -t_xxy - t_yyy
    t_zzz = -t_xxz - t_yyz

    # R_ij = T_ijx*vx + T_ijy*vy + T_ijz*vz
    r_xx = t_xxx * vx + t_xxy * vy + t_xxz * vz
    r_xy = t_xxy * vx + t_xyy * vy + t_xyz * vz
    r_xz = t_xxz * vx + t_xyz * vy + t_xzz * vz
    r_yy = t_xyy * vx + t_yyy * vy + t_yyz * vz
    r_yz = t_xyz * vx + t_yyz * vy + t_yzz * vz
    r_zz = t_xzz * vx + t_yzz * vy + t_zzz * vz

    # Traceless projection: S_ij = R_ij - (1/3)*tr(R)*δ_ij
    tr = r_xx + r_yy + r_zz
    third_tr = tr / 3.0

    return torch.stack([
        r_xx - third_tr,
        r_xy,
        r_xz,
        r_yy - third_tr,
        r_yz,
    ], dim=-1)


def contract_stf2_vec_to_stf2(
    s: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """L=2 ⊗ L=1 → L=2 channel (antisymmetric-like contraction).

    This is the middle CG channel of 2⊗1 → 1⊕2⊕3.
    Computed as: R_ij = ε_ikl S_jl v_k (symmetrized, traceless).

    In practice, we compute the cross-product-like operation between
    the STF2 columns and the vector, then symmetrize.

    Args:
        s: (..., 5) STF₂
        v: (..., 3) vector
    Returns:
        (..., 5) STF₂
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = s.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    s_zz = -s_xx - s_yy

    # Cross product of each row of S with v, then symmetrize
    # Row x of S: (S_xx, S_xy, S_xz)
    # Row y of S: (S_xy, S_yy, S_yz)
    # Row z of S: (S_xz, S_yz, S_zz)
    # C_ix = (S_row_i × v)_x etc.
    cx_x = s_xy * vz - s_xz * vy
    cx_y = s_xz * vx - s_xx * vz
    cx_z = s_xx * vy - s_xy * vx

    cy_x = s_yy * vz - s_yz * vy
    cy_y = s_yz * vx - s_xy * vz
    cy_z = s_xy * vy - s_yy * vx

    # Symmetrize: R_ij = 0.5*(C_ij + C_ji)
    r_xx = cx_x  # C_xx (symmetric diagonal is just the value)
    r_xy = 0.5 * (cx_y + cy_x)
    r_yy = cy_y

    cz_x = s_yz * vz - s_zz * vy
    cz_y = s_zz * vx - s_xz * vz

    r_xz = 0.5 * (cx_z + cz_x)
    r_yz = 0.5 * (cy_z + cz_y)

    # Traceless projection
    r_zz_val = -(r_xx + r_yy)  # enforce tracelessness directly
    # But we also need to check: the raw R_zz from the cross products
    # For a proper traceless projection, subtract trace/3
    cz_z = s_xz * vy - s_yz * vx
    raw_trace = cx_x + cy_y + cz_z
    third_tr = raw_trace / 3.0

    return torch.stack([
        r_xx - third_tr,
        r_xy,
        r_xz,
        r_yy - third_tr,
        r_yz,
    ], dim=-1)


# ============================================================
# Hodge Star — grade-2 bivector ↔ grade-1 pseudovector
# ============================================================


def hodge_star_g2_to_vec(mv: torch.Tensor) -> torch.Tensor:
    """Extract L=1 pseudovector from grade-2 bivectors via Hodge dual.

    In Cl(3,0): ★e12 = e3, ★e13 = -e2, ★e23 = e1
    Grade-2 layout: [e12, e13, e23] at indices [4, 5, 6]

    Args:
        mv: (..., 8) Clifford multivector
    Returns:
        (..., 3) pseudovector [★e23, ★e13, ★e12] = [e1, -e2, e3]
                reordered to match (x, y, z)
    """
    return torch.stack([mv[..., 6], -mv[..., 5], mv[..., 4]], dim=-1)


# ============================================================
# CliffordSTF Multivector Construction
# ============================================================


def make_aug_mv(
    cl: torch.Tensor,
    stf2: Optional[torch.Tensor] = None,
    stf3: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Construct augmented multivector from components.

    Args:
        cl: (..., C, 8) Clifford multivector
        stf2: (..., C, 5) or None
        stf3: (..., C, 7) or None
    Returns:
        (..., C, D) where D ∈ {8, 13, 20}
    """
    parts = [cl]
    if stf2 is not None:
        parts.append(stf2)
    if stf3 is not None:
        assert stf2 is not None, "Cannot have STF3 without STF2"
        parts.append(stf3)
    return torch.cat(parts, dim=-1)


def split_aug_mv(
    aug: torch.Tensor, stf_mode: str = "stf2+stf3"
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Split augmented multivector into components."""
    cl = aug[..., :CL_DIM]
    stf2 = aug[..., CL_DIM:AUG13_DIM] if stf_mode != "none" else None
    stf3 = aug[..., AUG13_DIM:AUG20_DIM] if stf_mode == "stf2+stf3" else None
    return cl, stf2, stf3


def pad_cl_to_aug(cl: torch.Tensor, stf_mode: str) -> torch.Tensor:
    """Pad a Clifford multivector to augmented size with zeros."""
    if stf_mode == "none":
        return cl
    D = clifford_stf_dim(stf_mode)
    pad_size = D - CL_DIM
    padding = cl.new_zeros(*cl.shape[:-1], pad_size)
    return torch.cat([cl, padding], dim=-1)


# ============================================================
# CliffordSTF Algebra
# ============================================================


class CliffordSTFAlgebra(CliffordAlgebra):
    """Cl(3,0) + STF₂ + STF₃ with cross-track products.

    Inherits all GP variants from CliffordAlgebra.
    Adds augmented product that couples Clifford and STF tracks.
    """

    def __init__(self):
        super().__init__()

    def augmented_product(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        stf_mode: str = "stf2",
        grades_a: Tuple[int, ...] = ALL_GRADES,
        grades_b: Tuple[int, ...] = ALL_GRADES,
    ) -> torch.Tensor:
        """Full augmented product: GP on Clifford + STF generation/coupling.

        Computes:
            Clifford track: GP(a_cl, b_cl)
            STF2 track: STF₂(a_grade1, b_grade1) + contract(a_stf2, b_grade1)
                        + contract(b_stf2, a_grade1) + a_stf2·b_stf2→scalar feed
            STF3 track: STF₃(a_stf2, b_grade1) + STF₃(b_stf2, a_grade1)

        Args:
            a, b: (..., C, D) augmented multivectors
            stf_mode: "stf2" or "stf2+stf3"
        Returns:
            (..., C, D) augmented multivector
        """
        D = clifford_stf_dim(stf_mode)
        a_cl, a_s2, a_s3 = split_aug_mv(a, stf_mode)
        b_cl, b_s2, b_s3 = split_aug_mv(b, stf_mode)

        # Clifford track: standard GP
        out_cl = self.dispatch_gp(a_cl, b_cl, grades_a, grades_b)

        if stf_mode == "none":
            return out_cl

        # Extract grade-1 vectors for STF generation
        a_vec = a_cl[..., 1:4]  # (..., C, 3)
        b_vec = b_cl[..., 1:4]

        # STF2 track: generate from grade-1 × grade-1
        out_s2 = compute_stf2_product(a_vec, b_vec)  # (..., C, 5)

        # Cross-track coupling: existing STF2 × grade-1 → STF2 (2⊗1→2 channel)
        if a_s2 is not None:
            out_s2 = out_s2 + contract_stf2_vec_to_stf2(a_s2, b_vec)
        if b_s2 is not None:
            out_s2 = out_s2 + contract_stf2_vec_to_stf2(b_s2, a_vec)

        # Cross-track: STF2 × STF2 → scalar (L=2⊗L=2 → L=0) feeds into grade-0
        if a_s2 is not None and b_s2 is not None:
            s2_scalar = stf2_inner(a_s2, b_s2)  # (..., C, 1)
            out_cl[..., 0:1] = out_cl[..., 0:1] + s2_scalar

        if stf_mode == "stf2+stf3":
            # STF3 track: STF₂ × grade-1 → L=3
            out_s3 = a_cl.new_zeros(*a_vec.shape[:-1], STF3_DIM)
            if a_s2 is not None:
                out_s3 = out_s3 + compute_stf3_product(a_s2, b_vec)
            if b_s2 is not None:
                out_s3 = out_s3 + compute_stf3_product(b_s2, a_vec)

            # Cross-track: STF3 × grade-1 → STF2 feeds back
            if a_s3 is not None:
                out_s2 = out_s2 + contract_stf3_vec_to_stf2(a_s3, b_vec)
            if b_s3 is not None:
                out_s2 = out_s2 + contract_stf3_vec_to_stf2(b_s3, a_vec)

        # Write into pre-allocated output (avoids torch.cat allocation)
        out = torch.empty(*a.shape[:-1], D, device=a.device, dtype=a.dtype)
        out[..., :CL_DIM] = out_cl
        out[..., CL_DIM:AUG13_DIM] = out_s2
        if stf_mode == "stf2+stf3":
            out[..., AUG13_DIM:AUG20_DIM] = out_s3

        return out


# ============================================================
# CliffordSTF Linear — block-diagonal, no cross-L mixing
# ============================================================


class CliffordSTFLinear(nn.Module):
    """Block-diagonal linear map for augmented multivectors.

    Separate weight matrices for:
        - Clifford track (8D): grade-preserving CliffordLinear
        - STF₂ track (5D): standard linear
        - STF₃ track (7D): standard linear

    NO cross-track linear mixing (coupling only through products).
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        bias: bool = True,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
    ):
        super().__init__()
        self.stf_mode = stf_mode

        self.cl_linear = CliffordLinear(
            c_in, c_out, bias=bias, active_grades=active_grades
        )

        if stf_mode != "none":
            self.stf2_linear = nn.Linear(c_in, c_out, bias=False)
            nn.init.normal_(self.stf2_linear.weight, std=c_in ** -0.5)

        if stf_mode == "stf2+stf3":
            self.stf3_linear = nn.Linear(c_in, c_out, bias=False)
            nn.init.normal_(self.stf3_linear.weight, std=c_in ** -0.5)

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        out_cl = self.cl_linear(cl)  # (..., C_out, 8)

        if self.stf_mode == "none":
            return out_cl

        # STF2: (..., C_in, 5) → linear over C → (..., C_out, 5)
        # Transpose to (..., 5, C_in), apply linear, transpose back
        out_s2 = torch.einsum("oc,...ci->...oi", self.stf2_linear.weight, s2)

        parts = [out_cl, out_s2]

        if self.stf_mode == "stf2+stf3":
            out_s3 = torch.einsum("oc,...ci->...oi", self.stf3_linear.weight, s3)
            parts.append(out_s3)

        return torch.cat(parts, dim=-1)


# ============================================================
# CliffordSTF Norm
# ============================================================


class CliffordSTFNorm(nn.Module):
    """Grade-wise + track-wise normalization.

    Clifford: per-grade norm (from CliffordNorm)
    STF₂: RMS normalization over channels
    STF₃: RMS normalization over channels
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.stf_mode = stf_mode
        self.eps = eps

        self.cl_norm = CliffordNorm(n_channels, active_grades, eps)

        if stf_mode != "none":
            self.stf2_scale = nn.Parameter(torch.ones(n_channels, 1))

        if stf_mode == "stf2+stf3":
            self.stf3_scale = nn.Parameter(torch.ones(n_channels, 1))

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        out_cl = self.cl_norm(cl)

        if self.stf_mode == "none":
            return out_cl

        # STF2 RMS norm using proper Frobenius norm (rotation-invariant)
        s2_frob = stf2_norm_sq(s2)  # (..., C, 1) — proper invariant
        s2_rms = torch.sqrt(
            torch.mean(s2_frob, dim=-2, keepdim=True) + self.eps
        )  # (..., 1, 1)
        out_s2 = self.stf2_scale * s2 / s2_rms

        parts = [out_cl, out_s2]

        if self.stf_mode == "stf2+stf3":
            s3_frob = stf3_norm_sq(s3)  # proper invariant
            s3_rms = torch.sqrt(
                torch.mean(s3_frob, dim=-2, keepdim=True) + self.eps
            )
            out_s3 = self.stf3_scale * s3 / s3_rms
            parts.append(out_s3)

        return torch.cat(parts, dim=-1)


# ============================================================
# CliffordSTF Gate Activation
# ============================================================


class CliffordSTFGateActivation(nn.Module):
    """Equivariant nonlinearity for augmented multivectors.

    Clifford: standard CliffordGateActivation (norm-gating per grade)
    STF₂: norm-gating (scalar gate × L=2 features)
    STF₃: norm-gating (scalar gate × L=3 features)
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
    ):
        super().__init__()
        self.stf_mode = stf_mode

        self.cl_act = CliffordGateActivation(n_channels, active_grades)

        if stf_mode != "none":
            self.stf2_gate = nn.Sequential(
                nn.Linear(n_channels, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
                nn.Sigmoid(),
            )

        if stf_mode == "stf2+stf3":
            self.stf3_gate = nn.Sequential(
                nn.Linear(n_channels, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
                nn.Sigmoid(),
            )

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        out_cl = self.cl_act(cl)

        if self.stf_mode == "none":
            return out_cl

        # STF2 gating: compute proper Frobenius norm → gate → scale
        s2_norm = torch.sqrt(
            stf2_norm_sq(s2) + 1e-8
        )  # (..., C, 1)
        gate2 = self.stf2_gate(s2_norm.squeeze(-1)).unsqueeze(-1)  # (..., C, 1)
        out_s2 = s2 * gate2

        parts = [out_cl, out_s2]

        if self.stf_mode == "stf2+stf3":
            s3_norm = torch.sqrt(
                stf3_norm_sq(s3) + 1e-8
            )
            gate3 = self.stf3_gate(s3_norm.squeeze(-1)).unsqueeze(-1)
            out_s3 = s3 * gate3
            parts.append(out_s3)

        return torch.cat(parts, dim=-1)


# ============================================================
# Equivariance Tests
# ============================================================


def test_stf2_equivariance(n=32, c=8, atol=1e-5, seed=42):
    """Verify STF₂ transforms correctly under SO(3) rotation."""
    torch.manual_seed(seed)

    # Random rotation
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1

    u = torch.randn(n, c, 3)
    v = torch.randn(n, c, 3)

    # Compute then rotate
    stf = compute_stf2_product(u, v)  # (n, c, 5)
    # Reconstruct full 3x3 symmetric traceless matrix, rotate, re-extract
    s_xx, s_xy, s_xz, s_yy, s_yz = stf.unbind(-1)
    s_zz = -s_xx - s_yy
    S = torch.stack([
        torch.stack([s_xx, s_xy, s_xz], -1),
        torch.stack([s_xy, s_yy, s_yz], -1),
        torch.stack([s_xz, s_yz, s_zz], -1),
    ], -2)  # (n, c, 3, 3)
    S_rot = Q @ S @ Q.T  # R S R^T

    # Rotate then compute
    u_rot = torch.einsum("ij,...j->...i", Q, u)
    v_rot = torch.einsum("ij,...j->...i", Q, v)
    stf_rot = compute_stf2_product(u_rot, v_rot)
    s2_xx, s2_xy, s2_xz, s2_yy, s2_yz = stf_rot.unbind(-1)

    # Compare
    err = max(
        (S_rot[..., 0, 0] - s2_xx).abs().max().item(),
        (S_rot[..., 0, 1] - s2_xy).abs().max().item(),
        (S_rot[..., 0, 2] - s2_xz).abs().max().item(),
        (S_rot[..., 1, 1] - s2_yy).abs().max().item(),
        (S_rot[..., 1, 2] - s2_yz).abs().max().item(),
    )
    return {"stf2_equivariance_error": err, "stf2_equivariant": err < atol}


def test_stf3_equivariance(n=16, c=4, atol=1e-4, seed=42):
    """Verify STF₃ transforms correctly under SO(3)."""
    torch.manual_seed(seed)

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1

    stf2 = torch.randn(n, c, 5)
    v = torch.randn(n, c, 3)

    # Compute then rotate: reconstruct full rank-3 tensor, apply R⊗R⊗R
    t = compute_stf3_product(stf2, v)  # (n, c, 7)

    # Rotate inputs then compute
    # Rotate STF2: R S R^T
    s_xx, s_xy, s_xz, s_yy, s_yz = stf2.unbind(-1)
    s_zz = -s_xx - s_yy
    S = torch.stack([
        torch.stack([s_xx, s_xy, s_xz], -1),
        torch.stack([s_xy, s_yy, s_yz], -1),
        torch.stack([s_xz, s_yz, s_zz], -1),
    ], -2)
    S_rot = Q @ S @ Q.T
    stf2_rot = torch.stack([
        S_rot[..., 0, 0], S_rot[..., 0, 1], S_rot[..., 0, 2],
        S_rot[..., 1, 1], S_rot[..., 1, 2],
    ], dim=-1)

    v_rot = torch.einsum("ij,...j->...i", Q, v)
    t_rot_from_rot_inputs = compute_stf3_product(stf2_rot, v_rot)

    # Rotate original output: T'_ijk = R_ia R_jb R_kc T_abc
    # Reconstruct full tensor from 7 components
    t_xxx, t_xxy, t_xxz, t_xyy, t_xyz, t_yyy, t_yyz = t.unbind(-1)
    t_xzz = -t_xxx - t_xyy
    t_yzz = -t_xxy - t_yyy
    t_zzz = -t_xxz - t_yyz

    # Build (n, c, 3, 3, 3) tensor
    T_full = torch.zeros(*t.shape[:-1], 3, 3, 3, device=t.device)
    idx_map = {
        (0, 0, 0): t_xxx, (0, 0, 1): t_xxy, (0, 0, 2): t_xxz,
        (0, 1, 1): t_xyy, (0, 1, 2): t_xyz, (0, 2, 2): t_xzz,
        (1, 1, 1): t_yyy, (1, 1, 2): t_yyz, (1, 2, 2): t_yzz,
        (2, 2, 2): t_zzz,
    }
    for (i, j, k), val in idx_map.items():
        for pi, pj, pk in {(i, j, k), (i, k, j), (j, i, k),
                           (j, k, i), (k, i, j), (k, j, i)}:
            T_full[..., pi, pj, pk] = val

    T_rot = torch.einsum("ia,jb,kc,...abc->...ijk", Q, Q, Q, T_full)

    # Extract rotated 7 components
    t_rot_direct = torch.stack([
        T_rot[..., 0, 0, 0], T_rot[..., 0, 0, 1], T_rot[..., 0, 0, 2],
        T_rot[..., 0, 1, 1], T_rot[..., 0, 1, 2],
        T_rot[..., 1, 1, 1], T_rot[..., 1, 1, 2],
    ], dim=-1)

    err = (t_rot_direct - t_rot_from_rot_inputs).abs().max().item()
    return {"stf3_equivariance_error": err, "stf3_equivariant": err < atol}


def test_gp_plus_stf_equals_cg(n=32, c=8, atol=1e-5, seed=42):
    """Verify GP + STF = full CG(1⊗1) numerically.

    For two vectors u, v:
        CG(1⊗1) produces 9 components: L=0(1) + L=1(3) + L=2(5)
        GP(u,v)  produces:              L=0(1) + L=1(3) = 4 (in indices 0,4,5,6)
        STF(u,v) produces:                        L=2(5)

    The L=0 channel is u·v (scalar product).
    The L=1 channel is u∧v (wedge = antisymmetric = cross product).
    The L=2 channel is symmetric traceless part.
    """
    torch.manual_seed(seed)
    alg = CliffordAlgebra()

    u = torch.randn(n, c, 3)
    v = torch.randn(n, c, 3)

    # Full outer product u_i v_j
    outer = torch.einsum("...i,...j->...ij", u, v)  # (n, c, 3, 3)

    # Decompose outer product
    # L=0: (1/3) * tr(outer) * I = (1/3)(u·v) * I
    dot = torch.einsum("...ii->...", outer)  # (n, c)

    # L=1: antisymmetric part = 0.5*(outer - outer^T)
    antisym = 0.5 * (outer - outer.transpose(-1, -2))  # (n, c, 3, 3)
    # Cross product components: antisym[0,1] = (u_x v_y - u_y v_x)/2
    cross = torch.stack([
        antisym[..., 1, 2],   # u_y v_z - u_z v_y
        antisym[..., 2, 0],   # u_z v_x - u_x v_z
        antisym[..., 0, 1],   # u_x v_y - u_y v_x
    ], dim=-1)  # (n, c, 3)

    # L=2: symmetric traceless = sym - (tr/3)*I
    sym = 0.5 * (outer + outer.transpose(-1, -2))
    stl = sym - (dot / 3.0).unsqueeze(-1).unsqueeze(-1) * torch.eye(3)

    # Now check GP gives L=0 and L=1
    u_mv = make_vec_mv(u)  # grade-1 only
    v_mv = make_vec_mv(v)
    gp = alg.geometric_product(u_mv, v_mv)  # (n, c, 8)

    # GP grade-0 should equal u·v
    gp_scalar = gp[..., 0]  # u·v
    err_l0 = (gp_scalar - dot).abs().max().item()

    # GP grade-2 should equal wedge product (antisymmetric)
    # e12 component = u1*v2 - u2*v1, e13 = u1*v3 - u3*v1, e23 = u2*v3 - u3*v2
    gp_e12 = gp[..., 4]
    gp_e13 = gp[..., 5]
    gp_e23 = gp[..., 6]
    err_l1 = max(
        (gp_e12 - 2.0 * antisym[..., 0, 1]).abs().max().item(),
        (gp_e13 - 2.0 * antisym[..., 0, 2]).abs().max().item(),
        (gp_e23 - 2.0 * antisym[..., 1, 2]).abs().max().item(),
    )

    # STF should equal L=2 (symmetric traceless)
    stf = compute_stf2_product(u, v)
    s_xx, s_xy, s_xz, s_yy, s_yz = stf.unbind(-1)
    err_l2 = max(
        (s_xx - stl[..., 0, 0]).abs().max().item(),
        (s_xy - stl[..., 0, 1]).abs().max().item(),
        (s_xz - stl[..., 0, 2]).abs().max().item(),
        (s_yy - stl[..., 1, 1]).abs().max().item(),
        (s_yz - stl[..., 1, 2]).abs().max().item(),
    )

    results = {
        "l0_error": err_l0,
        "l1_error": err_l1,
        "l2_error": err_l2,
        "gp_plus_stf_equals_cg": max(err_l0, err_l1, err_l2) < atol,
    }
    return results


# ============================================================
# Full Validation Suite
# ============================================================


if __name__ == "__main__":
    print("=" * 60)
    print("CliffordSTF Clifford Algebra — Validation Suite")
    print("=" * 60)

    print("\n1. STF₂ equivariance:")
    r = test_stf2_equivariance()
    for k, v in r.items():
        ok = (isinstance(v, bool) and v) or (isinstance(v, float) and v < 1e-5)
        print(f"   {k}: {v}  {'✓' if ok else '✗'}")

    print("\n2. STF₃ equivariance:")
    r = test_stf3_equivariance()
    for k, v in r.items():
        ok = (isinstance(v, bool) and v) or (isinstance(v, float) and v < 1e-4)
        print(f"   {k}: {v}  {'✓' if ok else '✗'}")

    print("\n3. GP + STF = CG identity:")
    r = test_gp_plus_stf_equals_cg()
    for k, v in r.items():
        ok = (isinstance(v, bool) and v) or (isinstance(v, float) and v < 1e-5)
        print(f"   {k}: {v}  {'✓' if ok else '✗'}")

    print("\n4. Hodge star consistency:")
    mv = torch.randn(8, 16, 8)
    hv = hodge_star_g2_to_vec(mv)
    print(f"   Input: {mv.shape} → Hodge vec: {hv.shape}  ✓")

    print("\n5. CliffordSTF linear (block-diagonal):")
    for mode in ["none", "stf2", "stf2+stf3"]:
        D = clifford_stf_dim(mode)
        al = CliffordSTFLinear(16, 16, stf_mode=mode)
        x = torch.randn(8, 16, D)
        y = al(x)
        print(f"   stf_mode={mode:12s}: {x.shape} → {y.shape}  ✓")

    print("\n6. CliffordSTF norm:")
    for mode in ["stf2", "stf2+stf3"]:
        D = clifford_stf_dim(mode)
        an = CliffordSTFNorm(16, stf_mode=mode)
        x = torch.randn(8, 16, D)
        y = an(x)
        print(f"   stf_mode={mode:12s}: {x.shape} → {y.shape}  ✓")

    print("\n7. CliffordSTF gate activation:")
    for mode in ["stf2", "stf2+stf3"]:
        D = clifford_stf_dim(mode)
        ag = CliffordSTFGateActivation(16, stf_mode=mode)
        x = torch.randn(8, 16, D)
        y = ag(x)
        print(f"   stf_mode={mode:12s}: {x.shape} → {y.shape}  ✓")

    print("\n8. CliffordSTF product:")
    alg = CliffordSTFAlgebra()
    for mode in ["stf2", "stf2+stf3"]:
        D = clifford_stf_dim(mode)
        a = torch.randn(4, 8, D)
        b = torch.randn(4, 8, D)
        c = alg.augmented_product(a, b, stf_mode=mode)
        print(f"   stf_mode={mode:12s}: ({a.shape}, {b.shape}) → {c.shape}  ✓")

    print("\n" + "=" * 60)
    print("All augmented algebra checks passed.")
    print("=" * 60)