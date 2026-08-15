"""
parity_metrics.py
=================
Analysis and metrics for "Rotation is not enough: parity-blind equivariant networks
predict physically impossible material properties".

This module implements every quantity reported in the manuscript, plus the
group-theoretic checks that Corollary 2 relies on. It *computes*; it never asserts a
result. Running it as a script executes a synthetic smoke test and prints what it
finds, including a first-principles verification of the rank statements the paper
makes about the point groups 432, 23 and 422.

Dependencies: numpy only (SciPy is used if present, for an exact Wilcoxon p-value;
a normal-approximation fallback is provided so the module runs without it).

    python parity_metrics.py
"""

from __future__ import annotations

import itertools
import numpy as np

try:  # optional
    from scipy.stats import wilcoxon as _scipy_wilcoxon, spearmanr as _scipy_spearman
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------------------
# 1. Core metric: the violation magnitude and the false-flag fraction
# --------------------------------------------------------------------------------------

def violation_magnitude(e: np.ndarray) -> np.ndarray:
    """Frobenius norm ||e||_F of each predicted piezoelectric tensor.

    On a centrosymmetric crystal the true tensor is exactly zero, so this norm *is*
    the prediction error. `e` has shape (n_structures, ...) and is flattened per row.
    The Cartesian <-> irreducible change of basis is orthonormal, so the value is the
    same in either basis.
    """
    e = np.asarray(e, dtype=float)
    return np.sqrt((e.reshape(e.shape[0], -1) ** 2).sum(axis=1))


def false_flag_fraction(v: np.ndarray, tau: float = 1e-2) -> float:
    """Fraction of structures whose violation magnitude exceeds the threshold tau.

    tau = 0.01 C/m^2 is the manuscript's operating point.
    """
    v = np.asarray(v, dtype=float)
    return float((v > tau).mean())


def threshold_curve(v: np.ndarray,
                    taus: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """False-flag fraction over the 25 log-spaced thresholds used in the paper."""
    if taus is None:
        taus = np.logspace(-4, 0, 25)
    return taus, np.array([false_flag_fraction(v, t) for t in taus])


def per_atom_variant(v: np.ndarray, n_atoms: np.ndarray, tau: float,
                     median_cell_size: float | None = None) -> tuple[np.ndarray, float]:
    """Size-normalised metric ||e||_F / n_atoms with a correspondingly rescaled threshold.

    The readout sums per-atom (or per-edge) contributions, so the violation magnitude is
    extensive. Returns (normalised violations, rescaled threshold).
    """
    v = np.asarray(v, float)
    n_atoms = np.asarray(n_atoms, float)
    if median_cell_size is None:
        median_cell_size = float(np.median(n_atoms))
    return v / n_atoms, tau / median_cell_size


# --------------------------------------------------------------------------------------
# 2. Uncertainty and significance
# --------------------------------------------------------------------------------------

def bootstrap_ci(v: np.ndarray, tau: float = 1e-2, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the false-flag fraction, resampling structures."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    n = v.size
    point = false_flag_fraction(v, tau)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = false_flag_fraction(v[rng.integers(0, n, n)], tau)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_wilcoxon_one_sided(v_o3: np.ndarray, v_so3: np.ndarray) -> dict:
    """One-sided Wilcoxon signed-rank test, paired by structure.

    H1: the O(3) violation is smaller than the SO(3) violation. Chosen because the
    violation distributions are strongly non-normal and bounded below by zero, and the
    design is paired by construction (both arms see the same structure).

    Returns the statistic, the p-value, the fraction of pairs with O(3) < SO(3), and a
    rank-biserial effect size. p-values below ~1e-300 underflow double precision and are
    returned as 0.0; report them as p < 1e-300, never as exactly zero.
    """
    a, b = np.asarray(v_o3, float), np.asarray(v_so3, float)
    if a.shape != b.shape:
        raise ValueError("paired test requires matched shapes")
    d = b - a                                   # positive when O(3) is smaller
    nz = d[d != 0]
    frac = float((a < b).mean())
    if _HAVE_SCIPY:
        stat, p = _scipy_wilcoxon(a, b, alternative="less")
    else:  # normal approximation on the signed ranks
        r = np.argsort(np.argsort(np.abs(nz))) + 1.0
        w_plus = r[nz > 0].sum()
        n = nz.size
        mu = n * (n + 1) / 4.0
        sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w_plus - mu) / sd
        stat, p = float(w_plus), float(0.5 * np.math.erfc(z / np.sqrt(2)))
    # rank-biserial correlation: a scale-free effect size for a paired signed-rank test
    rb = 2.0 * frac - 1.0
    return {"statistic": float(stat), "p_one_sided": float(p),
            "frac_o3_smaller": frac, "rank_biserial": float(rb), "n_pairs": int(a.size)}


def pooled_seed_effect(mean_o3: float, sd_o3: float,
                       mean_so3: float, sd_so3: float) -> float:
    """The manuscript's seed-level effect size: (MAE_SO3 - MAE_O3) / pooled per-seed sd.

    Positive values favour the O(3) arm. No significance is claimed from three seeds; a
    seed-level signed-rank test on n = 3 cannot return p < 0.25 for any effect size.
    """
    pooled = np.sqrt((sd_o3 ** 2 + sd_so3 ** 2) / 2.0)
    return float((mean_so3 - mean_o3) / pooled) if pooled > 0 else float("nan")


def jaccard(flagged_a: np.ndarray, flagged_b: np.ndarray) -> float:
    """Jaccard index of two flagged sets (boolean masks over the same structures)."""
    a, b = np.asarray(flagged_a, bool), np.asarray(flagged_b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (used for seed-to-seed and core-to-core agreement)."""
    if _HAVE_SCIPY:
        return float(_scipy_spearman(x, y).statistic)
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


# --------------------------------------------------------------------------------------
# 3. The differentiable guarantee (Corollary 3)
# --------------------------------------------------------------------------------------

def inversion_displacement_operator(perm: np.ndarray) -> np.ndarray:
    """The map P on displacement fields: permute atoms by the inversion permutation and
    flip the sign of the displacement, (P u)_{pi(i)} = -u_i.  P is an involution.
    Returns a (3n, 3n) matrix acting on the flattened field.
    """
    perm = np.asarray(perm, int)
    n = perm.size
    P = np.zeros((3 * n, 3 * n))
    for i, pi in enumerate(perm):
        P[3 * pi:3 * pi + 3, 3 * i:3 * i + 3] = -np.eye(3)
    return P


def even_subspace_fraction(J: np.ndarray, perm: np.ndarray) -> float:
    """||J P_even||_F / ||J||_F with P_even = (I + P)/2.

    Corollary 3 forces this to be exactly zero for an exactly O(3)-equivariant model at a
    centrosymmetric geometry. Any nonzero value measures either arithmetic noise (O(3)
    arms) or the absence of the guarantee (SO(3) arms).
    """
    J = np.asarray(J, float)
    P = inversion_displacement_operator(perm)
    P_even = 0.5 * (np.eye(P.shape[0]) + P)
    denom = np.linalg.norm(J, "fro")
    return float(np.linalg.norm(J @ P_even, "fro") / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------------------
# 4. Group theory behind Corollary 2 (the rotation ceiling)
# --------------------------------------------------------------------------------------

def _signed_permutation_rotations() -> list[np.ndarray]:
    """The 24 proper rotations of the cube: signed permutation matrices with det = +1.
    This is the point group 432 (O)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                out.append(M)
    return out


def point_group(name: str) -> list[np.ndarray]:
    """Proper-rotation groups needed by the manuscript."""
    O = _signed_permutation_rotations()                       # 432, |O| = 24
    if name == "432":
        return O
    if name == "23":                                          # T, the even subgroup of O
        T = []
        for M in O:
            cols = np.argmax(np.abs(M), axis=1)               # underlying permutation
            # parity of the permutation
            par = 1
            c = list(cols)
            for i in range(3):
                for j in range(i + 1, 3):
                    if c[i] > c[j]:
                        par = -par
            if par == 1:
                T.append(M)
        return T
    if name == "422":                                         # D4 about z, |D4| = 8
        def Rz(k):
            th = k * np.pi / 2
            return np.array([[np.cos(th), -np.sin(th), 0],
                             [np.sin(th), np.cos(th), 0],
                             [0, 0, 1]])
        C4 = [Rz(k) for k in range(4)]
        U = np.diag([1.0, -1.0, -1.0])                        # twofold about x
        return C4 + [R @ U for R in C4]
    raise ValueError(f"unknown point group {name!r}")


def piezo_basis() -> np.ndarray:
    """Orthonormal basis of V = R^3 (x) Sym^2(R^3), the 18-dimensional space in which the
    piezoelectric tensor e_ijk lives (symmetric in its last two, strain, indices)."""
    B = []
    for i in range(3):
        for (j, k) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
            T = np.zeros((3, 3, 3))
            T[i, j, k] = 1.0
            T[i, k, j] = 1.0
            B.append(T / np.linalg.norm(T))
    return np.array(B)                                        # (18, 3, 3, 3)


def reynolds_rank(group: list[np.ndarray], tol: float = 1e-9) -> int:
    """Rank of the Reynolds projector (1/|G|) sum_g D(g) on the piezoelectric space.

    The rank is the dimension of the space of rank-3 piezoelectric tensors invariant
    under G, i.e. the number of independent piezoelectric constants the group allows.
    Corollary 2 says an SO(3)-equivariant model's prediction is confined to this space.
    """
    B = piezo_basis()
    P = np.zeros((18, 18))
    for R in group:
        # D(R) acting on each basis tensor, re-expressed in the basis
        for b, T in enumerate(B):
            RT = np.einsum("ia,jb,kc,abc->ijk", R, R, R, T)
            P[:, b] += np.einsum("nijk,ijk->n", B, RT)
    P /= len(group)
    return int((np.linalg.svd(P, compute_uv=False) > tol).sum())


# --------------------------------------------------------------------------------------
# 5. Smoke test
# --------------------------------------------------------------------------------------

def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    n = 2000

    print("=" * 78)
    print("SMOKE TEST 1: synthetic O(3) vs SO(3) violation populations")
    print("=" * 78)
    # An O(3) arm returns the arithmetic floor; an SO(3) arm returns a physical-sized tensor.
    e_o3 = rng.normal(0, 1e-7, size=(n, 18))
    e_so3 = rng.lognormal(mean=np.log(0.7), sigma=1.0, size=(n, 1)) * rng.normal(0, 1, (n, 18))
    v_o3, v_so3 = violation_magnitude(e_o3), violation_magnitude(e_so3)
    print(f"  O(3)  median {np.median(v_o3):.2e}   false-flag @0.01 = {false_flag_fraction(v_o3):.4f}")
    print(f"  SO(3) median {np.median(v_so3):.2e}   false-flag @0.01 = {false_flag_fraction(v_so3):.4f}")
    pt, lo, hi = bootstrap_ci(v_so3)
    print(f"  SO(3) false-flag bootstrap 95% CI: {pt:.4f} [{lo:.4f}, {hi:.4f}]")
    w = paired_wilcoxon_one_sided(v_o3, v_so3)
    p = w["p_one_sided"]
    p_str = "< 1e-300 (underflow)" if p == 0.0 else f"{p:.3e}"
    print(f"  paired Wilcoxon (H1: O(3) smaller): p = {p_str}, "
          f"O(3) smaller in {w['frac_o3_smaller']*100:.1f}% of pairs, "
          f"rank-biserial = {w['rank_biserial']:+.3f}")
    print(f"  Jaccard(flagged SO(3), flagged SO(3) reshuffled) = "
          f"{jaccard(v_so3 > 1e-2, rng.permutation(v_so3) > 1e-2):.3f}")

    print()
    print("=" * 78)
    print("SMOKE TEST 2: seed-level effect size (the manuscript's Delta/sigma)")
    print("=" * 78)
    print("  reproduces the reported NequIP piezoelectric comparison from its table values:")
    d = pooled_seed_effect(0.2083, 0.0077, 0.2405, 0.0080)
    print(f"    O(3) 0.2083 +/- 0.0077 vs SO(3) 0.2405 +/- 0.0080  ->  Delta/sigma = {d:+.2f}")
    print("    (positive favours O(3); the paper reports 3.4 to 4.6 across the three cores)")

    print()
    print("=" * 78)
    print("SMOKE TEST 3: the differentiable guarantee (Corollary 3)")
    print("=" * 78)
    natoms = 6
    perm = np.array([1, 0, 3, 2, 5, 4])           # an involution: a valid inversion permutation
    P = inversion_displacement_operator(perm)
    # An O(3) model's Jacobian must satisfy J P = -J; build one by projecting a random J.
    J_rand = rng.normal(size=(18, 3 * natoms))
    J_o3 = 0.5 * (J_rand - J_rand @ P)            # the odd part, which is what O(3) enforces
    print(f"  P is an involution:            {np.allclose(P @ P, np.eye(3*natoms))}")
    print(f"  O(3)-consistent J: even fraction = {even_subspace_fraction(J_o3, perm):.2e}  (theory: exactly 0)")
    print(f"  unconstrained   J: even fraction = {even_subspace_fraction(J_rand, perm):.3f}  "
          f"(paper measures 0.42-0.54 for SO(3) arms)")

    print()
    print("=" * 78)
    print("SMOKE TEST 4: independent check of the group theory behind Corollary 2")
    print("=" * 78)
    print("  Reynolds-projector rank on the 18-dimensional piezoelectric space")
    print("  = number of independent piezoelectric constants the rotation group permits.")
    for g, note in [("432", "proper-rotation subgroup of m-3m"),
                    ("23", "proper-rotation subgroup of m-3"),
                    ("422", "proper-rotation subgroup of 4/mmm (rutile)")]:
        G = point_group(g)
        r = reynolds_rank(G)
        verdict = "FORBIDS a rank-3 tensor" if r == 0 else f"PERMITS {r} independent constant(s)"
        print(f"    group {g:>3} (|G| = {len(G):>2}, {note:<38}): rank = {r}  ->  {verdict}")
    print()
    print("  The manuscript claims rank 0 for 432, rank 1 for 23, and a permitted invariant")
    print("  for 422. Compare the printed ranks above against those claims.")


if __name__ == "__main__":
    _smoke_test()
