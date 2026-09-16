"""Numerical checks of the parity-gap theorem and output antisymmetrization."""

import numpy as np
import sympy as sp


def test_parity_gap_theorem_and_output_enforcement():
    fail = []

    def chk(label, ok, note=""):
        if not ok:
            fail.append(label)
        print(f"  [{'pass' if ok else 'FAIL'}] {label}  {note}")

    rng = np.random.default_rng(7)

    def centro(n=5):
        h = rng.normal(size=(n, 3))
        return np.vstack([h, -h, np.zeros((1, 3))])

    # Permutation- and translation-invariant polar readouts built from positions.
    f3 = lambda r: np.einsum("ni,nj,nk->ijk", r, r, r)
    f2 = lambda r: np.einsum("ni,nj->ij", r, r)

    print("INPUT IDENTITY  f(I.x) = f(x) under (A1),(A2)")
    w = max(np.abs(f3(-x) - f3(x)).max() for x in (centro() for _ in range(400)))
    chk("identity holds on 400 centrosymmetric configurations", w < 1e-12, f"max = {w:.2e}")

    print("\nTHEOREM (odd rank)  O(3) equivariance forces f(x) = 0")
    w = max(np.abs(f3(x)).max() for x in (centro() for _ in range(400)))
    chk("odd-rank output vanishes on 400 configurations", w < 1e-12, f"max |f| = {w:.2e}")
    v = max(np.abs(f2(centro())).max() for _ in range(50))
    chk("even-rank output is generically nonzero", v > 1e-3, f"max |f_even| = {v:.3f}")

    print("\nTHEOREM (index symmetry)  class 432 on the full and piezoelectric rank-3 spaces")
    c4z = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c3 = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    group = {sp.ImmutableMatrix(sp.eye(3)), sp.ImmutableMatrix(c4z), sp.ImmutableMatrix(c3)}
    grew = True
    while grew:
        grew = False
        for a in list(group):
            for b in list(group):
                p = sp.ImmutableMatrix(a * b)
                if p not in group:
                    group.add(p)
                    grew = True
    g432 = [np.array(m).astype(float) for m in group]
    chk("432 closes to 24 proper rotations", len(g432) == 24)

    reynolds = np.zeros((27, 27))
    for k in range(27):
        e = np.zeros(27)
        e[k] = 1
        reynolds[:, k] = (
            sum(np.einsum("ia,jb,kc,abc->ijk", r, r, r, e.reshape(3, 3, 3)) for r in g432)
            / len(g432)
        ).ravel()
    cols = []
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                t = np.zeros((3, 3, 3))
                t[i, j, k] += 1
                t[i, k, j] += 1
                cols.append(t.ravel())
    basis = np.array(cols).T
    chk(
        "dim V^432 = 1 on the full rank-3 space (Levi-Civita survives)",
        np.linalg.matrix_rank(reynolds, tol=1e-9) == 1,
    )
    chk(
        "dim V^432 = 0 on the piezoelectric subspace, so gamma_3(m-3m) = 0",
        np.linalg.matrix_rank(np.linalg.pinv(basis) @ reynolds @ basis, tol=1e-9) == 0,
    )

    print("\nOUTPUT ENFORCEMENT  A(x) = [f(x) - f(I.x)] / 2")
    antisym = lambda f, x: 0.5 * (f(x) - f(-x))
    odd = max(
        np.abs(antisym(f, -x) + antisym(f, x)).max()
        for x in (rng.normal(size=(7, 3)) for _ in range(300))
        for f in (f3, f2)
    )
    chk("A(I.x) = -A(x) for arbitrary inputs", odd < 1e-12, f"max = {odd:.2e}")
    zero = max(
        np.abs(antisym(f, x)).max() for x in (centro() for _ in range(300)) for f in (f3, f2)
    )
    chk("A(x) = 0 on centrosymmetric inputs", zero < 1e-12, f"max = {zero:.2e}")

    assert not fail, f"theory checks failed: {fail}"
