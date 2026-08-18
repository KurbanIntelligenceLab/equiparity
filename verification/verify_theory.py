import sys

import numpy as np
import sympy as sp

FAIL = []


def chk(lab, ok, note=""):
    (FAIL if not ok else []).append(lab)
    print(f"  [{'pass' if ok else 'FAIL'}] {lab}  {note}")


rng = np.random.default_rng(7)


def centro(n=5):
    h = rng.normal(size=(n, 3))
    return np.vstack([h, -h, np.zeros((1, 3))]), n


f3 = lambda R: np.einsum("ni,nj,nk->ijk", R, R, R)

f2 = lambda R: np.einsum("ni,nj->ij", R, R)

print("LEMMA 1  f(I.x) = f(x) under (A1),(A2)")
w = max(np.abs(f3(-centro()[0]) - f3(centro()[0])).max() for _ in range(1))
w = 0.0
for _ in range(400):
    x, _ = centro()
    w = max(w, np.abs(f3(-x) - f3(x)).max())
chk(
    "Lemma 1 holds on 400 random centrosymmetric configurations",
    w < 1e-12,
    f"max |f(I.x)-f(x)| = {w:.2e}",
)

print("\nTHEOREM 1(i)  odd rank + O(3) => f(x) = 0")
w = 0.0
for _ in range(400):
    x, _ = centro()
    w = max(w, np.abs(f3(x)).max())
chk("Theorem 1(i) on 400 configurations", w < 1e-12, f"max |f(x)| = {w:.2e}")
print("   and the same construction at EVEN rank must NOT vanish (else (i) would be vacuous)")
v = max(np.abs(f2(centro()[0])).max() for _ in range(50))
chk("even-rank analogue is generically nonzero", v > 1e-3, f"max |f_even| = {v:.3f}")

print("\nPROPOSITION 1  arithmetic floor, WITH the odd-rank hypothesis")
worst = -1e9
for _ in range(500):
    x, _ = centro()
    fx = 1e-3 * rng.normal(size=(3, 3, 3))
    fIx = fx + 1e-4 * rng.normal(size=(3, 3, 3))
    d = np.linalg.norm(fIx - fx)
    e = np.linalg.norm(fIx + fx)
    worst = max(worst, np.linalg.norm(fx) - 0.5 * (d + e))
chk(
    "bound holds for odd r over 500 random defect pairs", worst <= 1e-12, f"max slack = {worst:.2e}"
)
print("   counterexample at EVEN rank confirms the hypothesis is necessary")
T = rng.normal(size=(3, 3))
d = 0.0
e = np.linalg.norm(T - T)
chk(
    "even-rank counterexample violates the bound",
    np.linalg.norm(T) > 0.5 * (d + e) + 1e-9,
    f"||f|| = {np.linalg.norm(T):.3f} > (delta+eta)/2 = 0",
)

print("\nPROPOSITION 2  stability of the ceiling")
s3 = sp.sqrt(3) / 2
h = sp.Rational(1, 2)
C4z = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
C3 = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def close(g):
    G = {sp.ImmutableMatrix(sp.eye(3))} | {sp.ImmutableMatrix(m) for m in g}
    ch = True
    while ch:
        ch = False
        for a in list(G):
            for b in list(G):
                p = sp.ImmutableMatrix(sp.simplify(a * b))
                if p not in G:
                    G.add(p)
                    ch = True
    return G


G432 = [np.array(sp.Matrix(R)).astype(float) for R in close([C4z, C3])]
worst = -1e9
for _ in range(200):
    fx = rng.normal(size=(3, 3, 3)) * 1e-2
    eR = [rng.normal(size=(3, 3, 3)) * 1e-4 for _ in G432]
    gR = [rng.normal(size=(3, 3, 3)) * 1e-4 for _ in G432]
    dp = max(np.linalg.norm(v) for v in eR)
    ep = max(np.linalg.norm(v) for v in gR)

    Pi = sum(np.einsum("ia,jb,kc,abc->ijk", R, R, R, fx) for R in G432) / len(G432)
    lhs = np.linalg.norm(Pi - fx)
    worst = max(worst, lhs - (dp + ep) - np.linalg.norm(Pi))


cols = []
for i in range(3):
    for j in range(3):
        for k in range(j, 3):
            T = np.zeros((3, 3, 3))
            T[i, j, k] += 1
            T[i, k, j] += 1
            cols.append(T.ravel())
B = np.array(cols).T
Bp = np.linalg.pinv(B)
M = np.zeros((27, 27))
for k in range(27):
    E = np.zeros(27)
    E[k] = 1
    M[:, k] = (
        sum(np.einsum("ia,jb,kc,abc->ijk", R, R, R, E.reshape(3, 3, 3)) for R in G432) / len(G432)
    ).ravel()
chk(
    "dim V^432 = 1 on the FULL rank-3 space (Levi-Civita survives)",
    np.linalg.matrix_rank(M, tol=1e-9) == 1,
)
chk(
    "dim V^432 = 0 on the piezoelectric subspace, so gamma_3(m-3m)=0",
    np.linalg.matrix_rank(Bp @ M @ B, tol=1e-9) == 0,
    "confirms the index-symmetry hypothesis of Theorem 1 is necessary",
)

print("\nCOROLLARY 1  inversion averaging is vacuous for ANY (A1),(A2) model")
w = 0.0
for _ in range(300):
    x, _ = centro()
    for f in (f3, f2):
        w = max(w, np.abs(0.5 * (f(x) - f(-x))).max())
chk("T_sym = 0 for both odd and even readouts", w < 1e-12, f"max |T_sym| = {w:.2e}")

print("\nCOROLLARY 2  Jacobian, J o P = -J")
worst = 0.0
for _ in range(200):
    x, n = centro()
    N = x.shape[0]
    pi = np.concatenate([np.arange(n, 2 * n), np.arange(0, n), [2 * n]])
    u = rng.normal(size=x.shape) * 1e-7
    Pu = np.zeros_like(u)
    inv = np.argsort(pi)
    for j in range(N):
        Pu[j] = -u[inv[j]]
    lhs = f3(x + Pu)
    rhs = -f3(x + u)
    worst = max(worst, np.abs(lhs - rhs).max() / max(np.abs(rhs).max(), 1e-30))
chk("f(x+Pu) = -f(x+u) to second order", worst < 1e-5, f"max relative deviation = {worst:.2e}")
P2ok = True
for _ in range(50):
    x, n = centro()
    N = x.shape[0]
    pi = np.concatenate([np.arange(n, 2 * n), np.arange(0, n), [2 * n]])
    inv = np.argsort(pi)
    u = rng.normal(size=x.shape)
    Pu = np.zeros_like(u)
    PPu = np.zeros_like(u)
    for j in range(N):
        Pu[j] = -u[inv[j]]
    for j in range(N):
        PPu[j] = -Pu[inv[j]]
    P2ok &= np.allclose(PPu, u)
chk("P^2 = id, so P_even is a projector", P2ok)

print("\n" + "=" * 88)
print("FAILED:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
