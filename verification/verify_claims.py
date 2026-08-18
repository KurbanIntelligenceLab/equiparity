import math
import os
import sys

RAW_DIR = None
FAIL, PASS = [], []


def chk(label, got, want, tol=5e-4, note=""):
    ok = abs(got - want) <= tol
    (PASS if ok else FAIL).append((label, got, want, note))
    mark = "pass" if ok else "FAIL"
    print(f"  [{mark}] {label:<56s} computed={got:<13.6g} stated={want:<13.6g} {note}")
    return ok


def section_group_theory():
    import sympy as sp

    print("\n[1] CRYSTALLOGRAPHIC GROUP THEORY (exact, sympy)")
    s3, h, I3 = sp.sqrt(3) / 2, sp.Rational(1, 2), sp.eye(3)
    rz = lambda c, s: sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rx = lambda c, s: sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    C2z, C2x, C4z = rz(-1, 0), rx(-1, 0), rz(0, 1)
    C3z, C6z = rz(-h, s3), rz(h, s3)
    C3_111 = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    INV, Mz, Mx = -I3, sp.diag(1, 1, -1), sp.diag(-1, 1, 1)
    S4z = INV * C4z

    GEN = {
        "1": [],
        "-1": [INV],
        "2": [C2z],
        "m": [Mz],
        "2/m": [C2z, INV],
        "222": [C2z, C2x],
        "mm2": [C2z, Mx],
        "mmm": [C2z, C2x, INV],
        "4": [C4z],
        "-4": [S4z],
        "4/m": [C4z, INV],
        "422": [C4z, C2x],
        "4mm": [C4z, Mx],
        "-42m": [S4z, C2x],
        "4/mmm": [C4z, C2x, INV],
        "3": [C3z],
        "-3": [C3z, INV],
        "32": [C3z, C2x],
        "3m": [C3z, Mx],
        "-3m": [C3z, C2x, INV],
        "6": [C6z],
        "-6": [C3z, Mz],
        "6/m": [C6z, INV],
        "622": [C6z, C2x],
        "6mm": [C6z, Mx],
        "-6m2": [C3z, Mz, C2x],
        "6/mmm": [C6z, C2x, INV],
        "23": [C2z, C3_111],
        "m-3": [C2z, C3_111, INV],
        "432": [C4z, C3_111],
        "-43m": [S4z, C3_111],
        "m-3m": [C4z, C3_111, INV],
    }
    CENTRO = {"-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"}
    PROPER = {
        "-1": "1",
        "2/m": "2",
        "mmm": "222",
        "4/m": "4",
        "4/mmm": "422",
        "-3": "3",
        "-3m": "32",
        "6/m": "6",
        "6/mmm": "622",
        "m-3": "23",
        "m-3m": "432",
    }

    ORD = {
        "1": 1,
        "-1": 2,
        "2": 2,
        "m": 2,
        "2/m": 4,
        "222": 4,
        "mm2": 4,
        "mmm": 8,
        "4": 4,
        "-4": 4,
        "4/m": 8,
        "422": 8,
        "4mm": 8,
        "-42m": 8,
        "4/mmm": 16,
        "3": 3,
        "-3": 6,
        "32": 6,
        "3m": 6,
        "-3m": 12,
        "6": 6,
        "-6": 6,
        "6/m": 12,
        "622": 12,
        "6mm": 12,
        "-6m2": 12,
        "6/mmm": 24,
        "23": 12,
        "m-3": 24,
        "432": 24,
        "-43m": 24,
        "m-3m": 48,
    }

    def close(gens):
        G = {sp.ImmutableMatrix(I3)} | {sp.ImmutableMatrix(g) for g in gens}
        changed = True
        while changed:
            changed = False
            for a in list(G):
                for b in list(G):
                    p = sp.ImmutableMatrix(sp.simplify(a * b))
                    if p not in G:
                        G.add(p)
                        changed = True
            if len(G) > 200:
                raise RuntimeError("group closure diverged")
        return G

    cols = []
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                T = sp.zeros(27, 1)
                T[i * 9 + j * 3 + k] += 1
                T[i * 9 + k * 3 + j] += 1
                cols.append(T)
    B = sp.Matrix.hstack(*cols)
    Bp = (B.T * B).inv() * B.T

    def inv_dim(G):
        P = sp.zeros(27, 27)
        for R in G:
            P += sp.Matrix(sp.kronecker_product(sp.Matrix(R), sp.Matrix(R), sp.Matrix(R)))
        return sp.simplify(Bp * sp.simplify(P / len(G)) * B).rank()

    groups, dims = {}, {}
    for name, gens in GEN.items():
        G = close(gens)
        chk(f"|{name}| equals the crystallographic order", len(G), ORD[name], 0)
        groups[name], dims[name] = G, inv_dim(G)

    zero = sorted(g for g in dims if dims[g] == 0)
    expect = sorted(CENTRO | {"432"})
    chk(
        "V^G = {0} exactly for 11 centrosymmetric classes and 432",
        1.0 if zero == expect else 0.0,
        1.0,
        0,
        "" if zero == expect else f"got {zero}",
    )
    noncentro = [g for g in dims if g not in CENTRO]
    chk("number of non-centrosymmetric classes", len(noncentro), 21, 0)
    chk(
        "number of piezoelectric non-centrosymmetric classes",
        sum(1 for g in noncentro if dims[g] > 0),
        20,
        0,
        "432 is the sole exception",
    )

    print("  -- proper-rotation subgroup of each centrosymmetric class --")
    for cls, claimed in PROPER.items():
        Gp = {R for R in groups[cls] if sp.Matrix(R).det() == 1}
        chk(f"G+({cls}) = {claimed}", 1.0 if Gp == groups[claimed] else 0.0, 1.0, 0)
    chk("dim V^432 (Reynolds rank over m-3m proper subgroup)", dims["432"], 0, 0)
    chk("dim V^23  (Reynolds rank over m-3 proper subgroup)", dims["23"], 1, 0)
    chk(
        "dim V^422 (rutile parent 4/mmm proper subgroup)",
        dims["422"],
        1,
        0,
        "nonzero, so only parity forbids a rutile response",
    )

    print("  -- Theorem 2: parity gap gamma_r = dim V^{G+} - dim V^{G} --")

    def sym_basis(r):
        if r == 1:
            return sp.eye(3)
        if r == 2:
            cols = []
            for i in range(3):
                for j in range(i, 3):
                    T = sp.zeros(9, 1)
                    T[i * 3 + j] += 1
                    T[j * 3 + i] += 1
                    cols.append(T)
            return sp.Matrix.hstack(*cols)
        if r == 3:
            return B
        if r == 4:
            seen, cols = set(), []
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                            if key in seen:
                                continue
                            seen.add(key)
                            T = sp.zeros(81, 1)
                            for a, b in {(i, j), (j, i)}:
                                for c, d in {(k, l), (l, k)}:
                                    T[a * 27 + b * 9 + c * 3 + d] += 1
                                    T[c * 27 + d * 9 + a * 3 + b] += 1
                            cols.append(T)
            return sp.Matrix.hstack(*cols)

    def dimV(G, Bs, r):
        n = 3**r
        P = sp.zeros(n, n)
        for R in G:
            M = sp.Matrix(R)
            K = M
            for _ in range(r - 1):
                K = sp.Matrix(sp.kronecker_product(K, M))
            P += K
        Bp = (Bs.T * Bs).inv() * Bs.T
        return sp.simplify(Bp * sp.simplify(P / len(G)) * Bs).rank()

    TABLE1 = {
        "-1": (3, 0, 18, 0),
        "2/m": (1, 0, 8, 0),
        "mmm": (0, 0, 3, 0),
        "4/m": (1, 0, 4, 0),
        "4/mmm": (0, 0, 1, 0),
        "-3": (1, 0, 6, 0),
        "-3m": (0, 0, 2, 0),
        "6/m": (1, 0, 4, 0),
        "6/mmm": (0, 0, 1, 0),
        "m-3": (0, 0, 1, 0),
        "m-3m": (0, 0, 0, 0),
    }
    BS = {r: sym_basis(r) for r in (1, 2, 3, 4)}
    for cls, stated in TABLE1.items():
        Gp = {R for R in groups[cls] if sp.Matrix(R).det() == 1}
        for idx, r in enumerate((1, 2, 3, 4)):
            gap = dimV(Gp, BS[r], r) - dimV(groups[cls], BS[r], r)
            chk(f"gamma_{r}({cls})", gap, stated[idx], 0)
    even_ok = all(TABLE1[c][i] == 0 for c in TABLE1 for i in (1, 3))
    chk("parity gap vanishes identically at even rank", 1.0 if even_ok else 0.0, 1.0, 0)
    chk(
        "gamma_3 = 0 for exactly one centrosymmetric class",
        sum(1 for c in TABLE1 if TABLE1[c][2] == 0),
        1,
        0,
        "m-3m",
    )

    print("  -- Theorem 1(ii) tightness: orbit construction well defined --")
    import random as _rnd

    _rnd.seed(0)
    for name in ["1", "2", "222", "4", "422", "3", "32", "6", "622", "23", "432"]:
        Gp = groups[name]
        P27 = sp.zeros(27, 27)
        for R in Gp:
            M = sp.Matrix(R)
            P27 += sp.Matrix(sp.kronecker_product(M, M, M))
        Bp = (B.T * B).inv() * B.T
        Pi = sp.simplify(Bp * sp.simplify(P27 / len(Gp)) * B)
        if Pi.rank() == 0:
            chk(f"G+={name}: dim V^G+ = 0, nothing to realise", 0, 0, 0)
            continue
        v = sp.Matrix([sp.Rational(_rnd.randint(-5, 5)) for _ in range(18)])
        T27 = B * sp.simplify(Pi * v)
        worst = 0
        for R in Gp:
            M = sp.Matrix(R)
            d = sp.simplify(sp.Matrix(sp.kronecker_product(M, M, M)) * T27 - T27)
            worst = max(worst, max(abs(e) for e in d))
        chk(
            f"G+={name}: max |R^(x r) T - T| over G+",
            float(worst),
            0.0,
            0,
            f"dim V^G+ = {Pi.rank()}",
        )


def section_theorems(n_trials=2000, seed=0):
    import numpy as np

    print("\n[2] THEOREM STRESS TEST (numpy, synthetic centrosymmetric inputs)")
    rng = np.random.default_rng(seed)

    def centro(n=6):
        half = rng.normal(size=(n, 3))
        return np.vstack([half, -half, np.zeros((1, 3))])

    f = lambda R: np.einsum("ni,nj,nk->ijk", R, R, R)

    wl = wt = wp = wj = 0.0
    for _ in range(n_trials):
        x = centro()
        T, Ti = f(x), f(-x)
        wl = max(wl, np.abs(Ti - T).max())
        wt = max(wt, np.abs(T).max())
        d, e = np.linalg.norm(Ti - T), np.linalg.norm(Ti + T)
        wp = max(wp, np.linalg.norm(T) - 0.5 * (d + e))

        u = rng.normal(size=x.shape) * 1e-6
        n2 = (len(x) - 1) // 2
        P = np.vstack([-u[n2 : 2 * n2], -u[:n2], -u[2 * n2 :]])
        wj = max(wj, abs(np.linalg.norm(f(x + (u + P) / 2))))

    chk("Lemma 1   max |f(I.x) - f(x)|", wl, 0.0, 1e-10, "must vanish")
    chk("Theorem 1 max |f(x)| on centrosymmetric x", wt, 0.0, 1e-10, "must vanish")
    chk("Prop 1    max [|f| - (d+eta)/2]  (must be <= 0)", min(wp, 0.0), 0.0, 1e-12)
    chk("Cor 3     max |f(x + P_even u)| at |u| ~ 1e-6", wj, 0.0, 1e-12)

    import sympy as sp

    fx, eR, gR = sp.symbols("f e_R g_R")
    Rf = sp.solve(sp.Eq(fx + eR, sp.Symbol("Rf") + gR), sp.Symbol("Rf"))[0]
    chk(
        "Cor 1 displayed identity is (e_R - g_R)",
        1.0 if sp.simplify(Rf - fx - (eR - gR)) == 0 else 0.0,
        1.0,
        0,
        "the displayed identity is (e_R - g_R)",
    )


def section_reconcile():
    print("\n[3] NUMERIC RECONCILIATION (result records against each other)")
    N, M3M, M3, NC, SEEN, UNSEEN = 2000, 166, 18, 1816, 1232, 768
    chk("m-3m + m-3 + non-cubic = 2000", M3M + M3 + NC, N, 0)
    chk("crystals only parity protects", N - M3M, 1834, 0)
    chk("  as a percentage", 100 * (N - M3M) / N, 91.7, 0.05)
    chk("m-3m share of the population (%)", 100 * M3M / N, 8.3, 0.05)

    print("  -- Suppl. Table 7 (per family) reproduces Suppl. Table 2 (population) --")
    for c, (a, b, d, want) in {
        "NequIP": (0.0000, 0.9630, 0.9765, 0.8953),
        "Allegro": (0.0000, 0.9815, 0.9919, 0.9095),
        "MACE": (0.0000, 0.9444, 0.9903, 0.9077),
        "EqV2": (0.5341, 0.9630, 0.9956, 0.9570),
    }.items():
        chk(f"{c} SO(3) population ff", (M3M * a + M3 * b + NC * d) / N, want, 1e-3)

    print("  -- Suppl. Table 5 (seen/unseen) reproduces the same population ff --")
    for c, (s, u, want) in {
        "NequIP": (0.9042, 0.8811, 0.8953),
        "Allegro": (0.9075, 0.9128, 0.9095),
        "MACE": (0.9075, 0.9080, 0.9077),
        "EqV2": (0.9508, 0.9670, 0.9570),
    }.items():
        chk(f"{c} SO(3) baseline ff", (SEEN * s + UNSEEN * u) / N, want, 1e-3)

    print("  -- 'of the crystals where free to violate' --")
    for c, (ff, want) in {
        "NequIP": (0.8953, 97.6),
        "Allegro": (0.9095, 99.2),
        "MACE": (0.9077, 99.0),
    }.items():
        chk(f"{c} share of the 1,834", 100 * ff * N / 1834, want, 0.06)

    print("  -- headline separation --")
    chk("NequIP median ratio / 1e6", 6.66e-1 / 3.19e-7 / 1e6, 2.0, 0.15)
    chk("orders of magnitude separating the arms", math.log10(6.66e-1 / 3.19e-7), 6.0, 0.35)
    chk("largest O(3) prediction below tau by", 1e-2 / 1.3e-3, 7.6, 0.1)

    print("  -- effect sizes, Suppl. Table 4 --")
    for lab, mo, so, ms, ss, want in [
        ("NequIP U0", 53.65, 22.50, 51.66, 9.19, -0.12),
        ("NequIP dipole", 0.0519, 0.0027, 0.0530, 0.0026, 0.42),
        ("NequIP elastic", 24.33, 0.27, 24.49, 0.14, 0.74),
        ("NequIP piezo", 0.2083, 0.0077, 0.2405, 0.0080, 4.10),
        ("Allegro U0", 31.08, 12.89, 26.98, 4.88, -0.42),
        ("Allegro dipole", 0.0751, 0.0023, 0.0764, 0.0021, 0.59),
        ("Allegro elastic", 23.72, 0.25, 23.89, 0.31, 0.60),
        ("Allegro piezo", 0.2140, 0.0058, 0.2589, 0.0176, 3.43),
        ("MACE U0", 16.76, 6.29, 26.09, 12.49, 0.94),
        ("MACE dipole", 0.0484, 0.0018, 0.0500, 0.0045, 0.47),
        ("MACE elastic", 24.92, 0.54, 24.94, 0.60, 0.04),
        ("MACE piezo", 0.2222, 0.0066, 0.2567, 0.0084, 4.57),
    ]:
        chk(f"Delta/sigma {lab}", (ms - mo) / math.sqrt((so**2 + ss**2) / 2), want, 0.006)

    print("  -- budgets, splits, sweeps --")
    chk("260,779 s in GPU-hours", 260779 / 3600, 72.44, 0.01)
    chk("per-core hours sum", 7.62 + 5.06 + 40.64 + 19.12, 72.44, 0.01)
    chk("grid = 7 arms x 4 targets x 3 seeds", 7 * 4 * 3, 84, 0)
    chk("O(3) structure-runs", 3 * 3 * 2000, 18000, 0)
    chk("piezo split sums", 2649 + 331 + 332, 3312, 0)
    chk("elastic split sums", 10464 + 1308 + 1308, 13080, 0)
    chk("QM9 split sums", 110000 + 10000 + 10831, 130831, 0)
    chk("QM9 = 133,885 - 3,054 uncharacterised", 133885 - 3054, 130831, 0)
    chk("QM9 forced-zero dipoles (%)", 100 * 7 / 10831, 0.065, 0.001)
    chk("zero-injection median falls (fold)", 0.666 / 0.077, 8.6, 0.06)
    chk("W=100 reduces trained-zero ff by", 0.8953 - 0.6762, 0.22, 0.002)
    chk("U0 error as % of target s.d. (low)", 100 * 16.76 / 1085.57, 1.5, 0.05)
    chk("U0 error as % of target s.d. (high)", 100 * 53.65 / 1085.57, 4.9, 0.05)
    chk("frozen-head MAE gap (%)", 100 * (0.1697 - 0.1583) / 0.1583, 7, 0.3)

    print("  -- pooling arms and headline runs, each against its own source --")
    import json as _json
    import statistics as _st

    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    _f5 = _json.load(open(os.path.join(_root, "f5_pooling_arms.json")))
    _e1 = _json.load(open(os.path.join(_root, "e1_augmentation.json")))
    _ns, _nu = _e1["n_seen"], _e1["n_unseen"]
    _stated_pool = {
        "nequip": (0.8872, 0.0014),
        "allegro": (0.8982, 0.0021),
        "mace": (0.8978, 0.0006),
        "equiformer_v2": (0.9472, 0.0172),
    }
    _stated_head = {"nequip": 0.8953, "allegro": 0.9095, "mace": 0.9077, "equiformer_v2": 0.9570}
    _gaps = []
    for _c, (_m, _s) in _stated_pool.items():
        _arm = [p["ff_sum"] for p in _f5["pairs"] if p["core"] == _c and p["parity"] == "so3"]
        chk(f"pooling summed arm mean, {_c}", _st.mean(_arm), _m, 5e-5, "f5_pooling_arms.json")
        chk(f"pooling summed arm s.d., {_c}", _st.stdev(_arm), _s, 5e-5, "f5_pooling_arms.json")
        _head = [
            (x["ff_seen"] * _ns + x["ff_unseen"] * _nu) / (_ns + _nu)
            for x in _e1["arms"][_c + "_baseline_so3"]["per_seed"]
        ]
        chk(f"headline ff, {_c}", _st.mean(_head), _stated_head[_c], 5e-5, "e1_augmentation.json")
        _gaps.append(_st.mean(_head) - _st.mean(_arm))
    chk(
        "all four pooling-to-headline gaps positive",
        min(_gaps) > 0,
        True,
        0,
        "independent retrainings, offset in one direction",
    )
    chk(
        "gap spread stays inside one percentage point",
        max(_gaps) - min(_gaps) < 0.01,
        True,
        0,
        "0.008 to 0.011",
    )
    chk(
        "batch size equal within every pooling pair",
        _f5["summary"]["bs_mismatch"],
        0,
        0,
        "f5 summary",
    )

    print("  -- seen-group saturation (Allegro and MACE at 0.9075 +/- 0.0000) --")
    _sg = {
        r["index"]: r
        for r in _json.load(open(os.path.join(_root, "ood_spacegroups.json")))["records"]
    }
    _seen = _json.load(open(os.path.join(_root, "e1_eval_split.json")))["seen_indices"]
    _nc = sum(1 for i in _seen if _sg[i]["family"] == "non-cubic")
    chk("seen group size", len(_seen), 1232, 0, "e1_eval_split.json")
    chk("seen non-cubic count", _nc, 1119, 0, "joined to ood_spacegroups")
    chk("seen m-3m count", len(_seen) - _nc, 113, 0, "joined to ood_spacegroups")
    _e7 = _json.load(open(os.path.join(_root, "e7_rotation_subgroup.json")))["by_family"]
    for _a in ("allegro_so3", "mace_so3"):
        _counts = [
            round(x["ff_seen"] * len(_seen))
            for x in _e1["arms"][_a.replace("_so3", "_baseline_so3")]["per_seed"]
        ]
        chk(
            f"{_a} flagged-count spread across seeds",
            max(_counts) - min(_counts),
            0,
            0,
            "identical in all three seeds",
        )
        chk(
            f"{_a} flags 1,118 of the seen group",
            _counts[0],
            1118,
            0,
            "one short of every non-cubic crystal",
        )
        chk(
            f"{_a} m-3m false-flag fraction is zero",
            _e7[_a]["m-3m"]["false_flag_max"],
            0.0,
            0,
            "e7_rotation_subgroup.json",
        )


def smoke():

    print("[smoke] planting a deliberate mismatch; the checker must report FAIL")
    before = len(FAIL)
    chk("planted mismatch (expected to FAIL)", 1.0, 2.0, 1e-9)
    caught = len(FAIL) == before + 1
    FAIL.pop()
    print(f"[smoke] checker {'works' if caught else 'IS BROKEN'}")
    if not caught:
        sys.exit(2)
    section_theorems(n_trials=50)
    print("[smoke] done")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        smoke()
        section_group_theory()
        section_theorems()
        section_reconcile()
    print("\n" + "=" * 88)
    print(f"PASSED {len(PASS)}    FAILED {len(FAIL)}")
    for l, g, w, n in FAIL:
        print(f"  FAIL  {l}: computed {g:.6g} vs stated {w:.6g}   {n}")
    sys.exit(1 if FAIL else 0)
