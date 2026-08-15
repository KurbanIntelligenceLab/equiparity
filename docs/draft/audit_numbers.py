"""
audit_numbers.py
================
Internal-consistency audit of every quantitative claim in the manuscript that can be
checked against another quantitative claim in the manuscript.

This does NOT verify the numbers against the raw run outputs, which were not provided.
It verifies that the paper is consistent with itself. Where the paper reports the same
quantity twice, or reports enough to reconstruct a third quantity, this script does the
arithmetic. A paper whose tables were fabricated would fail these checks; a paper whose
tables were machine-generated from real runs should pass them all.

    python audit_numbers.py
"""

import math
import sys

PASS, FAIL = "OK  ", "FAIL"
results = []


def chk(name, got, want, tol=5e-3, note=""):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    results.append((PASS if ok else FAIL, name, f"{got:.6g}", f"{want:.6g}", note))
    return ok


print("=" * 104)
print("INTERNAL-CONSISTENCY AUDIT")
print("=" * 104)

# ---------------------------------------------------------------- dataset arithmetic
chk("QM9: 133,885 - 3,054 uncharacterized", 133885 - 3054, 130831)
chk("QM9 split sums to the corpus", 110000 + 10000 + 10831, 130831)
chk("Piezo split sums to 3,312", 2649 + 331 + 332, 3312)
chk("Piezo split is 80/10/10", 2649 / 3312, 0.80, tol=2e-3)
chk("Elastic split sums to 13,080", 10464 + 1308 + 1308, 13080)
chk("Elastic split is 80/10/10", 10464 / 13080, 0.80, tol=1e-6)
chk("Zero rows = 1,000 injected + 16 real", 1000 + 16, 1016)
chk("Grid = 7 arms x 4 targets x 3 seeds", 7 * 4 * 3, 84)
chk("Population families sum to 2,000", 166 + 18 + 1816, 2000)
chk("Seen + unseen = 2,000", 1232 + 768, 2000)

# ---------------------------------------------------------------- the normalization constant
# Targets are scaled by the training standard deviation with NO mean subtraction, i.e. an RMS.
# Freezing it at the unaugmented value (0.749134) and recomputing over the zero-augmented set
# must give sqrt(2649/3649) times it, because 1,000 exact zeros add no sum of squares.
recomputed = 0.749134 * math.sqrt(2649 / (2649 + 1000))
chk("Augmented normalization (Supp. Note 8)", recomputed, 0.638289, tol=1e-4,
    note="0.749134 x sqrt(2649/3649); confirms the split AND the augmentation size")

# ---------------------------------------------------------------- compute table
chk("GPU-hours sum to the reported total", 7.62 + 5.06 + 40.64 + 19.12, 72.44, tol=1e-3)
chk("Runs sum to the grid", 24 + 24 + 24 + 12, 84)
chk("260,779 s = 72.44 GPU-hours", 260779 / 3600, 72.44, tol=1e-3)

# ---------------------------------------------------------------- false-flag reconstruction
# Supp. Table 15 (per point-group family) must reproduce Supp. Table 3 (population) exactly.
fam = {"NequIP": (0.0000, 0.9630, 0.9765), "Allegro": (0.0000, 0.9815, 0.9919),
       "MACE": (0.0000, 0.9444, 0.9903), "EquiformerV2": (0.5341, 0.9630, 0.9956)}
pop = {"NequIP": 0.8953, "Allegro": 0.9095, "MACE": 0.9077, "EquiformerV2": 0.9570}
for k, (a, b, c) in fam.items():
    chk(f"{k}: per-family (T15) reproduces population (T3)",
        (166 * a + 18 * b + 1816 * c) / 2000, pop[k], tol=2e-4)

# Supp. Table 10 (seen/unseen) must ALSO reproduce Supp. Table 3, independently.
su = {"NequIP": (0.9042, 0.8811), "Allegro": (0.9075, 0.9128),
      "MACE": (0.9075, 0.9080), "EquiformerV2": (0.9508, 0.9670)}
for k, (s, u) in su.items():
    chk(f"{k}: seen/unseen (T10) reproduces population (T3)",
        (1232 * s + 768 * u) / 2000, pop[k], tol=3e-4)

# ---------------------------------------------------------------- the rotation ceiling
ceiling = (18 + 1816) / 2000
chk("Rotation ceiling = 1 - (m-3m fraction)", ceiling, 0.917, tol=1e-3)
chk("m-3m fraction", 166 / 2000, 0.083, tol=1e-3)
for k, s in [("NequIP", 0.976), ("Allegro", 0.992), ("MACE", 0.990)]:
    chk(f"{k}: fraction of the ceiling saturated", pop[k] / ceiling, s, tol=1e-3)
chk("EquiformerV2 excess over the ceiling (pp)", 100 * (pop["EquiformerV2"] - ceiling), 4.0, tol=0.03)
chk("...accounted for by its m-3m leakage (pp)", 100 * 0.5341 * 166 / 2000, 4.4, tol=0.02)

# ---------------------------------------------------------------- accuracy effect sizes
acc = {
    ("NequIP", "U0"): ((53.65, 22.50), (51.66, 9.19)),
    ("NequIP", "dipole"): ((0.0519, 0.0027), (0.0530, 0.0026)),
    ("NequIP", "elastic"): ((24.33, 0.27), (24.49, 0.14)),
    ("NequIP", "piezo"): ((0.2083, 0.0077), (0.2405, 0.0080)),
    ("Allegro", "U0"): ((31.08, 12.89), (26.98, 4.88)),
    ("Allegro", "dipole"): ((0.0751, 0.0023), (0.0764, 0.0021)),
    ("Allegro", "elastic"): ((23.72, 0.25), (23.89, 0.31)),
    ("Allegro", "piezo"): ((0.2140, 0.0058), (0.2589, 0.0176)),
    ("MACE", "U0"): ((16.76, 6.29), (26.09, 12.49)),
    ("MACE", "dipole"): ((0.0484, 0.0018), (0.0500, 0.0045)),
    ("MACE", "elastic"): ((24.92, 0.54), (24.94, 0.60)),
    ("MACE", "piezo"): ((0.2222, 0.0066), (0.2567, 0.0084)),
}
d = {}
for (c, t), ((mo, so), (ms, ss)) in acc.items():
    d[(c, t)] = (ms - mo) / math.sqrt((so ** 2 + ss ** 2) / 2)
chk("Max |delta/sigma| on the three inactive targets",
    max(abs(d[(c, t)]) for c in ("NequIP", "Allegro", "MACE") for t in ("U0", "dipole", "elastic")),
    0.94, tol=6e-3, note="paper: 'at most 0.94'")
chk("Piezo delta/sigma, minimum", min(d[(c, "piezo")] for c in ("NequIP", "Allegro", "MACE")), 3.4, tol=0.02)
chk("Piezo delta/sigma, maximum", max(d[(c, "piezo")] for c in ("NequIP", "Allegro", "MACE")), 4.6, tol=0.02)
dip = [d[(c, "dipole")] for c in ("NequIP", "Allegro", "MACE")]
ela = [d[(c, "elastic")] for c in ("NequIP", "Allegro", "MACE")]
results.append((PASS if all(x > 0 for x in dip) else FAIL,
                "Dipole: O(3) nominally better in all 3 cores", f"{min(dip):+.2f}..{max(dip):+.2f}", "all > 0",
                "the claim the paper originally got backwards"))
results.append((PASS if all(x > 0 for x in ela) else FAIL,
                "Elasticity: O(3) nominally better in all 3 cores", f"{min(ela):+.2f}..{max(ela):+.2f}", "all > 0", ""))
u0 = [d[(c, "U0")] for c in ("NequIP", "Allegro", "MACE")]
results.append((PASS if not (all(x > 0 for x in u0) or all(x < 0 for x in u0)) else FAIL,
                "U0: sign varies across cores", f"{min(u0):+.2f}..{max(u0):+.2f}", "mixed", ""))

# ---------------------------------------------------------------- U0 as a fraction of target sd
chk("U0 error as % of target s.d. (low)", 100 * 16.76 / 1085.57, 1.5, tol=0.04)
chk("U0 error as % of target s.d. (high)", 100 * 53.65 / 1085.57, 4.9, tol=0.02)

# ---------------------------------------------------------------- raw-variant tolerance
chk("Raw not centrosymmetric at 1e-4 (%)", 100 * (1 - 1956 / 2000), 2.2, tol=0.02)
chk("Raw not centrosymmetric at 1e-5 (%)", 100 * (1 - 1869 / 2000), 6.5, tol=0.06)
chk("O(3) raw false-flag = 1 structure in 1 of 3 seeds", (1 / 3) * (1 / 2000), 0.00017, tol=0.02)
chk("O(3) raw false-flag = 1 structure in 2 of 3 seeds", (2 / 3) * (1 / 2000), 0.00033, tol=0.02)

# ---------------------------------------------------------------- loss-weight sweep
chk("W=100: median shrinks ~9-fold", 0.1200 / 0.0140, 8.6, tol=0.02, note="paper says 0.120 -> 0.014")
chk("W=100 median vs O(3) floor: >4 orders", math.log10(0.0140 / 3e-7), 4.7, tol=0.02)

# ---------------------------------------------------------------- the arithmetic floor claim
chk("MACE O(3) floor / NequIP O(3) floor", 2.73e-6 / 3.13e-7, 8.7, tol=0.02, note="paper: 'roughly eight times'")
chk("Worst single O(3) prediction vs threshold", 0.01 / 1.32e-3, 7.6, tol=0.02, note="paper: 'a factor of seven below'")

# ---------------------------------------------------------------- irrep dimensions
chk("Piezoelectric: 2x1o + 1x2o + 1x3o = 18", 2 * 3 + 1 * 5 + 1 * 7, 18)
chk("Elasticity: 2x0e + 2x2e + 1x4e = 21", 2 * 1 + 2 * 5 + 1 * 9, 21)
chk("dim R^3 (x) Sym^2(R^3)", 3 * 6, 18)

# ---------------------------------------------------------------- print
w = max(len(x[1]) for x in results) + 2
nf = 0
for st, name, got, want, note in results:
    if st == FAIL:
        nf += 1
    print(f"  {st}  {name:<{w}} got {got:>10}   expected {want:>10}   {note}")
print("=" * 104)
print(f"  {len(results) - nf} consistent, {nf} inconsistent, out of {len(results)} checks")
if nf == 0:
    print("\n  Every quantity the manuscript reports twice, or reports enough to reconstruct,")
    print("  agrees with itself. This is a strong indication the tables were machine-generated")
    print("  from real runs. It is NOT a check against the raw outputs, which were not provided.")
sys.exit(1 if nf else 0)
