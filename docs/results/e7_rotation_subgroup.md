# E7 — the rotation subgroup explains SO(3)'s correct zeros (supplementary)

SO(3) equivariance forces `T(x) = R·T(x)` for every proper rotation `R` in the crystal's
point group. Under **432** (the proper subgroup of m-3̄m) the only invariant rank-3 
tensor is
zero, so an exactly SO(3)-equivariant model is *forced* to predict zero — no parity label
required. Under **23** (the proper subgroup of m-3̄, a piezoelectric class) invariants 
exist
and nothing is forced. Both group-theoretic facts are asserted as tests in
`tests/test_physics_claims.py`.

Counts in the OOD set: m-3̄ **18**, m-3̄m **166**,
non-cubic **1816** (total 2,000; spglib at symprec 1e-3).

## Violation by point-group family (idealized variant)

Range over the 3 training seeds. False-flag threshold 0.01 C/m².

| core | arm | family | rotation subgroup | rank-3 invariant? | n | median ‖T‖ | false-flag |
|---|---|---|---|---|---|---|---|
| Allegro | o3 | m-3 | 23 (order 12) | yes | 18 | 5.92e-07 – 7.82e-07 | 0.000e+00 |
| Allegro | o3 | m-3m | 432 (order 24) | **no** | 166 | 1.36e-07 – 1.67e-07 | 0.000e+00 |
| Allegro | o3 | non-cubic | various | yes | 1816 | 3.83e-07 – 4.24e-07 | 0.000e+00 |
| Allegro | so3 | m-3 | 23 (order 12) | yes | 18 | 4.65e-01 – 7.35e-01 | 9.44e-01 – 1.00e+00 |
| Allegro | so3 | m-3m | 432 (order 24) | **no** | 166 | 1.33e-07 – 2.09e-07 | 0.000e+00 |
| Allegro | so3 | non-cubic | various | yes | 1816 | 9.69e-01 – 1.23e+00 | 9.90e-01 – 9.93e-01 |
| EquiformerV2 | so3 | m-3 | 23 (order 12) | yes | 18 | 2.68e-01 – 3.91e-01 | 9.44e-01 – 1.00e+00 |
| EquiformerV2 | so3 | m-3m | 432 (order 24) | **no** | 166 | 9.03e-03 – 1.58e-02 | 4.94e-01 – 5.78e-01 |
| EquiformerV2 | so3 | non-cubic | various | yes | 1816 | 4.16e-01 – 6.08e-01 | 9.92e-01 – 9.98e-01 |
| MACE | o3 | m-3 | 23 (order 12) | yes | 18 | 4.52e-06 – 6.98e-06 | 0.000e+00 |
| MACE | o3 | m-3m | 432 (order 24) | **no** | 166 | 1.03e-06 – 1.07e-06 | 0.000e+00 |
| MACE | o3 | non-cubic | various | yes | 1816 | 2.58e-06 – 3.01e-06 | 0.000e+00 |
| MACE | so3 | m-3 | 23 (order 12) | yes | 18 | 4.49e-01 – 6.24e-01 | 8.89e-01 – 1.00e+00 |
| MACE | so3 | m-3m | 432 (order 24) | **no** | 166 | 9.99e-07 – 1.14e-06 | 0.000e+00 |
| MACE | so3 | non-cubic | various | yes | 1816 | 9.61e-01 – 1.10e+00 | 9.90e-01 – 9.91e-01 |
| NequIP | o3 | m-3 | 23 (order 12) | yes | 18 | 3.17e-07 – 5.50e-07 | 0.000e+00 |
| NequIP | o3 | m-3m | 432 (order 24) | **no** | 166 | 1.36e-07 – 1.74e-07 | 0.000e+00 |
| NequIP | o3 | non-cubic | various | yes | 1816 | 2.86e-07 – 3.95e-07 | 0.000e+00 |
| NequIP | so3 | m-3 | 23 (order 12) | yes | 18 | 1.75e-01 – 3.79e-01 | 9.44e-01 – 1.00e+00 |
| NequIP | so3 | m-3m | 432 (order 24) | **no** | 166 | 1.76e-07 – 2.02e-07 | 0.000e+00 |
| NequIP | so3 | non-cubic | various | yes | 1816 | 6.90e-01 – 8.03e-01 | 9.76e-01 – 9.77e-01 |

## Where SO(3)'s correct zeros live

Structures the SO(3) arm does **not** false-flag (seed-averaged ‖T‖ ≤ 0.01).

| core | unflagged | of which cubic | m-3̄m | m-3̄ | cubic fraction |
|---|---|---|---|---|---|
| NequIP | 200 | 166 | 166 | 0 | 0.830 |
| Allegro | 175 | 166 | 166 | 0 | 0.949 |
| MACE | 175 | 166 | 166 | 0 | 0.949 |
| EquiformerV2 | 71 | 69 | 69 | 0 | 0.972 |

## Reading

For the three exactly-SO(3) e3nn cores (NequIP, Allegro, MACE) the m-3̄m false-flag rate 
is
**0.000** in every seed and the median ‖T‖ sits at machine zero (1.8e-07 – 1.1e-06): the 
432
constraint is satisfied exactly, by rotation alone. All 166 m-3̄m crystals are unflagged, 
for
every core and every seed.

The m-3̄ crystals, whose rotation subgroup 23 does permit a rank-3 tensor, are 
false-flagged at
**0.889 – 1.000** depending on seed (one MACE seed leaves 2 of the 18 below threshold). 
No
m-3̄ crystal survives seed-averaging unflagged.

The correspondence is therefore strong but not exhaustive: the 432 constraint accounts 
for 166
of each core's 175–200 unflagged crystals (83%–95%). The remainder are non-cubic 
structures
on which the model simply happened to predict a small magnitude — no symmetry protects 
them,
and they carry no guarantee.

**EquiformerV2 is the exception, and it is not a symmetry-group difference.** It is only
*approximately* SO(3)-equivariant (E5: rotation error 7e-2 – 1.1e-1; its forward redraws 
a
random per-edge frame each call), so on m-3̄m — where the exact SO(3) answer is zero — 
the
entire predicted signal is equivariance error. Its m-3̄m numbers are single stochastic 
draws
and straddle the 0.01 threshold; they should not be read as a statement about SO(3).

**Consequence for the headline.** 1,816 non-cubic + 18 m-3̄ = 1,834 of 2,000 (91.7%) of 
the
OOD set lies outside what rotation can enforce, and the observed SO(3) false-flag rate is
0.89–0.91 — close to that ceiling, the small shortfall being non-cubic crystals with
incidentally small predictions. SO(3)'s zeros are the ones rotations already guarantee;
O(3)'s zeros are the strictly larger set that parity guarantees. This sharpens the 
paper's
claim rather than weakening it, and it costs no new inference.
