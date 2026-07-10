# E2 — symmetry-breaking curves

`x(δ) = x_parent + δ·Δx` along a [001] polar mode. δ = 0 is the 
centrosymmetric parent,
where the piezoelectric tensor is exactly zero; every δ > 0 is polar, where a 
response is
allowed. δ = 1 is the physical distortion amplitude.

All 33 frames per material were spglib-verified at symprec 1e-08.
At the selection tolerance 1e-3 the frames below δ ≈ 0.006 are *wrongly* reported
centrosymmetric — the maximum displacement falls below the tolerance. Same 
phenomenon as
the mp-1227949 raw-coordinate artifact (appendix A5).

## Which material can show the parity effect

The parent's **proper-rotation subgroup** decides whether SO(3) equivariance 
alone already
forbids a rank-3 tensor at δ = 0. Where it does, both arms start at machine zero and the
curve says nothing about parity.

| material | parent | polar | rotation subgroup | separates the arms at δ=0? |
|---|---|---|---|---|
| TiO₂ (rutile) | P4₂/mnm (136) | P4₂nm (102) | 422 — permits rank-3 | **yes** |
| BaTiO₃ | Pm-3̄m (221) | P4mm (99) | 432 — forbids rank-3 | no |
| PbTiO₃ | Pm-3̄m (221) | P4mm (99) | 432 — forbids rank-3 | no |

Values are ‖T‖_F, mean over 3 seeds. Full curves: `results/e2_symmetry_breaking.csv`.

## TiO2  (422 (permits rank-3))

| core | arm | δ=0 | δ=0.001 | δ=0.05 | δ=0.2 | δ=1 |
|---|---|---|---|---|---|---|
| allegro | o3 | 2.655e-07 | 1.034e-03 | 5.167e-02 | 2.054e-01 | 9.130e-01 |
| allegro | so3 | 2.825e-01 | 2.825e-01 | 2.871e-01 | 3.469e-01 | 7.919e-01 |
| equiformer_v2 | so3 | 9.214e-02 | 9.117e-02 | 1.302e-01 | 3.099e-01 | 1.402e+00 |
| mace | o3 | 9.077e-07 | 1.901e-03 | 9.501e-02 | 3.775e-01 | 1.612e+00 |
| mace | so3 | 1.414e-01 | 1.414e-01 | 1.627e-01 | 3.452e-01 | 1.465e+00 |
| nequip | o3 | 1.935e-07 | 1.090e-03 | 5.451e-02 | 2.173e-01 | 9.865e-01 |
| nequip | so3 | 1.441e-01 | 1.441e-01 | 1.565e-01 | 2.406e-01 | 7.762e-01 |

## BaTiO3  (432 (forbids rank-3))

| core | arm | δ=0 | δ=0.001 | δ=0.05 | δ=0.2 | δ=1 |
|---|---|---|---|---|---|---|
| allegro | o3 | 3.164e-07 | 2.932e-03 | 1.463e-01 | 5.681e-01 | 1.349e+00 |
| allegro | so3 | 4.772e-07 | 2.791e-03 | 1.394e-01 | 5.461e-01 | 1.825e+00 |
| equiformer_v2 | so3 | 2.269e-03 | 3.096e-03 | 1.000e-01 | 4.090e-01 | 1.354e+00 |
| mace | o3 | 1.189e-07 | 3.554e-03 | 1.774e-01 | 6.993e-01 | 2.596e+00 |
| mace | so3 | 1.210e-07 | 3.607e-03 | 1.800e-01 | 7.053e-01 | 2.405e+00 |
| nequip | o3 | 1.777e-07 | 4.141e-03 | 2.067e-01 | 8.083e-01 | 2.151e+00 |
| nequip | so3 | 4.963e-07 | 5.312e-03 | 2.653e-01 | 1.044e+00 | 3.394e+00 |

## PbTiO3  (432 (forbids rank-3))

| core | arm | δ=0 | δ=0.001 | δ=0.05 | δ=0.2 | δ=1 |
|---|---|---|---|---|---|---|
| allegro | o3 | 3.927e-07 | 2.049e-03 | 1.024e-01 | 4.066e-01 | 1.859e+00 |
| allegro | so3 | 4.238e-07 | 2.123e-03 | 1.061e-01 | 4.233e-01 | 2.045e+00 |
| equiformer_v2 | so3 | 2.740e-03 | 3.516e-03 | 8.429e-02 | 3.039e-01 | 1.291e+00 |
| mace | o3 | 1.130e-07 | 2.411e-03 | 1.206e-01 | 4.763e-01 | 1.836e+00 |
| mace | so3 | 6.615e-07 | 1.748e-03 | 8.745e-02 | 3.454e-01 | 1.380e+00 |
| nequip | o3 | 1.551e-07 | 2.227e-03 | 1.112e-01 | 4.403e-01 | 1.701e+00 |
| nequip | so3 | 2.433e-07 | 2.778e-03 | 1.388e-01 | 5.509e-01 | 2.290e+00 |

## Reading

**TiO₂ is the panel that carries the argument.** The O(3) arms start at 
machine zero and
rise smoothly as inversion breaks: the guarantee switches off exactly when the symmetry
does. The SO(3) arms start from an O(1) offset — a physically impossible response for a
centrosymmetric crystal — and that spurious floor dominates the true signal through the
small-δ regime, which is where displacive ferroelectrics actually live.

**The perovskite panels do not show this, and that is itself the finding.** Cubic Pm-3̄m
has
rotation subgroup 432, under which no rank-3 tensor is invariant, so the SO(3) arms are
forced to zero at δ = 0 without any parity label. Both arms therefore start 
at machine zero.
Had we run only BaTiO₃ — the material the experiment was originally designed 
around — we
would have concluded that SO(3) tracks the symmetry breaking correctly. It does not; a
cubic parent simply hides the defect (E7).

EquiformerV2's values are means over 5 seeded draws; its forward pass is stochastic (E5).
