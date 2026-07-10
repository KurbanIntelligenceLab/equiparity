# E4 — test-time inversion averaging

`T_sym(x) = [T(x) - T(I·x)] / 2`, evaluated on all 2,000 centrosymmetric crystals.

## Symmetrisation is vacuous for an exactly equivariant model — on *both* variants

If `x` is exactly centrosymmetric then `I·x` is the same periodic crystal up to a
permutation of atoms and a translation. Any permutation- and translation-invariant model
therefore returns `T(I·x) = T(x)`, and `T_sym ≡ 0` identically — regardless of parity.

The measurement confirms this and **corrects an expectation we held going in**: we 
assumed
the raw (DFT-relaxed) variant would retain a residual signal because its coordinates 
satisfy
inversion only to within a tolerance. It does not. For all three e3nn cores the SO(3)
false-flag rate collapses from ~0.90 to ~0.000 on the raw variant as well. Symmetrisation
does remove the false flags here; it simply does so for a reason that carries no 
information
about parity. **Do not present either zero as evidence that symmetrisation works.**

## The exception: symmetrisation fails for EquiformerV2

EquiformerV2's false-flag rate only falls from ~0.95 to ~0.82, and its `|T_sym|` stays at
~2.8e-02. It violates the very identity the fix relies on: `|T(I·x) − T(x)| / |T|` is
0.10–0.15 rather than ~1e-6, because it is only approximately rotation-equivariant (see 
E5).
The one deployed SO(3) model in this study is the one the proposed fix does not repair.

## Tables

| run | core | parity | variant | median &#124;T&#124; | median &#124;T_sym&#124; | false-flag | false-flag (sym) |
|---|---|---|---|---|---|---|---|
| allegro_o3_piezoelectric_seed0 | allegro | o3 | idealized | 5.221e-07 | 3.714e-07 | 0.0000 | 0.0000 |
| allegro_o3_piezoelectric_seed0 | allegro | o3 | raw | 5.870e-07 | 4.138e-07 | 0.0000 | 0.0000 |
| allegro_o3_piezoelectric_seed1 | allegro | o3 | idealized | 5.765e-07 | 4.007e-07 | 0.0000 | 0.0000 |
| allegro_o3_piezoelectric_seed1 | allegro | o3 | raw | 6.168e-07 | 4.416e-07 | 0.0000 | 0.0000 |
| allegro_o3_piezoelectric_seed2 | allegro | o3 | idealized | 5.390e-07 | 3.826e-07 | 0.0000 | 0.0000 |
| allegro_o3_piezoelectric_seed2 | allegro | o3 | raw | 5.935e-07 | 4.195e-07 | 0.0000 | 0.0000 |
| allegro_so3_piezoelectric_seed0 | allegro | so3 | idealized | 8.559e-01 | 5.677e-07 | 0.9095 | 0.0000 |
| allegro_so3_piezoelectric_seed0 | allegro | so3 | raw | 8.585e-01 | 6.383e-07 | 0.9095 | 0.0000 |
| allegro_so3_piezoelectric_seed1 | allegro | so3 | idealized | 1.103e+00 | 7.439e-07 | 0.9110 | 0.0000 |
| allegro_so3_piezoelectric_seed1 | allegro | so3 | raw | 1.122e+00 | 8.425e-07 | 0.9110 | 0.0005 |
| allegro_so3_piezoelectric_seed2 | allegro | so3 | idealized | 9.200e-01 | 6.298e-07 | 0.9080 | 0.0000 |
| allegro_so3_piezoelectric_seed2 | allegro | so3 | raw | 9.287e-01 | 6.938e-07 | 0.9080 | 0.0000 |
| equiformer_v2_so3_piezoelectric_seed0 | equiformer_v2 | so3 | idealized | 5.398e-01 | 2.775e-02 | 0.9525 | 0.8220 |
| equiformer_v2_so3_piezoelectric_seed0 | equiformer_v2 | so3 | raw | 5.485e-01 | 2.760e-02 | 0.9530 | 0.8265 |
| equiformer_v2_so3_piezoelectric_seed1 | equiformer_v2 | so3 | idealized | 4.306e-01 | 2.744e-02 | 0.9600 | 0.8320 |
| equiformer_v2_so3_piezoelectric_seed1 | equiformer_v2 | so3 | raw | 4.317e-01 | 2.728e-02 | 0.9605 | 0.8365 |
| equiformer_v2_so3_piezoelectric_seed2 | equiformer_v2 | so3 | idealized | 3.639e-01 | 2.933e-02 | 0.9525 | 0.8125 |
| equiformer_v2_so3_piezoelectric_seed2 | equiformer_v2 | so3 | raw | 3.662e-01 | 2.905e-02 | 0.9520 | 0.8100 |
| mace_o3_piezoelectric_seed0 | mace | o3 | idealized | 2.846e-06 | 2.853e-06 | 0.0000 | 0.0000 |
| mace_o3_piezoelectric_seed0 | mace | o3 | raw | 3.271e-06 | 3.292e-06 | 0.0005 | 0.0005 |
| mace_o3_piezoelectric_seed1 | mace | o3 | idealized | 2.481e-06 | 2.483e-06 | 0.0000 | 0.0000 |
| mace_o3_piezoelectric_seed1 | mace | o3 | raw | 2.844e-06 | 2.834e-06 | 0.0000 | 0.0000 |
| mace_o3_piezoelectric_seed2 | mace | o3 | idealized | 2.927e-06 | 2.929e-06 | 0.0000 | 0.0000 |
| mace_o3_piezoelectric_seed2 | mace | o3 | raw | 3.408e-06 | 3.384e-06 | 0.0005 | 0.0005 |
| mace_so3_piezoelectric_seed0 | mace | so3 | idealized | 9.813e-01 | 2.446e-06 | 0.9085 | 0.0000 |
| mace_so3_piezoelectric_seed0 | mace | so3 | raw | 9.887e-01 | 2.842e-06 | 0.9085 | 0.0005 |
| mace_so3_piezoelectric_seed1 | mace | so3 | idealized | 9.210e-01 | 2.430e-06 | 0.9070 | 0.0000 |
| mace_so3_piezoelectric_seed1 | mace | so3 | raw | 9.233e-01 | 2.823e-06 | 0.9070 | 0.0000 |
| mace_so3_piezoelectric_seed2 | mace | so3 | idealized | 8.724e-01 | 2.324e-06 | 0.9075 | 0.0000 |
| mace_so3_piezoelectric_seed2 | mace | so3 | raw | 8.784e-01 | 2.633e-06 | 0.9075 | 0.0000 |
| nequip_o3_piezoelectric_seed0 | nequip | o3 | idealized | 3.673e-07 | 2.561e-07 | 0.0000 | 0.0000 |
| nequip_o3_piezoelectric_seed0 | nequip | o3 | raw | 3.917e-07 | 2.773e-07 | 0.0005 | 0.0005 |
| nequip_o3_piezoelectric_seed1 | nequip | o3 | idealized | 2.948e-07 | 2.030e-07 | 0.0000 | 0.0000 |
| nequip_o3_piezoelectric_seed1 | nequip | o3 | raw | 3.251e-07 | 2.294e-07 | 0.0000 | 0.0000 |
| nequip_o3_piezoelectric_seed2 | nequip | o3 | idealized | 2.625e-07 | 1.852e-07 | 0.0000 | 0.0000 |
| nequip_o3_piezoelectric_seed2 | nequip | o3 | raw | 2.894e-07 | 2.067e-07 | 0.0000 | 0.0000 |
| nequip_so3_piezoelectric_seed0 | nequip | so3 | idealized | 6.929e-01 | 3.280e-07 | 0.8960 | 0.0000 |
| nequip_so3_piezoelectric_seed0 | nequip | so3 | raw | 6.971e-01 | 3.707e-07 | 0.8960 | 0.0005 |
| nequip_so3_piezoelectric_seed1 | nequip | so3 | idealized | 7.021e-01 | 3.199e-07 | 0.8955 | 0.0000 |
| nequip_so3_piezoelectric_seed1 | nequip | so3 | raw | 7.082e-01 | 3.560e-07 | 0.8955 | 0.0000 |
| nequip_so3_piezoelectric_seed2 | nequip | so3 | idealized | 6.042e-01 | 2.616e-07 | 0.8945 | 0.0000 |
| nequip_so3_piezoelectric_seed2 | nequip | so3 | raw | 6.090e-01 | 2.921e-07 | 0.8945 | 0.0000 |

## Which identity does `T(I·x)` obey?

Median relative residual of the two candidate laws, on the centrosymmetric set.
Permutation- and translation-invariance force the first on exactly centrosymmetric input.

The O(3) rows are **uninformative here** (both residuals ≈ 1.4): their `T` is machine 
noise,
so no identity about its direction can be measured. The O(3) parity law is tested in the
control table below, on non-centrosymmetric crystals where `T` is a real signal.

| run | parity | variant | &#124;T(I·x) − T(x)T&#124;/&#124;T&#124; | &#124;T(I·x) + T(x)T&#124;/&#124;T&#124; |
|---|---|---|---|---|
| allegro_o3_piezoelectric_seed0 | o3 | idealized | 1.401e+00 | 1.430e+00 |
| allegro_o3_piezoelectric_seed0 | o3 | raw | 1.442e+00 | 1.363e+00 |
| allegro_o3_piezoelectric_seed1 | o3 | idealized | 1.414e+00 | 1.415e+00 |
| allegro_o3_piezoelectric_seed1 | o3 | raw | 1.470e+00 | 1.382e+00 |
| allegro_o3_piezoelectric_seed2 | o3 | idealized | 1.414e+00 | 1.428e+00 |
| allegro_o3_piezoelectric_seed2 | o3 | raw | 1.446e+00 | 1.357e+00 |
| allegro_so3_piezoelectric_seed0 | so3 | idealized | 1.359e-06 | 2.000e+00 |
| allegro_so3_piezoelectric_seed0 | so3 | raw | 1.440e-06 | 2.000e+00 |
| allegro_so3_piezoelectric_seed1 | so3 | idealized | 1.286e-06 | 2.000e+00 |
| allegro_so3_piezoelectric_seed1 | so3 | raw | 1.405e-06 | 2.000e+00 |
| allegro_so3_piezoelectric_seed2 | so3 | idealized | 1.352e-06 | 2.000e+00 |
| allegro_so3_piezoelectric_seed2 | so3 | raw | 1.482e-06 | 2.000e+00 |
| equiformer_v2_so3_piezoelectric_seed0 | so3 | idealized | 9.881e-02 | 1.997e+00 |
| equiformer_v2_so3_piezoelectric_seed0 | so3 | raw | 9.764e-02 | 1.996e+00 |
| equiformer_v2_so3_piezoelectric_seed1 | so3 | idealized | 1.091e-01 | 1.995e+00 |
| equiformer_v2_so3_piezoelectric_seed1 | so3 | raw | 1.114e-01 | 1.995e+00 |
| equiformer_v2_so3_piezoelectric_seed2 | so3 | idealized | 1.490e-01 | 1.992e+00 |
| equiformer_v2_so3_piezoelectric_seed2 | so3 | raw | 1.484e-01 | 1.993e+00 |
| mace_o3_piezoelectric_seed0 | o3 | idealized | 1.995e+00 | 1.426e-01 |
| mace_o3_piezoelectric_seed0 | o3 | raw | 1.999e+00 | 1.299e-01 |
| mace_o3_piezoelectric_seed1 | o3 | idealized | 1.993e+00 | 1.383e-01 |
| mace_o3_piezoelectric_seed1 | o3 | raw | 1.999e+00 | 1.224e-01 |
| mace_o3_piezoelectric_seed2 | o3 | idealized | 1.993e+00 | 1.471e-01 |
| mace_o3_piezoelectric_seed2 | o3 | raw | 1.996e+00 | 1.281e-01 |
| mace_so3_piezoelectric_seed0 | so3 | idealized | 5.031e-06 | 2.000e+00 |
| mace_so3_piezoelectric_seed0 | so3 | raw | 6.040e-06 | 2.000e+00 |
| mace_so3_piezoelectric_seed1 | so3 | idealized | 5.230e-06 | 2.000e+00 |
| mace_so3_piezoelectric_seed1 | so3 | raw | 6.249e-06 | 2.000e+00 |
| mace_so3_piezoelectric_seed2 | so3 | idealized | 5.285e-06 | 2.000e+00 |
| mace_so3_piezoelectric_seed2 | so3 | raw | 6.380e-06 | 2.000e+00 |
| nequip_o3_piezoelectric_seed0 | o3 | idealized | 1.415e+00 | 1.414e+00 |
| nequip_o3_piezoelectric_seed0 | o3 | raw | 1.466e+00 | 1.359e+00 |
| nequip_o3_piezoelectric_seed1 | o3 | idealized | 1.395e+00 | 1.409e+00 |
| nequip_o3_piezoelectric_seed1 | o3 | raw | 1.452e+00 | 1.363e+00 |
| nequip_o3_piezoelectric_seed2 | o3 | idealized | 1.424e+00 | 1.420e+00 |
| nequip_o3_piezoelectric_seed2 | o3 | raw | 1.474e+00 | 1.358e+00 |
| nequip_so3_piezoelectric_seed0 | so3 | idealized | 8.670e-07 | 2.000e+00 |
| nequip_so3_piezoelectric_seed0 | so3 | raw | 9.657e-07 | 2.000e+00 |
| nequip_so3_piezoelectric_seed1 | so3 | idealized | 8.520e-07 | 2.000e+00 |
| nequip_so3_piezoelectric_seed1 | so3 | raw | 9.344e-07 | 2.000e+00 |
| nequip_so3_piezoelectric_seed2 | so3 | idealized | 8.117e-07 | 2.000e+00 |
| nequip_so3_piezoelectric_seed2 | so3 | raw | 8.915e-07 | 2.000e+00 |

## Control: the O(3) parity law on non-centrosymmetric crystals

Here `I·x` is a genuinely different crystal and `T(x)` a genuine nonzero prediction, so 
this
is a statement about the *model*, not the structure. An O(3) model must satisfy
`T(I·x) = -T(x)` (second column ≈ 1e-6): inversion is in its symmetry group. An SO(3) 
model
satisfies **neither** law (both columns O(1)) — inversion is simply not a symmetry it was
built to respect, so nothing constrains `T(I·x)` at all. This is the correctness check 
that
makes the `T_sym = T` identity meaningful for the O(3) arms.

| run | parity | n | &#124;T(I·x) − T(x)T&#124;/&#124;T&#124; | &#124;T(I·x) + T(x)T&#124;/&#124;T&#124; |
|---|---|---|---|---|
| allegro_o3_piezoelectric_seed0 | o3 | 25 | 2.000e+00 | 7.788e-07 |
| allegro_o3_piezoelectric_seed1 | o3 | 25 | 2.000e+00 | 7.237e-07 |
| allegro_o3_piezoelectric_seed2 | o3 | 25 | 2.000e+00 | 9.624e-07 |
| allegro_so3_piezoelectric_seed0 | so3 | 25 | 1.788e+00 | 1.166e+00 |
| allegro_so3_piezoelectric_seed1 | so3 | 25 | 1.799e+00 | 1.130e+00 |
| allegro_so3_piezoelectric_seed2 | so3 | 25 | 1.867e+00 | 1.113e+00 |
| equiformer_v2_so3_piezoelectric_seed0 | so3 | 25 | 1.759e+00 | 9.762e-01 |
| equiformer_v2_so3_piezoelectric_seed1 | so3 | 25 | 1.888e+00 | 8.812e-01 |
| equiformer_v2_so3_piezoelectric_seed2 | so3 | 25 | 1.959e+00 | 7.249e-01 |
| mace_o3_piezoelectric_seed0 | o3 | 25 | 2.000e+00 | 3.304e-07 |
| mace_o3_piezoelectric_seed1 | o3 | 25 | 2.000e+00 | 3.983e-07 |
| mace_o3_piezoelectric_seed2 | o3 | 25 | 2.000e+00 | 4.158e-07 |
| mace_so3_piezoelectric_seed0 | so3 | 25 | 1.892e+00 | 9.546e-01 |
| mace_so3_piezoelectric_seed1 | so3 | 25 | 1.877e+00 | 9.690e-01 |
| mace_so3_piezoelectric_seed2 | so3 | 25 | 1.899e+00 | 9.325e-01 |
| nequip_o3_piezoelectric_seed0 | o3 | 25 | 2.000e+00 | 1.091e-06 |
| nequip_o3_piezoelectric_seed1 | o3 | 25 | 2.000e+00 | 6.060e-07 |
| nequip_o3_piezoelectric_seed2 | o3 | 25 | 2.000e+00 | 4.771e-07 |
| nequip_so3_piezoelectric_seed0 | so3 | 25 | 1.842e+00 | 5.206e-01 |
| nequip_so3_piezoelectric_seed1 | so3 | 25 | 1.834e+00 | 1.299e+00 |
| nequip_so3_piezoelectric_seed2 | so3 | 25 | 1.853e+00 | 1.033e+00 |

## What the fix costs

Symmetrisation (a) doubles inference cost, (b) presupposes knowing the target's parity in
advance — exactly the knowledge an O(3) model's features already encode — and (c) 
repairs one
output while leaving the model's internal representations parity-blind for every other
quantity derived from them.
