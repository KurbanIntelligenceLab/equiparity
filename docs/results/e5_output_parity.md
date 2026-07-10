# E5 — output-level equivariance audit

Median relative error over 25 **non-centrosymmetric** structures. The mirror
law is `T(Mx) = D(M) T(x)`, with `D` built on the physical irreps `2x1o+1x2o+1x3o`.

The laws are measured on non-centrosymmetric inputs because they are *relative* errors: 
an
O(3) model predicts machine zero on centrosymmetric inputs (last column, 25
structures), so a ratio there would divide float noise by ~1e-7 and report a spurious 
O(1)
violation. The zero itself is reported separately.

| run | core | parity | mirror rel. err | rotation rel. err | determinism spread | median &#124;T&#124; on centro |
|---|---|---|---|---|---|---|
| allegro_o3_piezoelectric_seed0 | allegro | o3 | 8.275e-07 | 1.211e-06 | 2.147e-06 | 7.216e-07 |
| allegro_o3_piezoelectric_seed1 | allegro | o3 | 9.157e-07 | 1.160e-06 | 7.307e-06 | 7.767e-07 |
| allegro_o3_piezoelectric_seed2 | allegro | o3 | 8.178e-07 | 1.172e-06 | 2.418e-06 | 7.055e-07 |
| allegro_so3_piezoelectric_seed0 | allegro | so3 | 1.166e+00 | 1.289e-06 | 4.560e-06 | 9.404e-01 |
| allegro_so3_piezoelectric_seed1 | allegro | so3 | 1.130e+00 | 1.087e-06 | 1.097e-05 | 1.629e+00 |
| allegro_so3_piezoelectric_seed2 | allegro | so3 | 1.113e+00 | 1.337e-06 | 8.821e-06 | 1.052e+00 |
| equiformer_v2_so3_piezoelectric_seed0 | equiformer_v2 | so3 | 9.821e-01 | 7.274e-02 | 1.106e-01 | 7.136e-01 |
| equiformer_v2_so3_piezoelectric_seed1 | equiformer_v2 | so3 | 9.286e-01 | 9.373e-02 | 1.432e-01 | 4.842e-01 |
| equiformer_v2_so3_piezoelectric_seed2 | equiformer_v2 | so3 | 7.127e-01 | 1.064e-01 | 1.438e-01 | 3.534e-01 |
| mace_o3_piezoelectric_seed0 | mace | o3 | 2.953e-07 | 2.646e-06 | 1.024e-06 | 3.591e-06 |
| mace_o3_piezoelectric_seed1 | mace | o3 | 3.686e-07 | 4.172e-06 | 1.429e-06 | 3.477e-06 |
| mace_o3_piezoelectric_seed2 | mace | o3 | 3.760e-07 | 3.429e-06 | 8.368e-07 | 3.419e-06 |
| mace_so3_piezoelectric_seed0 | mace | so3 | 9.546e-01 | 3.030e-06 | 9.537e-07 | 1.379e+00 |
| mace_so3_piezoelectric_seed1 | mace | so3 | 9.690e-01 | 3.197e-06 | 7.302e-07 | 8.048e-01 |
| mace_so3_piezoelectric_seed2 | mace | so3 | 9.325e-01 | 3.892e-06 | 1.192e-06 | 1.442e+00 |
| nequip_o3_piezoelectric_seed0 | nequip | o3 | 5.252e-07 | 1.061e-06 | 1.088e-06 | 3.681e-07 |
| nequip_o3_piezoelectric_seed1 | nequip | o3 | 5.824e-07 | 1.002e-06 | 1.139e-06 | 3.754e-07 |
| nequip_o3_piezoelectric_seed2 | nequip | o3 | 6.039e-07 | 8.837e-07 | 8.763e-07 | 2.610e-07 |
| nequip_so3_piezoelectric_seed0 | nequip | so3 | 5.206e-01 | 9.673e-07 | 1.585e-06 | 9.201e-01 |
| nequip_so3_piezoelectric_seed1 | nequip | so3 | 1.299e+00 | 9.066e-07 | 1.027e-06 | 1.019e+00 |
| nequip_so3_piezoelectric_seed2 | nequip | so3 | 1.033e+00 | 8.569e-07 | 9.377e-07 | 6.791e-01 |

**Reading.**

- **Mirror.** O(3) arms satisfy the law to float precision (~6e-7). SO(3) arms 
violate it 
by
  O(1) (0.52 – 1.30): their head cannot represent the sign flip an improper operation
  demands. This is the parity defect, measured directly at the output.
- **Rotation.** Both arms of every e3nn core satisfy it (~1e-6). Parity labels are
  irrelevant to rotations, as they must be.
- **EquiformerV2** fails the rotation law too (8e-2 – 1.4e-1) and is the only
nondeterministic model. `models/equiformer_v2/edge_rot_mat.py` redraws a random per-edge
frame on every forward; an exactly SO(3)-equivariant network is frame-independent, so the
  determinism spread *is* its rotational equivariance error. Every EquiformerV2 number
  elsewhere in this study is therefore reported as a mean over seeded draws.
