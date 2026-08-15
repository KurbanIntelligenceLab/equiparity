# F4 — the non-e3nn O(3) control (HotPP), in numbers

The manuscript's principal stated limitation is that all three O(3) arms share the e3nn irrep implementation, so the parity finding cannot be fully separated from that implementation. A first attempt at a non-e3nn control, CliffordSTF (geometric algebra), was withdrawn — not because it disagreed with Theorem 1, but because its Cl(3,0) algebra reaches only grade 1, forcing an ill-conditioned cubic tensor product to reach a rank-3 output, which amplified the ~1e-6 residual coordinate asymmetry of real crystals by 3,000–25,000× into a false-flag fraction of 0.42 (`METHODS.md` §7). This closes the same gap with **HotPP** (arXiv:2402.15286, *Nature Communications* 15, 2024), a message-passing potential with zero e3nn dependency (grep-verified over the full package tree) whose native rank-3 features are built as a 3-fold outer product of the raw bond vector and reach the output through a **linear** channel-mixing readout (`TensorLinear = nn.Linear`), with no tensor product between input and output rank anywhere in the construction.

The prior scouting probe (`results/noneN3/probe_hotpp*.py`) established the mirror law and the structural zero on a **synthetic, non-periodic** cluster. This closes the gap to the manuscript's actual claim, which is about periodic crystals, by running the same tests on the same 9 idealized centrosymmetric crystals `scripts/f3_size_consistency.py` uses (`ase.spacegroup.crystal`, 6 space groups, spglib-verified centrosymmetric), fed to HotPP through periodic neighbour lists (`ase.neighborlist`, r_max = 5.0 Å, HotPP's native `offset = S @ cell` convention — no adapter needed).

**Route.** Random-init `MiaoNet` (HotPP's architecture), 2 layers, `max_out_way = 3`, `max_r_way = 3`, r_max = 5.0 Å, float64 throughout, seed 1, 2,489 parameters. Runs in a dedicated `hotpp-control` conda env (numpy/scipy/pyyaml/pytorch-cpu/ase/spglib/pytorch-lightning/tensorboard), importing nothing from `equiparity` — standalone in the same sense as `scripts/t2_backbone_probe.py`.

## The idealized crystals are exactly centrosymmetric, not ~1e-6

Unlike a relaxed real crystal, `ase.spacegroup.crystal` places every atom exactly on its space-group orbit. Measured directly via spglib's inversion operator, mapped through fractional coordinates:

| crystal | family | space group | max inversion residual (Å) |
|---|---|---|---|
| NaCl_rocksalt | m-3m | 225 (Fm-3m) | 5.42e-16 |
| MgO_rocksalt | m-3m | 225 (Fm-3m) | 0.0 |
| CsCl | m-3m | 221 (Pm-3m) | 0.0 |
| SrTiO3_perovskite | m-3m | 221 (Pm-3m) | 3.75e-16 |
| Si_diamond | m-3m | 227 (Fd-3m) | 1.28e-15 |
| CaF2_fluorite | m-3m | 225 (Fm-3m) | 0.0 |
| TiO2_rutile | non-cubic | 136 (P4₂/mnm) | 7.21e-16 |
| TiO2_anatase | non-cubic | 141 (I4₁/amd) | 6.06e-16 |
| Al2O3_corundum | non-cubic | 167 (R-3c) | 2.92e-15 |

Every crystal is centrosymmetric to float64 round-off (< 3e-15 Å), not the ~1e-6 Å scale of a real relaxed structure. As instructed, this script therefore also runs a controlled epsilon-displacement sweep (below) to emulate that real-crystal scale.

## The structural zero (Theorem 1) holds on all 9 periodic crystals

The sum-pooled rank-3 (way=3, odd-parity) feature — the system-level quantity Theorem 1 constrains — is measured at random init and compared to a scale-free reference: the same model's per-atom feature magnitude on a fixed, generic non-symmetric 6-atom cluster (Na, Cl, Ca, O, Ti, Al; reference scale = 0.2327).

| crystal | family | n atoms | sum-pooled ‖T‖ | ratio to reference scale |
|---|---|---|---|---|
| NaCl_rocksalt | m-3m | 8 | 2.96e-16 | 1.27e-15 |
| MgO_rocksalt | m-3m | 8 | 1.19e-15 | 5.11e-15 |
| CsCl | m-3m | 2 | 7.99e-17 | 3.43e-16 |
| SrTiO3_perovskite | m-3m | 5 | 1.20e-15 | 5.16e-15 |
| Si_diamond | m-3m | 8 | 3.46e-16 | 1.49e-15 |
| CaF2_fluorite | m-3m | 12 | 9.73e-16 | 4.18e-15 |
| TiO2_rutile | non-cubic | 6 | 1.17e-15 | 5.01e-15 |
| TiO2_anatase | non-cubic | 12 | 1.55e-15 | 6.68e-15 |
| Al2O3_corundum | non-cubic | 30 | 9.40e-15 | 4.04e-14 |

Every crystal — cubic and non-cubic alike — cancels to the float64 noise floor (max ratio to the reference scale: 4.04e-14, on the 30-atom Al2O3 corundum cell, where the summation of more terms accumulates more round-off). This is Theorem 1's prediction, on real periodic input, in a second, independent, non-e3nn implementation.

## The decisive test: does HotPP amplify residual coordinate asymmetry like CliffordSTF did?

Each crystal was displaced by a fixed random unit direction scaled by epsilon in {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3}, and the amplification factor was measured as

```
amplification = (violation / reference_scale) / (epsilon / nearest_neighbor_distance)
```

i.e. relative output violation divided by relative input coordinate displacement — the same normalized quantity CliffordSTF's 3,000–25,000× applies to.

| crystal | d_nn (Å) | amplification factor (constant across ε = 1e-8..1e-3) |
|---|---|---|
| NaCl_rocksalt | 2.820 | 4.04 |
| MgO_rocksalt | 2.106 | 12.20 |
| CsCl | 3.571 | 10.82 |
| SrTiO3_perovskite | 1.952 | 10.21 |
| CaF2_fluorite | 2.366 | 2.22 |
| TiO2_rutile | 1.946 | 12.15 |
| TiO2_anatase | 1.934 | 21.51 |
| Al2O3_corundum | 1.855 | 17.10 |

**HotPP's amplification factor is 2.2–21.5× — order-1, not the 3,000–25,000× CliffordSTF produced.** It is also constant to better than 0.02% relative deviation across five decades of epsilon (1e-8 to 1e-3; the largest measured deviation is 1.1e-4 relative, on TiO2_rutile at the largest epsilon, from higher-order terms starting to contribute), confirming the response is genuinely linear (order-1 in the perturbation), the textbook signature of a well-conditioned readout, in sharp contrast to CliffordSTF's cubic tensor product.

**Si_diamond is excluded from this statistic and reported separately.** Its violation scales as epsilon^2.00 (log-log fit, R² effectively 1), not epsilon^1 like every other crystal — a favourable symmetry cancellation specific to the diamond lattice's local site point group (-43m), which forces the leading-order (linear) term to vanish identically, not a conditioning artifact. Two of its six epsilon points (1e-8, 1e-7) fall below the noise floor (perturbed pooled-norm not yet 10× above the exact-crystal float64 noise floor) and are flagged accordingly.

## Mirror-law probe on periodic input

The same rotation/improper-operation probe from `results/noneN3/probe_hotpp.py` was re-run directly on periodic crystals (cell and positions co-rotated), rather than only on the free-cluster geometry:

| crystal | ‖T₃‖ (informative?) | rotation rel. error | improper rel. error |
|---|---|---|---|
| Si_diamond | 2.75e-01 | 7.29e-16 | 4.78e-16 |
| CaF2_fluorite | 2.37e-01 | 2.09e-15 | 1.61e-15 |
| TiO2_rutile | 1.14e+00 | 3.17e-16 | 2.68e-16 |
| TiO2_anatase | 1.92e+00 | 1.54e-16 | 2.46e-16 |
| Al2O3_corundum | 2.14e+00 | 2.72e-16 | 2.40e-16 |
| NaCl_rocksalt | 3.08e-16 (noise floor) | — | — |
| MgO_rocksalt | 1.24e-15 (noise floor) | — | — |
| CsCl | 7.12e-17 (noise floor) | — | — |
| SrTiO3_perovskite | 8.70e-16 (noise floor) | — | — |

On rock-salt NaCl/MgO, CsCl, and the cubic SrTiO3 perovskite, the **per-atom** way=3 feature is itself forced to machine zero by the local site point group (Oh at every ion in these four structures) — this is the same rotation-ceiling effect `scripts/t2_backbone_probe.py`/`scripts/f3_size_consistency.py` document for other cores, and a relative error computed against a machine-zero denominator divides noise by noise. These four are flagged uninformative rather than misreported. On the five crystals where the per-atom feature is well above the noise floor, both the rotation law and the improper-operation law pass to float64 machine precision (2–3e-15 worst case) — confirming the mirror law holds on periodic input, not just the free-cluster probe.

## Reading

HotPP satisfies, on real periodic centrosymmetric crystals, in a second implementation independent of e3nn: (1) the mirror law (rotation and improper-operation equivariance) to float64 precision; (2) the structural zero of Theorem 1, on all 9 crystals, at the float64 noise floor; and (3) an order-1 (2.2–21.5×) amplification of residual coordinate asymmetry, in sharp contrast to CliffordSTF's 3,000–25,000×. This is the scientifically decisive difference: CliffordSTF's failure was a conditioning failure of its readout (a cubic tensor product forced by an algebra that stops at grade 1), not evidence against Theorem 1, and HotPP demonstrates that a native rank-3 construction reached through a linear readout does not reproduce that failure.

This is a random-init structural control on 9 idealized crystals, not a trained model evaluated on the manuscript's 2,000-crystal Materials Project population, and should not be overclaimed as a full fourth toggleable arm of the main study. It directly answers the question the withdrawal of CliffordSTF left open: whether a non-e3nn O(3)-equivariant implementation *can* deliver Theorem 1's guarantee at real-crystal-scale coordinate noise without amplifying it into a false-flag artifact. The measured answer is yes.

**Provenance.** HotPP (github.com/yongwongxx/Hotpp, MIT license, arXiv:2402.15286, vendored at `scratch_hotpp/hotpp/`; no e3nn dependency, grep-verified), `hotpp-control` conda env, `scripts/f4_noneN3_control.py`, seed 1, float64, random init (no training — Theorem 1 holds at any parameter values), 9 idealized crystals from `scripts/f3_size_consistency.py:build_crystals`, full results in `results/f4_noneN3_control.json`.
