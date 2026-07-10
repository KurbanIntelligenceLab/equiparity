# Checkpoint 8 — claims validated

V0 required that every physics claim this package relies on be **verified by a test in the
repository, not asserted**. This table lists each claim, the test that checks it, and the result.
Where validation contradicted the claim as written, the correction is recorded and the downstream
experiment was changed.

All tests: `uv run --extra nequip --extra data python -m pytest tests/test_physics_claims.py`
(20 passed).

## Claims that held

| # | Claim | Test | Result |
|---|---|---|---|
| 1 | The piezoelectric tensor decomposes into parity-**odd** irreps only (`2x1o+1x2o+1x3o`) | `test_piezoelectric_irreps_are_all_parity_odd` | every irrep has `p = -1` |
| 2 | The SO(3) arm differs from O(3) only by relabelling odd irreps even | `test_so3_output_head_relabels_odd_irreps_even` | `2x1o+1x2o+1x3o` → `2x1e+1x2e+1x3e` |
| 3 | No rank-3 tensor is invariant under rotation subgroup **432** (point group m-3̄m) | `test_rotation_subgroup_432_forbids_any_rank_three_tensor` | Reynolds projector is the zero map (`max|P| < 1e-5`), rank 0, group order 24 |
| 4 | Rotation subgroup **23** (point group m-3̄) *does* permit a rank-3 invariant | `test_rotation_subgroup_23_permits_a_rank_three_tensor` | rank 1, group order 12 |
| 5 | Cubic BaTiO₃/PbTiO₃ are centrosymmetric (221) at every tolerance | `test_cubic_perovskite_is_centrosymmetric_at_every_tolerance` | 221 at symprec 1e-3, 1e-5, 1e-8 |
| 6 | For an O(3) model, `J∘P = −J` at a centrosymmetric point; all active singular vectors score −1 | `test_o3_jacobian_is_purely_inversion_odd` | identity to 1e-9; top-5 scores `−1.0000 ± 1e-9`; rank ≤ odd-subspace dim |
| 7 | The even-subspace energy fraction separates the arms | `test_even_subspace_energy_fraction_separates_the_arms` | O(3) < 1e-9, SO(3) > 1e-2 |
| 8 | Inversion averaging is identically zero on an exactly centrosymmetric input, for **both** arms | `test_inversion_averaging_is_trivially_zero_on_an_exactly_centrosymmetric_input` | `|T_sym| < 1e-10` for O(3) and SO(3) alike |
| 9 | An O(3) toy outputs machine zero on a centrosymmetric cloud but is non-degenerate | `test_o3_toy_output_vanishes_at_centrosymmetric_configuration` | `|T(centro)| < 1e-12`, `|T(random)| > 1e-3` |
| 10 | An SO(3) toy outputs nonzero on the same cloud | `test_so3_toy_output_is_nonzero_at_centrosymmetric_configuration` | `|T(centro)| > 1e-3` |

## Claims that were WRONG as written, and what changed

| # | Claim as written | What we measured | Consequence |
|---|---|---|---|
| A | *"The [001] distortion registers as P4mm (99) for all δ > 0"* (V0.2) | False at symprec 1e-3: frames below **δ ≈ 0.006** are reported as the centrosymmetric parent, because the max displacement (0.12·δ Å) falls under the tolerance. Correct at symprec 1e-8. | E2 verifies every frame at **symprec 1e-8**. Regression-guarded by `test_spglib_symprec_is_a_distance_tolerance_not_a_symmetry_test`. Same phenomenon as the mp-1227949 artifact (A5). |
| B | *"E4's idealized column is trivially zero; the raw variant carries the content"* | The raw variant is **also** trivially zero (`T(I·x) = T(x)` to 1e-6 by permutation invariance). SO(3) false-flag collapses ~0.90 → ~0.000 on **both** variants. | E4's argument is restated: symmetrisation does remove the flags, for a reason carrying no parity information. EquiformerV2 is the exception (0.95 → 0.82) because it violates the identity the fix relies on. |
| C | *"E3: SO(3) parity scores are broadly distributed"* | False for *trained* models. A trained SO(3) model is approximately odd: median top-5 score **−0.76**, with 17% of leading vectors within 1e-2 of −1. (The mixed scores in the toy are a property of **random** weights.) | E3's primary statistic became the **even-subspace energy fraction** `‖J·P_even‖/‖J‖`: O(3) 1.4e-07, SO(3) 0.42–0.54 — five to six orders of magnitude, no overlap. Scores demoted to a secondary check. |
| D | *"E2 on BaTiO₃ shows O(3) starting at zero and SO(3) at an O(1) offset"* | Both arms start at machine zero on BaTiO₃. Cubic Pm-3̄m's rotation subgroup is **432**, which forbids a rank-3 tensor outright (claim 3), so SO(3) is forced to zero there by rotation alone — no parity involved. | E2 gained **rutile TiO₂** (P4₂/mnm 136 → P4₂nm 102), rotation subgroup **422**, which permits a rank-3 invariant. There the arms separate: at δ = 0 the O(3) arms give 1.9e-07 – 1.4e-06 (machine zero) and the SO(3) arms give 0.092 – 0.282 (seed-averaged per core). The perovskite panels are reported with the caveat stated. |
| E | *"E6: shortlist ~8 familiar centrosymmetric materials from the OOD set"* | Most textbook compounds are absent from the OOD 2,000; the two present (diamond, KCl) are **not** false-flagged. | E6 became two panels, and every row is annotated with its point group. The m-3̄m entries are near-zero for the exact-SO(3) cores too (claim 3 again). The rows that carry the argument are the non-cubic ones — Al₂O₃, TiO₂. |
| F | *"EquiformerV2's Jacobian can be recovered by patching `edge_distance_vec`"* | `so3.py:410-411` detaches the Wigner-D matrices, which depend on positions. Autograd yields only the radial part: **FD disagrees by 45%**, versus 9e-4 for NequIP under the identical check. Its forward is also nondeterministic in `eval()` (`edge_rot_mat.py:15` redraws a random per-edge frame). | **EquiformerV2 excluded from E3**, with the FD table as the documented reason. Every EquiformerV2 number elsewhere is a mean over 5 seeded draws. The planned "patched vs unpatched agree bit-for-bit" check was **retired as vacuous**: `shift` absorbs the sign at `pos₀`, so both edge orientations agree to ~4e-06 and only the gradient distinguishes them. |
| G | *"run_label is unique per (core, parity, target, seed), so there are no cross-box collisions"* (`flatten_results.py` docstring) | False once two datasets share a target. The E1 augmented runs produced identical labels and **overwrote the headline metrics** on sync; `results/stats.json` moved. | Caught by the byte-identity guard before anything was committed. `run_key` (dataset-qualified) added; flatten, `analyze_results`, and `find_piezo_runs` all filter on dataset. Regression test: `test_run_label_collides_across_datasets_but_run_key_does_not`. |

## Standing-rule check

None of the corrections above is a physics failure — each is a protocol, tooling, or expectation
error found by executing the claim instead of trusting it. Correction **D** is the one that would
have produced a *false negative in the paper*: running E2 only on BaTiO₃, as designed, would have
shown SO(3) tracking the symmetry breaking correctly and we would have reported that. Correction
**G** would have silently corrupted the headline table.

## Verification artefacts

| Claim | Generated evidence |
|---|---|
| 3, 4 | `results/e7_rotation_subgroup.json`, `docs/results/e7_rotation_subgroup.md` |
| 6, 7, C | `results/e3_jacobian.json`, `docs/results/e3_jacobian.md` |
| 8, B | `results/e4_inversion_averaging.json`, `docs/results/e4_inversion_averaging.md` |
| A, D | `results/e2_symmetry_breaking.{json,csv}`, `docs/results/e2_symmetry_breaking.md` |
| E | `results/e6_named_materials.json`, `docs/results/e6_named_materials.md` |
| F | `results/e5_output_parity.json`, `docs/results/e5_output_parity.md` |
| G | `tests/test_physics_claims.py`, `results/stats.json` (byte-identical) |
