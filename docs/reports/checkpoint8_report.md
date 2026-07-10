# Checkpoint 8 — post-headline strengthening package

This is the consolidated report for the package defined in `docs/new_additions.md`: a mandatory
validation gate (V0), three experiments (E1–E3), two rebuttal checks (E4–E5), figure assets (E6), a
rotation-subgroup analysis added during the work (E7), the prevalence audit (Task 0.4), and framing
plus corrections (F1–F2). Nothing here retrained a headline arm; the 84-run grid and
`results/stats.json` are byte-identical throughout.

Each section links to its generated evidence. Every number below was cross-checked against the source
JSON (validation sweep, commit `93b2ae6`).

## Headline, unchanged

On the 2,000 spglib-verified centrosymmetric crystals (idealized variant), where the piezoelectric
tensor is exactly zero by Neumann's principle:

| core | O(3) false-flag | O(3) median ‖e‖ | SO(3) false-flag | SO(3) median ‖e‖ |
|---|---|---|---|---|
| NequIP | 0.0000 | 3.3e-07 | 0.8953 | 0.738 |
| Allegro | 0.0000 | 4.0e-07 | 0.9095 | 1.028 |
| MACE | 0.0000 | 2.8e-06 | 0.9077 | 0.985 |
| EquiformerV2 | — (SO(3)-only) | — | 0.9570 | 0.489 |

Compute: 72.44 GPU-hours over the 84 headline runs. Full tables and threshold curves in
[`RESULTS.md`](../../RESULTS.md).

## V0 — claims validated before implementing

The full table is [`checkpoint8_claims.md`](checkpoint8_claims.md). Twenty physics claims are checked
by `tests/test_physics_claims.py` (20 passed). Ten held; **seven were wrong as written** and each
changed a downstream experiment. Two of those corrections prevented a defect in the paper:

- **E2 on BaTiO₃ alone would have been a false negative.** Cubic Pm-3̄m's rotation subgroup is 432,
  under which no rank-3 tensor is invariant, so SO(3) is forced to zero at δ=0 by rotation alone — we
  would have reported SO(3) tracking symmetry breaking correctly. Fixed by adding rutile TiO₂.
- **A `run_label` collision silently overwrote the headline metrics** when the E1 side-study synced,
  because `run_label` omits the dataset. Caught by the `stats.json` byte-identity guard; fixed with a
  dataset-qualified `run_key` and a regression test.

## E1 — augmentation does not buy the zero (off-cycle)

[`e1_augmentation.md`](../results/e1_augmentation.md) ·
[off-cycle report](checkpoint8_offcycle_e1.md)

12 runs (SO(3) only, 4 cores × 3 seeds) on the piezoelectric set augmented with 1,000 centrosymmetric
crystals labelled exactly zero, drawn from SGs `{2,12,14,15,62,225}`. Val/test inherited unchanged;
target scale frozen at 0.749134. The measured outcome is the third the design flagged — no drop even
on SEEN-SG — so it ships with an off-cycle report.

The decisive control: on the 1,000 zero-labelled crystals each model **trained on**, the false-flag
rate is 0.895–0.922. The zeros are fit *better* than the real tensors (train MAE 0.014–0.045 vs
0.086–0.128, mean target magnitude 0.145), and the model is not underfitting (train MAE is 0.4× test
MAE). Gradient descent pushes towards zero and cannot arrive. Held-out false-flag change baseline →
augmented: −0.000 to −0.021 on SEEN-SG, −0.013 to −0.032 on UNSEEN-SG. O(3), never retrained: 0.0000
on both. Honest counterweight: augmentation *does* improve non-centrosymmetric test MAE for every core
(e.g. EquiformerV2 0.2157 → 0.1774) — better regressors that still predict impossible values.

## E2 — symmetry-breaking curves

[`e2_symmetry_breaking.md`](../results/e2_symmetry_breaking.md)

`x(δ) = x_parent + δ·Δx` along a [001] polar mode. **Rutile TiO₂** (P4₂/mnm 136 → P4₂nm 102) is the
load-bearing panel: its rotation subgroup 422 permits a rank-3 invariant, so only parity forbids a
response at δ=0. There the arms separate — O(3) 1.9e-07–9.1e-07, SO(3) 0.092–0.282 (seed-averaged per
core), the SO(3) floor roughly equal to its own response at δ=0.05. The BaTiO₃/PbTiO₃ panels are
reported with the caveat that their cubic parent (432) forbids rank-3 by rotation alone, so both arms
start at machine zero and the panel cannot show the parity effect. All 33 frames per material verified
at symprec 1e-8.

## E3 — the parity guarantee is differentiable

[`e3_jacobian.md`](../results/e3_jacobian.md)

`J = dT/dr` by autograd at the centrosymmetric geometry of 20 crystals across 19 space groups, both
arms of the three e3nn cores, 3 seeds (360 Jacobians, each FD-checked to 6.6e-05–1.4e-04). Primary
statistic: even-subspace energy fraction `‖J·P_even‖/‖J‖` — O(3) 1.4e-07–9.7e-07, SO(3) 0.42–0.54,
five to six orders of magnitude with no overlap. Parity scores of the top singular vectors were
demoted to secondary: they only partially separate the arms (O(3) exactly −1.00000, but trained SO(3)
is approximately odd, median −0.76). EquiformerV2 is excluded, with cause — `so3.py` detaches the
Wigner-D matrices, so its autograd Jacobian is missing the angular path (FD disagrees by 45%).

## E4 — test-time inversion averaging

[`e4_inversion_averaging.md`](../results/e4_inversion_averaging.md)

`T_sym = [T(x) − T(I·x)]/2`. Vacuous on both OOD variants for an exactly equivariant model
(`T(I·x)=T(x)` to 1e-6 by permutation invariance), correcting the design's expectation that the raw
variant would retain signal. SO(3) false-flag collapses ~0.90 → ~0.000 — for a reason carrying no
parity information. EquiformerV2 is the exception (0.95 → 0.82), because it violates the identity the
fix relies on. The fix also presupposes knowing the target's parity in advance — exactly what O(3)
features encode.

## E5 — output-level equivariance audit

[`e5_output_parity.md`](../results/e5_output_parity.md)

All 21 piezo models. O(3) satisfies the mirror law to ~6e-7; SO(3) violates it by O(1). Every e3nn
core satisfies the rotation law in both arms (~1e-6) and is deterministic. EquiformerV2 fails both:
rotation error 7e-2–1.1e-1, and it is the only nondeterministic model (a random per-edge frame each
forward) — the deferred EquiformerV2 positive control, and a direct measure of its equivariance error.

## E6 — named materials

[`e6_named_materials.md`](../results/e6_named_materials.md)

Ten familiar centrosymmetric compounds, each annotated with its point group. The headline case is
**corundum Al₂O₃ (sapphire)**, non-cubic, where every SO(3) model predicts a substantial response
(NequIP 1.13, Allegro 0.72, MACE 0.84, EquiformerV2 0.21) against an exact zero. The cubic m-3̄m
compounds (NaCl, diamond, Si, MgO, CaF₂, CsCl, SrTiO₃) are near-zero for the exact-SO(3) cores too —
reported rather than dropped, because that is the rotation-subgroup effect (E7), not a counterexample.

## E7 — the rotation subgroup explains SO(3)'s correct zeros (supplementary)

[`e7_rotation_subgroup.md`](../results/e7_rotation_subgroup.md)

For the three e3nn cores the m-3̄m false-flag rate is **0.000** in every seed (all 166 m-3̄m crystals,
rotation subgroup 432 forbids rank-3), while m-3̄ (subgroup 23, permits rank-3) is false-flagged
0.889–1.000. 1,834 of 2,000 crystals (91.7%) lie outside what rotation can enforce — close to the
observed SO(3) false-flag rate. SO(3)'s zeros are the ones rotations already guarantee; O(3)'s are the
strictly larger set that parity guarantees. Scoped as supplementary; the headline framing of the 9%
is unchanged.

## Task 0.4 — prevalence audit (Table 1)

[`checkpoint8_prevalence_audit.md`](checkpoint8_prevalence_audit.md)

14 models classified by reading released source at a pinned version/commit, with the deciding line
recorded: 5 parity-aware, 3 SO(3)-only, 3 vector-only, 3 invariant, 1 undetermined (GotenNet, left
open honestly). Equiformer v1 as released is SO(3)-only; NequIP's own docstring shows `parity=False`
does not strip parity labels — corroborating the Checkpoint-1 finding.

## F1 / F2 — framing and corrections

[`f1_related_work.md`](../f1_related_work.md) · [`f2_per_atom.md`](../results/f2_per_atom.md)

F1 positions the work against the symmetry-breaking literature, with every characterisation checked
against the cited abstract. Key distinction: Smidt et al. already invoke Curie's principle, so ours is
the complement — upholding it for tensor properties *requires* improper-operation equivariance, which
SO(3) lacks. The dielectric-tensor readout-on-embeddings paper is rank-2 parity-even, so it is *not*
an instance of the failure; cited only as evidence the pattern is standard practice.

F2 corrections applied to `RESULTS.md`: U₀ reworded to "no consistent direction across cores" (not a
tested null; verified — deltas −1.99/−4.10/+9.33 eV); EquiformerV2's stochastic evaluation disclosed;
per-atom variant of the headline metric confirms the conclusion is unchanged (O(3) stays 0.0000, SO(3)
moves ≤0.0177); Table 1 linked.

## Status

| item | state |
|---|---|
| V0 gate | 20/20 tests pass |
| E1–E7, Task 0.4, F1, F2 | complete, cross-checked |
| Headline `results/stats.json` | byte-identical |
| E1 GPU instances | verified local, destroyed |
| Remaining | fold E1–E7 into `RESULTS.md`/`METHODS.md` narrative (writing phase) |
