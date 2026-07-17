# Checkpoint 10 — final pre-submission experiment package (running report)

V0 discipline: every prediction in this report is declared BEFORE its measurement; every claim
that enters the paper is backed by a `results/*.json` artifact with source attribution.

## Tier 0 (complete — see RED_FLAG_LIST.txt resolution log)

- T0-1 contamination: overlap 0/0/0; the 16 zero train rows are non-centrosymmetric (12x class
  432) — the declared prediction ("432-class or similar rotation-forbidden") CONFIRMED.
  `results/t0_contamination.json`.
- T0-2 epoch evidence: premise correction (nothing per-epoch was logged); two-point checkpoint
  evidence shows the false-flag fraction RISES with training in all 12 runs
  (`results/h1_best_vs_final.json`); full curves from the re-instrumented retrain are landing
  (box B MACE complete and verified: plateau at 0.90 from epoch ~10, endpoints match released
  values to ±0.001).
- T0-3 bootstrap CIs: in `results/stats.json` and Supp Tables 3/15.

## Tier 2 (complete)

Declared: eSEN / UMA / EquiformerV3 are SO(3)-only (eSCN lineage); rotation law near machine
precision, mirror violation O(1). EquiformerV3 declared strictly rotation-equivariant per its
title.

Measured (`results/t2_random_init.json`, random init, standard rank-3 head, 2,000 crystals):
ALL DECLARATIONS CONFIRMED. Rotation 2e-5..1e-4, deterministic, mirror violated 1.8-2.0;
m-3m/non-cubic violation ratios 1.2e-6..1.9e-5 (the rotation-ceiling signature with no
training). EquiformerV3 rotation-exact AND deterministic — the approximate-equivariance
confound is removed. Audit -> 18 architectures.

## Tier 1 — declarations (recorded BEFORE measurement, 2026-07-17)

Runnable set: **GMTNet** (AIRS 4a16c68, in-repo `piezo_model.pt`, `--use_mask` toggle) and
**CEITNet** (80ab259, in-repo `piezo.pt`, `--zero_mask` toggle). NOT runnable at their pinned
releases: **GoeCTP** — the released journal-version repo (b5e4946) is dielectric-only, no
rank-3 head or piezo data path exists; **EATGNN** (5a854e9) is a 2D in-plane dielectric model
(target shapes hardwired to [3]/[3,3]) despite the paper's bulk-piezo claims; **PCGNN** — no
public code found. Per the prespecified rule, we proceed with the runnable two and state the
rest exactly as above. The non-releasability of the piezo variants is itself evidence for the
benchmark-blindness framing.

Declared per-model predictions (advisor head-algebra, to be verified on a toy centrosymmetric
input before any full run):

1. **CEITNet, mask OFF**: its head assembles odd-parity Cartesian bases with invariant weights
   and invariant pooling → predicted STRUCTURALLY ZERO on exactly centrosymmetric (idealized)
   inputs, no mask needed. On the raw variant: near zero, degraded only by the residual
   coordinate asymmetry.
2. **GMTNet, mask ON**: zero via the space-group zero-mask lookup (`feature_mask` projection
   built from spglib rotations) — a symmetry-database repair, not a property of the features.
3. **GMTNet, mask OFF**: the genuine measurement — no prediction declared beyond the paper's
   own Table-9-style success rates (its dielectric variant satisfies symmetry zeros only
   44-78% without correction), which suggest nonzero violations.
4. Both models sit on the ComformerConv INVARIANT backbone: any exact zeros they achieve are
   readout-level (head algebra or mask), never feature-level — the Jacobian even-fraction
   (only if FD-validated) is predicted to show no odd-subspace structure.

Protocol: both models on the 2,000 idealized AND raw variants, mask ON and OFF, released
checkpoints; report false-flag @ 0.01 C/m^2 + 25-threshold curves + violation medians →
`results/t1_sota_predictors.json`.

### Tier 1 — MEASURED (2026-07-17; declared-vs-measured)

Toy gate (exact centro inputs, mask off): GMTNet 1.3e-8/1.3e-7, CEITNet 9.6e-9/1.7e-7 on
NaCl/rutile — structural zeros at float32 noise. Declaration (1) holds on exact inputs; the
open GMTNet case (3) resolves the same way there.

Full population (2,000; false-flag @ 0.01 C/m^2 | median | max):

| config | ff | median | max |
|---|---|---|---|
| GMTNet idealized mask-off | **0.2345** | 9.9e-8 | 8.72 |
| GMTNet idealized mask-on  | 0.0000 | 0 | 5.5e-4 |
| GMTNet raw mask-off       | **0.2170** | 1.1e-7 | 10.64 |
| GMTNet raw mask-on        | **0.0105** | 0 | 0.265 |
| CEITNet idealized mask-off| **0.1195** | 6.2e-8 | 0.276 |
| CEITNet idealized mask-on | 0.0915 | 5.5e-8 | 0.276 |
| CEITNet raw mask-off      | **0.1100** | 7.4e-8 | 0.276 |
| CEITNet raw mask-on       | 0.0935 | 6.7e-8 | 0.276 |

Declared-vs-measured verdicts:
- (1) CEITNet "structurally zero, no mask needed": PARTIALLY OVERTURNED. True on the toy
  inputs and for ~68% of the population (norms < 1e-5), but 12.0% false-flag on exactly
  symmetric idealized inputs — the head algebra does not extend to all symmetry classes.
  Its own zero-mask barely helps (12.0% -> 9.2%): the mask is a no-op for fully-forbidden
  point groups (their `infer_forced_zero_mask...` returns all-False when the group-average
  projector is identically zero — released semantics, kept).
- (2) GMTNet mask-on: CONFIRMED on idealized (0.0000) — but the repair leaks 1.05% (21
  crystals, max 0.265 C/m^2) on raw coordinates, where spglib finds a subgroup: the
  symmetry-database repair is only as reliable as the symmetry determination feeding it,
  the manuscript's pre-filtering fragility point measured on the field's own mechanism.
- (4) Invariant-backbone consequence, the sharpest finding: the family split INVERTS the
  SO(3) pattern. CEITNet mask-off false-flags **68.7% of m-3m** crystals (28% of m-3, 6.6%
  of non-cubic) — worst exactly where rotation-equivariant models are forced correct.
  Cause: the benchmark protocol removes zero tensors from training, so the most-forbidden
  classes are the classes these models never saw. Benchmark blindness, measured.
- Flagged sets of the two models overlap only partially (469 vs 239 flagged on idealized,
  137 shared): the failures are model-specific learned artifacts, not a shared physical
  signal.

Artifacts: `results/t1/*.npy` (norm vectors + full 3x6 tensors), summary json verified
vector-by-vector (json_match=True for every config). Driver: `scripts/t1_sota_eval.py`
(their pipelines, their checkpoints, their eval semantics; deviations documented in the
script docstring).

## Pending

- Tier 3 learning curve (box A, in flight) — title verdict rule prespecified.
- T4-1 frozen backbones: MACE-MP-0 (parity-labelled `128x0e+128x1o+128x2e+128x3o`, verified
  loadable) vs eSEN. Declared: MACE-MP-0 head structurally zero; eSEN head false-flags ~0.9.
- H-1 full curves (nequip/allegro/EqV2 retrains, box A).
