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

## T4-1 — COMPLETE (both arms measured; flag 3 cleared)

| frozen pretrained backbone | declared | measured false-flag @ 0.01 | test MAE |
|---|---|---|---|
| MACE-MP-0 (parity-aware) | ~0 structural | **0.0000 ± 0.0000** (exact zeros, all crystals, all seeds) | 0.158 |
| eSEN-30M-OAM (parity-blind) | ~0.9 | **0.8997 ± 0.0034** | 0.170 |

Both declared predictions confirmed. Same head, same protocol; the parity typing fixed at
pretraining time is the only difference and it decides everything. eSEN checkpoint:
`esen_30m_oam.pt` from the gated facebook/OMAT24 HF repo (user-authorized token); its CPU
forward carries single-ULP nondeterminism (1.2e-6 on scale-8 features), gated at the noise
floor and documented. `results/t4_frozen_backbone.json`, `results/t4/{mace_mp0,esen}_seed*`.

## Tier 3 — MEASURED (title verdict: SURVIVES)

Five points, three seeds each, nested sets: held-out ff 0.895 -> 0.858±0.020 over a 64x
range of zero-labelled data; SEEN-SG flat at 0.90; in-training control 0.88–0.90 for every
N >= 1,000 (0.883±0.011 at N=16,000, trained-zeros median 0.12 -> 0.06 C/m^2, six times
threshold). Prespecified rule applied neutrally: plateau well above zero -> "Not all symmetry
can be learned" STANDS. `results/t3_learning_curve.json`, Supp Table stab:t3.

## H-1 — MEASURED (flag 18 cleared; the last experimental red flag)

All 12 SO(3) retrains under per-epoch instrumentation: every core >= 0.85 within its first
third of training (Allegro epoch 1, MACE 4, EqV2 23, NequIP 34), flat at the headline value
thereafter. Deterministic endpoints reproduce released values to ±0.003 (3 seeds exact to
four decimals); EqV2 within its measured draw spread. All 21 grid runs passed the
verification battery. `results/h1_epoch_curve.json`, Supp Fig sfig:epochcurves.

## Pending

- Tier 3 learning curve (box A, in flight) — title verdict rule prespecified.
- T4-1 frozen backbones — HALF MEASURED. MACE-MP-0 (parity-labelled
  `128x0e+128x1o+128x2e+128x3o`) frozen + trained O(3) head: **false-flag 0.0000, OOD
  violations exactly 0.0 for all 2,000 crystals, all 3 seeds** (declared structural zero
  CONFIRMED); test MAE 0.158 with a stated expressivity ceiling (a linear head from these
  irreps has no path to the 2o output component — noted honestly, does not affect the zero).
  `results/t4_frozen_backbone.json`, `results/t4/mace_mp0_seed*_ood.npy`.
  eSEN arm BLOCKED on a gated checkpoint: every released eSEN model lives in the gated
  HF repo facebook/OMAT24 (401 anonymously). USER ACTION: accept the FAIR Chemistry License
  at https://huggingface.co/facebook/OMAT24, create a read token, then
  `HF_TOKEN=<token> third_party/venvs/fairchem1/bin/python scripts/t4_cache_esen_features.py`
  and `uv run python scripts/t4_frozen_backbone.py --backbone esen`. Scripts ready.
- H-1 full curves (nequip/allegro/EqV2 retrains, box A).
