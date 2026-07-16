# EXPERIMENTS TO RUN

Deep experimental audit, 2026-07-14. Every item below is specified so it can be executed
without further design work. Ordered by what a Nature Communications referee will kill the
paper for.

---

## What you already have, and it is strong

Your experimental *methodology* is better than most papers at this venue. Do not change it.

| | |
|---|---|
| Matched-pair ablation, identical in every respect but one bit | **the gold standard for mechanism isolation** |
| Positive control (verification gate, hand-built all-even construction) | present |
| Negative control (the dipole: parity-odd but never symmetry-forbidden) | present, and decisive |
| Capacity control (SO(3) has *more* parameters, ratios 1.00–1.40) | present |
| Deployed-model control (EquiformerV2, unmodified, reproduced against pinned upstream) | present |
| Mechanism evidence (Jacobian, point-group split, continuous distortion sweep) | three independent lines |
| Reproducibility (provenance manifests, content hashes, refuses to run on a dirty tree) | exemplary |

The gap is not in *how* you ran the experiments. It is in **which models you ran them on**,
and in **what you never compared against**.

---

# BLOCKING

## B-1. The SOTA crystal-tensor predictors are never tested, and you criticize them

**This is the one that will kill the paper.**

There is an established benchmark for piezoelectric tensor prediction, with published
baselines and a standard metric (Frobenius-norm error `Fnorm`, and error-within-threshold
`EwT`). From CEITNet (arXiv 2602.04323), Table 5:

| Method | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---|---|---|---|
| ETGNN | 0.873 | 0.00% | 0.00% | 0.00% |
| GMTNet | 0.752 | 6.29% | 1.48% | 1.11% |
| GeoCTP | 0.778 | 2.59% | 1.14% | 0.04% |
| CEITNet | **0.517** | **21.98%** | **5.80%** | **2.72%** |

Your Discussion argues that this entire family performs "readout-level repairs" that
"work only for the quantity someone remembered to repair". **You never run one.** A referee
from this community, and there will be one, opens with that.

### The experiment

Run **GMTNet**, **GoeCTP**, **EATGNN** and **CEITNet** on your 2,000 centrosymmetric crystals.
All four have released code. This is inference on released checkpoints, or a short retrain
on your split. Report, for each:

1. The **false-flag fraction** at τ = 0.01 C/m², next to your own arms in Fig. 2.
2. The **even-subspace energy fraction of the Jacobian** (your Corollary 2 statistic).

### Why you cannot lose

- **If they output exactly zero:** the field already solves the *output* problem, and you must
  say so. Your contribution then narrows, correctly, to: the features underneath are still
  parity-blind, and the Jacobian proves it. That is still a paper, and it is an honest one.
- **If they output nonzero:** you have shown that even the *dedicated symmetry-enforcing*
  state of the art fails a label-free physical-validity test. That is a substantially bigger
  result than the one you currently claim, and it is the headline.

Either way you are stronger than you are now, where you are simply exposed.

**Cost:** days, not weeks. Four inference runs.

---

## B-2. The models you test are two to four years old

You train NequIP, Allegro, MACE (2022–23) and EquiformerV2 (2023). In 2026, people deploy
**eSEN** (Feb 2025), **UMA** (June 2025) and **EquiformerV3** (April 2026). Your headline
claim is about *deployed* models. A referee will name UMA.

### The experiment, and it needs no training at all

Theorem 1 says the O(3) guarantee holds **at any parameter values, including random
initialization**. So:

1. Instantiate eSEN, UMA and EquiformerV3 at random init (or load the released checkpoint).
2. Attach a rank-3 readout head (the same one you use for the matched cores).
3. Evaluate on the 2,000 centrosymmetric crystals.
4. Report the violation magnitude distribution.

A parity-typed backbone returns the arithmetic floor. A parity-blind one does not. **No
training, no data, no tuning.** This is an afternoon, and it extends your headline from
"three architectures from 2022" to "what the field runs today".

Then add the three rows to Supplementary Table 1 and the audit becomes eighteen models.

**Cost:** hours.

---

## B-3. Your new title makes a learnability claim and you have one data point

The title is now **"Not all symmetry can be learned"**. On the learning axis you have:

- one augmentation size (1,000 zero-labelled crystals),
- a loss-weight sweep (W = 1, 10, 100) on **one** architecture.

A referee reading that title will immediately ask for **the learning curve**. If violations
fall off fast with more zero-labelled data, the title is wrong.

### The experiment

Retrain the NequIP SO(3) arm with

    N_zero in {0, 250, 1000, 4000, 16000}

(16,000 is roughly six times your real piezoelectric training set, which is the point). Three
seeds each. Plot, against N_zero:

- false-flag fraction on the **trained-on** zeros (the in-training control),
- false-flag fraction on **held-out** centrosymmetric crystals,
- violation **median**,
- non-centrosymmetric **test MAE** (to show the regressor keeps improving).

**What you expect, and what the theory predicts:** the median falls, the false-flag fraction
*plateaus*, and the curve flattens well above zero. If it does, your title is bulletproof and
this becomes Figure 4c, the most-cited panel in the paper. If it plummets towards zero,
**change the title**, and be glad you found out before a referee did.

**Cost:** 15 runs, same scale as your existing augmentation experiment.

---

## B-4. The contamination check you never made

Methods verifies that the **1,000 augmentation crystals** are disjoint from the evaluation
population and the training identifiers. It says **nothing** about whether the **2,000
evaluation crystals** are disjoint from the **3,312 piezoelectric train/val/test structures**.

And they might not be. Your own text says the training set contains **16 exactly-zero rows**,
which means it contains centrosymmetric crystals.

### The check

1. Intersect the 2,000 evaluation identifiers with the 2,649 / 331 / 332 splits.
2. Report the overlap. If it is nonzero, report the false-flag fractions with those crystals
   removed.

The direction of the bias works *against* you (a model shown a zero label predicts lower
there), so this is very unlikely to hurt. But it is a standard leakage check, it is one line
of set arithmetic, and a rigor referee will ask for it. **State the answer in Methods either
way.**

**Cost:** minutes.

---

# HIGH VALUE

## H-1. Does the violation plateau during training?

You train 150 epochs for the piezoelectric target. **You already logged the metrics.** Plot
the false-flag fraction of the SO(3) arms **against epoch**.

- If it plateaus at ~0.90, "training cannot buy the zero" is nailed, and the objection
  "your models are undertrained" dies on the spot.
- If it is still falling at epoch 150, a referee will say so, and they will be right.

**Cost:** a plot from data you already have.

## H-2. Where does your accuracy sit?

You report piezoelectric test MAE of 0.21–0.26 C/m². Nobody else reports MAE; the field
reports **Fnorm** and **EwT** (table in B-1). Convert your predictions to those metrics on the
standard split and state where you land. If you are within ~2x of GMTNet, say so and the
"your models are weak, of course they err" objection is dead. If you are 5x worse, you need to
know that now.

## H-3. The pretrained-backbone experiment your Discussion promises

Your Discussion's most consequential practical claim is that the parity decision is
"inherited by every downstream task" when property heads are attached to pretrained universal
potentials. **You never test it.** Every model in the paper is trained from scratch.

Fine-tune a piezoelectric head on two pretrained backbones, one parity-aware (MACE-MP-0) and
one parity-blind (eSEN or UMA), freezing the backbone. Show the zeros and the false flags.
That is the scenario practitioners are actually in, and it is the figure they will remember.

## H-4. Loss-weight sweep on all cores

Currently NequIP only, and your limits section concedes it. Six more runs.

## H-5. Bootstrap confidence intervals on the false-flag fractions

`parity_metrics.py::bootstrap_ci` already does it. Three seeds is thin; a percentile bootstrap
over the 2,000 structures costs nothing and pre-empts the n=3 objection.

## H-6. More materials in the distortion sweep

Rutile carries the whole mechanism argument in Fig. 5a. Add two more polar distortion paths
(BaTiO3 cubic→tetragonal, PbTiO3) so that one material is not load-bearing.

---

# What is NOT missing

Do not let a referee talk you into these:

- **Experimental validation.** The ground truth here is a symmetry identity, not a measurement.
  There is nothing to validate against.
- **More seeds.** Three is thin, but all your statistical weight comes from the structure-level
  paired tests over n = 2,000, and you say so. Defend it; do not spend a month on it.
- **Hyperparameter tuning per architecture.** That would *break* the matched-pair design, which
  is the whole point. Defend it.
