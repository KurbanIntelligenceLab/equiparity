# E1 — the augmentation rebuttal

The obvious objection to the headline is: *just train the SO(3) model on centrosymmetric
crystals labelled zero.* This measures what that buys.

Training set: 2,649 real piezoelectric tensors + 1000 fresh
centrosymmetric insulators labelled **exactly zero**, drawn only from space groups
[2, 12, 14, 15, 62, 225]. Validation and test partitions are inherited unchanged from the
headline split, and the target normalisation scale is frozen at the un-augmented value
(0.749134), so every number below is directly comparable to the main table. 
SO(3) arms only,
3 seeds, hyperparameters identical to the headline runs.

## The control that decides what this experiment means

Before asking whether learned zeros *generalize*, ask whether they are learned at all.
Below: what each model predicts on the very crystals it trained on with 
exact-zero labels.

| core | median ‖T‖ on trained zeros | false-flag on trained zeros | train MAE (zero rows) | train MAE (real rows) | mean &#124;target&#124; (real rows) |
|---|---|---|---|---|---|
| NequIP | 0.1200 | 0.8953 | 0.0181 | 0.0882 | 0.1454 |
| Allegro | 0.3291 | 0.8990 | 0.0445 | 0.1284 | 0.1454 |
| EquiformerV2 | 0.1153 | 0.9219 | 0.0203 | 0.1197 | 0.1454 |
| MACE | 0.0870 | 0.8960 | 0.0140 | 0.0863 | 0.1454 |

The zero rows *are* fit better than the real-tensor rows — train MAE 
0.014–0.045 against
0.086–0.128, on targets whose mean component magnitude is 0.145. The model is 
not ignoring
them. **But it still false-flags ~90% of the crystals it was explicitly trained to call
zero.** Gradient descent drives the prediction towards zero; it cannot make it 
zero. An O(3)
model does not have to try: the zero is structural.

## Transfer: does it help on space groups seen in training?

Evaluation splits the untouched OOD 2,000 by space group — **SEEN-SG** (1232),
**UNSEEN-SG** (768). Neither overlaps the training ids. Mean ± std over 3 seeds.

| core | arm | false-flag SEEN-SG | false-flag UNSEEN-SG | median SEEN | median UNSEEN | test MAE |
|---|---|---|---|---|---|---|
| Allegro | augmented_so3 | 0.9067 ± 0.0000 | 0.8919 ± 0.0045 | 5.384e-01 | 2.736e-01 | 0.2303 ± 0.0038 |
| Allegro | baseline_o3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 4.030e-07 | 3.389e-07 | 0.2140 ± 0.0058 |
| Allegro | baseline_so3 | 0.9075 ± 0.0000 | 0.9128 ± 0.0039 | 1.182e+00 | 5.862e-01 | 0.2589 ± 0.0176 |
| EquiformerV2 | augmented_so3 | 0.9302 ± 0.0049 | 0.9353 ± 0.0106 | 1.489e-01 | 1.020e-01 | 0.1774 ± 0.0060 |
| EquiformerV2 | baseline_so3 | 0.9508 ± 0.0048 | 0.9670 ± 0.0066 | 5.433e-01 | 2.947e-01 | 0.2157 ± 0.0096 |
| MACE | augmented_so3 | 0.9067 ± 0.0008 | 0.8924 ± 0.0015 | 5.339e-01 | 2.594e-01 | 0.2221 ± 0.0107 |
| MACE | baseline_o3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2.974e-06 | 2.298e-06 | 0.2222 ± 0.0066 |
| MACE | baseline_so3 | 0.9075 ± 0.0000 | 0.9080 ± 0.0020 | 1.156e+00 | 5.084e-01 | 0.2567 ± 0.0084 |
| NequIP | augmented_so3 | 0.9042 ± 0.0024 | 0.8685 ± 0.0039 | 4.486e-01 | 2.540e-01 | 0.2278 ± 0.0015 |
| NequIP | baseline_o3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 3.319e-07 | 2.884e-07 | 0.2083 ± 0.0077 |
| NequIP | baseline_so3 | 0.9042 ± 0.0021 | 0.8811 ± 0.0040 | 8.258e-01 | 4.222e-01 | 0.2405 ± 0.0080 |

### Change in false-flag rate, baseline → augmented

| core | SEEN-SG | UNSEEN-SG |
|---|---|---|
| NequIP | 0.9042 → 0.9042 (-0.0000) | 0.8811 → 0.8685 (-0.0126) |
| Allegro | 0.9075 → 0.9067 (-0.0008) | 0.9128 → 0.8919 (-0.0208) |
| MACE | 0.9075 → 0.9067 (-0.0008) | 0.9080 → 0.8924 (-0.0156) |
| EquiformerV2 | 0.9508 → 0.9302 (-0.0206) | 0.9670 → 0.9353 (-0.0317) |

## Reading

**Augmentation does not buy the zero.** The false-flag rate on SEEN-SG — the 
space groups
the augmentation was drawn from — falls by between 0.0000 and 0.0206. On 
UNSEEN-SG it falls
by 0.013–0.032. There is no meaningful transfer advantage for the space groups 
the model was
trained on, because there is barely any improvement to transfer.

What augmentation *does* do is shrink violation magnitudes roughly uniformly, 
by about half
(e.g. NequIP SEEN 0.826 → 0.449, UNSEEN 0.422 → 0.254). The predictions move 
towards zero
everywhere and reach it nowhere. Since the physical answer is exactly zero, a 
factor-of-two
reduction in an impossible quantity is not a fix.

**A caveat on SEEN vs UNSEEN.** The two subsets differ in composition, not only in space
group: median ‖T‖ is higher on SEEN-SG than on UNSEEN-SG in the *baseline* 
runs too, before
any augmentation exists. The interpretable quantity is therefore the change relative to
baseline within each subset, which is what the table above reports — not the 
SEEN/UNSEEN
difference itself.

**The O(3) rows were not retrained.** They are quoted from the headline runs, 
and they are
0.0000 on both subsets at machine-zero medians (3e-07 – 3e-06). Their zeros 
hold for any
weights and any training data; that is the entire point, and it is why 
retraining them would
answer nothing.

**Augmentation is not free of benefits.** Non-centrosymmetric test MAE 
*improves* for every
core (NequIP 0.2405 → 0.2278; EquiformerV2 0.2157 → 0.1774): a thousand extra 
crystals is a
thousand extra crystals. The augmented models are better regressors that still predict
physically impossible values.

## Off-cycle note

The design anticipated two outcomes — SEEN-SG false-flags drop substantially 
(fix requires
curated data), or they drop and UNSEEN-SG does not (learned zeros do not generalize). The
measured outcome is the third one the plan flagged: **no drop even on SEEN-SG**, which
triggers the standing rule. See `docs/reports/checkpoint8_offcycle_e1.md`.
