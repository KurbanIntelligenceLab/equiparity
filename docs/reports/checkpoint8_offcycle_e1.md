# Off-cycle report — E1 landed outside its anticipated outcomes

## Why this report exists

The approved E1 design named two acceptable outcomes and one that triggers the standing rule:

> *Expected: SEEN-SG false-flags drop substantially; UNSEEN-SG remains materially higher. Either
> outcome supports the thesis… A result outside both (e.g., no drop even on SEEN-SG) triggers the
> standing rule.*

The measured outcome is the third. Reporting before writing, as required.

## What was run

12 runs, SO(3) arms only, 3 seeds × 4 cores, hyperparameters identical to the headline. Training set
= the 2,649 real piezoelectric tensors + **1,000 fresh centrosymmetric insulators labelled exactly
zero**, drawn only from space groups `{2, 12, 14, 15, 62, 225}`. Validation and test partitions
inherited unchanged. Target scale frozen at the un-augmented 0.749134.

All 12 runs completed 150 epochs, 0 failures. Data integrity verified before analysis: train split
3,649 = 2,649 + 1,000; `train ∩ OOD = ∅`; val/test byte-identical to the headline split; every run
wrote both OOD violation vectors (2,000 finite values each) and a final-epoch checkpoint.

## The result

**Augmentation does not buy the zero, anywhere — including on the crystals it trained on.**

| core | false-flag on the 1,000 **trained** zero-labelled crystals | median ‖T‖ there |
|---|---|---|
| NequIP | 0.8953 | 0.1200 |
| Allegro | 0.8990 | 0.3291 |
| MACE | 0.8960 | 0.0870 |
| EquiformerV2 | 0.9219 | 0.1153 |

Change in false-flag rate on held-out OOD crystals, baseline → augmented:

| core | SEEN-SG (1,232) | UNSEEN-SG (768) |
|---|---|---|
| NequIP | 0.9042 → 0.9042 (−0.0000) | 0.8811 → 0.8685 (−0.0126) |
| Allegro | 0.9075 → 0.9067 (−0.0008) | 0.9128 → 0.8919 (−0.0208) |
| MACE | 0.9075 → 0.9067 (−0.0008) | 0.9080 → 0.8924 (−0.0156) |
| EquiformerV2 | 0.9508 → 0.9302 (−0.0206) | 0.9670 → 0.9353 (−0.0317) |

O(3) arms, not retrained, on both subsets: **0.0000**, median 3e-07 – 3e-06.

## Is this a bug or a result?

Checked, in this order.

1. **Did the runs use the augmented data?** Yes. `dataset: mp_piezoelectric_augmented`, train split
   3,649 (= baseline 2,649 + 1,000), and the violation medians move substantially versus baseline
   (NequIP 0.693 → 0.379), which cannot happen if the data were unchanged.
2. **Did the runs use the frozen target scale?** Yes — verified empirically, because
   `_config_snapshot` did not serialise `training.target_scale` at the time (now fixed). Reloading a
   checkpoint and rescaling by the frozen 0.749134 reproduces the committed OOD median to relative
   **3.9e-08**; the recomputed 0.638289 is 15% off. The runs used the frozen value.
3. **Did the model simply fail to fit its own zero examples?** No. Train MAE on the zero rows is
   **0.014–0.045**, against **0.086–0.128** on the real-tensor rows, on targets whose mean component
   magnitude is 0.145. The zeros are fit *better* than the real tensors. The model is pushing them
   towards zero and cannot arrive.
4. **Is the model underfitting overall?** No. Train MAE on real rows (0.088) is 0.4× the test MAE
   (0.227): it fits its training data considerably better than held-out data, so it has the capacity
   to memorise. What it cannot memorise is an exact zero.
5. **Is the SEEN/UNSEEN split confounded?** Partly, and this is stated in the report. Median ‖T‖ is
   higher on SEEN-SG than UNSEEN-SG in the *baseline* runs, before any augmentation exists — the
   subsets differ in composition, not only in space group. The interpretable quantity is therefore
   the change relative to baseline **within** each subset, which is what we report. The SEEN/UNSEEN
   difference on its own is not interpretable.

**Conclusion: this is a result, not a bug.** The failure to drop even on SEEN-SG is the honest
finding, and it is stronger than either anticipated outcome.

## What it means for the paper

The anticipated outcomes both conceded something: either "the fix works but requires curated data"
or "the fix works on seen classes and fails to transfer". Neither concession is needed.

Gradient descent can push an SO(3) model's prediction towards zero — magnitudes halve, roughly
uniformly, on seen and unseen space groups alike (NequIP SEEN 0.826 → 0.449, UNSEEN 0.422 → 0.254).
It cannot reach zero, on any crystal, including the ones in its own training set. The physical answer
is *exactly* zero, so halving an impossible quantity is not a fix. Meanwhile the O(3) arms are at
machine zero for free, at any weights, having never seen a centrosymmetric crystal in training.

One honest counterweight: **augmentation is not worthless.** Non-centrosymmetric test MAE improves
for every core (NequIP 0.2405 → 0.2278, MACE 0.2567 → 0.2221, EquiformerV2 0.2157 → 0.1774). A
thousand extra crystals is a thousand extra crystals. The augmented models are better regressors
that still predict physically impossible values.

## Limits of this experiment

- One augmentation budget (1,000 crystals, ~27% of the training set) and one schedule (150 epochs,
  identical to the headline). We did **not** sweep the augmentation fraction, up-weight the zero
  rows in the loss, or train longer. A reviewer may reasonably ask whether an aggressive enough
  regime forces the false-flag rate down; we have not tested that, and the report does not claim it
  cannot be forced down. The claim is narrower and safe: **under matched hyperparameters, with a
  quarter of the training set labelled zero, SO(3) still false-flags ~90% of the crystals it was
  explicitly trained to call zero, while O(3) is at 0.0000 without being trained on any.**
- Six space groups define SEEN-SG. A different choice would change the subset composition.

## Recommendation

Report as measured. If the reviewer response warrants it, the cheapest follow-up is a loss-weight
sweep on the zero rows (one core, three weights, ~3 GPU-hours), which would bound how hard one must
push to buy a zero that O(3) has by construction.
