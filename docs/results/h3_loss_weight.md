# H3 — the loss-weight sweep

Can up-weighting the zero-labelled rows force an SO(3) model's centrosymmetric violation to
zero? NequIP SO(3) on the augmented set (2,649 real tensors + 1,000 injected zeros), with the
exactly-zero-target rows (1,016: 1,000 injected + 16 real) weighted by W in the MSE. W=1 is
the committed E1 run; W=10, 100 are the sweep. Mean ± std over 3 seeds.

| W | ff trained zeros | median trained zeros | ff SEEN-SG | ff UNSEEN-SG | test MAE |
|---|---|---|---|---|---|
| 1 | 0.8953 ± 0.0020 | 0.1200 | 0.9042 ± 0.0024 | 0.8685 ± 0.0039 | 0.2278 ± 0.0015 |
| 10 | 0.8730 ± 0.0026 | 0.0463 | 0.9031 ± 0.0012 | 0.8420 ± 0.0072 | 0.2180 ± 0.0027 |
| 100 | 0.6762 ± 0.0154 | 0.0140 | 0.8715 ± 0.0031 | 0.7604 ± 0.0060 | 0.2142 ± 0.0055 |

For reference, O(3) untrained: 0.0000 on every population, median ~3e-07 (structural).

## Reading

**Extreme reweighting does not buy the zero.** As W goes 1 -> 10 -> 100, the false-flag rate on the crystals the model was *trained to call zero* falls only 0.895 -> 0.873 -> 0.676, and on held-out SEEN-SG 0.904 -> 0.871. Even at 100x it still false-flags roughly two-thirds of its own trained-zero crystals and ~87% of held-out seen ones -- far above O(3)'s 0.0000.

The median violation does shrink with weight (0.120 -> 0.014 on the trained zeros, ~9x), and non-centrosymmetric test MAE is not harmed -- it slightly improves (0.2278 -> 0.2142). So the fix costs nothing in regression quality; it simply cannot reach zero. Gradient descent pushes the prediction towards zero and cannot arrive, because the physical answer is *exactly* zero -- which only the O(3) structure delivers, for free, at any weight.

No weight setting achieved ~0.00 false-flag with intact MAE, so no off-cycle report is triggered. The E1 conclusion stands, sharpened: augmentation with even 100x loss weighting reduces but does not remove the impossible predictions.
