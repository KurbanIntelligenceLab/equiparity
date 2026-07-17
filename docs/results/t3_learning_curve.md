# T3 — the N_zero learning curve

NequIP SO(3), 3 seeds per point, nested augmentation sets; held-out 2,000 population.

| N_zero | ff (global) | ff SEEN | ff UNSEEN | median | test MAE | ff trained zeros |
|---|---|---|---|---|---|---|
| 0 | 0.8953 ± 0.0008 | 0.9042 | 0.8811 | 0.6664 | 0.2405 | - |
| 250 | 0.8937 ± 0.0034 | 0.9058 | 0.8741 | 0.5222 | 0.2318 | 0.7519 ± 0.0038 |
| 1000 | 0.8905 ± 0.0030 | 0.9042 | 0.8685 | 0.3725 | 0.2278 | 0.8953 ± 0.0020 |
| 4000 | 0.8818 ± 0.0035 | 0.9018 | 0.8498 | 0.1779 | 0.2098 | 0.8865 ± 0.0014 |
| 16000 | 0.8582 ± 0.0201 | 0.8904 | 0.8064 | 0.0768 | 0.2110 | 0.8827 ± 0.0109 |

Title rule: plateau well above zero -> the learnability title stands; falling towards
zero -> revert. The verdict is stated in the Checkpoint-10 report once N=16000 lands.
