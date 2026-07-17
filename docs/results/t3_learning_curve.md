# T3 — the N_zero learning curve

NequIP SO(3), 3 seeds per point, nested augmentation sets; held-out 2,000 population.

| N_zero | ff (global) | ff SEEN | ff UNSEEN | median | test MAE | ff trained zeros |
|---|---|---|---|---|---|---|
| 0 | 0.8953 ± 0.0008 | 0.9042 | 0.8811 | 0.6664 | 0.2405 | - |
| 250 | 0.8937 ± 0.0034 | 0.9058 | 0.8741 | 0.5222 | 0.2318 | - |
| 1000 | 0.8905 ± 0.0030 | 0.9042 | 0.8685 | 0.3725 | 0.2278 | - |
| 4000 | 0.8800 ± 0.0021 | 0.9014 | 0.8457 | 0.1764 | 0.2089 | - |
| 16000 | (pending) | | | | | |

Title rule: plateau well above zero -> the learnability title stands; falling towards
zero -> revert. The verdict is stated in the Checkpoint-10 report once N=16000 lands.
