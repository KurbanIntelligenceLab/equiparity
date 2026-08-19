# Compute

Every number here is read from [`results/appendix_stats.json`](../results/appendix_stats.json)
(`compute` block), written by the grid aggregation script. Per-run wall-clock, peak memory, GPU
model, CUDA version and driver version are recorded in each run's `outputs/<experiment_id>/manifest.json`.

## Hardware

Two GPU classes carried the 84-run matched-pair grid:

| GPU | Train hours | Cores run on it |
|---|---|---|
| RTX 5090 (Blackwell, sm_120, CUDA 12.8) | 31.8 | NequIP, Allegro, EquiformerV2 |
| RTX PRO 6000 Blackwell WS (96 GB) | 40.6 | MACE |

**The two are not interchangeable, so per-core wall-clock is not a like-for-like architecture
comparison.** MACE's 40.6 hours and NequIP's 7.6 hours were measured on different silicon. Compare
cores on accuracy and on false-flag fraction, never on training time.

Memory is not a constraint anywhere in this study. Models are 50–120k parameters; QM9 molecules
are at most 29 atoms; training crystals at most 288; the largest evaluation crystal is 444 atoms.
Peak GPU memory across all four cores stayed under 5.5 GB, so no A100-class card is required.

## Cost of the full grid

| | |
|---|---|
| Total training | 72.4 GPU-hours across 84 runs |
| Total test evaluation | 644 s |
| Total OOD evaluation (2,000 crystals) | 320 s |
| Upper bound on rental cost | $72 |

The cost is an upper bound, not an estimate: the price actually paid was not recorded, and runs
were launched against a filter that accepts any offer under $1.00/GPU-hour.

## Per-core throughput

| Core | Runs | Train hours | s / epoch | Structures / s | Peak GPU MB |
|---|---|---|---|---|---|
| NequIP | 24 | 7.6 | 10.3 | 1603 | 1225 |
| Allegro | 24 | 5.1 | 6.7 | 2346 | 5495 |
| MACE | 24 | 40.6 | 55.5 | 252 | 2020 |
| EquiformerV2 | 12 | 19.1 | 50.3 | 354 | 2253 |

EquiformerV2 contributes 12 runs rather than 24 because it is a fixed rotation-only
representative: it has no parity-labelled arm to pair against.

## Numerical precision

Training runs in **float32 mixed precision**, the default in every committed config. At production
size (128 features, `l_max=3`) this measured 9.8 ms/step against 66.9 ms/step in float64 — a 6.8×
speedup. Set `precision: float64` in a config only when a measurement needs the extra headroom;
the structural-zero results do not, because the O(3) floor sits far above float32 epsilon.

Exact numerical reproducibility is not guaranteed across torch, CUDA or cuDNN releases, or across
devices. Seeds, library versions and device metadata are recorded per run so a divergence can be
attributed rather than guessed at.
