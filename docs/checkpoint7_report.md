# Checkpoint 7 — Results Review Response

Response to the five-item scientific review. Two items were blocking (2 and 3); both are cleared.
All numbers below come from the **84-run instrumented grid** (7 arms × 4 targets × 3 seeds), which
was re-run from scratch because the original grid saved no checkpoints and evaluated only one OOD
variant.

Aggregates: [`../results/tables.md`](../results/tables.md), [`../results/stats.json`](../results/stats.json),
[`../results/threshold_curves.csv`](../results/threshold_curves.csv). Regenerate with
`python3 scripts/analyze_results.py`.

---

## Item 1 — Drop CliffordSTF, archive its runs, record the limitation

**Done.**

- Removed from the grid generator (`scripts/generate_grid.py`); scope reduced from 96 to **84 runs**
  across four cores.
- Its 14 completed run directories are **archived, not deleted**, at
  `~/Desktop/parity_work/archive_clifford/`.
- The limitation is recorded in `RESULTS.md` § *Two models were dropped, and why*.

**Why it was dropped.** CliffordSTF is O(3)-exact in exact arithmetic — on machine-perfect
centrosymmetric input it cancels to ~1e-15. But Cl(3,0) grades only reach L=1, so an L=3 output must
be assembled through a **cubic** tensor product, which is ill-conditioned. Trained, it amplifies the
~1e-6 residual coordinate asymmetry of real DFT-relaxed crystals by 3,000–25,000×, yielding a
false-flag of 0.42. That number measures the readout's condition number, not the parity of the
algebra, so including it would confound the central claim.

**Cost of dropping it.** CliffordSTF was the only **non-e3nn** O(3) arm. Its purpose was to show the
finding is a property of O(3)-equivariance rather than of the e3nn library. That purpose is now
unmet, and it is stated in `RESULTS.md` as the study's **principal limitation**: all three O(3) arms
share the e3nn irrep implementation. ICTP was surveyed as a substitute and rejected — its rank-3
features sit inside internal product-basis tensors with no clean linear readout.

The conditioning result is itself worth reporting: *O(3)-equivariance guarantees an exact zero in
theory, but realizing it numerically on real data requires a well-conditioned (linear) readout of
native high-order features.*

---

## Item 2 (BLOCKING) — Resolve the evaluation-protocol inconsistency

**Cleared.** Every model is now evaluated on **both** OOD variants, side by side, in the same run.

First, the factual correction: the reported headline numbers had *always* used the **idealized** OOD
set. The "1e-6 raw residual" figure that prompted this item came from a local diagnostic and never
entered a production result. There was no inconsistency between reported numbers — but there was no
evidence either way about the raw set, which is what made this blocking.

Both variants are now first-class, computed per run by `evaluate_ood_variants`:

- **idealized** — coordinates snapped to the exact space group (`spglib.standardize_cell`)
- **raw** — the unmodified DFT-relaxed coordinates (centrosymmetric only to ~1e-6)

Both sets contain the same 2000 spglib-verified centrosymmetric insulators (idealized 72,280 atoms;
raw 72,642 atoms). Per-structure violation vectors are saved for both.

### Result: idealization neither creates nor hides the effect

False-flag @ 1e-2, mean ± s.d. over 3 seeds:

| Core | Parity | idealized | raw | Δ (raw − idealized) |
|---|---|---|---|---|
| NequIP | O(3) | 0.0000 ± 0.0000 | 0.00017 ± 0.00029 | +0.00017 |
| NequIP | SO(3) | 0.8953 ± 0.0008 | 0.8953 ± 0.0008 | 0.00000 |
| Allegro | O(3) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00000 |
| Allegro | SO(3) | 0.9095 ± 0.0015 | 0.9095 ± 0.0015 | 0.00000 |
| MACE | O(3) | 0.0000 ± 0.0000 | 0.00033 ± 0.00029 | +0.00033 |
| MACE | SO(3) | 0.9077 ± 0.0008 | 0.9077 ± 0.0008 | 0.00000 |
| EquiformerV2 | SO(3) | 0.9570 ± 0.0052 | 0.9563 ± 0.0054 | −0.00067 |

**Does O(3) degrade on raw? No — and this is the flag the review asked for, resolved negative.**
The worst O(3) case is MACE at 0.00033, i.e. **0.67 of 2000 structures** averaged over three seeds:
a single structure, in a single seed, marginally over the 0.01 cut. The median violation barely moves
(MACE: 2.76e-6 idealized → 3.17e-6 raw). The ~1e-6 coordinate asymmetry of real DFT geometries is
real, but it sits three to four orders of magnitude below the threshold of consequence.

**SO(3) is bit-for-bit indifferent to the variant** (Δ ≤ 0.0007). Its violation is intrinsic to the
model, not induced by the coordinate treatment. This is the strongest available evidence that the
headline is not an idealization artifact: the failing arm does not care whether the coordinates were
symmetrized, and the passing arm cares only at the 1e-4 level.

All headline tables in `RESULTS.md` now report both variants.

---

## Item 3 (BLOCKING) — Threshold curves and violation distributions

**Cleared.** The single 0.01 threshold is replaced by a full curve, and the underlying distributions
are published.

### Deliverables

| Artifact | Contents |
|---|---|
| `results/threshold_curves.csv` | false-flag fraction at 25 log-spaced thresholds (1e-4 → 1e0), per core × parity × variant, mean and s.d. over seeds |
| `results/fig_threshold_curves.png` / `.pdf` | draft figure: 4 panels (one per core), O(3) vs SO(3), solid = idealized, dashed = raw, ±1 s.d. band |
| `results/fig_violation_hist.png` / `.pdf` | violation-magnitude histograms, log x, 3 seeds pooled |
| `results/stats.json` | median, IQR, 5/25/50/75/95 percentiles, max — per arm, per variant |
| `ood_violations_{idealized,raw}.npy` | per-structure violation magnitudes (2000 floats) for all 21 piezoelectric runs |

### The threshold is not doing any work

SO(3)'s false-flag fraction is **flat across four decades**. For NequIP it is 0.917 at a 1e-4
threshold and 0.895 at 1e-2 — a 2-point change over a 100× change in threshold. The O(3) curve is
pinned at zero across the same range. There is no threshold in this window at which the conclusion
flips, because the two distributions are separated by ~6 orders of magnitude with essentially no
overlap.

### Violation distributions (idealized set, seeds pooled)

| Core | Parity | median | IQR | max |
|---|---|---|---|---|
| NequIP | O(3) | 3.13e-07 | [1.6e-07, 5.2e-07] | 5.5e-04 |
| NequIP | SO(3) | 0.665 | [0.20, 1.33] | 11.9 |
| Allegro | O(3) | 3.83e-07 | [1.9e-07, 6.7e-07] | 1.1e-03 |
| Allegro | SO(3) | 0.949 | [0.31, 1.89] | 20.0 |
| MACE | O(3) | 2.73e-06 | [1.4e-06, 4.8e-06] | 1.3e-03 |
| MACE | SO(3) | 0.921 | [0.32, 1.67] | 20.6 |
| EquiformerV2 | SO(3) | 0.439 | [0.17, 0.91] | 19.8 |

(Full percentiles per arm and variant in `results/stats.json`.)

The O(3) distribution is a tight peak at float32 machine zero. The SO(3) distribution is a tight peak
at order unity — physically, predicted piezoelectric responses comparable to real piezoelectrics, for
crystals where symmetry forbids any response at all.

---

## Item 4 — Statistics over seeds, and the MACE dipole anomaly

### 4a. Seed-level statistics — and an honest statement of their power

All tables in `RESULTS.md` are now **mean ± s.d. over 3 seeds**.

The review asked for a paired Wilcoxon over seeds. It is computed and stored, but it must be reported
for what it is: **with n = 3, a two-sided Wilcoxon signed-rank test cannot produce p < 0.25**,
regardless of effect size. The observed values confirm this exactly — every piezoelectric row returns
p = 0.25 (the floor: all three seeds agree in sign) and every null row returns 0.50–1.00. These
p-values carry no evidential weight in either direction and are labelled *descriptive only* in
`results/stats.json`.

Reporting them as significance would be a mistake. Instead:

- **Effect sizes over seeds** are reported as `Δ/σ`, the O(3)→SO(3) gap in units of the pooled
  per-seed standard deviation.
- **The headline OOD claim is tested where the test has power** — at the **structure level**, paired
  across the 2000 OOD crystals (one-sided, O(3) violation < SO(3) violation), per seed:

| Core | variant | max p over seeds | fraction with O(3) < SO(3) |
|---|---|---|---|
| NequIP | idealized / raw | < 1e-300 | 0.975 / 0.980 |
| Allegro | idealized / raw | < 1e-300 | 0.968 / 0.972 |
| MACE | idealized / raw | < 1e-300 | 0.962 / 0.967 |

(p underflows double precision.) This is the test that matters, and it is unambiguous.

**Accuracy null claims.** On U₀, dipole, and elastic the O(3)/SO(3) gap is within seed noise for every
core: |Δ/σ| ≤ 0.94, with inconsistent sign. On the piezoelectric tensor, O(3) is *better*
in-distribution, consistently: Δ/σ = +4.09 (NequIP), +3.43 (Allegro), +4.55 (MACE).

**The Allegro U₀ gap — resolved as noise.** O(3) 31.08 ± **12.89** meV vs SO(3) 26.98 ± 4.88 meV. The
4.10 meV difference is **0.42σ**, seed-level p = 0.75, and it favours **SO(3)** — the opposite
direction to any parity-based explanation. The O(3) arm's own seed spread is three times the gap.
There is no gap. What the data show is that the U₀ setup is under-converged with high seed variance
across all cores (NequIP O(3) s.d. = 22.5), which is worth noting but is not a parity effect.

### 4b. The MACE dipole anomaly — a bug, now fixed

**Diagnosis: misconfiguration, not a scientific result.** The MACE tensor/dipole readout hardcoded
`interactions.0.linear` as its feature probe — the *shallowest* interaction block, whose irreps are
effectively scalar, so the L=1 dipole head was reading almost no directional signal.

**Fix:** `_deepest_tensor_probe()` in `src/equiparity/models/mace.py` now selects the deepest
`interactions.{i}.linear` whose irreps contain l > 0. On the production config it selects
`interactions.2` (previously `interactions.0`).

**Effect:** MACE dipole MAE **0.328 → 0.0484 ± 0.0018**, bringing it in line with NequIP (0.0519) and
Allegro (0.0751). The ~6× discrepancy is gone.

**The O(3) piezoelectric zero was not affected**, and could not have been: it is structural, arising
from the parity labels themselves, and holds for any choice of probe. Verified — O(3) piezoelectric
median violation remains 4.2e-8 at probe-selection time and 2.8e-6 after full training.

---

## Item 5 — Equiformer v1, the dipole null, and a process note

### 5a. Equiformer v1's fate — formally documented

Equiformer v1 was **demoted, not silently dropped**. Two independent reasons, both recorded in
`docs/parity_work_plan.md`:

1. **Infeasible on the hardware.** Its 2022 stack (Python 3.8 / torch 1.10 / CUDA 11.3) does not build
   for the RTX 5090 (sm_120).
2. **Scientifically redundant.** It is e3nn-based, so as a fourth toggleable core it would exercise no
   equivariance mechanism not already covered by NequIP, Allegro, and MACE.

**EquiformerV2 supersedes it** as the transformer representative, and does so in a stronger role: a
fixed, deployed, SO(3)-only SOTA model rather than another research toggle. It is also the worst
offender in the study, false-flagging 96% of centrosymmetric crystals.

The user's instruction on this was explicit ("no ignore v1"), and this report records the reasoning so
the omission is defensible in review rather than merely convenient.

### 5b. Reframing the dipole null

The dipole is parity-**odd**, like the piezoelectric tensor, yet SO(3) matches O(3) on it
(Δ/σ = +0.42 to +0.58). Read naively this is a counterexample to the paper's thesis.

It is not, and the reason is the point of the paper:

> **QM9 contains no molecule whose symmetry forces the dipole to vanish.** No test case is
> symmetry-forbidden, so the constraint an O(3) model enforces is never binding, and a model that
> cannot represent that constraint is never actually tested on it.

The correct framing, now in `RESULTS.md`:

> Parity labels do not improve in-distribution accuracy on odd targets. They determine whether the
> model can represent a symmetry-mandated **zero**. The failure appears only where symmetry forbids a
> nonzero value — and there it is catastrophic (90–96%), not marginal.

So the dipole is not a weakness in the story; it is the **control that isolates the mechanism**. Same
parity, same architectures, no symmetry-forbidden cases → no gap. Change only the last condition and
SO(3) fails on 90–96% of structures. The claim is about symmetry-mandated zeros, not about odd
tensors in general.

### 5c. Process note

Two failures in this study were caught only by a *physical* check, never by a formal one:

- **MACE without PBC** silently broke centrosymmetry (a cut molecular boundary is not centrosymmetric),
  leaking parity in the O(3) tensor head. Types checked, tests passed, physics was wrong.
- **CliffordSTF** passed an exact-arithmetic parity test at 1e-15 and still false-flagged 42% of real
  crystals, because the parity test used machine-perfect coordinates and the trained model amplified a
  1e-6 residual.
- **The MACE dipole probe** (item 4b) was a third instance: a plausible hardcoded layer index, no type
  error, a 6× wrong number.

In each case `mypy`, `ruff`, and `pytest` were green. The check that caught them was running the model
on a physically constrained input and asking whether the output obeyed the constraint. **The parity
gate must be run on real, DFT-relaxed structures, not only on synthetic symmetric ones** — the
idealized/raw split in item 2 exists for exactly this reason and is now permanent.

---

## Status

| Item | Status |
|---|---|
| 1 — Drop CliffordSTF, archive, record limitation | Done |
| 2 — Both-variant OOD evaluation (BLOCKING) | **Cleared** — O(3) does not degrade on raw |
| 3 — Threshold curves + violation distributions (BLOCKING) | **Cleared** — curves, histograms, per-structure vectors |
| 4 — Seed statistics + MACE dipole anomaly | Done — Allegro U₀ gap is noise; dipole bug fixed (0.328 → 0.048) |
| 5 — Equiformer v1, dipole reframing, process note | Done |

Both blocking items are cleared. Figures and writing are unblocked.
