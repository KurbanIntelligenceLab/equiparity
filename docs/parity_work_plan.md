# Implementation Plan: When Parity Matters

**Can SO(3)-equivariant models predict physically impossible material properties?**

Target venue: Nature Machine Intelligence | Backup: NeurIPS Datasets & Benchmarks
Compute: ~1,350–1,650 A100-hours total
Scope: **4 toggleable architectures × 2 parity modes × 3 datasets × 3 seeds, plus EquiformerV2 as a fixed SO(3)-only deployed representative on the headline experiment**

Everything in this document is required; nothing is optional. Optional extensions are listed at the end and should not be started until all core deliverables are complete.

---

## The One-Paragraph Goal

Centrosymmetric crystals cannot be piezoelectric — Neumann's principle forces the tensor to exact zero. O(3)-equivariant models (features carry parity labels) produce these zeros by construction; this is already acknowledged in the literature (EATGNN, npj Comput. Mater. 2025). What nobody has shown is the converse: that SO(3)-equivariant models (no parity labels) **fail** this test — predicting nonzero piezoelectric tensors for centrosymmetric crystals they have never seen. Demonstrating that failure, quantifying when parity matters (never for energies, somewhat for dipoles, categorically for odd-parity tensors), and turning it into a practitioner rule is the entire paper.

---

## Model Zoo Role Assignment

The available zoo is: DimeNet++, EquiformerV2, FAENet, GotenNet, ICTP, MACE, NequIP, PaiNN, SchNet, TorchMD-Net, ViSNet. The core ablation requires architectures whose internal features are irreps that can be built **with or without parity labels** — that is the experimental variable. Two models in the zoo satisfy this; two more toggleable architectures (Allegro, Equiformer v1) are added by installation to bring the core to four distinct families. EquiformerV2 plays a special fixed role. The rest are audit entries, not training targets.

| Model | Role | Reason |
|---|---|---|
| **NequIP** | Core (toggleable) | Equivariant convolutional MPNN. `parity: true/false` boolean — the cleanest O(3)/SO(3) switch in any released code. |
| **Allegro** (install: `mir-group/allegro`) | Core (toggleable) | Strictly local descriptors, no message passing — a genuinely different family. Inherits NequIP's `parity` flag, so the toggle is verified and near-zero engineering. |
| **MACE** | Core (toggleable) | Body-ordered MPNN. Irreps set via `--hidden_irreps` string: default alternating-parity vs manual all-even. |
| **Equiformer v1** (install: `atomicarchitects/equiformer`) | Core (toggleable) | Equivariant transformer. Natively supports both E(3) features (type-(L,p), with parity) and SE(3) features (type-L, no parity) — toggled via the irreps specification in configs. PBC support from the OC20 lineage. No boolean flag, so the 0.3 verification gate is critical here. |
| **EquiformerV2** | Fixed SO(3) representative (piezoelectric experiment only) | Built on SO(3) irreps with eSCN convolutions; carries no parity labels by design and cannot be switched to O(3) — the eSCN efficiency trick is defined on parity-free irreps. It is the field's dominant high-throughput lineage (OC20-scale), so its failure on the OOD test shows the finding applies to deployed practice, not just toggled research configs. Native PBC support. |
| DimeNet++, SchNet | Audit only | Invariant features; no equivariant tensor output, no parity concept to toggle. |
| PaiNN, TorchMD-Net, ViSNet | Audit only | Scalar + Cartesian-vector features (polar vectors only, no pseudovector channel); no irrep parity machinery to switch. |
| FAENet | Audit only | Frame-averaging equivariance; no irreps, no parity toggle. |
| GotenNet | Audit only | E(3)-equivariant via geometric tensor representations, explicitly built without irreps or Clebsch-Gordan transforms; no parity toggle. |
| ICTP | Audit only | Irreducible Cartesian tensors (MACE-like); parity exists in its math but there is no supported SO(3) switch, and modifying research code for it is out of scope. |

**No further models are needed.** Four toggleable cores spanning four architecture families (convolutional MPNN, local descriptors, body-ordered MPNN, transformer), one deployed SO(3) representative, and eight audit entries for the prevalence table. Do not train the audit-only models. The scope is fixed at what is listed here — if a reviewer later asks for more architectures, that is a revision task, not a first-submission task.

---

## Task 0: Setup and Verification

### 0.1 Environment

- Single repository. Pin the current NequIP/Allegro framework generation: `nequip` v0.18.x (the post-2025 rewrite with the `NequIPGNNModel` API — **not** the legacy v0.6.x this plan originally specified), `nequip-allegro` v0.4.x+ matching the installed `nequip` (the package is now published as `nequip-allegro`, superseding the old `allegro` v0.3.x), `mace-torch` v0.3.x, and `e3nn` at the version the installed `nequip` requires (the plan's `e3nn` v0.5.x pin is superseded). Clone Equiformer v1 (`github.com/atomicarchitects/equiformer`) at a fixed commit. Record exact resolved versions and commit hashes. The `parity` flag survives into the new NequIP API, so the O(3)/SO(3) toggle is unchanged; verify it empirically in Task 0.3 against the actually-installed build.
- One config generator: YAML template → {architecture × parity mode × dataset × seed} configs.
- Log everything to W&B (or equivalent): configs, parameter counts, training curves, final metrics.

### 0.2 The Two Parity Modes — CORRECTED (Checkpoint-1 decision)

**The original `parity: true/false` boolean toggle is wrong for the e3nn cores** and has been
replaced. See `docs/reports/checkpoint1_offcycle_parity_toggle.md`. The boolean `parity=False`
in NequIP/Allegro keeps *honest natural-parity* irreps (`0e,1o,2e`) and the model stays fully
O(3)-equivariant (reflection error ~1e-16) — it is NOT an SO(3) model. e3nn tensor products
always conserve parity; to remove parity you must relabel the **edge spherical harmonics** (and
hidden irreps) as **all-even**, which is what genuinely breaks reflection equivariance.

Both arms are built as a **matched pair** through the raw-irreps route, identical in every
hyperparameter (multiplicity, `l_max`, layers); the ONLY difference is the parity labeling of
the edge SH and hidden irreps.

| Mode | NequIP | Allegro | MACE |
|---|---|---|---|
| **O(3)** | `FullNequIPGNNModel`, natural SH `0e+1o+2e`, hidden `Nx0e+Nx1o+Nx2e` | `FullAllegroModel`, same | `use_so3=False`, hidden `Nx0e+Nx1o+Nx2e` |
| **SO(3)** | `FullNequIPGNNModel`, all-even SH `0e+1e+2e`, hidden `Nx0e+Nx1e+Nx2e` | `FullAllegroModel`, same | `use_so3=True` (evens SH, `p=1`), hidden `Nx0e+Nx1e+Nx2e` |

Only three toggleable cores remain. **Equiformer v1 is demoted to optional extensions** (its
2022 stack — Python 3.8 / torch 1.10 / CUDA 11.3 — cannot run on current hardware or coexist with
the environment; being e3nn-based it adds no new mechanism, per the gate's own fallback clause).
The three cores span three architecture families: convolutional MPNN (NequIP), local descriptors
(Allegro), body-ordered MPNN (MACE).

### 0.3 Parity Verification Gate — quantitative, runs before any training

The gate lives in `src/equiparity/verification/` and runs in CI (`tests/verification/`). For each
core it builds both matched arms, then on a toy structure applies a random proper rotation and a
random improper reflection and measures how far the model's internal equivariant features deviate
from the parity-aware prediction `features @ D(g).T`:

- **Rotation error** must be below threshold in BOTH arms (every model must be rotation-equivariant).
- **Reflection error** classifies the arm: small => O(3); large => genuine SO(3).

Thresholds (float64): O(3) reflection `< 1e-12`, SO(3) reflection `> 1e-4`; anything between, or a
failed rotation check, is a FAIL. MACE runs at ~1e-7 precision (its symmetric-contraction tensors
stay float32), so it uses the float32 thresholds (`< 1e-5` / `> 1e-2`). Parameter counts per arm are
recorded; the SO(3) arm is slightly larger (all-even opens more tensor-product paths).

**Gate: no experiment starts until all three cores pass in both arms.** Verified results are in
`docs/reports/checkpoint1_parity_gate.md`.

The end-to-end analogue — checking that a trained model's predicted *odd-parity tensor* transforms
with the correct sign under reflection (and is exactly zero for centrosymmetric inputs in O(3)) — is
deferred to the tensor head (Task 2.2/2.3), where it also serves as the EquiformerV2 positive control.

### 0.4 Prevalence Mini-Audit (no compute)

Classify 14 models — the zoo (DimeNet++, EquiformerV2, FAENet, GotenNet, ICTP, MACE, NequIP, PaiNN, SchNet, TorchMD-Net, ViSNet) plus Allegro, Equiformer v1, and MatTen — as parity-aware / SO(3)-only / vector-only / invariant, by inspecting released source code. Record commit hashes for every classification. Output: one table for the paper's introduction. Do not expand beyond these 14 models.

---

## Task 1: Data Pipelines

### QM9 (control + vector signal)
- Source: MoleculeNet. Split 110k/10k/~10.8k.
- Targets: **U₀** (parity-even scalar, the control) and **dipole μ** (parity-odd vector, the first signal).
- **Dipole must use a direct equivariant L=1 output head — not charge × position.** The μ = Σ qᵢrᵢ route produces correct odd parity automatically (0e × 1o = 1o) and would mask the SO(3)/O(3) difference. This design choice must be stated explicitly in the paper.

### MP Elastic (~10k, parity-even tensor control)
- Materials Project API. Rank-4 elasticity tensor, Voigt notation. 80/10/10 random split.
- Decomposition (for the output head): 2×0e ⊕ 2×2e ⊕ 1×4e = 21 components, all even.

### MP Piezoelectric + OOD evaluation set (the headline)
- Fetch MP piezoelectric dataset via API: expect **~3,300 entries**, all non-centrosymmetric, all with full DFPT 3×6 tensors. If the API returns far fewer, the query is wrong (old documentation cites ~941 — outdated). Fallback source: JARVIS-DFT (5,015 entries).
- Decomposition (for the output head): 2×1o ⊕ 1×2o ⊕ 1×3o = 18 components, all odd.
- Build the **OOD evaluation set**: ~2,000 centrosymmetric insulators from the broader MP database. Filters: centrosymmetric space group (2, 10–15, 47–74, 83–88, 123–142, 147–148, 162–167, 175–176, 191–194, 200–206, 221–230), bandgap > 0.1 eV. These crystals never appear in training. Their true piezoelectric tensor is exactly zero by symmetry — no labels needed.
- Verify space-group assignment with spglib on every OOD structure. A single non-centrosymmetric structure leaking into the eval set invalidates the headline figure.

---

## Task 2: Experiments

All experiments: 4 toggleable architectures (NequIP, Allegro, MACE, Equiformer v1) × 2 parity modes × 3 seeds. EquiformerV2 joins experiment 2.3 only, in the SO(3) column only.

### 2.1 QM9 (48 runs, ~480 A100-hrs)
- Train separately on U₀ and on dipole μ (24 runs each).
- Metrics: U₀ MAE (meV), dipole MAE (Debye).
- Expected: no gap on U₀ (this is the sanity check — a gap here means the comparison is broken); O(3) measurably better on dipole.

### 2.2 MP Elastic (24 runs, ~480 A100-hrs)
- Tensor output head: adapt the MatTen codebase (`github.com/wengroup/matten`, NequIP-compatible backbone with elastic output already working). Implementation order: NequIP first, then Allegro (same framework, near-trivial port), then MACE, then Equiformer v1 (separate codebase, budget the most porting effort here).
- Metrics: component MAE (GPa), Frobenius error, symmetry compliance rate (% of predictions respecting the crystal point group).
- Expected: small or no MAE gap (target is parity-even); possibly better symmetry compliance for O(3).

### 2.3 Piezoelectric OOD (27 runs, ~300 A100-hrs) — THE HEADLINE
- Train on the ~3,300 non-centrosymmetric MP structures only. No zero tensors exist anywhere in training.
- Evaluate on the ~2,000-crystal centrosymmetric OOD set.
- Runs: all four toggleable architectures in both parity modes (24 runs) **plus EquiformerV2 × 3 seeds (3 runs, SO(3) column only — it has no O(3) mode, which is the point)**. EquiformerV2 needs the tensor output head ported to its irreps features (L = 1, 2, 3 components, no parity labels); its OC20 lineage handles PBC natively, and the Equiformer v1 head from 2.2 is the starting point.
- Metrics:
  1. Component MAE (C/m²) on a held-out non-centrosymmetric test split (standard regression quality — proves the models are competent, not broken).
  2. **Violation magnitude:** ||predicted tensor||_F per centrosymmetric structure. O(3) must give exact zeros (machine precision); verify this, it doubles as a correctness check.
  3. **False-flag fraction:** % of centrosymmetric structures predicted above a materials-relevance threshold. Report the full fraction-vs-threshold curve, not one number.
  4. **Named examples:** pick 2–3 familiar centrosymmetric materials (e.g., a rocksalt-structure compound) and record the SO(3) model's predicted magnitude for each. These go in the headline figure and abstract.

### 2.4 Parameter-Matched Ablation (~9 runs, ~120 A100-hrs)
- NequIP only. Reduce SO(3) channel count until parameters ≈ O(3) count. Repeat on all three datasets, 3 seeds.
- Purpose: kills the "unfair parameter count" objection. One architecture is sufficient for this; the other three are not needed.

**Run order: 2.1 → 2.2 → 2.3 → 2.4.** The QM9 U₀ null result must be confirmed before spending compute on tensors.

---

## Task 3: Analysis and Figures

### Analyses
- Master table: architecture × parity mode × dataset, mean ± std over seeds. Wilcoxon signed-rank across seeds where a gap is claimed.
- Violation-magnitude histogram for SO(3) models on the OOD set (the signature visualization).
- Wall-clock and memory per parity mode (SO(3) is cheaper — this feeds the decision rule).

### Four Main-Text Display Items
1. **Fig 1 — Concept.** Parity labels in irreps + Neumann's principle: rank-3 tensor under inversion gains (−1)³ = −1, so centrosymmetric ⇒ zero. Plus the prevalence mini-audit table.
2. **Fig 2 — The parity sensitivity spectrum.** SO(3)−O(3) gap for the four targets in order: U₀ (even scalar, no gap) → dipole (odd vector, moderate) → elastic (even tensor, no/small gap) → piezoelectric (odd tensor, categorical failure). Note the pattern follows parity character, not tensor rank — elastic is rank-4 and shows no gap.
3. **Fig 3 — The smoking gun.** Frobenius norms on the centrosymmetric OOD set: O(3) at exact zero, SO(3) scattered nonzero, named materials annotated. Include the EquiformerV2 column alongside the four toggled architectures — it shows the failure holds for the field's deployed high-throughput lineage, not just toggled research configs. Inset: false-flag fraction vs threshold.
4. **Fig 4 — Decision rule.** Parity-even target → SO(3) is sufficient and cheaper. Parity-odd target → O(3) is required; no amount of data substitutes for the architectural guarantee.

### Paper skeleton (short version)
- Abstract: lead with the impossible-prediction finding and the false-flag number.
- Results in spectrum order: QM9 → elastic → piezoelectric OOD (the centerpiece).
- Discussion thesis: physical law built into architecture holds everywhere; physical law learned from data holds only where the data reached.
- Methods: derivations (both tensor decompositions), OOD set construction, all hyperparameters.

---

## Deliverables Checklist

- [x] Repository with pinned environment and one-command reproduction (`equiparity run <config>`); config *generator* for the full grid still to add
- [x] Parity verification report (Task 0.3) — `docs/reports/checkpoint1_parity_gate.md`, **three** cores (Equiformer v1 demoted, see below); gate in CI
- [ ] Prevalence audit table (14 models) with commit-hash evidence — not started
- [x] OOD evaluation set + construction script (`scripts/prepare_mp.py`) — 2000 structures, spglib-verified, 0 leaks
- [ ] All training runs logged with final metrics — **infrastructure built and smoke-verified locally; full-scale A100 runs pending**
- [ ] Master results table + the four figures — not started
- [ ] Trained checkpoints for the piezoelectric experiment — pending full training
- [ ] Draft results section text (numbers filled in) — pending full runs

## Go/No-Go Gates

| Gate | Condition | Status | If it fails |
|---|---|---|---|
| After Task 0 | All toggleable architectures pass parity verification | **PASSED** for three cores (Equiformer v1 demoted per the fallback clause). | Debug before anything else. For MACE, manual irreps strings in mainline are the fallback. For Equiformer v1, if the irreps toggle cannot be verified after reasonable effort, demote it to optional extensions and proceed with three cores — do not let it block the pipeline. |
| After Task 1 | MP returns ~3,300 piezo entries; OOD set spglib-verified | **PASSED** (3,322 piezo entries; OOD 0 leaks). | Fix the API query; fallback to JARVIS-DFT. |
| After 2.1 (U₀) | SO(3) ≈ O(3) on U₀ | **PASSED (smoke)** — 342 vs 352 eV; confirm at full scale. | Comparison is broken — find the confound before training tensors. |
| After 2.3 | SO(3) nonzero on centrosymmetric OOD, O(3) exact zero | **PASSED (smoke)** — O(3) ~1e-15, SO(3) 70% false-flag; confirm at full scale. | If SO(3) is also ≈ zero: verify with larger violations threshold and per-component analysis; if genuinely null, report as negative result — still publishable with reframing. Escalate before writing. |

## Progress Reporting Checkpoints

Ten checkpoints, one per completed step. At each checkpoint the student prepares a short report using the template below, presents it for peer discussion, and does not proceed past a decision point until the outcome is agreed. Checkpoints are milestone-based, not time-based. The Go/No-Go gates above are technical pass/fail conditions; these checkpoints are the discussion and decision layer on top of them.

**Report template (every checkpoint, one page maximum):**
1. What was completed (bullet list against the plan)
2. Key numbers/artifacts (tables, figures, logs — attached, not described)
3. Deviations from the plan and why
4. Blockers or risks discovered
5. Proposed next action

### Checkpoint status (implementation to date)

The experimental *machinery* for every target and core is built, smoke-verified on the local
RTX 5090, and committed. The remaining work is *scale* — full-size, multi-seed runs on the A100
cluster to produce final numbers — plus the prevalence audit, param-matched ablation, and figures.

| # | Checkpoint | Status |
|---|---|---|
| 1 | Environment + parity verification | **Done, with a correction.** Off-cycle escalation (`docs/reports/checkpoint1_offcycle_parity_toggle.md`) found the `parity` boolean is NOT the SO(3) toggle; adopted all-even-SH matched pairs (advisor-approved). Verified for NequIP, Allegro, MACE (`checkpoint1_parity_gate.md`). Equiformer v1 **demoted** (2022 stack, incompatible with the RTX 5090; e3nn-based so adds no new mechanism). EquiformerV2 deferred to output-level (2.3). Three cores. |
| 2 | Prevalence audit (Task 0.4) | **Not started.** |
| 3 | Data pipelines | **Done.** QM9 (130,831), MP elastic (13,080, garbage-filtered), MP piezo (3,312 — matches ~3.3k), OOD (2,000, spglib-airtight, 0 leaks re-verified). Manifests + splits committed. |
| 4 | QM9 U₀ control | **Infrastructure done + smoke-verified.** O(3)≈SO(3) (342 vs 352 eV on a small run) — the null result holds. Full 3-seed runs pending. |
| 5 | QM9 dipole | **Infrastructure done + smoke-verified.** O(3) 0.27 vs SO(3) 0.28 D; O(3) reflection-correct (`1o`), SO(3) not (`1e`) — the behavioral signal, in a test. Full runs pending. |
| 6 | Elastic tensor | **Infrastructure done + smoke-verified.** O(3) 40.4 vs SO(3) 38.2 GPa — no gap (even tensor), as expected. Full runs pending. |
| 7 | Piezoelectric OOD | **Infrastructure done + smoke-verified — the headline.** Trained: O(3) OOD violation ~1e-15 (exact zero), SO(3) false-flags 70%; both equally competent on the non-centrosymmetric test. EquiformerV2 column and full runs pending. |
| 8 | Parameter-matched ablation | **Not started** (param gap is small and SO(3)-larger; matched runs still to do). |
| 9 | Analysis + figures | **Not started.** |
| 10 | Final package | **Not started.** |

| # | Checkpoint | Trigger | Present to peers | Discussion questions | Possible outcomes |
|---|---|---|---|---|---|
| 1 | Environment + parity verification | Task 0.1–0.3 complete | Verification report: reflection test results for all four toggleable architectures in both modes, internal irrep inspection, parameter counts per mode | Do all four toggles behave exactly as intended? Is the parameter gap between modes acceptable or does it change the ablation design? Should Equiformer v1 be demoted if its toggle resists verification? | Proceed to data; or debug; or fall back to manual MACE irreps strings; or drop to three cores |
| 2 | Prevalence audit | Task 0.4 complete | 14-model classification table with commit-hash evidence | Does the field-level story hold (SO(3)/vector-only models are common)? Does any classification look contestable? | Lock the introduction framing; or adjust motivation before writing anything |
| 3 | Data pipelines | Task 1 complete | Dataset statistics: MP piezo entry count, OOD set size and space-group distribution, spglib verification log, QM9/elastic split summaries | Is the OOD set airtight (zero non-centrosymmetric leakage)? Are dataset sizes as expected? | Approve datasets and freeze splits; or fix API queries / fall back to JARVIS-DFT |
| 4 | QM9 U₀ control | First half of 2.1 done | U₀ MAE table, O(3) vs SO(3), all four architectures, 3 seeds | Is SO(3) ≈ O(3) within seed noise? If not, what is the confound? | Comparison validated, continue; or halt all tensor work and debug the setup |
| 5 | QM9 dipole | 2.1 complete | Dipole MAE table + gap with confidence intervals | Is the O(3) advantage statistically meaningful? Is the direct-L=1-head choice defensible as implemented? | Record the vector data point; decide whether effect size changes the paper's emphasis |
| 6 | Elastic tensor | 2.2 complete | Component MAE, Frobenius error, symmetry compliance rates | Does the parity-even prediction hold (no/small gap)? Any symmetry-compliance difference worth highlighting? | Confirm the spectrum's even-tensor point; flag anything unexpected before the headline run |
| 7 | Piezoelectric OOD | 2.3 complete | Violation-magnitude distributions, false-flag vs threshold curve, EquiformerV2 column, exact-zero verification for O(3), candidate named materials | Is the headline result clean? Which named materials and which threshold to feature? Is the EquiformerV2 result consistent with toggled SO(3)? | Lock the headline figure; or run the null-result escalation path from the gates table |
| 8 | Parameter-matched ablation | 2.4 complete | Matched-parameter results across all three datasets | Does the finding survive parameter matching? | Objection closed; or add matched runs for MACE before submission |
| 9 | Analysis + figures | Task 3 complete | Master results table, all four figures in near-final form, statistical tests | Does each figure carry its intended message to a non-specialist? Is the spectrum figure convincing? | Approve figures; or iterate on specific panels with concrete feedback |
| 10 | Final package | Deliverables checklist fully checked | Repository, reproduction commands, checkpoints, draft results text | Is everything reproducible from a clean environment? Ready for advisor review and paper writing? | Hand off to writing; or close remaining checklist items |

**Standing rule:** any result that contradicts an "Expected" statement in Task 2 triggers an immediate off-cycle report using the same template — do not wait for the next checkpoint, and do not spend further compute on downstream experiments until the discrepancy is discussed.

---

## Optional Extensions (only after every checkbox above is done)

1. Full-O(3) parity mode on the piezoelectric experiment only.
2. EquiformerV2 on QM9 dipole (3 runs, ~30 A100-hrs) — adds the deployed-representative data point to the vector experiment.
3. 5 seeds instead of 3 on the piezoelectric OOD runs.
4. Parameter-matched ablation on a second architecture (MACE) if reviewers request it.

---

## Compute Budget

| Experiment | Runs | Hrs/Run | A100-hrs |
|---|---|---|---|
| QM9 (U₀ + μ) — 4 toggleable architectures | 48 | ~10 | ~480 |
| MP Elastic — 4 toggleable architectures | 24 | ~20 | ~480 |
| Piezo OOD — 4 toggleable architectures | 24 | ~10 | ~240 |
| Piezo OOD — EquiformerV2 (SO(3) only) | 3 | ~15 | ~45 |
| Param-matched (NequIP) | 9 | ~13 | ~120 |
| **TOTAL** | **108** | | **~1,365** |

Budget ~1,350–1,650 A100-hours with margin. Crystal runs and Equiformer v1 (transformer, heavier per step) are the bottlenecks.
