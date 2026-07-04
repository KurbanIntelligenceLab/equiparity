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

### 0.2 The Two Parity Modes

| Mode | NequIP | Allegro | MACE | Equiformer v1 |
|---|---|---|---|---|
| **O(3) (default)** | `parity: true` | `parity: true` | default `--hidden_irreps='128x0e + 128x1o + 128x2e'` | type-(L,p) irreps in config, e.g. `128x0e + 128x1o + 128x2e` |
| **SO(3)** | `parity: false` | `parity: false` | `--hidden_irreps='128x0e + 128x1e + 128x2e'` (all-even, manual string) | type-L irreps without parity labels (SE(3) mode, all-even equivalent) |

### 0.3 Parity Toggle Verification — do this before any training run

On 5–10 toy structures:
1. Reflect all coordinates (x → −x). O(3)-mode scalar outputs must be identical; SO(3)-mode outputs may differ. Confirm both behaviors.
2. Inspect internal feature irreps and confirm they match the intended mode.
3. Record parameter counts per mode at identical channel counts — this documents the parameter confound exactly.

**Gate: no experiment starts until all four architectures pass all three checks in both modes.** NequIP and Allegro share the flag, so their verification is nearly free. MACE and especially Equiformer v1 (no boolean flag, irreps-string toggle) are where bugs will hide.

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

- [ ] Repository with pinned environment, config generator, and one-command reproduction per experiment
- [ ] Parity verification report (Task 0.3 outputs, all four toggleable architectures)
- [ ] Prevalence audit table (14 models) with commit-hash evidence
- [ ] OOD evaluation set + construction script (reproducible from MP API)
- [ ] All 108 training runs logged with final metrics
- [ ] Master results table + the four figures
- [ ] Trained checkpoints for the piezoelectric experiment
- [ ] Draft results section text (numbers filled in)

## Go/No-Go Gates

| Gate | Condition | If it fails |
|---|---|---|
| After Task 0 | All four toggleable architectures pass parity verification | Debug before anything else. For MACE, manual irreps strings in mainline are the fallback. For Equiformer v1, if the irreps toggle cannot be verified after reasonable effort, demote it to optional extensions and proceed with three cores — do not let it block the pipeline. |
| After Task 1 | MP returns ~3,300 piezo entries; OOD set spglib-verified | Fix the API query; fallback to JARVIS-DFT. |
| After 2.1 (U₀) | SO(3) ≈ O(3) on U₀ | Comparison is broken — find the confound before training tensors. |
| After 2.3 | SO(3) nonzero on centrosymmetric OOD, O(3) exact zero | If SO(3) is also ≈ zero: verify with larger violations threshold and per-component analysis; if genuinely null, report as negative result — still publishable with reframing. Escalate before writing. |

## Progress Reporting Checkpoints

Ten checkpoints, one per completed step. At each checkpoint the student prepares a short report using the template below, presents it for peer discussion, and does not proceed past a decision point until the outcome is agreed. Checkpoints are milestone-based, not time-based. The Go/No-Go gates above are technical pass/fail conditions; these checkpoints are the discussion and decision layer on top of them.

**Report template (every checkpoint, one page maximum):**
1. What was completed (bullet list against the plan)
2. Key numbers/artifacts (tables, figures, logs — attached, not described)
3. Deviations from the plan and why
4. Blockers or risks discovered
5. Proposed next action

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
