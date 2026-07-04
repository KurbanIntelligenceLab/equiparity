# Checkpoint 1 — Parity Verification Gate (complete)

Closes the Checkpoint-1 off-cycle escalation (`checkpoint1_offcycle_parity_toggle.md`) after the
advisor approved the all-even-SH redefinition of the SO(3) arm. Date: 2026-07-04.

## 1. What was completed

- Redefined the O(3)/SO(3) toggle as a **matched pair**: both arms built through the raw-irreps
  route, identical in every hyperparameter, differing ONLY in the parity labeling of the edge
  spherical harmonics and hidden irreps (Task 0.2, corrected).
- Built a **quantitative equivariance gate** (`src/equiparity/verification/`) and wired it into CI
  (Task 0.3, corrected).
- Audited, formalized, and tested all three toggleable cores (NequIP, Allegro, MACE).
- Demoted Equiformer v1; deferred EquiformerV2 to the output-level check (Task 2.3).

## 2. Key numbers / artifacts

Reproduce: `uv run --extra nequip python scripts/parity_audit.py` and `uv run --extra mace python
scripts/parity_audit.py`. Errors are `max|feat(g·x) - D(g)·feat(x)|` at an intermediate equivariant
layer; float64 except MACE (~1e-7, float32 thresholds).

| Core | Arm | irreps | rotation err | reflection err | params | verdict |
|---|---|---|---|---|---|---|
| NequIP | O(3) | `16x0e+16x1o+16x2e` | 4.7e-16 | 4.7e-16 | 42,304 | O3 |
| NequIP | SO(3) | `16x0e+16x1e+16x2e` | 4.7e-16 | 9.7e-03 | 45,376 | SO3 |
| Allegro | O(3) | `1x0e+1x1o+1x2e` | 5.4e-15 | 3.3e-15 | 12,608 | O3 |
| Allegro | SO(3) | `1x0e+1x1e+1x2e` | 9.3e-15 | 4.9e+00 | 12,640 | SO3 |
| MACE | O(3) | `16x0e+16x1o+16x2e` | 2.4e-07 | 9.4e-08 | 43,280 | O3 |
| MACE | SO(3) | `16x0e+16x1e+16x2e` | 2.4e-07 | 1.3e+00 | 50,512 | SO3 |

Every core: rotation preserved in both arms; only the SO(3) arm breaks reflection. Parameter
counts near-matched, SO(3) slightly larger (all-even labeling opens more tensor-product paths).

Positive control for the probe: a hand-built all-even construction gives reflection error ~1e-2
(genuine SO(3)) while the natural-parity build gives ~1e-16 — the probe distinguishes them cleanly.

## 3. Mechanism per core

- **NequIP / Allegro**: the preset `parity` boolean is NOT an SO(3) toggle (both settings stay
  O(3)-equivariant; source builds natural-parity SH unconditionally). The SO(3) arm requires the
  raw route (`FullNequIPGNNModel` / `FullAllegroModel`) with all-even edge SH.
- **MACE**: exposes a correct native toggle, `use_so3=True`, which builds all-even SH (`p=1`). No
  patch needed. Runs at ~1e-7 precision, so the gate uses the float32 thresholds for it.

## 4. Blockers / risks

- Equiformer v1 demoted (incompatible 2022 stack; e3nn-based so no new mechanism).
- EquiformerV2 positive control deferred to the output-level test with the tensor head (Task 2.3);
  its eSCN features are not parity-labeled irreps, so the internal-feature probe does not apply.
- MACE precision caveat is documented and encoded in the gate thresholds.

## 5. Next action

Resume the pipeline: Task 1 (data) then Task 2.1 (QM9 U0 control), building on the three verified
matched-pair cores. All Task-0 gates pass.
