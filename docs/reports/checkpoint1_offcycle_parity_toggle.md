# Off-Cycle Report — Checkpoint 1: the SO(3) toggle does not toggle SO(3)

Triggered by the plan's standing rule: a result contradicts an "Expected" assumption in Task 0.2/0.3,
so downstream work is paused pending discussion. Date: 2026-07-04.

## 1. What was completed

- Repository scaffolded to CODING_RULES.md (`equiparity` package, uv, src layout, CI gate green).
- Environment installed and GPU-verified on the RTX 5090: two uv profiles (`.[nequip]`: nequip 0.18.0 +
  nequip-allegro 0.8.3 + e3nn 0.6.0; `.[mace]`: mace-torch 0.3.16 + e3nn 0.4.4).
- NequIP builder implemented and the parity toggle exercised (Task 0.3 checks 2 and 3: internal irreps
  and per-mode parameter counts).
- Reflection-based parity discriminator built and run (Task 0.3 check 1, done correctly — see item 4).

## 2. Key numbers / artifacts

Reproduce with `uv run --extra nequip python scripts/parity_toggle_probe.py`. Reflection-equivariance
error is `max|feat(Mx) - D(M).feat(x)|` at the second convolution layer, `M` a mirror reflection,
`D` e3nn's parity-aware representation of `M`. Near machine-zero => O(3) (parity is respected);
large => genuine parity violation (true SO(3)).

| Construction | Realized irreps | Refl. error | Verdict |
|---|---|---|---|
| preset `parity=True` (plan's O(3)) | `0e+1e+1o+2e+2o` | 7.4e-18 | O(3) |
| preset `parity=False` (plan's SO(3) toggle) | `0e+1o+2e` | 8.2e-18 | **O(3), not SO(3)** |
| full, natural-parity edge SH | `0e+1o+2e` | 8.2e-18 | O(3) |
| full, **all-even edge SH** | `0e+1e+2e` | **1.3e-2** | **genuine SO(3) (parity-violating)** |

Supporting facts already verified: the `parity` boolean lives only in the preset `NequIPGNNModel`
(the low-level `FullNequIPGNNModel` takes raw irreps); and the toggle changes nothing below 3 layers
(shallow-net no-op, locked in `tests/models/test_nequip.py`).

## 3. Deviations from the plan and why

- Task 0.2's parity-mode table specifies the SO(3) arm for NequIP/Allegro as `parity: false`. Evidence
  shows `parity: false` remains O(3)-equivariant to machine precision. It is not the SO(3) arm the study
  needs; on the piezoelectric OOD test it would produce the same symmetry-forced zeros as O(3), showing
  no effect and silently voiding the headline comparison.
- Root cause: e3nn tensor products always conserve parity. `parity: false` merely restricts to honest
  natural-parity irreps (`0e,1o,2e`) — lower capacity, same O(3) symmetry. Removing parity as a
  constraint (what EquiformerV2 does structurally via eSCN) requires relabelling the edge spherical
  harmonics themselves as all-even, which is only reachable through `FullNequIPGNNModel`, not the boolean.
- The pins were already corrected (nequip 0.18, nequip-allegro 0.8.3, e3nn per-profile) in a prior update.

## 4. Blockers / risks discovered

- The "cleanest boolean switch" premise for NequIP/Allegro is invalid. Allegro inherits the same flag,
  so it is presumed affected but not yet probed. MACE's plan spec (`0e+1e+2e`) evens the hidden features
  but its edge SH have not been checked; Equiformer v1's SE(3) mode is likewise unverified. Any core
  whose SO(3) config leaves the edge SH at natural parity is not actually SO(3).
- The committed NequIP builder (`src/equiparity/models/nequip.py`) maps SO(3) to `parity=False` and is
  therefore wrong for the SO(3) arm. Left unchanged pending this discussion.
- Positive result: the reflection-equivariance probe is a clean, machine-precision parity discriminator
  (O(3) ~1e-18 vs SO(3) ~1e-2). This is the correct Task 0.3 gate and the tool to audit every core.

## 5. Proposed next action

1. Redefine the SO(3) arm for all e3nn cores as all-even irreps **including the edge spherical
   harmonics** (`FullNequIPGNNModel`-style), and update Task 0.2/0.3.
2. Audit Allegro, MACE, and Equiformer v1 with the probe; record which plan configs are genuinely SO(3).
3. Rewrite the NequIP builder's SO(3) path and promote the probe into the package verification gate.
4. Only then resume the training pipeline (Task 1 onward).

Decision requested: approve the all-even-SH redefinition of the SO(3) arm, or discuss alternatives
before any code/plan change.
