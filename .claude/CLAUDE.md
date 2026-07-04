# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT RULES — read and follow [RULES.md](RULES.md) before doing anything else.**
**CODING RULES — read and follow [CODING_RULES.md](CODING_RULES.md) before writing or modifying code.** CODING_RULES.md is authoritative for layout, tooling, typing, and reproducibility; this file follows it and supplies the project-specific bindings it refers to.

## Project

One study, one repo: **When Parity Matters — can SO(3)-equivariant models predict physically impossible material properties?** The scientific plan, scope, experiments, gates, and checkpoints live in [`docs/parity_work_plan.md`](../docs/parity_work_plan.md) and are the source of truth for what to build. Everything in that plan is required; nothing is optional until the core deliverables are complete.

The finding: O(3)-equivariant models (features carry parity labels) produce exact zeros for piezoelectric tensors of centrosymmetric crystals by construction; SO(3)-equivariant models (no parity labels) fail — predicting nonzero tensors for symmetry-forbidden cases. The core ablation is **4 toggleable architectures × 2 parity modes × 3 datasets × 3 seeds**, plus EquiformerV2 as a fixed SO(3)-only representative on the headline piezoelectric experiment.

Target venue: Nature Machine Intelligence (backup: NeurIPS Datasets & Benchmarks).

## Package name

The package is **`equiparity`**. Everywhere CODING_RULES.md writes `<package>`, substitute `equiparity`:

- Importable code lives only in `src/equiparity/`.
- Tests mirror it under `tests/`.
- `pyproject.toml` `name = "equiparity"`.

CI type-checking targets (from CODING_RULES.md §B, substituted):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/equiparity/domain src/equiparity/io src/equiparity/data src/equiparity/reproducibility.py
uv run pytest
```

## Environment

`pyproject.toml` + `uv.lock`, `requires-python = "==3.12.*"` (paper-reproduction repo, exact pinning allowed). Use `uv sync` and `uv run <cmd>`. No `pip install`, `requirements.txt`, or ad hoc conda.

### Two conflicting install profiles

MACE pins `e3nn==0.4.4`; NequIP/Allegro require `e3nn>=0.6`. They **cannot** share one environment. `pyproject.toml` declares them as **uv conflicting extras** — one repo, one `uv.lock`, two mutually-exclusive install profiles over a shared base:

```bash
uv sync --extra nequip   # nequip + nequip-allegro (e3nn 0.6)
uv sync --extra mace      # mace-torch (e3nn 0.4.4)
uv sync --extra jarvis    # + JARVIS-DFT fallback source (composes with either)
```

Every run's manifest must record which profile it used. `equiparity` package code stays profile-agnostic; only the model wrappers are profile-specific.

### Resolved and verified pins (installed and GPU-tested on the RTX 5090)

| Dependency | Resolved | Profile | Role / correction |
|---|---|---|---|
| `torch` | `2.11.0+cu128` (from the `pytorch-cu128` index) | shared base | Backend; needed for RTX 5090 sm_120. Seed control per CODING_RULES.md §E.1 |
| `nequip` | `0.18.0` — post-2025 rewrite | `nequip` | Core toggleable. **Plan's 0.6.x superseded.** Boolean `parity` survives only in the preset `NequIPGNNModel`; the low-level `FullNequIPGNNModel` takes irreps strings instead |
| `nequip-allegro` | `0.8.3` | `nequip` | Core toggleable; `AllegroModel` exposes `parity` + `l_max`. **Plan's `allegro 0.3.x` superseded** (package renamed `nequip-allegro`) |
| `e3nn` | `0.6.0` (nequip) / `0.4.4` (mace) | split | Irreps/parity substrate. **Plan's single 0.5.x pin is wrong** — the version differs per profile |
| `mace-torch` | `0.3.16` | `mace` | Core toggleable (irreps string: alternating-parity vs all-even); imports and runs on torch 2.11 |
| Equiformer v1 (`atomicarchitects/equiformer`) | fixed commit (TBD) | own profile | Core toggleable (type-(L,p) vs type-L irreps) — not yet installed |
| EquiformerV2 | vendored / fixed commit (TBD) | own profile | Fixed SO(3)-only representative, piezoelectric experiment only — not yet installed |

Proven reference: `~/agents-mlip` runs NequIP (parity flag), MACE, and EquiformerV2 on this hardware. Selected wrappers are ported into `src/equiparity/`, re-verified against these resolved pins — not trusted as-is. Note the agents-mlip wrapper targets an older NequIP API, so the preset-vs-full builder split above must be handled during the port.

Data/materials dependencies (shared base, verified): `pymatgen 2026.5.4`, `mp-api 0.46.4`, `spglib 2.7.0` (space-group verification — load-bearing for the OOD set), `ase 3.29.0`; `jarvis-tools` via the `jarvis` extra. Prefer `pyarrow`/Parquet for tables.

Equiformer v1 is a cloned research codebase at a fixed commit, not a PyPI package — treat its integration as a boundary and keep our code on our side of it.

## Datasets and splits

Raw data is external; commit only manifests under `data/manifests/` and splits under `data/splits/` (CODING_RULES.md §F). Bindings for this study:

- **QM9** (MoleculeNet): targets **U₀** (parity-even scalar control) and **dipole μ** (parity-odd vector). Split 110k/10k/~10.8k. Dipole uses a **direct equivariant L=1 head, never charge × position** — this is a scientific choice, not an implementation detail.
- **MP Elastic** (~10k): rank-4 elasticity tensor, Voigt. 80/10/10 random split. Output head decomposition 2×0e ⊕ 2×2e ⊕ 1×4e (21 even components).
- **MP Piezoelectric** (~3,300 non-centrosymmetric, DFPT 3×6): the headline training set. Output head 2×1o ⊕ 1×2o ⊕ 1×3o (18 odd components). If the API returns far fewer, the query is wrong (~941 is outdated). Fallback: **JARVIS-DFT** (5,015 entries).
- **OOD evaluation set** (~2,000 centrosymmetric insulators from MP): true piezoelectric tensor is exactly zero by symmetry, no labels needed. Every structure's space group is verified with spglib before use — a single non-centrosymmetric leak invalidates the headline figure.

Splits are generated once, saved, and referenced by ID in every result table. Never tune on test.

## Reproducibility and provenance

Follow CODING_RULES.md §E in full. Per-experiment `outputs/<experiment_id>/manifest.json` is the CI-enforced source of truth; experiment ID is `<short_git_sha>_<config_hash_prefix>_<utc_timestamp>`. Final-result runs fail on dirty git. Scientific parameters live in versioned YAML under `configs/`, loaded once at the entrypoint and validated into frozen dataclasses — one config generator expands the {architecture × parity mode × dataset × seed} grid from a template.

**No experiment tracker is a required dependency.** Log to files and manifests. A tracker may be added later for live monitoring only, and must never become part of the reproduction path.

## Hardware

Training targets NVIDIA A100 GPUs; full grid budget ~1,350–1,650 A100-hours (108 runs). Crystal runs and Equiformer v1 (transformer) are the per-step bottlenecks. Record `gpu_model`, `cuda_version`, and `driver_version` in every manifest. GPU tests are marked and skipped when no GPU is present.

## Gates before compute

From the work plan — do not skip:

- **Parity toggle verification (Task 0.3)** on all four toggleable architectures in both modes, before any training: reflection test, internal irrep inspection, per-mode parameter counts. NequIP/Allegro share the flag; MACE and Equiformer v1 (irreps-string toggle, no boolean) are where bugs hide.
- **QM9 U₀ null result** must be confirmed (SO(3) ≈ O(3)) before spending compute on any tensor experiment. A gap there means the comparison is broken.
- Run order is fixed: 2.1 → 2.2 → 2.3 → 2.4.
