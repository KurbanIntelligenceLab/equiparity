# CODING_RULES.md

Rules for Python research repositories in AI for materials science. These rules are CI-enforced unless marked as guidance. Project-specific bindings such as package name, dependency list, dataset names, file formats, cluster details, and paper metadata live in `CLAUDE.md`.

Wherever this file says `<package>`, substitute the package name from `CLAUDE.md`.

---

## 0. Principles

1. **One coherent study per repo.** A paper repo contains one scientific study. Shared code may stay only while it directly supports the paper; if reused across projects, move it to a real package.
2. **Boring code over clever code.** Prefer explicit types, small modules, simple composition, and direct control flow.
3. **Functional core, object-oriented shell.** Use pure functions for stateless numerical logic. Use classes for state, configuration, workflows, I/O, and domain objects.
4. **Reproducibility is a feature.** Every reported output traces to a commit, environment, config, seed, data manifest, split manifest, and hardware/software context.
5. **Validate at boundaries.** Raw files, API responses, CLI input, YAML, JSON, and environment variables are untyped. Convert them immediately into typed objects.
6. **No hidden science.** Scientific choices live in versioned configs, manifests, tests, and documented methods, not notebooks or ad hoc scripts.

---

## A. Repository layout

Use the `src/` layout.

```text
repo/
  .claude/CLAUDE.md
  pyproject.toml
  uv.lock
  README.md
  LICENSE
  CITATION.cff
  CHANGELOG.md
  src/<package>/
  tests/
  configs/
  data/manifests/
  data/splits/
  scripts/
  notebooks/
  outputs/
```

Rules:

- `src/<package>/` is the only location for importable package code.
- `scripts/` may call the package but must not contain scientific logic.
- `notebooks/` are exploratory only. Anything important moves into `src/` with tests.
- `tests/` mirrors `src/<package>/`.
- Raw datasets, checkpoints, trajectories, and large binary artifacts are never committed.
- No `utils.py`, `helpers.py`, or catch-all modules.

Recommended package structure:

```text
src/<package>/
  domain/
  io/
  data/
  features/
  models/
  training/
  evaluation/
  workflows/
  reproducibility.py
  logging_config.py
  cli.py
```

---

## B. Environment and tooling

Use `pyproject.toml` plus `uv.lock`.

```bash
uv init
uv add <runtime-deps>
uv add --dev ruff mypy pytest pytest-cov hypothesis pre-commit
uv sync
uv run pytest
```

Rules:

- Commit `uv.lock`.
- Use `uv sync` to create/update the environment.
- Use `uv run <command>` in docs, scripts, and CI.
- Do not use `pip install`, `requirements.txt`, or ad hoc conda environments inside this repo.
- If a dependency requires conda-forge, document why in `CLAUDE.md` and use `pixi` only with lead approval.
- For paper reproduction repos, exact Python pinning is allowed: `requires-python = "==3.12.*"`.
- For reusable packages, use a supported-version policy instead of exact pinning.
- Pre-commit must run ruff check, ruff format, and mypy.

CI must run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/<package>/domain src/<package>/io src/<package>/data src/<package>/reproducibility.py
uv run pytest
```

---

## C. Code organization

### C.1 Use functions and classes for the right jobs

Allowed module-level functions:

- stateless numerical kernels
- metrics
- transforms
- validators
- featurization primitives
- plotting primitives
- small pure helpers local to a focused module

Use classes for:

- objects with state or lifecycle
- domain entities
- config objects
- dataset loaders
- model trainers
- workflow runners
- provenance writers
- external API clients
- resource-owning objects

Do not create static-method classes just to satisfy OOP. A pure `rmse(...)` function is better than `MathUtils.rmse(...)`.

### C.2 Type everything

Rules:

- Every `.py` file starts with `from __future__ import annotations`.
- Every public function, method, and class attribute is typed.
- No bare `list`, `dict`, `tuple`, or `set`; use `list[Record]`, `dict[str, float]`, etc.
- Avoid `Any`. If required at I/O boundaries, convert immediately into typed objects.
- Use `# type: ignore[specific-error]` only with a specific error code and a reason.

Type checking is enforced as a two-tier policy. Boundary and contract code is held to `mypy --strict`; this is the CI-enforced set and covers `domain/`, `io/`, `data/`, and `reproducibility.py`. Numerical, featurization, and plotting code (`features/`, plotting primitives, and similar modules built on libraries with weak or missing stubs such as `pymatgen`, `ase`, `matminer`, and `scipy`) is held to a relaxed mypy profile rather than strict mode. The boundaries carry the typing contract; the numerical interior is not taxed for stub gaps it cannot fix.

### C.3 Structured data

Use frozen dataclasses for structured results.

```python
@dataclass(frozen=True, slots=True)
class MetricSummary:
    mae: float
    rmse: float
    r2: float
    n_samples: int
```

Rules:

- Do not return bare tuples for multi-value results.
- Do not pass untyped dictionaries through the interior.
- Result dataclasses live next to the producer.
- Shared domain entities live in `domain/`, not in a giant `types.py`.
- Mutable dataclasses require a documented reason.

### C.4 Interfaces

- Do not introduce abstract base classes until there are at least two concrete implementations.
- Use `Protocol` only when it clarifies a boundary, such as `Featurizer`, `Predictor`, or `StructureStore`.
- No metaclasses, stateful decorators, global registries, import-time side effects, or `__getattr__` magic.

---

## D. Configuration

All scientific configuration lives in versioned YAML under `configs/`.

Rules:

- Load config once at the entrypoint.
- Validate config into frozen dataclasses.
- Do not hard-code experimental parameters in scripts.
- Do not use environment variables for scientific parameters.
- Use environment variables or secret managers only for secrets: API keys, tokens, credentials.
- Save a config snapshot into every output directory.

Example config dataclass:

```python
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    dataset_manifest: Path
    split_manifest: Path
    output_dir: Path
    model_name: str
```

---

## E. Reproducibility and provenance

### E.1 Seed control

Every executable experiment calls one seed function exactly once near startup. It must seed `random`, `numpy`, PyTorch when used, CUDA when used, and `PYTHONHASHSEED` where applicable.

For PyTorch:

- call `torch.manual_seed`
- call `torch.cuda.manual_seed_all` when CUDA is available
- use deterministic algorithms where possible
- set cuDNN deterministic behavior where relevant
- document that exact reproducibility is not guaranteed across releases, devices, CUDA, or cuDNN versions

### E.2 Output manifests

Every experiment directory in `outputs/` has a `manifest.json` recording the run's provenance. This per-experiment manifest is the CI-enforced source of truth.

Required fields:

```json
{
  "git_sha": "...",
  "git_dirty": false,
  "config_hash": "sha256:...",
  "config_path": "configs/experiment.yaml",
  "dataset_manifest": "data/manifests/dataset.yaml",
  "split_manifest": "data/splits/split.yaml",
  "seed": 42,
  "python_version": "...",
  "package_version": "...",
  "timestamp_utc": "...",
  "hostname": "...",
  "gpu_model": "...",
  "cuda_version": "...",
  "driver_version": "..."
}
```

Rules:

- If `git_dirty == true`, final-result runs must fail unless explicitly marked debug.
- Store logs, figures, tables, and model checkpoints under one experiment directory.
- Experiment ID format: `<short_git_sha>_<config_hash_prefix>_<utc_timestamp>`.
- Checkpoints carry their own provenance record per Section G.
- Guidance: per-artifact sibling files (`<filename>.meta.json`) for individual figures and tables are encouraged where artifact-level traceability matters, but are not CI-enforced. The per-experiment manifest plus checkpoint records are the required floor.

### E.3 Experiment tracking

Manifests are the source of truth for provenance, not an experiment tracker. A tracker such as Weights & Biases or MLflow may be used for live monitoring and convenience, provided every reported result still traces fully to the committed manifest, config, and split. A tracker must not become a required dependency for reproducing results, and tracker run IDs are supplementary, not a substitute for the manifest.

---

## F. Data and materials-science rules

### F.1 Data manifests

Raw data is external. Commit only manifests. Each manifest records dataset name, source URL/DOI/database, license/access restrictions, version/release/query date, exact query/filter code, file names and SHA-256 hashes, schema and units, structure format, cleaning/exclusion rules, and known limitations.

A loader must verify hashes before use.

### F.2 Materials-specific validation

Validate domain assumptions explicitly:

- units
- compositions
- oxidation-state assumptions
- structure validity
- duplicate structures
- disorder handling
- charged systems
- periodic boundary assumptions
- symmetry tolerances
- NaNs, infinities, and missing targets

Never silently drop rows. Log and report all exclusions.

### F.3 Splits and leakage

Train/validation/test splits are artifacts.

Rules:

- Generate splits once and save them under `data/splits/`.
- Each split manifest records seed, method, group keys, target distribution, and sample counts.
- Never tune on the test set.
- Use group-aware splits when leakage is plausible.
- For materials ML, consider composition-based, structure-based, scaffold/group-based, time-based, or source-held-out splits.
- Report the split ID in every result table.

### F.4 Benchmarks

Use an established benchmark when one exists.

Rules:

- For property prediction, prefer benchmark-style reporting before custom splits.
- For generated materials, report validity, uniqueness, novelty, and coverage unless a better domain-specific metric is justified.
- Always compare to simple baselines.
- Report mean, uncertainty, number of seeds, and exact split IDs.

---

## G. Models, training, and checkpoints

Rules:

- Model initialization is controlled by config and seed.
- Checkpoints include model state, optimizer state, scheduler state, epoch/step, config hash, dataset manifest, split manifest, git SHA, and package version.
- Do not save only `state_dict` without metadata for final-result models.
- Training code must support resume or explicitly document why not.
- Evaluation code must load a checkpoint without rerunning training.
- Final paper metrics come from scripts or workflow entrypoints, not notebook cells.
- The released model ships with a one-page model card recording intended use, training data, evaluation data and metrics, known limitations, and out-of-distribution caveats.

---

## H. Logging and outputs

Rules:

- Use `logging`, never `print`, except in CLI help or one-off local debugging.
- Configure logging only at entrypoints.
- Library code emits records but does not configure handlers.
- Use `pathlib.Path`, never stringly typed paths.
- Use JSON logs for final-result runs.

Output structure:

```text
outputs/<experiment_id>/
  config_snapshot.yaml
  manifest.json
  metrics.parquet
  figures/
  logs/
  checkpoints/
  model_card.md
```

Preferred formats: Parquet for tables, JSON for metadata, YAML for configs/manifests, SVG/PDF for vector figures, and PNG only when raster is required.

---

## I. Testing

Test tiers:

- Unit tests: fast, no GPU, no external network, no large files.
- Integration tests: small fixtures and class-class interaction.
- Smoke tests: full minimal pipeline.
- GPU tests: explicitly marked and skipped when GPU is unavailable.
- Slow tests: explicitly marked and excluded from default local runs.

For numerical and scientific code, the CI-enforced requirement is property-focused: test shapes, dtypes, units, invalid inputs, NaNs/infinities, deterministic behavior, physical invariants, split leakage, hash verification, and provenance writing. Use `hypothesis` for load-bearing numerical invariants when it adds real coverage.

Guidance: aim for at least one test per public function or method. This is a target for completeness, not a merge gate; the enforced bar is the property-focused coverage above, which is where defects actually surface.

---

## J. Documentation

Rules:

- Every module starts with a short docstring describing its purpose.
- Every public class has a docstring.
- Every public function and method has a Google-style docstring unless the signature is fully obvious.
- README contains setup, reproduction, data access, commands, expected outputs, and citation.
- `CLAUDE.md` contains project-specific bindings and conventions.
- The released model ships with a model card per Section G.
- The paper methods section must match the repository commands and configs.

Minimal README sections: title, purpose, installation, data access, reproduce main results, repository layout, citation, license, and known limitations.

---

## K. FAIR and release rules

A paper-supporting release must include a GitHub release, Zenodo archive and DOI, `CITATION.cff`, `LICENSE`, `README.md`, changelog entry, frozen `uv.lock`, final configs, data manifests, split manifests, result manifests, a model card, and instructions to regenerate tables and figures.

Default license: MIT or Apache-2.0 unless project constraints require otherwise.

---

## L. What this repo does not require

Do not add these unless there is a concrete need:

- dependency injection frameworks
- abstract base classes with one implementation
- complex plugin systems
- global registries
- metaclasses
- stateful decorators
- premature multiprocessing
- premature distributed training
- notebooks as production code
- convenience scripts that duplicate package logic
- experiment trackers as a required reproduction dependency

---

## M. Minimal `pyproject.toml`

```toml
[project]
name = "<package>"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = []

[dependency-groups]
dev = [
  "ruff>=0.8",
  "mypy>=1.13",
  "pytest>=8.0",
  "pytest-cov",
  "hypothesis",
  "pre-commit>=3.0",
]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "ANN", "RUF", "SIM", "PTH", "RET"]
ignore = ["ANN101", "ANN102"]

[tool.ruff.format]
docstring-code-format = true
quote-style = "double"

# Strict typing for boundary and contract code.
[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
warn_redundant_casts = true

# Relaxed typing for numerical, featurization, and plotting code built on
# libraries with weak or missing stubs.
[[tool.mypy.overrides]]
module = [
  "<package>.features.*",
  "<package>.plotting.*",
]
disallow_untyped_defs = false
disallow_incomplete_defs = false
warn_return_any = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=src/<package>"
markers = [
  "integration: integration tests",
  "smoke: end-to-end smoke tests",
  "gpu: requires GPU",
  "slow: slow tests",
]
```

---

## N. CI gates

A PR cannot merge unless formatting, linting, strict type checking of boundary code, property-focused numerical tests, and smoke tests pass. It also must not commit raw data or checkpoints, must validate manifests, must keep final-result runs clean-git only, and must allow generated figures/tables to be reproduced from committed configs and manifests.