# syntax=docker/dockerfile:1
# ==============================================================
# equiparity — lean GPU image for cloud training (vast.ai / any CUDA host).
#
# Size strategy (compared against agents-mlip and E3-GRAND):
#   - python:3.12-slim base, NOT nvidia/cuda:*-devel. The torch cu128 wheel bundles the
#     CUDA runtime, so a devel base just double-ships ~5-6 GB of CUDA. (E3-GRAND's slim
#     image proved this: ~40 GB -> ~9 GB.)
#   - uv sync --frozen from the committed uv.lock: reproducible, no pip --no-deps hacks.
#   - --no-dev drops ruff/mypy/pytest; the `data` extra (pymatgen/mp-api/spglib, data-prep
#     only) is NOT installed here. Training reads the baked npz splits via numpy.
#   - the 62 MB processed npz are baked in (self-contained, no runtime S3 dependency); the raw
#     668 MB QM9 xyz stays out. scripts/docker/fetch_data.sh remains as an optional override.
#
# Result: ~7 GB uncompressed / ~3.5 GB pulled. The CUDA+torch+triton floor (~6.4 GB) is
# irreducible for GPU training on Blackwell (sm_120 needs CUDA 12.8).
#
# PREREQUISITE: the processed npz must exist before building (they are gitignored). On a fresh
# clone run first:  uv sync --extra nequip --extra data  &&  uv run python scripts/prepare_qm9.py
#   &&  MP_TOKEN=... uv run python scripts/prepare_mp.py
#
# Build one profile at a time (nequip and mace pin incompatible e3nn versions):
#   docker build --build-arg PROFILE=nequip -t equiparity:nequip .
#   docker build --build-arg PROFILE=mace   -t equiparity:mace   .
# ==============================================================

FROM --platform=linux/amd64 python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
# git: reproducibility records the commit sha in each run manifest.
# curl/aria2/ca-certificates: fast dataset download from S3 in the runtime fetch script.
# openssh-client/server, tmux, rsync: REQUIRED for vast.ai reachability — vast sets up SSH
# INSIDE the container and its /.launch invokes ssh; a slim base without these leaves the
# instance unreachable ("ssh: command not found") or commands silently exit 127 (the tmux
# auto-exec trap). rsync is needed for pulling results back. (E3-GRAND hard-won gotcha.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates aria2 openssh-client openssh-server tmux rsync \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv binary (matches the host uv that produced uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

ARG PROFILE=nequip
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1 \
    PATH=/workspace/.venv/bin:$PATH

WORKDIR /workspace

# --- Dependency layer (cached unless pyproject.toml / uv.lock change) ---
# Installs the locked environment for one model profile: base (numpy/torch/ase) + the profile's
# e3nn stack. torch cu128 and its bundled CUDA libs come from the pytorch index declared in
# pyproject. No dev tools, no data-prep libraries.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra ${PROFILE}

# --- Project layer ---
COPY LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data/manifests ./data/manifests
COPY data/splits ./data/splits
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ${PROFILE}

# --- Data layer: bake the 62 MB processed npz (self-contained image) ---
# The OOD set ships in BOTH variants: `_processed.npz` is idealized onto the exact space group,
# `_processed_raw.npz` keeps the DFT-relaxed coordinates. The tensor trainers evaluate both
# (`_raw_variant()` silently skips the raw one if it is missing), so omitting it would quietly
# drop half the headline OOD table. `_augmented` is the E1 rebuttal training set.
COPY data/raw/qm9/qm9_processed.npz ./data/raw/qm9/
COPY data/raw/mp/mp_piezoelectric_processed.npz data/raw/mp/mp_elastic_processed.npz \
     data/raw/mp/mp_ood_centrosymmetric_processed.npz \
     data/raw/mp/mp_ood_centrosymmetric_processed_raw.npz \
     data/raw/mp/mp_piezoelectric_augmented_processed.npz ./data/raw/mp/

# Sanity: the CLI wires up and torch imports. CUDA availability is checked at runtime on the GPU host.
RUN equiparity --version && python -c "import torch; print('torch', torch.__version__)"

# Data is baked in — just run an experiment (set EQUIPARITY_DATA_URL + run fetch_data.sh only to override):
#   equiparity run configs/<name>.yaml
CMD ["bash", "-c", "echo 'equiparity image. Data baked in. Run: equiparity run configs/<name>.yaml'"]
