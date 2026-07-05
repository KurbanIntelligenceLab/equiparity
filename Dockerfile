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
#     only) is NOT installed here. Training reads committed npz splits via numpy.
#   - datasets are fetched at runtime from S3 (scripts/docker/fetch_data.sh), never baked in.
#
# Result: ~7 GB uncompressed / ~3.5 GB pulled. The CUDA+torch+triton floor (~6.4 GB) is
# irreducible for GPU training on Blackwell (sm_120 needs CUDA 12.8).
#
# Build one profile at a time (nequip and mace pin incompatible e3nn versions):
#   docker build --build-arg PROFILE=nequip -t equiparity:nequip .
#   docker build --build-arg PROFILE=mace   -t equiparity:mace   .
# ==============================================================

FROM --platform=linux/amd64 python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
# git: reproducibility records the commit sha in each run manifest.
# curl/aria2/ca-certificates: fast dataset download from S3 in the runtime fetch script.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates aria2 \
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
RUN uv sync --frozen --no-dev --no-install-project --extra ${PROFILE}

# --- Project layer ---
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data/manifests ./data/manifests
COPY data/splits ./data/splits
RUN uv sync --frozen --no-dev --extra ${PROFILE}

# Sanity: the CLI wires up and torch imports. CUDA availability is checked at runtime on the GPU host.
RUN equiparity --version && python -c "import torch; print('torch', torch.__version__)"

# Datasets are not in the image. Fetch them at runtime, then run an experiment:
#   bash scripts/docker/fetch_data.sh && equiparity run configs/<name>.yaml
CMD ["bash", "-lc", "echo 'equiparity image (profile=${PROFILE}). Fetch data: bash scripts/docker/fetch_data.sh ; then: equiparity run configs/<name>.yaml'"]
