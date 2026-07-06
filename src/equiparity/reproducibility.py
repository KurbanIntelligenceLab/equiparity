"""Seed control and provenance collection (CODING_RULES.md Section E).

Every executable experiment calls :func:`seed_everything` exactly once near startup and
writes a :class:`~equiparity.domain.provenance.RunManifest` into its output directory via
:func:`write_manifest`. Torch is seeded when present but is not required to import this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from equiparity import __version__
from equiparity.domain.provenance import RunManifest


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility.

    Seeds ``PYTHONHASHSEED``, the stdlib ``random`` module, and NumPy unconditionally, and
    PyTorch (plus CUDA and deterministic algorithms) when torch is importable. Exact
    reproducibility is not guaranteed across torch/CUDA/cuDNN releases or devices.

    Args:
        seed: The integer seed applied to every RNG.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed)


def _seed_torch(seed: int) -> None:
    """Seed PyTorch and CUDA if torch is available; no-op otherwise."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hash_config(config_text: str) -> str:
    """Return ``sha256:<hex>`` for the given config text."""
    digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _run_git(args: list[str]) -> str:
    """Run a git command in the repo and return stripped stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def _git_sha() -> str:
    # EQUIPARITY_GIT_SHA lets cloud runs record the source commit even when .git is absent
    # (the docker image ships code without the repo). Falls back to live git, then "unknown".
    return os.environ.get("EQUIPARITY_GIT_SHA") or _run_git(["rev-parse", "HEAD"]) or "unknown"


def _git_dirty() -> bool:
    return bool(_run_git(["status", "--porcelain"]))


def _query_gpu() -> tuple[str, str]:
    """Return ``(gpu_model, driver_version)`` from nvidia-smi, or ``("none", "none")``."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none", "none"
    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not first:
        return "none", "none"
    parts = [p.strip() for p in first.split(",")]
    gpu_model = parts[0] if parts else "unknown"
    driver = parts[1] if len(parts) > 1 else "unknown"
    return gpu_model, driver


def _cuda_version() -> str:
    """Return the torch-reported CUDA runtime version, or ``none`` when unavailable."""
    try:
        import torch
    except ImportError:
        return "none"
    cuda: str | None = torch.version.cuda
    return cuda if cuda is not None else "none"


def collect_provenance(
    *,
    config_text: str,
    config_path: str,
    dataset_manifest: str,
    split_manifest: str,
    seed: int,
    timestamp: datetime | None = None,
) -> RunManifest:
    """Collect environment provenance into a :class:`RunManifest`.

    Args:
        config_text: The raw config content, hashed into ``config_hash``.
        config_path: Repo-relative path to the config file.
        dataset_manifest: Repo-relative path to the dataset manifest.
        split_manifest: Repo-relative path to the split manifest.
        seed: The run seed.
        timestamp: UTC timestamp for the run; defaults to now.

    Returns:
        A fully populated manifest. Environment fields degrade to ``"none"``/``"unknown"``
        when the corresponding tool (git, nvidia-smi, torch) is unavailable.
    """
    gpu_model, driver_version = _query_gpu()
    ts = timestamp or datetime.now(UTC)
    return RunManifest(
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        config_hash=hash_config(config_text),
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        seed=seed,
        python_version=platform.python_version(),
        package_version=__version__,
        timestamp_utc=ts.isoformat(),
        hostname=platform.node(),
        gpu_model=gpu_model,
        cuda_version=_cuda_version(),
        driver_version=driver_version,
    )


def write_manifest(manifest: RunManifest, output_dir: Path, *, allow_dirty: bool = False) -> Path:
    """Write ``manifest.json`` into ``output_dir`` and return its path.

    Args:
        manifest: The provenance record to serialize.
        output_dir: Experiment output directory (created if missing).
        allow_dirty: When False, a dirty git tree raises to protect final-result runs
            (CODING_RULES.md Section E.2). Set True only for debug runs.

    Raises:
        RuntimeError: If ``manifest.git_dirty`` is True and ``allow_dirty`` is False.
    """
    if manifest.git_dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing to write a final-result manifest with a dirty git tree. "
            "Commit changes or pass allow_dirty=True for a debug run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return path
