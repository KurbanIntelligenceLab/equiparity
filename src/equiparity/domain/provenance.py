"""Run provenance: the per-experiment manifest that is the source of truth for reproducibility.

Every experiment directory in ``outputs/`` carries a
``manifest.json`` built from :class:`RunManifest`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Provenance record for a single experiment run.

    All fields are required; collection
    of the environment-dependent fields lives in :mod:`equiparity.reproducibility`.
    """

    git_sha: str
    git_dirty: bool
    config_hash: str
    config_path: str
    dataset_manifest: str
    split_manifest: str
    seed: int
    python_version: str
    package_version: str
    timestamp_utc: str
    hostname: str
    gpu_model: str
    cuda_version: str
    driver_version: str

    @property
    def experiment_id(self) -> str:
        """Experiment ID: ``<short_git_sha>_<config_hash_prefix>_<utc_timestamp>``.

        The timestamp is normalized to a filesystem-safe form (``:`` and ``+`` removed).
        """
        short_sha = self.git_sha[:8]
        config_prefix = self.config_hash.removeprefix("sha256:")[:8]
        safe_ts = self.timestamp_utc.replace(":", "").replace("+", "").replace("-", "")
        return f"{short_sha}_{config_prefix}_{safe_ts}"

    def to_dict(self) -> dict[str, str | bool | int]:
        """Serialize to a plain JSON-ready dict."""
        return asdict(self)
