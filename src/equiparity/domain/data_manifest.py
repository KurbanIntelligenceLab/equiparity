"""Dataset and split manifests: the committed provenance for external data.

Raw data is never committed; only these manifests are. A loader verifies file hashes against
:class:`DatasetManifest` before use; a split is an artifact described by :class:`SplitManifest`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Provenance for an external dataset.

    Attributes:
        name: Dataset identifier.
        source: URL, DOI, or database name.
        version: Release, query date, or version string.
        license: License or access restriction.
        file_hashes: Mapping of file name to SHA-256 hex digest; loaders verify these.
        schema: Field-to-description mapping (columns, units).
        structure_format: How structures are stored (e.g. ``xyz``, ``pymatgen Structure``).
        query: Reference to the exact query/filter code that produced the dataset.
        cleaning: Cleaning and exclusion rules applied.
        limitations: Known limitations.
    """

    name: str
    source: str
    version: str
    license: str
    file_hashes: dict[str, str]
    schema: dict[str, str]
    structure_format: str
    query: str = ""
    cleaning: str = ""
    limitations: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain JSON/YAML-ready dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Provenance for a train/val/test split.

    Attributes:
        split_id: Stable identifier referenced in every result table.
        dataset: Name of the dataset this split partitions.
        method: Split method (e.g. ``random``, ``group``, ``time``, ``source-held-out``).
        seed: RNG seed used to generate the split.
        counts: Sample counts per partition (``train``/``val``/``test``).
        group_keys: Grouping keys used to prevent leakage, if any.
        target_distribution: Optional summary stats of the target per partition.
    """

    split_id: str
    dataset: str
    method: str
    seed: int
    counts: dict[str, int]
    group_keys: tuple[str, ...] = ()
    target_distribution: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain JSON/YAML-ready dict."""
        return asdict(self)
