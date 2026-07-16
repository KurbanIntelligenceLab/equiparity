"""Load and validate an experiment configuration from YAML (I/O boundary).

The untyped YAML mapping is converted immediately into the frozen
:class:`~equiparity.domain.experiment.ExperimentConfig` (CODING_RULES.md Section D).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from equiparity.domain.experiment import (
    ConfigError,
    ExperimentConfig,
    ModelHyperparams,
    TrainingParams,
)
from equiparity.domain.parity import ParityMode


def _require(mapping: dict[str, Any], key: str) -> Any:  # noqa: ANN401 (YAML boundary)
    if key not in mapping:
        raise ConfigError(f"missing required config key: {key!r}")
    return mapping[key]


def parse_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Convert a raw config mapping into a validated :class:`ExperimentConfig`."""
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    parity = ParityMode(str(_require(raw, "parity")))
    return ExperimentConfig(
        seed=int(_require(raw, "seed")),
        core=str(_require(raw, "core")),
        parity=parity,
        target=str(_require(raw, "target")),
        dataset=str(_require(raw, "dataset")),
        processed_npz=Path(str(_require(raw, "processed_npz"))),
        split_npz=Path(str(_require(raw, "split_npz"))),
        output_dir=Path(str(raw.get("output_dir", "outputs"))),
        model=ModelHyperparams(**model_raw),
        training=TrainingParams(**training_raw),
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment config YAML file into a validated :class:`ExperimentConfig`."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return parse_experiment_config(raw)
