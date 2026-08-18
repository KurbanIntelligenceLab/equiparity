"""Tests for experiment config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from equiparity.domain.experiment import ConfigError
from equiparity.domain.parity import ParityMode
from equiparity.io.config import parse_experiment_config

_BASE = {
    "seed": 42,
    "core": "nequip",
    "parity": "o3",
    "target": "U0",
    "dataset": "qm9",
    "processed_npz": "data/raw/qm9/qm9_processed.npz",
    "split_npz": "data/splits/qm9_split.npz",
    "model": {"num_layers": 4, "l_max": 2, "num_features": 16},
    "training": {"batch_size": 8, "epochs": 3, "max_train_samples": 100},
}


def test_parse_valid() -> None:
    cfg = parse_experiment_config(dict(_BASE))
    assert cfg.parity is ParityMode.O3
    assert cfg.core == "nequip"
    assert cfg.model.num_layers == 4
    assert cfg.training.max_train_samples == 100
    assert cfg.processed_npz == Path("data/raw/qm9/qm9_processed.npz")
    assert cfg.run_label == "nequip_o3_U0_seed42"


def test_invalid_core_raises() -> None:
    with pytest.raises(ConfigError, match="core must be"):
        parse_experiment_config({**_BASE, "core": "schnet"})


def test_invalid_target_raises() -> None:
    with pytest.raises(ConfigError, match="target must be"):
        parse_experiment_config({**_BASE, "target": "bandgap"})


def test_missing_key_raises() -> None:
    incomplete = {k: v for k, v in _BASE.items() if k != "seed"}
    with pytest.raises(ConfigError, match="missing required config key"):
        parse_experiment_config(incomplete)


def test_pooling_defaults_to_sum() -> None:
    """Non-negotiable: an existing config with no `pooling` key must keep reproducing sum."""
    cfg = parse_experiment_config(dict(_BASE))
    assert cfg.model.pooling == "sum"


def test_pooling_mean_loads_from_yaml() -> None:
    cfg = parse_experiment_config({**_BASE, "model": {**_BASE["model"], "pooling": "mean"}})
    assert cfg.model.pooling == "mean"


def test_invalid_pooling_raises() -> None:
    with pytest.raises(ConfigError, match=r"model\.pooling must be"):
        parse_experiment_config({**_BASE, "model": {**_BASE["model"], "pooling": "max"}})
