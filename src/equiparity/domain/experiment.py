"""Experiment configuration: the single typed contract for one training run.

Loaded once at the entrypoint and validated into this frozen dataclass (CODING_RULES.md
Section D). A run differs from its matched-pair partner only in ``parity``; the config
generator expands the {core x parity x dataset x seed} grid from a template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from equiparity.domain.parity import ParityMode
from equiparity.domain.target import TARGETS

CORES = ("nequip", "allegro", "mace", "equiformer_v2", "clifford_stf")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ModelHyperparams:
    """Core-agnostic model hyperparameters shared by both parity arms."""

    num_layers: int = 3
    l_max: int = 2
    num_features: int = 32
    r_max: float = 5.0


@dataclass(frozen=True, slots=True)
class TrainingParams:
    """Optimization and run-control parameters."""

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    # Model/output precision. "float32" trains in nequip mixed precision (fp32 weights, fp64
    # geometry) — ~6.8x faster than float64 on consumer GPUs at production size. "float64" is
    # kept for high-precision verification runs.
    precision: str = "float32"
    max_train_samples: int | None = None  # cap for smoke runs; None uses the full split
    max_eval_samples: int | None = None
    # Override the target normalization scale instead of recomputing it from the training split.
    # Needed by the E1 augmentation study: adding zero-labelled crystals shrinks the recomputed
    # std (0.749 -> 0.638), and freezing it keeps augmented violations directly comparable to the
    # main runs. None (the default) recomputes, which is what every headline run did.
    target_scale: float | None = None
    # Up-weight the exactly-zero-target training rows in the MSE by this factor. 1.0 (default) is a
    # numeric no-op == plain MSELoss. The H3 loss-weight sweep raises it to probe whether a large
    # enough weight can force an SO(3) model's centrosymmetric violation to zero.
    zero_row_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.precision not in ("float32", "float64"):
            raise ValueError(f"precision must be float32 or float64, got {self.precision!r}")
        if self.target_scale is not None and self.target_scale <= 0:
            raise ValueError(f"target_scale must be positive, got {self.target_scale!r}")
        if self.zero_row_loss_weight <= 0:
            raise ValueError(
                f"zero_row_loss_weight must be positive, got {self.zero_row_loss_weight!r}"
            )


# The datasets the headline grid uses. Anything else (e.g. the E1 augmented piezoelectric set) is
# a side study and must never overwrite a headline run's flattened metrics.
CANONICAL_DATASETS = frozenset({"qm9", "mp_elastic", "mp_piezoelectric"})


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A fully specified training run."""

    seed: int
    core: str
    parity: ParityMode
    target: str
    dataset: str
    processed_npz: Path
    split_npz: Path
    output_dir: Path
    model: ModelHyperparams = field(default_factory=ModelHyperparams)
    training: TrainingParams = field(default_factory=TrainingParams)

    def __post_init__(self) -> None:
        if self.core not in CORES:
            raise ConfigError(f"core must be one of {CORES}, got {self.core!r}")
        if self.target not in TARGETS:
            raise ConfigError(f"target must be one of {sorted(TARGETS)}, got {self.target!r}")

    @property
    def run_label(self) -> str:
        """Stable label for this run, e.g. ``nequip_o3_U0_seed42``.

        Note this does NOT identify the dataset: two datasets sharing a target produce the same
        label. Use :attr:`run_key` when runs from different datasets may be mixed.
        """
        return f"{self.core}_{self.parity.value}_{self.target}_seed{self.seed}"

    @property
    def run_key(self) -> str:
        """``run_label`` qualified by dataset when the dataset is not the target's canonical one."""
        if self.dataset in CANONICAL_DATASETS:
            return self.run_label
        return f"{self.run_label}__{self.dataset}"
