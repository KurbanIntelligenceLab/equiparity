"""Experiment configuration: the single typed contract for one training run.

Loaded once at the entrypoint and validated into this frozen dataclass. A run differs from its
matched-pair partner only in ``parity``; the config generator expands the {core x parity x dataset
x seed} grid from a template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from equiparity.domain.parity import ParityMode
from equiparity.domain.target import TARGETS

CORES = ("nequip", "allegro", "mace")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ModelHyperparams:
    """Core-agnostic model hyperparameters shared by both parity arms."""

    num_layers: int = 3
    l_max: int = 2
    num_features: int = 32
    r_max: float = 5.0
    # Mean pooling divides by the number of atoms (edges for Allegro).
    pooling: str = "sum"
    tensor_head: str = "linear"
    use_magnetic_magnitudes: bool = False

    def __post_init__(self) -> None:
        if self.tensor_head not in ("linear", "complete"):
            raise ConfigError("model.tensor_head must be linear or complete")
        if self.pooling not in ("sum", "mean"):
            raise ConfigError(f"model.pooling must be 'sum' or 'mean', got {self.pooling!r}")


@dataclass(frozen=True, slots=True)
class TrainingParams:
    """Optimization and run-control parameters."""

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    # Precision for model weights; geometry may use higher precision.
    precision: str = "float32"
    max_train_samples: int | None = None  # cap for smoke runs; None uses the full split
    max_eval_samples: int | None = None
    # Zero-label runs use the scale estimated from the unaugmented training split.
    target_scale: float | None = None
    # Per-row multiplier applied to exactly zero training targets.
    zero_row_loss_weight: float = 1.0
    loss: str = "mse"
    huber_delta: float = 1.0
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    min_lr: float = 1e-6
    gradient_clip: float = 10.0
    evaluate_test: bool = False

    def __post_init__(self) -> None:
        if self.loss not in ("mse", "tensor_huber"):
            raise ConfigError("training.loss must be mse or tensor_huber")
        if self.scheduler not in ("none", "plateau"):
            raise ConfigError("training.scheduler must be none or plateau")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.lr,
                self.huber_delta,
                self.min_lr,
                self.gradient_clip,
            )
            <= 0
        ):
            raise ConfigError(
                "training counts, learning rates, loss scale and clip must be positive"
            )
        if self.scheduler_patience < 0 or self.min_lr > self.lr:
            raise ConfigError("invalid scheduler patience or minimum learning rate")
        if self.precision not in ("float32", "float64"):
            raise ValueError(f"precision must be float32 or float64, got {self.precision!r}")
        if self.target_scale is not None and self.target_scale <= 0:
            raise ValueError(f"target_scale must be positive, got {self.target_scale!r}")
        if self.zero_row_loss_weight <= 0:
            raise ValueError(
                f"zero_row_loss_weight must be positive, got {self.zero_row_loss_weight!r}"
            )


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
    protocol: str = "final_epoch"

    def __post_init__(self) -> None:
        if self.protocol not in ("final_epoch", "validation_selected"):
            raise ConfigError("protocol must be final_epoch or validation_selected")
        if self.protocol == "final_epoch" and self.model.tensor_head != "linear":
            raise ConfigError("complete tensor heads require protocol validation_selected")
        if self.protocol == "validation_selected":
            if self.target not in ("piezoelectric", "elastic"):
                raise ConfigError("validation_selected supports matched crystal tensor cores only")
            if self.model.tensor_head != "complete" or self.model.pooling != "mean":
                raise ConfigError(
                    "validation_selected requires complete tensor heads and mean pooling"
                )
        if self.core not in CORES:
            raise ConfigError(f"core must be one of {CORES}, got {self.core!r}")
        if self.target not in TARGETS:
            raise ConfigError(f"target must be one of {sorted(TARGETS)}, got {self.target!r}")

    @property
    def run_label(self) -> str:
        """Stable label for this run, e.g. ``nequip_o3_U0_seed42``.

        The label identifies the backbone, target, symmetry arm and seed.
        """
        return f"{self.core}_{self.parity.value}_{self.target}_seed{self.seed}"
