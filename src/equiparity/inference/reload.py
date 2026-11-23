"""Load validation-selected crystal weights or final-epoch sum-pooled weights for inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

# e3nn 0.4.4 (the MACE profile) unpickles `constants.pt` at *import* time, and torch >= 2.6 refuses
# the `slice` global by default. This must run before anything imports e3nn, hence module scope.
torch.serialization.add_safe_globals([slice])

from equiparity.domain.experiment import ExperimentConfig  # noqa: E402
from equiparity.domain.structure import AtomicStructure  # noqa: E402
from equiparity.domain.target import TARGETS  # noqa: E402
from equiparity.io.config import parse_experiment_config  # noqa: E402
from equiparity.io.mp_dataset import (  # noqa: E402
    CrystalDataset,
    load_crystal_dataset,
    load_split,
)

if TYPE_CHECKING:
    from equiparity.domain.sample import LabeledStructure

# Recomputed from the piezoelectric training split; asserted on every reload as a tripwire.
PIEZO_SCALE = 0.749134
PIEZO_AVG_NEIGHBORS = 36.5717

_E3NN_CORES = ("nequip", "allegro")


@dataclass(frozen=True)
class TrainedModel:
    """A rebuilt model plus everything needed to turn its output into physical units."""

    model: Any
    scale: float
    config: ExperimentConfig
    core: str
    parity: str
    run_dir: Path
    device: torch.device
    _extra: dict[str, Any]

    def predict(self, structures: list[AtomicStructure]) -> np.ndarray:
        """Un-normalised tensor predictions in the orthonormal irreps basis, shape ``(n, dim)``."""
        return _predict_for_core(self, structures)

    def violations(self, structures: list[AtomicStructure]) -> np.ndarray:
        """Per-structure violation magnitude ``||T||_F``, shape ``(n,)``."""
        preds = self.predict(structures)
        return np.sqrt((preds**2).sum(axis=1))


def load_trained(
    run_dir: Path,
    *,
    repo_root: Path,
    device: torch.device | None = None,
    scale: float | None = None,
) -> TrainedModel:
    """Rebuild a run from its selected checkpoint (final epoch for sum-pooled runs).

    ``scale`` overrides the training-split normalization of sum-pooled models.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
    config = parse_experiment_config(raw)

    if config.protocol == "validation_selected":
        from equiparity.training.crystal_regression import build_crystal_model

        selected = torch.load(
            run_dir / "checkpoint_selected.pt", map_location="cpu", weights_only=False
        )
        identity = selected["identity"]
        if identity["config"] != raw:
            raise ValueError("selected checkpoint configuration differs from run snapshot")
        model, extra = build_crystal_model(config, identity["model_metadata"], dev)
        model.load_state_dict(selected["model"])
        model.eval()
        return TrainedModel(
            model=model,
            scale=scale if scale is not None else identity["scale"],
            config=config,
            core=config.core,
            parity=config.parity.value,
            run_dir=run_dir,
            device=dev,
            _extra=extra,
        )

    data = load_crystal_dataset(repo_root / config.processed_npz, (config.target,))
    train_ds = CrystalDataset(data, load_split(repo_root / config.split_npz, "train"))
    resolved_scale = scale if scale is not None else _recompute_scale(train_ds, config)

    builder = {
        "nequip": _build_e3nn,
        "allegro": _build_e3nn,
        "mace": _build_mace,
    }[config.core]
    model, extra = builder(config, train_ds, dev, repo_root)

    state = torch.load(run_dir / "checkpoint_latest.pt", map_location=dev, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    return TrainedModel(
        model=model,
        scale=resolved_scale,
        config=config,
        core=config.core,
        parity=config.parity.value,
        run_dir=run_dir,
        device=dev,
        _extra=extra,
    )


def _recompute_scale(train_ds: CrystalDataset, config: ExperimentConfig) -> float:
    """The scale a run used: its frozen override if set, else recomputed from the training split."""
    from equiparity.training.nequip_tensor import _irreps_targets

    if config.training.target_scale is not None:
        return float(config.training.target_scale)
    scale = float(_irreps_targets(train_ds, config.target, config.target).std()) or 1.0
    sum_pooled_piezo = config.target == "piezoelectric" and config.dataset == "mp_piezoelectric"
    if sum_pooled_piezo and abs(scale - PIEZO_SCALE) > 1e-5:
        raise RuntimeError(f"piezo scale drifted: {scale!r} != {PIEZO_SCALE}")
    return scale


def _build_e3nn(
    config: ExperimentConfig, train_ds: CrystalDataset, dev: torch.device, _repo: Path
) -> tuple[Any, dict[str, Any]]:
    from nequip.utils.global_state import set_global_state

    from equiparity.training.nequip_data import avg_num_neighbors, to_atomic_data
    from equiparity.training.nequip_tensor import build_tensor_model, periodic_type_map

    set_global_state(allow_tf32=False)
    dtype = torch.float64
    type_names, z_map = periodic_type_map()
    graphs = [
        to_atomic_data(train_ds[i].structure, z_map, config.model.r_max, dtype)
        for i in range(len(train_ds))
    ]
    avg_neigh = avg_num_neighbors(graphs)
    if config.dataset == "mp_piezoelectric" and abs(avg_neigh - PIEZO_AVG_NEIGHBORS) > 1e-3:
        raise RuntimeError(f"avg_num_neighbors drifted: {avg_neigh!r}")
    model = build_tensor_model(config, type_names, avg_neigh, TARGETS[config.target].irreps, dev)
    return model, {"z_map": z_map, "dtype": dtype}


def _build_mace(
    config: ExperimentConfig, train_ds: CrystalDataset, dev: torch.device, repo: Path
) -> tuple[Any, dict[str, Any]]:
    from mace import tools

    from equiparity.models.mace import MACEConfig, MACETensorModel
    from equiparity.training.mace_scalar import _to_mace_data
    from equiparity.training.mace_tensor import _avg_neighbors, _elements_of

    data = load_crystal_dataset(repo / config.processed_npz, (config.target,))
    val_ds = CrystalDataset(data, load_split(repo / config.split_npz, "val"))
    test_ds = CrystalDataset(data, load_split(repo / config.split_npz, "test"))
    ood_npz = repo / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
    ood = CrystalDataset(load_crystal_dataset(ood_npz))
    elements = _elements_of(train_ds, val_ds, test_ds, ood)
    z_table = tools.AtomicNumberTable(elements)

    r_max = config.model.r_max
    graphs = [_to_mace_data(train_ds[i].structure, z_table, r_max) for i in range(len(train_ds))]
    avg_neigh = _avg_neighbors(graphs)
    if config.dataset == "mp_piezoelectric" and abs(avg_neigh - PIEZO_AVG_NEIGHBORS) > 1e-3:
        raise RuntimeError(f"avg_num_neighbors drifted: {avg_neigh!r}")

    mace_cfg = MACEConfig(
        r_max=r_max,
        atomic_numbers=tuple(elements),
        num_interactions=config.model.num_layers,
        l_max=config.model.l_max,
        num_features=config.model.num_features,
        avg_num_neighbors=avg_neigh,
        seed=config.seed,
        model_dtype=config.training.precision,
        pooling=config.model.pooling,
    )
    model = MACETensorModel(mace_cfg, config.parity, TARGETS[config.target].irreps).to(dev)
    dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    return model, {"z_table": z_table, "dtype": dtype}


def _predict_for_core(trained: TrainedModel, structures: list[AtomicStructure]) -> np.ndarray:
    core, cfg = trained.core, trained.config
    batch_size = cfg.training.batch_size

    if core in _E3NN_CORES:
        from equiparity.training.nequip_tensor import predict_tensors

        return (
            predict_tensors(
                trained.model,
                _AdHocDataset(structures),
                trained._extra["z_map"],
                r_max=cfg.model.r_max,
                device=trained.device,
                dtype=trained._extra["dtype"],
                batch_size=batch_size,
            )
            * trained.scale
        )

    if core == "mace":
        from equiparity.training.mace_scalar import _to_mace_data
        from equiparity.training.mace_tensor import _predict as _mace_predict

        graphs = [_to_mace_data(s, trained._extra["z_table"], cfg.model.r_max) for s in structures]
        preds = _mace_predict(
            trained.model, graphs, batch_size, trained._extra["dtype"], trained.device
        )
        return preds * trained.scale

    raise ValueError(f"unsupported core {core}")


class _AdHocDataset:
    """Minimal CrystalDataset stand-in over a plain list of structures (predict_tensors only)."""

    def __init__(self, structures: list[AtomicStructure]) -> None:
        self._structures = structures

    def __len__(self) -> int:
        return len(self._structures)

    def __getitem__(self, index: int) -> LabeledStructure:
        from equiparity.domain.sample import LabeledStructure

        return LabeledStructure(
            structure=self._structures[index], targets={}, identifier=str(index)
        )
