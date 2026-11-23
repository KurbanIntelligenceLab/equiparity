"""Magnetic norms influence predictions without changing spatial covariance."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from equiparity.domain.experiment import ExperimentConfig, ModelHyperparams, TrainingParams
from equiparity.domain.parity import ParityMode
from equiparity.domain.structure import AtomicStructure
from equiparity.domain.target import TARGETS
from equiparity.training.crystal_regression import build_crystal_model

CORES = [
    c
    for c, pkg in [("nequip", "nequip"), ("allegro", "allegro"), ("mace", "mace")]
    if importlib.util.find_spec(pkg)
]


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("parity", [ParityMode.O3, ParityMode.SO3])
def test_magnitude_conditioning_covariance_gradients_and_reload(
    core: str, parity: ParityMode, tmp_path: Path
) -> None:
    torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    cfg = ExperimentConfig(
        seed=3,
        core=core,
        parity=parity,
        target="elastic",
        dataset="test",
        processed_npz=Path("unused"),
        split_npz=Path("unused"),
        output_dir=tmp_path,
        protocol="validation_selected",
        model=ModelHyperparams(
            num_features=4,
            num_layers=3,
            l_max=2,
            r_max=3,
            pooling="mean",
            tensor_head="complete",
            use_magnetic_magnitudes=True,
        ),
        training=TrainingParams(device="cpu"),
    )
    metadata = {"avg_neighbors": 8.0, "elements": [1, 6, 8]}
    model, extra = build_crystal_model(cfg, metadata, torch.device("cpu"))
    atoms = Atoms(
        numbers=[8, 1, 6, 1],
        positions=[[0, 0, 0], [1.2, 0.1, 0.2], [0.3, 1.4, 0.1], [0.1, 0.2, 1.3]],
        cell=np.diag([4.1, 4.3, 4.5]),
        pbc=True,
    )
    magnitudes = np.array([0.2, 1.1, 2.4, 0.6])

    def structure(a: Atoms, m: np.ndarray) -> AtomicStructure:
        return AtomicStructure(a.numbers, a.positions, a.cell.array, True, m)

    def batch(a: Atoms, m: np.ndarray) -> dict[str, torch.Tensor]:
        s = structure(a, m)
        if core == "mace":
            from equiparity.training.mace_scalar import _batches, _to_mace_data

            return next(
                iter(_batches([_to_mace_data(s, extra["z_table"], 3.0)], 1, extra["dtype"]))
            )
        from nequip.data import AtomicDataDict

        from equiparity.training.nequip_data import to_atomic_data

        return AtomicDataDict.batched_from_list(
            [to_atomic_data(s, extra["z_map"], 3.0, torch.float64)]
        )

    pred = model(batch(atoms, magnitudes))
    pred.square().sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.condition.parameters()) > 1e-10
    assert not torch.allclose(pred, model(batch(atoms, np.zeros(4))), atol=1e-9, rtol=1e-5)
    q = np.diag([-1.0, 1.0, 1.0]) if parity == ParityMode.O3 else o3.rand_matrix().numpy()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ q.T
    rotated.set_cell(atoms.cell.array @ q.T)
    representation = (
        o3.Irreps(TARGETS["elastic"].irreps)
        .D_from_matrix(torch.tensor(q, dtype=pred.dtype))
        .to(pred.dtype)
    )
    torch.testing.assert_close(
        model(batch(rotated, magnitudes)), pred @ representation.T, atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(
        model(batch(atoms.repeat((2, 1, 1)), np.tile(magnitudes, 2))), pred, atol=3e-5, rtol=3e-4
    )
    permutation = [2, 0, 3, 1]
    torch.testing.assert_close(
        model(batch(atoms[permutation], magnitudes[permutation])), pred, atol=3e-5, rtol=3e-4
    )
    import yaml

    from equiparity.inference.reload import load_trained
    from equiparity.workflows.run_experiment import _config_snapshot

    raw = _config_snapshot(cfg)
    (tmp_path / "config_snapshot.yaml").write_text(yaml.safe_dump(raw))
    torch.save(
        {
            "model": model.state_dict(),
            "identity": {"config": raw, "scale": 1.0, "model_metadata": metadata},
        },
        tmp_path / "checkpoint_selected.pt",
    )
    loaded = load_trained(tmp_path, repo_root=tmp_path, device=torch.device("cpu"))
    np.testing.assert_allclose(
        loaded.predict([structure(atoms, magnitudes)]), pred.detach().numpy(), atol=3e-5, rtol=3e-4
    )
