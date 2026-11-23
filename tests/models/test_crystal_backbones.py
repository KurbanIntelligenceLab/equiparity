"""Exercise the complete tensor head on real backbones, including equivalent supercells."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from equiparity.domain.parity import ParityMode
from equiparity.domain.structure import AtomicStructure
from equiparity.domain.target import TARGETS

_CORES = [
    c
    for c, package in [("nequip", "nequip"), ("allegro", "allegro"), ("mace", "mace")]
    if importlib.util.find_spec(package)
]


def _structure(atoms: Atoms) -> "AtomicStructure":

    return AtomicStructure(
        atomic_numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell.array, pbc=True
    )


@pytest.mark.parametrize("core", _CORES)
@pytest.mark.parametrize("parity", [ParityMode.O3, ParityMode.SO3])
@pytest.mark.parametrize(
    "target,lmax,block", [("piezoelectric", 3, slice(6, 11)), ("elastic", 2, slice(12, 21))]
)
def test_complete_backbones_have_all_blocks_and_replication_invariance(
    core: str, target: str, lmax: int, block: slice, parity: ParityMode, tmp_path: Path
) -> None:
    torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    from equiparity.domain.experiment import ExperimentConfig, ModelHyperparams, TrainingParams
    from equiparity.training.crystal_regression import build_crystal_model

    torch.manual_seed(42)
    config = ExperimentConfig(
        seed=42,
        core=core,
        parity=parity,
        target=target,
        dataset="test",
        processed_npz=Path("unused"),
        split_npz=Path("unused"),
        output_dir=Path("unused"),
        protocol="validation_selected",
        model=ModelHyperparams(
            num_features=4,
            num_layers=3,
            l_max=lmax,
            r_max=3.0,
            pooling="mean",
            tensor_head="complete",
        ),
        training=TrainingParams(device="cpu", precision="float32"),
    )
    model, extra = build_crystal_model(
        config, {"avg_neighbors": 8.0, "elements": [1, 6, 8]}, torch.device("cpu")
    )
    atoms = Atoms(
        numbers=[8, 1, 6, 1],
        positions=[[0, 0, 0], [1.2, 0.1, 0.2], [0.3, 1.4, 0.1], [0.1, 0.2, 1.3]],
        cell=np.diag([4.1, 4.3, 4.5]),
        pbc=True,
    )

    def predict(atoms: Atoms) -> torch.Tensor:
        if core == "mace":
            from equiparity.training.mace_scalar import _batches, _to_mace_data

            batch = next(
                iter(
                    _batches(
                        [_to_mace_data(_structure(atoms), extra["z_table"], 3.0)], 1, extra["dtype"]
                    )
                )
            )
        else:
            from nequip.data import AtomicDataDict

            from equiparity.training.nequip_data import to_atomic_data

            batch = AtomicDataDict.batched_from_list(
                [to_atomic_data(_structure(atoms), extra["z_map"], 3.0, torch.float64)]
            )
        return model(batch)

    pred = predict(atoms)
    assert bool(torch.all(model.readout.output_mask == 1))
    assert pred[:, block].abs().max() > 1e-9
    pred[:, block].square().sum().backward()
    assert model.readout.product.weight.grad.abs().sum() > 0
    torch.testing.assert_close(predict(atoms.repeat((2, 1, 1))), pred, atol=3e-5, rtol=3e-4)
    q = np.diag([-1.0, 1.0, 1.0]) if parity == ParityMode.O3 else o3.rand_matrix().numpy()
    reflected = atoms.copy()
    reflected.positions = atoms.positions @ q.T
    reflected.set_cell(atoms.cell.array @ q.T)
    representation = (
        o3.Irreps(TARGETS[target].irreps)
        .D_from_matrix(torch.tensor(q, dtype=pred.dtype))
        .to(pred.dtype)
    )
    torch.testing.assert_close(predict(reflected), pred @ representation.T, atol=3e-5, rtol=3e-4)
    if parity == ParityMode.O3 and target == "piezoelectric":
        vectors = np.array([[0.7, 0.1, 0.2], [0.2, 0.8, 0.1], [0.1, 0.3, 0.9]])
        centro = Atoms(
            numbers=[8, 1, 6, 8, 1, 6],
            positions=np.concatenate([vectors, -vectors]),
            cell=atoms.cell,
            pbc=True,
        )
        assert float(predict(centro).detach().abs().max()) < 3e-6

    # Reload must use the selected state and persisted scale without reading a dataset.
    import yaml

    from equiparity.inference.reload import load_trained
    from equiparity.workflows.run_experiment import _config_snapshot

    snapshot = _config_snapshot(config)
    (tmp_path / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot))
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 1,
            "identity": {
                "config": snapshot,
                "scale": 2.5,
                "model_metadata": {"avg_neighbors": 8.0, "elements": [1, 6, 8]},
            },
        },
        tmp_path / "checkpoint_selected.pt",
    )
    restored = load_trained(tmp_path, repo_root=tmp_path, device=torch.device("cpu"))
    np.testing.assert_allclose(
        restored.predict([_structure(atoms)]), pred.detach().numpy() * 2.5, atol=3e-5, rtol=3e-4
    )
