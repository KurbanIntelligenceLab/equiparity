"""Per-class parity diagnostics for trained and matched random tensor heads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from equiparity.inference.reload import load_trained
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.training.crystal_regression import build_crystal_model
from equiparity.training.tensor_optimization import atomic_json
from equiparity.verification.parity_gap import parity_gap_rows
from equiparity.workflows.prepare_data import sha256_file
from equiparity.workflows.tensor_datasets import validate_tensor_dataset


def class_summary(classes: np.ndarray, magnitudes: np.ndarray) -> list[dict[str, Any]]:
    """Retain all eleven classes, explicitly recording empty populations."""
    if classes.shape != magnitudes.shape or not np.isfinite(magnitudes).all():
        raise ValueError("finite, aligned class labels and norms required")
    rows = []
    for theory in parity_gap_rows():
        values = magnitudes[classes == theory["point_group"]]
        rows.append(
            {
                "point_group": theory["point_group"],
                "gamma3": theory["gap_rank3_piezoelectric"],
                "n": len(values),
                "false_flag_fraction": float((values > 0.01).mean()) if len(values) else None,
                "median_norm": float(np.median(values)) if len(values) else None,
                "max_norm": float(values.max()) if len(values) else None,
            }
        )
    if sum(r["n"] for r in rows) != len(classes):
        raise ValueError("evaluation contains a non-centrosymmetric class")
    return rows


def evaluate_crystal_symmetry(
    root: Path, run_dir: Path, output_dir: Path, device: str = "cuda"
) -> dict[str, Any]:
    """Compare selected weights with a fresh matched initialization, without test targets."""
    import spglib

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    trained = load_trained(run_dir, repo_root=root, device=torch.device(device))
    cfg = trained.config
    if cfg.protocol != "validation_selected" or cfg.target != "piezoelectric":
        raise ValueError("parity audit requires a crystal piezoelectric checkpoint")
    audit = validate_tensor_dataset(root / cfg.processed_npz, root / cfg.split_npz, cfg.target)
    selected = torch.load(
        run_dir / "checkpoint_selected.pt", map_location="cpu", weights_only=False
    )
    for key in ("data_sha256", "split_sha256"):
        if selected["identity"][key] != audit[key]:
            raise ValueError(f"selected checkpoint {key} mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    populations = {}
    data_hashes = {}
    for name, suffix in [("idealized", ""), ("raw", "_raw")]:
        path = root / f"data/raw/mp/mp_ood_centrosymmetric_processed{suffix}.npz"
        populations[name] = CrystalDataset(load_crystal_dataset(path))
        data_hashes[name] = sha256_file(path)
    ideal = populations["idealized"]
    classes = []
    for sample in ideal:
        s = sample.structure
        dataset = spglib.get_symmetry_dataset(
            (s.cell, s.positions @ np.linalg.inv(s.cell), s.atomic_numbers), symprec=1e-5
        )
        if dataset is None:
            raise ValueError(f"{sample.identifier}: no symmetry classification")
        classes.append(dataset.pointgroup)
    classes_array = np.asarray(classes)
    ids = np.asarray([sample.identifier for sample in ideal])
    if not np.array_equal(ids, np.asarray([s.identifier for s in populations["raw"]])):
        raise ValueError("raw and idealized population IDs differ")
    arrays = {"ids": ids, "point_group": classes_array}
    records = {}
    for state in ["selected", "random"]:
        if state == "random":
            trained.model.to("cpu")
            torch.manual_seed(cfg.seed)
            random_model, extra = build_crystal_model(
                cfg, selected["identity"]["model_metadata"], torch.device(device)
            )
            random_model.eval()
            trained = replace(trained, model=random_model, _extra=extra)
        for variant, population in populations.items():
            prediction = trained.predict([s.structure for s in population])
            norms = np.linalg.norm(prediction, axis=1)
            key = state + "_" + variant
            arrays[key] = norms
            records[key] = class_summary(classes_array, norms)
    # Training-zero residuals at both selected and final weights identify underfitting.
    data = load_crystal_dataset(root / cfg.processed_npz, (cfg.target,))
    train = CrystalDataset(data, load_split(root / cfg.split_npz, "train"))
    zeros = [s for s in train if np.all(s.targets[cfg.target] == 0)]
    if zeros:
        trained.model.to("cpu")
        trained = load_trained(run_dir, repo_root=root, device=torch.device(device))
        arrays["training_zero_ids"] = np.asarray([s.identifier for s in zeros])
        for state in ["selected", "final"]:
            if state == "final":
                checkpoint = torch.load(
                    run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False
                )
                trained.model.load_state_dict(checkpoint["model"])
            norms = trained.violations([s.structure for s in zeros])
            arrays["training_zero_" + state] = norms
            records["training_zero_" + state] = {
                "n": len(zeros),
                "false_flag_fraction": float((norms > 0.01).mean()),
                "median_norm": float(np.median(norms)),
                "max_norm": float(norms.max()),
            }
    np.savez_compressed(output_dir / "parity_vectors.npz", **arrays)
    result = {
        "run": str(run_dir),
        "core": cfg.core,
        "parity": cfg.parity.value,
        "seed": cfg.seed,
        "threshold_c_m2": 0.01,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoint_selected.pt"),
        "data_hashes": data_hashes,
        "classes_defined_on": "idealized geometry at symprec=1e-5 A",
        "random_normalization": "same saved training RMS as selected model",
        "test_targets_used": False,
        "records": records,
    }
    atomic_json(result, output_dir / "parity_diagnostics.json")
    return result
