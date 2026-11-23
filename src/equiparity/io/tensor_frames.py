"""Coordinate-frame validation at the Materials Project boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from equiparity.domain.sample import LabeledStructure
from equiparity.features.tensor_irreps import elastic_voigt_to_cartesian, piezo_voigt_to_cartesian
from equiparity.io.materials_project import tensor_sample


def elastic_ieee_target(
    structure: Any,  # noqa: ANN401
    raw: npt.NDArray[np.float64],
    published: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """Recover full precision using MP's documented target construction.

    Emmet publishes round(voigt_symmetrized(raw).convert_to_ieee(structure)).
    Retain site metadata and require agreement with the 1 GPa quantization
    interval, rather than projecting a label to match model constraints.
    """
    from pymatgen.analysis.elasticity.elastic import ElasticTensor

    if raw.shape != (6, 6) or published.shape != (6, 6):
        raise ValueError("invalid elastic source tensor shape")
    if not np.isfinite(raw).all() or not np.isfinite(published).all():
        raise ValueError("nonfinite elastic source tensor")
    canonical = ElasticTensor.from_voigt(raw).voigt_symmetrized.convert_to_ieee(structure)
    values = np.asarray(canonical.voigt, dtype=np.float64)
    error = float(np.abs(values - published).max())
    if error > 0.500001:
        raise ValueError("elastic reconstruction disagrees with published IEEE rounding interval")
    return values, {"published_rounding_error_max_gpa": error}


def ieee_sample(
    structure: Any,  # noqa: ANN401
    voigt: npt.NDArray[np.float64],
    target: str,
    identifier: str,
    *,
    tolerance: float = 0.01,
    tensor_frame: str = "ieee",
) -> tuple[LabeledStructure, dict[str, Any]]:
    """Rotate the property-associated structure into the tensor's IEEE frame.

    Never infer a rotation by minimizing disagreement with the label. The structure
    must be the one attached to the property document, not a summary-endpoint proxy.
    Large symmetry residuals are rejected rather than projected out of the label.
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core.tensors import Tensor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    expected = (3, 6) if target == "piezoelectric" else (6, 6)
    if voigt.shape != expected or not np.isfinite(voigt).all():
        raise ValueError(f"{identifier}: invalid {target} tensor shape or nonfinite entries")
    if target == "elastic" and not np.allclose(voigt, voigt.T, atol=1e-6):
        raise ValueError(f"{identifier}: elastic tensor lacks major symmetry")
    raw_rotation = np.asarray(Tensor.get_ieee_rotation(structure), dtype=np.float64)
    if np.linalg.norm(raw_rotation.T @ raw_rotation - np.eye(3)) > 1e-5:
        raise ValueError(f"{identifier}: IEEE transformation is not orthogonal")
    left, _, right = np.linalg.svd(raw_rotation)
    rotation = left @ right
    site_properties = {}
    if target == "elastic":
        moments = np.asarray(
            structure.site_properties.get("magmom", np.zeros(len(structure))), dtype=np.float64
        )
        magnitudes = np.abs(moments) if moments.ndim == 1 else np.linalg.norm(moments, axis=1)
        if magnitudes.shape != (len(structure),) or not np.isfinite(magnitudes).all():
            raise ValueError(f"{identifier}: invalid magnetic site metadata")
        # The norm is invariant under rotations, reflections and time reversal.
        # Audit exactly the scalar-decorated input supplied to the model.
        site_properties = {"magmom": magnitudes.tolist(), "magnetic_magnitude": magnitudes.tolist()}
    rotated = Structure(
        np.asarray(structure.lattice.matrix) @ rotation.T,
        structure.species,
        np.asarray(structure.cart_coords) @ rotation.T,
        coords_are_cartesian=True,
        site_properties=site_properties,
    )
    converter = (
        piezo_voigt_to_cartesian if target == "piezoelectric" else elastic_voigt_to_cartesian
    )
    if tensor_frame == "structure":
        voigt = np.asarray(Tensor(converter(voigt)).rotate(rotation).voigt)
    elif tensor_frame != "ieee":
        raise ValueError("tensor_frame must be ieee or structure")
    tensor = converter(voigt)
    norm = float(np.linalg.norm(tensor))
    residuals = {}
    symmetries = {}
    # Include the tolerance used to determine IEEE axes. Tight tolerances alone
    # can return only the identity for slightly distorted DFT structures.
    for symprec in (0.1, 1e-3, 1e-5):
        analyzer = SpacegroupAnalyzer(rotated, symprec=symprec, angle_tolerance=-1)
        ops = analyzer.get_symmetry_operations(cartesian=True)
        rotations = []
        corrections = []
        for op in ops:
            # Approximate symmetries of a strained DFT cell can contain tiny
            # nonorthogonal terms. Test the nearest orthogonal operation and
            # record its correction; never transform a tensor as if strain
            # were a rotation, and never alter the target label.
            left, _, right = np.linalg.svd(op.rotation_matrix)
            orthogonal = left @ right
            rotations.append(orthogonal)
            corrections.append(float(np.linalg.norm(orthogonal - op.rotation_matrix)))
        symmetries[str(symprec)] = {
            "spacegroup": analyzer.get_space_group_number(),
            "n_operations": len(ops),
            "max_operation_orthogonalization": max(corrections),
        }
        projected = np.mean([Tensor(tensor).rotate(rotation) for rotation in rotations], axis=0)
        residual = float(np.linalg.norm(tensor - projected) / max(norm, 1e-12))
        residuals[str(symprec)] = residual
        if symprec < 0.1 and residual > tolerance:
            raise ValueError(
                f"{identifier}: IEEE frame symmetry residual {residual:.6g} at {symprec}"
            )
    sample = tensor_sample(rotated, voigt, target, identifier)
    return sample, {
        "input_site_features": ["magnetic_magnitude"] if target == "elastic" else [],
        "rotation": rotation.tolist(),
        "rotation_orthogonalization": float(np.linalg.norm(rotation - raw_rotation)),
        "source_tensor_frame": tensor_frame,
        "approximate_symmetry_flag": residuals["0.1"] > tolerance,
        "relative_residuals": residuals,
        "symmetry": symmetries,
    }
