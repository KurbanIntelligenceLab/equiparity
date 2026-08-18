"""Tests for the Voigt <-> irreps tensor conversion (requires e3nn)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("e3nn")

from equiparity.features.tensor_irreps import (
    elastic_voigt_to_cartesian,
    piezo_voigt_to_cartesian,
    voigt_to_irreps,
)


def test_piezo_dims_and_norm_preserved() -> None:
    rng = np.random.default_rng(0)
    voigt = rng.standard_normal((3, 6))
    irreps = voigt_to_irreps(voigt, "piezoelectric")
    assert irreps.shape == (18,)  # 2x1o + 1x2o + 1x3o
    cartesian = piezo_voigt_to_cartesian(voigt)
    # orthonormal change of basis preserves the Frobenius norm
    np.testing.assert_allclose(np.linalg.norm(irreps), np.linalg.norm(cartesian), rtol=1e-6)


def test_elastic_dims_and_norm_preserved() -> None:
    rng = np.random.default_rng(1)
    voigt = rng.standard_normal((6, 6))
    voigt = (voigt + voigt.T) / 2  # elastic Voigt matrix is symmetric
    irreps = voigt_to_irreps(voigt, "elastic")
    assert irreps.shape == (21,)  # 2x0e + 2x2e + 1x4e
    cartesian = elastic_voigt_to_cartesian(voigt)
    np.testing.assert_allclose(np.linalg.norm(irreps), np.linalg.norm(cartesian), rtol=1e-6)


def test_piezo_cartesian_symmetric_in_strain_pair() -> None:
    rng = np.random.default_rng(2)
    cartesian = piezo_voigt_to_cartesian(rng.standard_normal((3, 6)))
    np.testing.assert_array_equal(cartesian, np.transpose(cartesian, (0, 2, 1)))


def test_zero_tensor_maps_to_zero() -> None:
    irreps = voigt_to_irreps(np.zeros((3, 6)), "piezoelectric")
    np.testing.assert_array_equal(irreps, np.zeros(18))
