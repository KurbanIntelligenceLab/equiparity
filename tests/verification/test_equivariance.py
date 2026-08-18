"""Unit tests for the parity classification thresholds (no models required)."""

from __future__ import annotations

from equiparity.verification.equivariance import THRESHOLDS, classify


def test_classify_o3_float64() -> None:
    t = THRESHOLDS["float64"]
    assert classify(rotation_error=1e-15, reflection_error=1e-15, thresholds=t) == "O3"


def test_classify_so3_float64() -> None:
    t = THRESHOLDS["float64"]
    # Rotation preserved, reflection broken -> genuine SO(3).
    assert classify(rotation_error=1e-15, reflection_error=1e-2, thresholds=t) == "SO3"


def test_classify_fail_ambiguous_reflection() -> None:
    t = THRESHOLDS["float64"]
    # Reflection error between the O(3) and SO(3) bounds is a FAIL.
    assert classify(rotation_error=1e-15, reflection_error=1e-8, thresholds=t) == "FAIL"


def test_classify_fail_broken_rotation() -> None:
    t = THRESHOLDS["float64"]
    # A model that breaks rotations is broken regardless of reflection error.
    assert classify(rotation_error=1e-2, reflection_error=1e-15, thresholds=t) == "FAIL"
    assert classify(rotation_error=1e-2, reflection_error=1e-2, thresholds=t) == "FAIL"


def test_float32_thresholds_are_looser() -> None:
    f32, f64 = THRESHOLDS["float32"], THRESHOLDS["float64"]
    assert f32.o3_reflection_max > f64.o3_reflection_max
    assert f32.so3_reflection_min > f64.so3_reflection_min
