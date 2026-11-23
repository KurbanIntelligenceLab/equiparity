"""Zero training must not reuse OOD or original tensor materials."""

import json
from pathlib import Path

import numpy as np
import pytest

from equiparity.workflows.zero_supervision import choose_zero_ids


def test_zero_cohort_size_uniqueness_and_disjointness(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/mp"
    splits = tmp_path / "data/splits"
    raw.mkdir(parents=True)
    splits.mkdir(parents=True)
    (tmp_path / "results").mkdir()
    np.savez(raw / "mp_piezoelectric_ieee_processed.npz", ids=["base"])
    np.savez(raw / "mp_ood_centrosymmetric_processed.npz", ids=["zero0", "zero1"])
    cohort = splits / "zero_label_ids.json"
    ids = [f"zero{i}" for i in range(2, 1002)]
    cohort.write_text(json.dumps(ids))
    assert choose_zero_ids(tmp_path) == ids
    for invalid in (ids[:-1], [*ids[:-1], "base"], [*ids[:-1], "zero0"], [*ids[:-1], ids[0]]):
        cohort.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="distinct and disjoint"):
            choose_zero_ids(tmp_path)
