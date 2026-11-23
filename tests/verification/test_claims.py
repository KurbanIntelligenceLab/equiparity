"""Recompute the reported measurements from released arrays and scientific records."""

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from equiparity.workflows.symmetry_masks import measure_masks
from equiparity.workflows.tensor_datasets import validate_tensor_dataset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def record(stem):
    return json.loads((RESULTS / f"{stem}.json").read_text())


@pytest.fixture(scope="module")
def predictions():
    with zipfile.ZipFile(RESULTS / "predictions.zip") as archive:
        yield archive


def load_array(archive, name):
    return np.load(io.BytesIO(archive.read(name)), allow_pickle=False)


def run_name(run):
    stem = "_".join(run[k] for k in ("core", "target", "parity", "loss"))
    control = f"_zero_w{run['weight']}" if run["augmented"] else ""
    return f"{stem}{control}_seed{run['seed']}"


def check_summary(measurements, summary):
    np.testing.assert_allclose(np.mean(measurements), summary["mean"])
    np.testing.assert_allclose(np.std(measurements, ddof=1), summary["sample_sd"], atol=1e-12)


def test_regression_and_percentage_reductions(predictions):
    data = record("crystal_regression")
    assert len(data["runs"]) == 54 and len(data["configurations"]) == 18
    grouped = defaultdict(list)
    keys = ("target", "core", "parity", "loss", "augmented", "weight")
    for r in data["runs"]:
        stem = run_name(r)
        with load_array(predictions, f"heldout/{stem}.npz") as raw:
            target = raw["target"].astype(float)
            mae = np.mean(np.abs(raw["prediction"].astype(float) - target))
            split = "piezoelectric_ieee_augmented" if r["augmented"] else r["target"]
            with np.load(ROOT / f"data/splits/mp_{split}_split.npz") as partitions:
                np.testing.assert_array_equal(raw["ids"], partitions["test"])
            np.testing.assert_allclose(np.abs(target).mean(), r["zero"], atol=1e-6)
        assert len(target) == r["test_n"]
        np.testing.assert_allclose(r["mae"], mae, rtol=0, atol=1e-9)
        baseline = min(r[k] for k in ("zero", "training_mean", "training_median"))
        assert mae < baseline
        np.testing.assert_allclose(100 * (1 - mae / baseline), r["reduction_percent"])
        grouped[tuple(r[k] for k in keys)].append(r)
    for group in data["configurations"]:
        runs = grouped[tuple(group[k] for k in keys)]
        assert {r["seed"] for r in runs} == {0, 1, 2}
        for metric in ("mae", "reduction_percent"):
            check_summary([r[metric] for r in runs], group[metric])
        values = [r["reduction_percent"] for r in runs]
        np.testing.assert_allclose(
            [min(values), max(values)], [group["reduction_percent"][k] for k in ("min", "max")]
        )


def test_class_flags_norms_and_percentiles(predictions):
    data = record("class_symmetry")
    quantiles = record("norm_percentiles")
    summaries = defaultdict(list)
    for row in data["runs"]:
        summaries[row["run"]].append(row)
    quant = {(r["core"], r["parity"], r["seed"]): r for r in quantiles["runs"]}
    primary = []
    changed = 0
    for name, rows in summaries.items():
        with load_array(predictions, f"symmetry/{name}.npz") as raw:
            labels = raw["point_group"]
            assert len(labels) == 2000 and len(set(labels)) == 11
            arrays = {
                k: raw[k]
                for k in ("selected_idealized", "selected_raw", "random_idealized", "random_raw")
            }
            for row in rows:
                values = arrays[f"{row['state']}_{row['geometry']}"][labels == row["group"]]
                assert len(values) == row["n"]
                np.testing.assert_allclose(100 * np.mean(values > 0.01), row["flag_percent"])
                np.testing.assert_allclose(np.median(values), row["median_norm"], rtol=1e-6)
            r = rows[0]
            if r["augmented"] or r["loss"] != "tensor_huber":
                continue
            values = arrays["selected_idealized"]
            q = quant[r["core"], r["arm"], r["seed"]]
            np.testing.assert_allclose(
                np.percentile(values, [5, 25, 50, 75, 95]),
                [q[k] for k in ("p05", "p25", "p50", "p75", "p95")],
            )
            if r["arm"] == "so3":
                primary.extend(rows)
                changed += np.sum((values > 0.01) != (arrays["selected_raw"] > 0.01))
            else:
                assert np.max(values) <= 0.01 and np.max(arrays["selected_raw"]) <= 0.01
    assert len(summaries) == 36 and changed == 1 and len(quant) == 18
    for field, keys in (
        ("by_class", ("group", "geometry", "state")),
        ("by_backbone", ("core", "group", "geometry", "state")),
    ):
        for g in data[field]:
            rows = [r for r in primary if all(r[k] == g[k] for k in keys)]
            for metric in ("flag_percent", "median_norm"):
                check_summary([r[metric] for r in rows], g[metric])
    for g in quantiles["by_backbone"]:
        rs = [r for r in quant.values() if (r["core"], r["parity"]) == (g["core"], g["parity"])]
        for k in ("p05", "p25", "p50", "p75", "p95"):
            check_summary([r[k] for r in rs], g[k])


def test_zero_supervision_and_external_masks(predictions):
    data = record("zero_supervision")
    for row in data["runs"]:
        with load_array(predictions, f"symmetry/{row['run']}.npz") as raw:
            assert len(raw["training_zero_ids"]) == data["n_training_zeros"] == 1016
            assert set(raw["ids"]).isdisjoint(raw["training_zero_ids"])
            for key, column in (
                ("train_selected", "training_zero_selected"),
                ("train_final", "training_zero_final"),
                ("unseen_selected", "selected_idealized"),
            ):
                np.testing.assert_allclose(100 * np.mean(raw[column] > 0.01), row[key])
        with load_array(predictions, f"heldout/{row['run']}.npz") as raw:
            assert len(raw["ids"]) == data["n_regression_test"]
            np.testing.assert_allclose(
                np.abs(raw["prediction"].astype(float) - raw["target"]).mean(), row["test_mae"]
            )
    for g in data["configurations"]:
        rs = [r for r in data["runs"] if (r["arm"], r["weight"]) == (g["arm"], g["weight"])]
        for k in ("train_selected", "train_final", "unseen_selected", "test_mae"):
            check_summary([r[k] for r in rs], g[k])
    assert measure_masks(RESULTS / "predictions.zip") == record("symmetry_masks")


def test_independent_cohort_and_displacement(predictions):
    with (RESULTS / "populations.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cohort = sorted(
        [r for r in rows if r["initialization_family"]],
        key=lambda r: int(r["initialization_index"]),
    )
    assert len(cohort) == sum(bool(r["symmetry_class"]) for r in rows) == 2000
    assert sum(bool(r["symmetry_class"]) for r in cohort) == 94
    families = np.asarray([r["initialization_family"] for r in cohort])
    for model, summary in record("initialization_controls").items():
        values = load_array(predictions, f"initialization/{model}.npy")
        median = np.median(values[families == "non-cubic"])
        np.testing.assert_allclose(
            np.mean(values > 0.01 * median), summary["fraction_above_1pct_of_noncubic_median"]
        )
        for family in set(families):
            v = values[families == family]
            assert len(v) == summary[family]["n"]
            np.testing.assert_allclose(
                [np.max(v), np.median(v)], [summary[family]["max"], summary[family]["median"]]
            )
    path = record("rutile_distortion")
    assert len(path["space_groups"]) == 33 and len(path["runs"]) == 33 * 18
    for g in path["mean_by_displacement"]:
        rs = [r for r in path["runs"] if all(r[k] == g[k] for k in ("core", "parity", "delta"))]
        assert {r["seed"] for r in rs} == {0, 1, 2}
        check_summary([r["norm_T"] for r in rs], g["norm_T"])
    assert all(r["false_flag_symmetrized"] == 0 for r in record("inversion_averaging").values())


def test_training_sources_and_squared_error():
    for target, stem in (
        ("elastic", "elastic_ieee"),
        ("piezoelectric", "piezoelectric_ieee"),
        ("piezoelectric", "piezoelectric_ieee_augmented"),
    ):
        path = ROOT / f"data/raw/mp/mp_{stem}_processed.npz"
        split = ROOT / f"data/splits/mp_{stem if 'augmented' in stem else target}_split.npz"
        validate_tensor_dataset(path, split, target)
        if stem != "piezoelectric_ieee":
            continue
        with np.load(path) as d, np.load(split) as parts:
            lookup = {mid: i for i, mid in enumerate(d["ids"])}
            train = (
                d[target][[lookup[mid] for mid in parts["train"]]].astype(float).reshape(-1, 3, 6)
            )
        power = np.sum(train**2 * [1, 1, 1, 2, 2, 2], axis=(1, 2))
        r = record("training_error_concentration")
        assert len(power) == r["n_train"] == 2649
        np.testing.assert_allclose(
            np.sort(power)[-r["n_largest"] :].sum() / power.sum(), r["fraction_of_squared_error"]
        )
