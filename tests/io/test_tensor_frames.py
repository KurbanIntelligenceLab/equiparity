"""Frame corrections preserve labels and reject inconsistent property documents."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen.core")
from pymatgen.core import Lattice, Structure
from pymatgen.core.tensors import Tensor

from equiparity.io.tensor_frames import ieee_sample
from equiparity.workflows.tensor_datasets import prepare_tensor_dataset, validate_tensor_dataset


def _property() -> tuple[Structure, np.ndarray]:
    structure = Structure(Lattice.tetragonal(3.1, 5.2), [30, 8], [[0, 0, 0], [0, 0, 0.31]])
    cart = np.zeros((3, 3, 3))
    cart[2, 2, 2] = 3
    cart[2, 0, 0] = cart[2, 1, 1] = 1
    cart[0, 0, 2] = cart[0, 2, 0] = cart[1, 1, 2] = cart[1, 2, 1] = 2
    angle = 0.47
    q = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rotated = Structure(
        structure.lattice.matrix @ q.T,
        structure.species,
        structure.cart_coords @ q.T,
        coords_are_cartesian=True,
    )
    native = Tensor(cart).rotate(q)
    ieee = native.convert_to_ieee(rotated, initial_fit=False).voigt
    return rotated, np.array(ieee)


def test_align_property_geometry_without_projecting_labels() -> None:
    structure, labels = _property()
    sample, audit = ieee_sample(structure, labels, "piezoelectric", "test")
    np.testing.assert_array_equal(sample.targets["piezoelectric"], labels.ravel())
    assert max(audit["relative_residuals"].values()) < 1e-8
    assert not np.allclose(sample.structure.cell, structure.lattice.matrix)
    bad = labels.copy()
    bad[0, 0] += 10
    with pytest.raises(ValueError, match="symmetry residual"):
        ieee_sample(structure, bad, "piezoelectric", "bad")


def test_tensor_datasets_preserves_splits_and_rejects_stale_bytes(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/mp"
    split_dir = tmp_path / "data/splits"
    raw.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    from emmet.core.mpid import AlphaID

    ids = ["mp-1", "mp-2", "mp-3"]
    cohort = tmp_path / "data/manifests/piezoelectric_ids.json"
    cohort.parent.mkdir()
    cohort.write_text(json.dumps(ids))
    split = split_dir / "mp_piezoelectric_split.npz"
    np.savez(split, train=ids[:1], val=ids[1:2], test=ids[2:])
    structure, labels = _property()
    docs = [
        {
            "material_id": str(AlphaID(mid, padlen=8)),
            "structure": structure.as_dict(),
            "total": labels.tolist(),
            "origins": [{"name": "piezoelectric", "task_id": "task-" + mid}],
        }
        for mid in ids
    ]
    source = tmp_path / "documents.json"
    source.write_text(json.dumps(docs))
    result = prepare_tensor_dataset(tmp_path, "piezoelectric", source)
    path = Path(result["path"])
    assert validate_tensor_dataset(path, split, "piezoelectric")["n"] == 3
    audit = validate_tensor_dataset(path, split, "piezoelectric")
    archived_source = path.parent / audit["source_file"]
    original_source = archived_source.read_bytes()
    archived_source.write_bytes(original_source + b" ")
    with pytest.raises(ValueError, match="source digest"):
        validate_tensor_dataset(path, split, "piezoelectric")
    archived_source.write_bytes(original_source)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        validate_tensor_dataset(path, split, "piezoelectric")


def test_missing_property_geometry_fails_without_new_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/mp"
    raw.mkdir(parents=True)
    cohort = tmp_path / "data/manifests/elastic_ids.json"
    cohort.parent.mkdir()
    cohort.write_text(json.dumps(["missing"]))
    source = tmp_path / "documents.json"
    source.write_text('[{"material_id": "missing"}]')
    with pytest.raises(ValueError, match="records rejected"):
        prepare_tensor_dataset(tmp_path, "elastic", source)
    assert not (raw / "mp_elastic_ieee_processed.npz").exists()


def test_origin_structure_lookup_uses_only_unambiguous_property_task() -> None:
    from equiparity.workflows.tensor_datasets import _attach_origin_structures

    docs = [
        {"origins": [{"name": "piezoelectric", "task_id": "mp-1"}]},
        {"origins": [{"name": "structure", "task_id": "mp-2"}]},
        {
            "origins": [
                {"name": "piezoelectric", "task_id": "mp-3"},
                {"name": "piezoelectric", "task_id": "mp-4"},
            ]
        },
    ]

    class Tasks:
        available_fields = ("task_id", "output")

        def search(self, task_ids: list[str], fields: list[str]) -> list[dict]:
            assert task_ids == ["mp-1"]
            return [{"task_id": "mp-1", "output": {"structure": {"source": "exact task"}}}]

    _attach_origin_structures(docs, Tasks())
    assert docs[0]["structure"] == {"source": "exact task"}
    assert "structure" not in docs[1]
    assert "structure" not in docs[2]


def test_interrupted_download_retries_and_reuses_completed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from requests.exceptions import ChunkedEncodingError

    from equiparity.workflows.tensor_datasets import _search_cached

    monkeypatch.setattr("equiparity.workflows.tensor_datasets.time.sleep", lambda seconds: None)

    class Route:
        calls = 0

        def search(self, **query: object) -> list[dict]:
            self.calls += 1
            if self.calls == 1:
                raise ChunkedEncodingError("interrupted transfer")
            assert query == {"material_ids": ["mp-1"], "fields": ["material_id"]}
            return [{"material_id": "mp-1"}]

    route = Route()
    args = (tmp_path, route, "test", ["mp-1"], ["material_id"], "material_ids")
    assert _search_cached(*args) == [{"material_id": "mp-1"}]
    assert route.calls == 2
    assert _search_cached(*args) == [{"material_id": "mp-1"}]
    assert route.calls == 2


def test_slightly_strained_operations_are_checked_as_rotations() -> None:
    structure, labels = _property()
    strain = np.eye(3)
    strain[0, 1] = 0.0007
    distorted = Structure(
        structure.lattice.matrix @ strain.T, structure.species, structure.frac_coords
    )
    sample, audit = ieee_sample(distorted, labels, "piezoelectric", "distorted")
    np.testing.assert_array_equal(sample.targets["piezoelectric"], labels.ravel())
    assert audit["symmetry"]["0.1"]["n_operations"] > 1
    assert audit["symmetry"]["0.1"]["max_operation_orthogonalization"] > 0
    assert max(audit["relative_residuals"].values()) < 0.01


def test_structure_frame_elastic_tensor_rotates_jointly_without_rounding() -> None:
    from equiparity.features.tensor_irreps import elastic_voigt_to_cartesian

    structure, _ = _property()
    tensor = np.einsum("ij,kl->ijkl", np.eye(3), np.eye(3)) * 7.12345
    tensor += (
        np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3))
        + np.einsum("il,jk->ijkl", np.eye(3), np.eye(3))
    ) * 2.34567
    axis = structure.lattice.matrix[2] / np.linalg.norm(structure.lattice.matrix[2])
    tensor += 3.789 * np.einsum("i,j,k,l->ijkl", axis, axis, axis, axis)
    labels = Tensor(tensor).voigt
    sample, audit = ieee_sample(structure, labels, "elastic", "elastic", tensor_frame="structure")
    actual = elastic_voigt_to_cartesian(sample.targets["elastic"].reshape(6, 6))
    rotation = np.asarray(audit["rotation"])
    expected = np.einsum("ia,jb,kc,ld,abcd->ijkl", rotation, rotation, rotation, rotation, tensor)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(actual), np.linalg.norm(tensor), atol=1e-12)
    assert not np.allclose(sample.targets["elastic"], np.round(sample.targets["elastic"]))


def test_elastic_reconstruction_must_match_published_rounding() -> None:
    from pymatgen.analysis.elasticity.elastic import ElasticTensor

    from equiparity.io.tensor_frames import elastic_ieee_target

    structure, _ = _property()
    raw = np.diag([12.345, 12.345, 18.123, 3.456, 3.456, 4.567])
    reference = ElasticTensor.from_voigt(raw).voigt_symmetrized.convert_to_ieee(structure).voigt
    recovered, audit = elastic_ieee_target(structure, raw, np.round(reference))
    np.testing.assert_allclose(recovered, reference)
    assert audit["published_rounding_error_max_gpa"] <= 0.5
    with pytest.raises(ValueError, match="rounding interval"):
        elastic_ieee_target(structure, raw, np.round(reference) + 2)


def test_elastic_magnetic_magnitudes_survive_archive_roundtrip(tmp_path: Path) -> None:
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset
    from equiparity.workflows.prepare_data import _save_crystal_dataset

    structure = Structure(
        Lattice.orthorhombic(3.1, 4.2, 5.3),
        [26, 26],
        [[0, 0, 0], [0.3, 0.2, 0.4]],
        site_properties={"magmom": [-1.5, 2.0]},
    )
    # An isotropic rank-four tensor satisfies every site-decorated symmetry.
    tensor = 100 * np.eye(6)
    tensor[3:, 3:] *= 0.5
    sample, audit = ieee_sample(structure, tensor, "elastic", "magnetic")
    np.testing.assert_array_equal(sample.structure.magnetic_magnitudes, [1.5, 2.0])
    np.testing.assert_array_equal(sample.targets["elastic"], tensor.ravel())
    assert audit["input_site_features"] == ["magnetic_magnitude"]
    _save_crystal_dataset(tmp_path, "magnetic", [sample], "elastic")
    data = load_crystal_dataset(tmp_path / "data/raw/mp/magnetic_processed.npz", ("elastic",))
    np.testing.assert_array_equal(CrystalDataset(data)[0].structure.magnetic_magnitudes, [1.5, 2.0])
    structure.add_site_property("magmom", [1.5, -2.0])
    flipped, _ = ieee_sample(structure, tensor, "elastic", "flipped")
    np.testing.assert_array_equal(
        flipped.structure.magnetic_magnitudes, sample.structure.magnetic_magnitudes
    )
