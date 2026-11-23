"""Create validated, versioned tensor data from property-associated MP structures."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from equiparity.domain.sample import LabeledStructure
from equiparity.io.tensor_frames import elastic_ieee_target, ieee_sample
from equiparity.workflows.prepare_data import _mp_token, _save_crystal_dataset, sha256_file

_log = logging.getLogger(__name__)


def _search_cached(
    root: Path,
    route: Any,  # noqa: ANN401
    kind: str,
    ids: list[str],
    fields: list[str],
    key: str,
) -> list[dict[str, Any]]:
    """Keep completed API batches and retry interrupted reads without losing progress."""
    from mp_api.client.core.exceptions import MPRestError
    from requests.exceptions import RequestException

    cache = root / ".mp_cache/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    size = 25 if key == "task_ids" else 100
    for start in range(0, len(ids), size):
        query = {key: ids[start : start + size], "fields": fields}
        digest = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        path = cache / f"{kind}_{digest}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            for attempt in range(3):
                try:
                    payload = route.search(**query)
                    break
                except (MPRestError, RequestException):
                    if attempt == 2:
                        raise
                    _log.warning("retrying %s batch %d", kind, start // size + 1)
                    time.sleep(2**attempt)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload) + "\n")
            temporary.replace(path)
        result.extend(payload)
        _log.info("%s: retrieved %d/%d requested IDs", kind, min(start + size, len(ids)), len(ids))
    return result


def _documents(root: Path, target: str, ids: list[str]) -> list[dict[str, Any]]:
    from mp_api.client import MPRester

    with MPRester(
        _mp_token(root), mute_progress_bars=True, use_document_model=False, timeout=45
    ) as mpr:
        route = (
            mpr.materials.piezoelectric if target == "piezoelectric" else mpr.materials.elasticity
        )
        field = "total" if target == "piezoelectric" else "elastic_tensor"
        fields = ["material_id", field, "origins"]
        if "structure" in route.available_fields:
            fields.append("structure")
        docs = _search_cached(root, route, target, ids, fields, "material_ids")
        # Piezoelectric structures may be excluded from the property API schema.
        # Only the exact originating calculation is an acceptable fallback.
        if target == "piezoelectric":
            _attach_origin_structures(docs, mpr.materials.tasks, root)
        else:
            provenance = _search_cached(
                root,
                route,
                "elastic_provenance",
                ids,
                ["material_id", "fitting_data.optimization_task", "fitting_data.deformation_tasks"],
                "material_ids",
            )
            by_id = {d["material_id"]: d.get("fitting_data") for d in provenance}
            for doc in docs:
                doc["fitting_data"] = by_id.get(doc["material_id"])
        return docs


def _attach_origin_structures(
    docs: list[dict[str, Any]],
    tasks: Any,  # noqa: ANN401
    root: Path | None = None,
) -> None:
    from emmet.core.mpid import AlphaID

    requested: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        if doc.get("structure"):
            continue
        origins = {
            str(o["task_id"])
            for o in doc.get("origins", [])
            if o.get("name") == "piezoelectric" and o.get("task_id")
        }
        if len(origins) == 1:
            requested.setdefault(next(iter(origins)), []).append(doc)
    by_number = {int(AlphaID(task_id)): group for task_id, group in requested.items()}
    ids = list(requested)
    fields = ["task_id", "structure" if "structure" in tasks.available_fields else "output"]
    records = (
        _search_cached(root, tasks, "piezo_tasks", ids, fields, "task_ids")
        if root is not None
        else tasks.search(task_ids=ids, fields=fields)
        if ids
        else []
    )
    for task in records:
        task_id = str(task["task_id"])
        structure = task.get("structure") or (task.get("output") or {}).get("structure")
        if structure:
            for doc in by_number.get(int(AlphaID(task_id)), []):
                doc["structure"] = structure
                doc["structure_source"] = {
                    "task_id": task_id,
                    "document": {
                        k: task[k] for k in ("task_id", "structure", "output") if k in task
                    },
                }


def prepare_tensor_dataset(
    root: Path, target: str, documents: Path | None = None
) -> dict[str, Any]:
    """Preserve original IDs/splits; publish no dataset unless every record passes.

    ``documents`` accepts an archived JSON list of property documents. This allows
    provenance-preserving ingestion without account credentials on a GPU worker.
    """
    from pymatgen.core.structure import Structure

    if target not in ("piezoelectric", "elastic"):
        raise ValueError("crystal data target must be piezoelectric or elastic")
    cohort = root / f"data/manifests/{target}_ids.json"
    ids = json.loads(cohort.read_text())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("original dataset identifiers must be nonempty and unique")
    docs = (
        json.loads(
            gzip.decompress(documents.read_bytes())
            if documents.suffix == ".gz"
            else documents.read_bytes()
        )
        if documents
        else _documents(root, target, ids)
    )
    if not isinstance(docs, list):
        raise ValueError("documents must contain a JSON list of property documents")
    aliases = {mid: mid for mid in ids}
    if any(str(d["material_id"]) not in aliases for d in docs):
        from emmet.core.mpid import AlphaID

        aliases.update({str(AlphaID(mid, padlen=8)): mid for mid in ids})
    by_id = {aliases.get(str(d["material_id"]), str(d["material_id"])): d for d in docs}
    if len(by_id) != len(docs):
        raise ValueError("duplicate property document identifiers")
    out_dir = root / "data/property_sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_bytes = (json.dumps(docs, sort_keys=True, indent=2) + "\n").encode()
    source_path = out_dir / f"{target}.json.gz"
    compressed = gzip.compress(source_bytes, mtime=0)
    if source_path.exists() and gzip.decompress(source_path.read_bytes()) != source_bytes:
        digest = hashlib.sha256(source_bytes).hexdigest()
        source_path = out_dir / f"{target}_{digest}.json.gz"
    if not source_path.exists():
        source_path.write_bytes(compressed)
    samples: list[LabeledStructure] = []
    accepted, rejected = [], []
    for mid in ids:
        doc = by_id.get(mid)
        try:
            if not doc or not doc.get("structure"):
                raise ValueError(f"{mid}: missing property-associated structure")
            fitting = doc.get("fitting_data") or {}
            if not doc.get("origins") and not (
                target == "elastic" and fitting.get("deformation_tasks")
            ):
                raise ValueError(f"{mid}: missing property calculation provenance")
            structure = Structure.from_dict(doc["structure"])
            precision_audit: dict[str, float] = {}
            if target == "piezoelectric":
                values = np.asarray(doc["total"], dtype=np.float64)
            else:
                values, precision_audit = elastic_ieee_target(
                    structure,
                    np.asarray(doc["elastic_tensor"]["raw"], dtype=np.float64),
                    np.asarray(doc["elastic_tensor"]["ieee_format"], dtype=np.float64),
                )
            sample, audit = ieee_sample(
                structure,
                np.asarray(values, dtype=np.float64),
                target,
                mid,
            )
            samples.append(sample)
            accepted.append(
                {
                    "material_id": mid,
                    "api_material_id": str(doc["material_id"]),
                    "origins": doc.get("origins", []),
                    "fitting_data": fitting,
                    **precision_audit,
                    **audit,
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"material_id": mid, "reason": str(exc)})
    report: dict[str, Any] = {
        "protocol": "validation_selected",
        "target": target,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sha256_file(source_path),
        "source_file": os.path.relpath(source_path, root / "data/raw/mp"),
        "cohort_sha256": sha256_file(cohort),
        "cohort_file": os.path.relpath(cohort, root / "data/raw/mp"),
        "frame": "IEEE tensor and property-associated geometry",
        "packages": {
            n: importlib.metadata.version(n) for n in ("pymatgen", "spglib", "mp-api", "emmet-core")
        },
        "accepted": accepted,
        "rejected": rejected,
        "passed": not rejected,
    }
    if rejected:
        report_path = root / ".mp_cache" / f"{target}_audit.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise ValueError(
            f"{len(rejected)}/{len(ids)} records rejected; inspect {report_path}; "
            "no dataset published"
        )
    name = f"mp_{target}_ieee"
    path = root / f"data/raw/mp/{name}_processed.npz"
    split = root / f"data/splits/mp_{target}_split.npz"
    _validate_splits(ids, split)
    with tempfile.TemporaryDirectory(dir=out_dir) as staging:
        staging_root = Path(staging)
        _save_crystal_dataset(staging_root, name, samples, target)
        (staging_root / f"data/raw/mp/{name}_processed.npz").replace(path)
    report.update(
        {"data_sha256": sha256_file(path), "split_sha256": sha256_file(split), "n": len(samples)}
    )
    report_path = path.with_suffix(".frame.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {"path": str(path), "audit": str(report_path), "n": len(samples), "passed": True}


def validate_tensor_dataset(path: Path, split: Path, target: str) -> dict[str, Any]:
    """Reject unvalidated, stale, or differently partitioned crystal data before training."""
    report = json.loads(path.with_suffix(".frame.json").read_text())
    if (
        report.get("protocol") != "validation_selected"
        or not report.get("passed")
        or report.get("target") != target
    ):
        raise ValueError("crystal dataset has not passed the frame audit")
    if report.get("data_sha256") != sha256_file(path) or report.get("split_sha256") != sha256_file(
        split
    ):
        raise ValueError("crystal dataset/split digest does not match the frame audit")
    source = path.parent / report["source_file"]
    if report.get("source_sha256") != sha256_file(source):
        raise ValueError("property source digest does not match the frame audit")
    if (
        "cohort_file" in report
        and sha256_file(path.parent / report["cohort_file"]) != report["cohort_sha256"]
    ):
        raise ValueError("cohort digest does not match the frame audit")
    for parent in report.get("parents", []):
        if sha256_file(path.parent / parent["path"]) != parent["sha256"]:
            raise ValueError("parent dataset/source digest mismatch")
    with np.load(path, allow_pickle=False) as data:
        _validate_splits(data["ids"].tolist(), split)
    return dict(report)


def _validate_splits(identifiers: list[str], split: Path) -> None:
    ids = set(identifiers)
    if len(ids) != len(identifiers):
        raise ValueError("duplicate dataset identifiers")
    with np.load(split, allow_pickle=False) as partitions:
        parts = [set(partitions[k].tolist()) for k in ("train", "val", "test")]
        if any(
            len(p) != len(partitions[k])
            for p, k in zip(parts, ("train", "val", "test"), strict=True)
        ):
            raise ValueError("duplicate split identifiers")
        if any(parts[i] & parts[j] for i in range(3) for j in range(i)) or set.union(*parts) != ids:
            raise ValueError("crystal splits must partition the dataset without overlap")
