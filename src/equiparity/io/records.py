"""Released result records: the ``results/index.json`` catalogue and repository-root discovery.

The index is the single machine-readable map from each record to what it contains, the manuscript
items it backs, and, where a CLI command rebuilds it, that command. ``results/README.md`` is its
human-readable rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

INDEX_PATH = Path("results/index.json")
# Files in results/ that describe the records rather than being records.
_NON_RECORDS = frozenset({"README.md", "index.json"})


@dataclass(frozen=True)
class Record:
    """One released result record."""

    file: str
    contents: str
    manuscript_items: tuple[str, ...]
    producer: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain mapping for JSON output."""
        out: dict[str, object] = {
            "file": self.file,
            "contents": self.contents,
            "manuscript_items": list(self.manuscript_items),
        }
        if self.producer is not None:
            out["producer"] = self.producer
        return out


def find_repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor of ``start`` (default: the working directory) holding the results index."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / INDEX_PATH).is_file():
            return candidate
    raise FileNotFoundError(
        f"no {INDEX_PATH} found above {here}; run from an equiparity checkout or pass --root"
    )


def load_records(root: Path) -> list[Record]:
    """Parse ``results/index.json`` under ``root``."""
    raw = json.loads((root / INDEX_PATH).read_text())
    return [
        Record(
            file=str(r["file"]),
            contents=str(r["contents"]),
            manuscript_items=tuple(str(i) for i in r["manuscript_items"]),
            producer=r.get("producer"),
        )
        for r in raw["records"]
    ]


def get_record(root: Path, name: str) -> Record:
    """Look up a record by file name, with or without its extension."""
    records = load_records(root)
    for record in records:
        if name in (record.file, Path(record.file).stem):
            return record
    raise KeyError(f"no record {name!r}; known: {', '.join(r.file for r in records)}")


def check_index(root: Path) -> list[str]:
    """Problems with the index: listed records missing on disk, or records on disk not listed."""
    results = root / "results"
    listed = {r.file for r in load_records(root)}
    on_disk = {p.name for p in results.iterdir() if p.is_file() and p.name not in _NON_RECORDS}
    return [f"indexed but missing: results/{f}" for f in sorted(listed - on_disk)] + [
        f"not indexed: results/{f}" for f in sorted(on_disk - listed)
    ]
