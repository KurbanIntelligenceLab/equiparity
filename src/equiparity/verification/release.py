"""Release checks: reconcile the released records with the theory, the index, and the proofs.

Three independent checks, each runnable without a GPU or trained model:

- ``theory``: numerical checks of the parity-gap identities (``tests/verification/test_theory.py``)
  and the released parity-gap table against :mod:`equiparity.verification.parity_gap`.
- ``claims``: every reported quantity against the committed records
  (``tests/verification/test_claims.py``) and the completeness of ``results/index.json``.
- ``proofs``: build and audit the Lean formalization (``proofs/verify.py``; needs ``lake``).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from equiparity.io.records import check_index
from equiparity.verification.parity_gap import check_parity_gap_record

_log = logging.getLogger(__name__)

CHECKS = ("theory", "claims", "proofs")


@dataclass
class CheckResult:
    """Outcome of one release check."""

    name: str
    passed: bool
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Plain mapping for JSON output."""
        return {"check": self.name, "passed": self.passed, "problems": self.problems}


def _pytest(root: Path, test_file: str) -> list[str]:
    path = root / "tests/verification" / test_file
    if not path.is_file():
        return [f"missing {path.relative_to(root)}"]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(path),
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return []
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    return [f"{test_file} failed (exit {proc.returncode})", *tail]


def _theory(root: Path) -> list[str]:
    return check_parity_gap_record(root / "results/parity_gap_table.csv") + _pytest(
        root, "test_theory.py"
    )


def _claims(root: Path) -> list[str]:
    return check_index(root) + _pytest(root, "test_claims.py")


def _proofs(root: Path) -> list[str]:
    proofs = root / "proofs"
    if shutil.which("lake") is None:
        return ["lake not found: install the Lean toolchain with elan (see proofs/README.md)"]
    # Fetch mathlib's precompiled build first; without it `lake build` compiles mathlib from source.
    for step in (["lake", "exe", "cache", "get"], [sys.executable, "verify.py"]):
        proc = subprocess.run(step, cwd=proofs, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
            return [f"`{' '.join(step)}` failed in proofs/ (exit {proc.returncode})", *tail]
    return []


_RUNNERS = {"theory": _theory, "claims": _claims, "proofs": _proofs}


def run_release_checks(root: Path, checks: tuple[str, ...]) -> list[CheckResult]:
    """Run the named checks against the checkout at ``root``."""
    results = []
    for name in checks:
        if name not in _RUNNERS:
            raise ValueError(f"unknown check {name!r}; choose from {CHECKS}")
        _log.info("running %s check", name)
        problems = _RUNNERS[name](root)
        results.append(CheckResult(name=name, passed=not problems, problems=problems))
    return results
