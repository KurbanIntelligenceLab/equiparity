"""Build all proofs and reject unfinished proofs or nonstandard axiom dependencies."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT / "ParityGap/Guarantees.lean").read_text()
expected = {"ParityGap." + n for n in re.findall(r"^theorem (\w+)", source, re.M)}
required = {
    "ParityGap.antisymmetrize_odd",
    "ParityGap.antisymmetrize_zero",
    "ParityGap.invariant_value_attainable",
    "ParityGap.parity_gap_even_rank",
    "ParityGap.parity_gap_odd_rank",
    "ParityGap.periodic_inversion_identity",
    "ParityGap.polarInversion_sign",
    "ParityGap.symmetry_forced_zero",
    "ParityGap.value_in_fixedSpace",
}
assert expected == required, (expected, required)
for path in [ROOT / "ParityGap.lean", ROOT / "ParityGap/Guarantees.lean", ROOT / "Audit.lean"]:
    code = re.sub(r"/\-.*?\-/", "", path.read_text(), flags=re.S)
    code = re.sub(r"--[^\n]*", "", code)
    assert not re.search(r"\b(sorry|admit|axiom|native_decide|unsafe|implemented_by)\b", code), path


def run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert not re.search(r"\b(?:warning|error):", output, re.I), output
    return output


print(run("lake", "build"), end="")
audit = run("lake", "env", "lean", "Audit.lean")
entries = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", audit)
assert len(entries) == len(expected) and {n for n, _ in entries} == expected, audit
allowed = {"propext", "Classical.choice", "Quot.sound"}
for name, dependencies in entries:
    axioms = {v.strip() for v in dependencies.split(",") if v.strip()}
    assert axioms <= allowed, (name, axioms)
print(audit, end="")
print(
    "PASS: 9 theorems checked; no unfinished proofs, custom axioms or native evaluation shortcuts."
)
