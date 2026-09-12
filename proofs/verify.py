"""Build all proofs and reject unfinished proofs or nonstandard axiom dependencies."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'ParityGap/Guarantees.lean').read_text()
expected = {'ParityGap.' + n for n in re.findall(r'^theorem (\w+)', source, re.M)}
assert len(expected) == 17
for path in [ROOT / 'ParityGap.lean', ROOT / 'ParityGap/Guarantees.lean', ROOT / 'Audit.lean']:
    code = re.sub(r'/\-.*?\-/', '', path.read_text(), flags=re.S)
    code = re.sub(r'--[^\n]*', '', code)
    assert not re.search(r'\b(sorry|admit|axiom|native_decide|unsafe|implemented_by)\b', code), path

def run(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    output = result.stdout + result.stderr
    assert not re.search(r'\b(?:warning|error):', output, re.I), output
    return output

print(run('lake', 'build'), end='')
audit = run('lake', 'env', 'lean', 'Audit.lean')
entries = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", audit)
assert len(entries) == len(expected) and {n for n, _ in entries} == expected, audit
allowed = {'propext', 'Classical.choice', 'Quot.sound'}
for name, dependencies in entries:
    axioms = {v.strip() for v in dependencies.split(',') if v.strip()}
    assert axioms <= allowed, (name, axioms)
print(audit, end='')
print('PASS: 17 theorems checked; no unfinished proofs, custom axioms or native evaluation shortcuts.')
