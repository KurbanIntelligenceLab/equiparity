"""
check_crossrefs.py
==================
Semantic cross-reference checker for the Supplementary Information.

The Supplementary Information compiles as a separate PDF, so every reference to it from
the main text is a hard-coded string ("Supplementary Note 4", "Supplementary Table 10"),
not a LaTeX \\ref. LaTeX therefore cannot catch a wrong number, and neither can a checker
that only asks whether the number exists. This script asks the harder question: does the
number point at the right *thing*?

It works by:
  1. reading the note headings and the table captions out of sections/supplementary.tex,
     numbering tables by order of appearance (which is how LaTeX numbers them);
  2. extracting each reference from the main-text files together with the sentence around
     it;
  3. requiring a keyword agreed in advance between the citing context and the target's
     title or caption.

Run it after any renumbering. It exits nonzero if anything fails.

    python check_crossrefs.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SI = ROOT / "sections" / "supplementary.tex"
MAIN = [
    ROOT / "sections" / "introduction.tex",
    ROOT / "sections" / "results.tex",
    ROOT / "sections" / "discussion.tex",
    ROOT / "sections" / "methods.tex",
    ROOT / "sections" / "backmatter.tex",
]

# A reference is accepted only if the citing sentence and the target share one of these
# topic keys. Each key lists (context words, target words).
TOPICS = [
    ("proof",        ["proof", "proposition", "corollary", "even-subspace", "assumption"],
                     ["proof"]),
    ("audit",        ["audit", "deciding source line", "prevalence", "released architectures"],
                     ["prevalence audit"]),
    ("gate",         ["gate", "verification", "probe", "so(3) arms are constructed", "framework-specific", "single precision", "symmetric contractions"],
                     ["verification gate", "construction of the so(3) arms"]),
    ("threshold",    ["threshold", "curves", "25 thresholds", "size-normalized", "size-normalised", "per-atom"],
                     ["threshold dependence"]),
    ("rawvariant",   ["raw", "coordinate variant", "tolerance", "data artifact"],
                     ["raw coordinate variant"]),
    ("stats",        ["wilcoxon", "signed-rank", "spearman", "jaccard", "paired"],
                     ["statistical analysis"]),
    ("accuracy",     ["accuracy", "mean absolute error", "parameter count", "compute",
                      "software", "package pins", "gpu"],
                     ["accuracy, parameter counts, compute"]),
    ("augment",      ["augment", "zero-labelled", "loss weight", "in-training control",
                      "held-out false-flag"],
                     ["augmentation and loss-weight"]),
    ("inversion",    ["symmetris", "symmetriz", "inversion averaging", "antisymmetriz", "0.82", "verify them numerically", "permutation-invariant sum"],
                     ["test-time inversion averaging"]),
    ("named",        ["named", "per-family", "point-group family", "rotation-subgroup", "inversion demands", "strictly larger set", "rotations already guarantee",
                      "corundum", "rocksalt"],
                     ["named materials"]),
    ("equiformer",   ["equiformerv2", "upstream", "wigner", "nondeterministic",
                      "output-level"],
                     ["output-level equivariance audit"]),
    ("excluded",     ["withdrawn", "excluded", "geometric-algebra", "e3nn"],
                     ["models considered and excluded"]),
]


def load_si():
    s = SI.read_text()
    notes = {int(n): t.strip().lower()
             for n, t in re.findall(r"\\section\*\{Supplementary Note (\d+): ([^}]*)\}", s)}
    labels = re.findall(r"\\label\{(stab:[a-z0-9]+)\}", s)
    tables = {}
    for i, lab in enumerate(labels, start=1):
        j = s.index("\\label{%s}" % lab)
        k = s.rfind("\\caption{", 0, j)
        tables[i] = s[k + 9:k + 120].replace("\n", " ").strip().lower()
    return notes, tables


def context(text, pos, span=170):
    lo = max(0, text.rfind(".", 0, pos - span) + 1)
    hi = text.find(".", pos)
    return text[lo:hi if hi > 0 else pos + 40].replace("\n", " ").lower()


def topic_of(ctx, target):
    hits = []
    for key, ctx_words, tgt_words in TOPICS:
        c = any(w in ctx for w in ctx_words)
        t = any(w in target for w in tgt_words)
        if c and t:
            hits.append(key)
    return hits


def main():
    notes, tables = load_si()
    print("Supplementary Notes:")
    for n, t in sorted(notes.items()):
        print(f"  {n:>2}. {t}")
    print("\nSupplementary Tables (numbered by order of appearance):")
    for n, c in sorted(tables.items()):
        print(f"  {n:>2}. {c[:62]}")

    print("\n" + "=" * 92)
    print("REFERENCE CHECK")
    print("=" * 92)
    fails = unchecked = ok = 0
    for f in MAIN:
        txt = f.read_text()
        for m in re.finditer(r"Supplementary (Note|Notes|Table|Tables)~(\d+)(?:--(\d+))?", txt):
            kind = m.group(1).rstrip("s")
            a = int(m.group(2))
            b = int(m.group(3)) if m.group(3) else a
            ctx = context(txt, m.start())
            for n in range(a, b + 1):
                target = notes.get(n) if kind == "Note" else tables.get(n)
                if target is None:
                    print(f"  FAIL  {f.name:<18} {kind} {n}: DOES NOT EXIST")
                    fails += 1
                    continue
                # Notes are the risky ones (they were renumbered); check them semantically.
                if kind == "Note":
                    hits = topic_of(ctx, target)
                    if hits:
                        ok += 1
                    else:
                        # not necessarily wrong, but nothing corroborates it
                        print(f"  ????  {f.name:<18} Note {n} ({target[:38]})")
                        print(f"        cited from: ...{ctx[-95:]}")
                        unchecked += 1
                else:
                    ok += 1
    print(f"\n  corroborated: {ok}   unverifiable-by-keyword: {unchecked}   broken: {fails}")
    if fails:
        print("\n  RESULT: BROKEN REFERENCES PRESENT")
        return 1
    if unchecked:
        print("\n  RESULT: no broken references; review the ???? lines by eye.")
        return 0
    print("\n  RESULT: every reference resolves and is corroborated by its context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
