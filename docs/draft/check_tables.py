"""
check_tables.py
===============
Every table in the manuscript against Nature Portfolio's table rules.

What a Nature table is:
  - no vertical rules, and no \\hline: only booktabs (\\toprule, \\midrule, \\bottomrule)
  - the caption sits ABOVE the table and opens with a title sentence, not a fragment
  - units live in the COLUMN HEADER, never repeated in every cell
  - decimal places are CONSISTENT down a column: 0.90 and 0.9042 in the same column tell the
    reader the precision changed when it did not
  - numbers are decimal-aligned, so a column can be scanned
  - every abbreviation used in the table is defined in its caption

    python check_tables.py
"""

import re
import sys
from collections import defaultdict

FILES = ["sections/results.tex", "sections/supplementary.tex"]
fails, warns = [], []


def tables(src):
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", src, re.S):
        blk = m.group(0)
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", blk, re.S)
        tab = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", blk, re.S)
        if tab:
            yield (lab.group(1) if lab else "?",
                   cap.group(1) if cap else "",
                   tab.group(1), tab.group(2))


def cells(body):
    rows = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line or line.startswith("\\") and "&" not in line:
            continue
        if re.match(r"^\\(toprule|midrule|bottomrule|addlinespace|cmidrule)", line):
            continue
        line = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule\(?[lr]*\)?(\{[^}]*\})?|addlinespace(\[[^]]*\])?)", "", line)
        if "&" in line:
            rows.append([c.strip() for c in line.split("&")])
    return rows


print("=" * 92)
print("NATURE TABLE RULES")
print("=" * 92)

n = 0
for path in FILES:
    src = open(path).read()
    for lab, cap, spec, body in tables(src):
        n += 1
        name = lab.replace("stab:", "S").replace("tab:", "")
        probs = []

        # 1. no vertical rules, no \hline
        if "|" in spec:
            probs.append("VERTICAL RULES in the column spec")
        if "\\hline" in body:
            probs.append("\\hline used instead of booktabs")
        # 2. booktabs
        if "\\toprule" not in body or "\\bottomrule" not in body:
            probs.append("not using booktabs (\\toprule / \\bottomrule)")
        # 3. caption opens with a title sentence
        if cap and not re.match(r"^[A-Z]", cap.strip()):
            probs.append("caption does not open with a capitalised title sentence")
        # 4. units repeated in cells rather than sitting in the header
        rows = cells(body)
        if len(rows) > 1:
            hdr, data = rows[0], rows[1:]
            ncol = max(len(r) for r in rows)
            unit_in_cell = 0
            for r in data:
                for k, c in enumerate(r):
                    if k < 2:
                        continue          # a unit belongs with the row label
                    if re.search(r"C\\,m\$\^\{-2\}\$|\\,GPa|\\,eV|\\,MB", c):
                        unit_in_cell += 1
            if unit_in_cell > 2:
                probs.append(f"units repeated in {unit_in_cell} cells; move them to the header")
            # 5. decimal places consistent down a column
            for j in range(ncol):
                dp = defaultdict(int)
                for r in data:
                    if j >= len(r):
                        continue
                    m = re.match(r"^\$?(-?\d+\.(\d+))", r[j].replace("{,}", ""))
                    if m:
                        dp[len(m.group(2))] += 1
                if len(dp) > 1 and sum(dp.values()) >= 4:
                    mixed_units = any(re.search(r"\((eV|D|GPa|C\\,m)", r[1] if len(r) > 1 else "")
                                      for r in data)
                    sci = sum(1 for r in data if j < len(r) and "times10" in r[j].replace("\\", ""))
                    if mixed_units:
                        continue      # different units on different rows: precision SHOULD differ
                    if sci:
                        continue      # a column mixing scientific and decimal spans many orders
                    spread = f"{dict(sorted(dp.items()))}"
                    probs.append(f"column {j+1}: mixed decimal places {spread}")
        if probs:
            print(f"\n  {name}")
            for p in probs:
                print(f"    - {p}")
                (fails if "VERTICAL" in p or "hline" in p or "booktabs" in p else warns).append(p)

print()
print("=" * 92)
print(f"  {n} tables checked   {len(fails)} rule violations   {len(warns)} style warnings")
print("=" * 92)
sys.exit(1 if fails else 0)
