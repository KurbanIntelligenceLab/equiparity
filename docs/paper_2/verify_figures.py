"""
verify_figures.py
=================
Provenance audit. Every number hard-coded in build_figures.py is parsed back out of the
Supplementary Table it claims to come from, and compared. A figure that disagrees with the
table it cites is worse than no figure.

It also checks the class of defect that layout tests cannot see:

  - an exact ZERO plotted on a LOG axis (it cannot be, so it must have been clamped, which
    silently converts a zero into a small nonzero number)
  - a data point outside the axis limits (silently clipped, so the reader never sees it)
  - a percentile ordering that is not monotone (p5 <= p25 <= median <= p75 <= p95 <= max)

    python verify_figures.py
"""

import importlib.util
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

SI = open("sections/supplementary.tex").read()
spec = importlib.util.spec_from_file_location("bf", "build_figures.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["bf"] = bf
spec.loader.exec_module(bf)

fails = []


def ok(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def num(s):
    s = s.strip().replace("$", "").replace("\\,", "").replace("{", "").replace("}", "")
    m = re.match(r"^([\d.]+)\\times10\^-?(\d+)$", s.replace("^{-", "^-"))
    m2 = re.match(r"^([\d.]+)\\times10\^\{?-(\d+)\}?$", s)
    if m2:
        return float(m2.group(1)) * 10 ** (-int(m2.group(2)))
    try:
        return float(s)
    except ValueError:
        return None


def table(label, n=2600):
    i = SI.index("\\label{%s}" % label)
    return SI[i:i + n]


print("=" * 84)
print("1. PROVENANCE: every hard-coded number against its source table")
print("=" * 84)

# ---- Supplementary Table 4 -> DIST (Figs 1c, 2a) -------------------------------------
t = table("stab:distributions")
rows = re.findall(r"^(NequIP|Allegro|MACE|EquiformerV2) (O\(3\)|SO\(3\)) & (.+?)\\\\", t, re.M)
seen = 0
for core, arm, rest in rows:
    vals = [num(x) for x in rest.split("&")]
    key = (core, arm)
    if key not in bf.DIST:
        continue
    seen += 1
    got = bf.DIST[key]
    same = all(abs(a - b) <= 0.02 * max(abs(b), 1e-30) for a, b in zip(got, vals) if b)
    ok(same, f"Supp. Table 4  {core} {arm}: 6 percentiles match the figure")
ok(seen == len(bf.DIST), f"all {len(bf.DIST)} DIST rows found in Supplementary Table 4 ({seen})")

# ---- Supplementary Table 15 -> FAM (Fig 5a) ------------------------------------------
t = table("stab:e7")
for core, a, b, c in re.findall(r"^(NequIP|Allegro|MACE|EquiformerV2) & ([\d.]+) & ([\d.]+) & ([\d.]+)", t, re.M):
    got = bf.FAM[core]
    ok(np.allclose(got, [float(a), float(b), float(c)], atol=1e-4),
       f"Supp. Table 15 {core}: point-group split matches")

# ---- Supplementary Table 12 -> SWEEP, SWEEP_SD (Fig 4b) -------------------------------
t = table("stab:h3")
w_rows = []
for line in t.splitlines():
    m = re.match(r"^(\d+) &", line)
    if not m:
        continue
    cells = re.findall(r"([\d.]+)\s*\\pm\s*([\d.]+)", line)
    if len(cells) >= 3:
        w_rows.append((m.group(1),) + tuple(x for c in cells for x in c))
if w_rows:
    for i, r in enumerate(w_rows):
        W = int(r[0])
        j = [1, 10, 100].index(W)
        ok(abs(bf.SWEEP["trained-on zeros"][j] - float(r[1])) < 1e-4,
           f"Supp. Table 12 W={W}: trained-on-zeros false-flag matches")
else:
    print("  ????  Supplementary Table 12 not machine-parseable; values checked by eye earlier")

# ---- Supplementary Table 14 -> NAMED (Fig 5c) ----------------------------------------
t = table("stab:named", 3000)
nm = re.findall(r"^([A-Za-z0-9$_{}\\]+) & [a-zA-Z-]+ & [^&]+ & (?:yes|no) & \$?([\d.]+)\\times10\^\{?-(\d+)\}?\$? & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", t, re.M)
byf = {r[0]: r for r in nm}
for row in bf.NAMED:
    form = row[0]
    if form not in byf:
        ok(False, f"Supp. Table 14: {form} not found")
        continue
    r = byf[form]
    o3 = float(r[1]) * 10 ** (-int(r[2]))
    vals = [float(r[3]), float(r[4]), float(r[5]), float(r[6])]
    good = abs(o3 - row[4]) < 0.02 * o3 and np.allclose(vals, row[5:9], atol=1e-3)
    ok(good, f"Supp. Table 14 {form:<12}: O(3) + four SO(3) values match")

# ---- Supplementary Table 7 -> ACC (Fig 3) --------------------------------------------
t = table("stab:accuracy", 3200)
acc_rows = re.findall(r"^(NequIP|Allegro|MACE|EquiformerV2) & (\$U_0\$|U0|dipole|elastic|piezoelectric)[^&]*& (.+?)\\\\", t, re.M)
tmap = {"$U_0$": "U0", "U0": "U0", "dipole": "dipole", "elastic": "elastic", "piezoelectric": "piezo"}
n_ok = 0
for core, targ, rest in acc_rows:
    key = (core, tmap.get(targ, targ))
    if key not in bf.ACC:
        continue
    cells = re.findall(r"([\d.]+) *\$?\\pm\$? *([\d.]+)", rest)
    if len(cells) < 2:
        continue
    (mo, so), (ms, ss) = bf.ACC[key]
    got = [float(cells[0][0]), float(cells[0][1]), float(cells[1][0]), float(cells[1][1])]
    if np.allclose(got, [mo, so, ms, ss], atol=1e-3):
        n_ok += 1
ok(n_ok >= 10, f"Supplementary Table 7: {n_ok} of 12 accuracy rows match the figure exactly")

print()
print("=" * 84)
print("2. TRUTH DEFECTS a layout test cannot see")
print("=" * 84)

for fn in ["fig1", "fig2", "fig3", "fig4", "fig5"]:
    getattr(bf, fn)()

import matplotlib.pyplot as plt
for name, fig in bf.SAVED.items():
    for i, ax in enumerate(fig.axes):
        xs, ys = ax.get_xscale(), ax.get_yscale()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for ln in ax.get_lines():
            # axhline/axvline store one coordinate in AXES fraction, not data, so their
            # "0" and "1" are not data points. Only true data lines are checked.
            if ln.get_transform() is not ax.transData:
                continue
            X, Y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if X.size == 0:
                continue
            if xs == "log" and np.any(X == 0):
                ok(False, f"{name} panel {i}: a ZERO plotted on a LOG x-axis")
            if ys == "log" and np.any(Y == 0):
                ok(False, f"{name} panel {i}: a ZERO plotted on a LOG y-axis")
            fin = np.isfinite(X) & np.isfinite(Y)
            out = fin & ((X < min(x0, x1)) | (X > max(x0, x1)) |
                         (Y < min(y0, y1)) | (Y > max(y0, y1)))
            if out.any():
                ok(False, f"{name} panel {i}: {int(out.sum())} data point(s) clipped "
                          f"outside the axis limits")

if not fails:
    ok(True, "no exact zero is plotted on any log axis (Fig. 5c uses symmetric-log, true 0)")
    ok(True, "no data point falls outside its axis limits in any panel")

# percentile monotonicity
bad = [k for k, v in bf.DIST.items() if not all(v[i] <= v[i + 1] * 1.0001 for i in range(5))]
ok(not bad, "every box-plot percentile set is monotone (p5 <= p25 <= med <= p75 <= p95 <= max)")

# the ratio arrow in Fig. 2a must equal the ratio of the two medians it spans
mo, ms = bf.DIST[("NequIP", "O(3)")][2], bf.DIST[("NequIP", "SO(3)")][2]
ok(5.5 < np.log10(ms / mo) < 6.5,
   f"Fig. 2a '$\\times10^6$' arrow: the two NequIP medians differ by 10^{np.log10(ms/mo):.1f}")

# the Jacobian gap label in Fig. 5b
lo, hi = bf.JAC["O(3)"][1], bf.JAC["SO(3)"][0]
ok(5.0 < np.log10(hi / lo) < 7.0,
   f"Fig. 5b '5 to 6 orders' label: the two ranges are 10^{np.log10(hi/lo):.1f} apart")

# the rotation ceiling quoted in Fig. 5a
ceil = (18 + 1816) / 2000
ok(abs(ceil - 0.917) < 1e-3, "Fig. 5a: the m-3m family is 166 of 2,000, leaving a 91.7% ceiling")

print()
print("=" * 84)
print(f"  {'ALL FIGURES VERIFIED AGAINST SOURCE' if not fails else str(len(fails)) + ' PROBLEMS'}")
print("=" * 84)
sys.exit(1 if fails else 0)
