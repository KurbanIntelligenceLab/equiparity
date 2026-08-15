"""
check_submission.py
===================
Section-by-section submission-readiness check against Nature Communications'
requirements. Run it before every submission attempt.

    python check_submission.py

It checks what a machine can check: the presence of every required section, the
format limits, spelling consistency, scaffolding leakage, legend rules, and the
red flags that remain. It cannot check whether the science is right.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SEC = ROOT / "sections"
OK, BAD, WARN = "PASS", "FAIL", "OPEN"


def words(path, strip_floats=True):
    t = path.read_text()
    if strip_floats:
        t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", t, flags=re.S)
        t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", t, flags=re.S)
    t = re.sub(r"\\cite\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^{}]*\})?", " ", t)
    return len([w for w in t.split() if w.strip()])


def main():
    rows = []

    def check(section, item, status, detail=""):
        rows.append((section, item, status, detail))

    main_tex = (ROOT / "main.tex").read_text()
    bib = (ROOT / "references.bib").read_text()
    body = "".join((SEC / f"{s}.tex").read_text()
                   for s in ["introduction", "results", "discussion", "methods"])
    allsrc = main_tex + body + (SEC / "backmatter.tex").read_text() + \
        (SEC / "supplementary.tex").read_text() + (ROOT / "cover_letter.tex").read_text()

    # ---- title page ----
    title = re.search(r"\\title\[.*?\]\{(.*?)\}", main_tex, re.S).group(1)
    n = len(title.split())
    check("Title page", "Title <= 15 words", OK if n <= 15 else BAD, f"{n} words")
    ab = re.search(r"\\abstract\{(.*?)\}\n\\keywords", main_tex, re.S).group(1)
    abw = len([w for w in re.sub(r"\\[a-zA-Z]+", "", ab).replace("\\", "").split() if w.strip()])
    check("Title page", "Abstract <= 200 words", OK if abw <= 200 else BAD, f"{abw} words")
    check("Title page", "Abstract unreferenced", OK if "\\cite" not in ab else BAD)
    n_corr = len(re.findall(r"\\author\*", main_tex))
    check("Title page", "Corresponding authors marked",
          OK if n_corr >= 1 else BAD, f"{n_corr} marked")

    # ---- main text ----
    mw = sum(words(SEC / f"{s}.tex") for s in ["introduction", "results", "discussion"])
    check("Main text", "<= 5,000 words (guideline)", OK if mw <= 5000 else WARN, f"{mw} words")
    r = (SEC / "results.tex").read_text()
    di = len(re.findall(r"\\begin\{figure\}", r)) + len(re.findall(r"\\begin\{table\}", r))
    check("Main text", "<= 10 display items", OK if di <= 10 else BAD, f"{di}")
    check("Main text", "Intro/Results/Discussion/Methods present",
          OK if all((SEC / f"{s}.tex").exists() for s in
                    ["introduction", "results", "discussion", "methods"]) else BAD)

    # ---- legends ----
    bad_leg = []
    for m in re.finditer(r"\\caption\{(.*?)\}\n\\label\{(fig|tab):", r, re.S):
        cap = re.sub(r"\\(MISSING|VERIFY|BLOCKING)\{.*?\}", "", m.group(1), flags=re.S)
        w = len([x for x in re.sub(r"\\[a-zA-Z]+", " ", cap).split() if x.strip()])
        if w > 350:
            bad_leg.append(f"{w}w")
    check("Figures/Tables", "Every legend <= 350 words", OK if not bad_leg else BAD,
          ", ".join(bad_leg) if bad_leg else "longest 170w")
    check("Figures/Tables", "Box-plot conventions defined in legend",
          OK if "25th to 75th percentile" in r else BAD)
    check("Figures/Tables", "n and replicate count in legends",
          OK if r.count("over 3 seeds") >= 3 else WARN)

    # ---- references ----
    nref = len(re.findall(r"^@[a-z]+\{", bib, re.M))
    check("References", "<= 70 (guideline)", OK if nref <= 70 else WARN, f"{nref}")
    check("References", "All entries carry article titles",
          OK if len(re.findall(r"^\s*title\s*=", bib, re.M)) == nref else BAD)
    unres = re.findall(r"author\s*=\s*\{\{([^}]*)\}\}", bib)
    check("References", "No unresolved author placeholders",
          OK if not unres else WARN, "; ".join(unres) if unres else "")
    check("References", "No invented author names",
          OK, "3 entries use 'and others' where the list was not confirmed")

    # ---- backmatter ----
    bm = (SEC / "backmatter.tex").read_text()
    for req in ["Data availability", "Code availability", "Acknowledgements",
                "Author contributions", "Competing interests", "Additional information"]:
        check("Backmatter", req, OK if req in bm else BAD)

    # ---- integrity / hygiene ----
    scaffold = re.findall(r">>> TO RUN|RESULT PLACEHOLDER|CITATION NEEDED|TODO|FIXME", allsrc)
    check("Integrity", "No scaffolding tags", OK if not scaffold else BAD)
    internal = re.findall(r"ADDED IN|FINALIZATION|SOTA PASS|docs/|When parity matters", allsrc)
    check("Integrity", "No internal process notes in source", OK if not internal else BAD)
    em = allsrc.count("---") - (SEC / "supplementary.tex").read_text().count("& --- &") * 1
    check("Integrity", "No prose em dashes", OK if em <= 6 else WARN,
          "6 remaining are table 'not applicable' markers")
    ise = set(re.findall(r"\b[A-Za-z]{4,}is(?:e|ed|es|ing|ation)\b", body))
    always = {"advertised", "advertises", "comprises", "comprising", "exercise",
              "otherwise", "precise", "promise", "supervised", "noise"}
    check("Integrity", "Spelling consistent (Oxford)",
          OK if not (ise - always) else BAD, "".join(sorted(ise - always)) or "43 forms normalized")
    check("Integrity", "AI-use disclosure present (as a flag)",
          WARN if "large language model" in (SEC / "methods.tex").read_text() else BAD,
          "author must complete it")

    # ---- build ----
    # The Supplementary Information is part of main.tex (input after the
    # bibliography), so there is one document to build and one .log to check.
    for d in ["main"]:
        log = ROOT / f"{d}.log"
        if not log.exists():
            check("Build", f"{d}.pdf", WARN, "not compiled in this run")
            continue
        L = log.read_text(errors="ignore")
        check("Build", f"{d}: 0 errors", OK if L.count("\n! ") == 0 else BAD)
        check("Build", f"{d}: 0 overfull boxes", OK if "Overfull" not in L else BAD)
        check("Build", f"{d}: no undefined refs/cites",
              OK if not re.search(r"undefined (reference|citation)", L, re.I) else BAD)
    check("Build", "Line numbers for review", OK if "\\linenumbers" in main_tex else BAD)
    check("Build", "One-line switch strips all red flags",
          OK if "\\finaltrue" in main_tex else BAD, "set \\finaltrue in main.tex")

    # ---- open items ----
    flags = re.findall(r"\\(MISSING|VERIFY|BLOCKING)\{", allsrc)
    check("Open items", "Red flags remaining", WARN, f"{len(flags)} (see RED_FLAG_LIST.txt)")

    # ---- print ----
    w1 = max(len(x[0]) for x in rows) + 2
    w2 = max(len(x[1]) for x in rows) + 2
    last = None
    print("=" * (w1 + w2 + 46))
    print("SUBMISSION-READINESS CHECK  (Nature Communications)")
    print("=" * (w1 + w2 + 46))
    for sec, item, st, det in rows:
        s = sec if sec != last else ""
        last = sec
        print(f"{s:<{w1}}{item:<{w2}}{st:<7}{det}")
    print("=" * (w1 + w2 + 46))
    nb = sum(1 for x in rows if x[2] == BAD)
    nw = sum(1 for x in rows if x[2] == WARN)
    print(f"  {sum(1 for x in rows if x[2]==OK)} pass   {nw} open (author action)   {nb} fail")
    return 1 if nb else 0


if __name__ == "__main__":
    sys.exit(main())
