"""
audit_fig1.py  -- pixel-exact overlap audit.

Two earlier checkers passed a broken figure:
  1. text-vs-text bounding boxes   -> blind to a label sitting on a box
  2. text-vs-artist bounding boxes -> blind to strokes and dashes, and it skipped large patches

This one cannot be fooled. It renders the figure TWICE: once normally, once with every Text
hidden. The second render is the ink that belongs to the DRAWING. It then takes each label's
bounding box and counts how much drawing-ink lies underneath it. Ink under a label means the
label is printed on top of something.

White type on a filled box is intentional and exempt. Everything else is a defect.

    python audit_fig1.py [fig1|fig2|fig3|fig4|fig5]
"""
import importlib.util, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
from matplotlib.text import Text

DPI = 400
SCALE_IN_MS = 131.2 / 180.0


def _walk(obj, out):
    """every Text anywhere in the tree: ax.text, tick labels, axis labels, titles, LEGEND."""
    if isinstance(obj, Text):
        if obj.get_visible() and obj.get_text().strip():
            out.append(obj)
        return
    for ch in getattr(obj, "get_children", lambda: [])():
        _walk(ch, out)


def _labels_of(ax):
    out = []
    _walk(ax, out)
    return out


def _labels(fig):
    out = []
    for ax in fig.axes:
        _walk(ax, out)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_visible() and t.get_text().strip():
                out.append(t)
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if t.get_visible() and t.get_text().strip():
                out.append(t)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                if t.get_visible() and t.get_text().strip():
                    out.append(t)
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append(t)
    seen, uniq = set(), []
    for t in out:
        if id(t) not in seen:
            seen.add(id(t)); uniq.append(t)
    return uniq


def render(fig, hide_text=False):
    hidden = []
    if hide_text:
        for a in _labels(fig):
            a.set_visible(False); hidden.append(a)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float).mean(axis=2)
    for a in hidden:
        a.set_visible(True)
    return buf


def audit(fig, name, ink=215, cover=0.0018):
    fig.set_dpi(DPI)
    full = render(fig, False)
    art = render(fig, True) < ink          # strokes and fills only; a pale background band is not "ink"
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    H, W = full.shape
    problems, smallest = [], 1e9
    axbb = {id(ax): ax.get_window_extent(renderer=r) for ax in fig.axes}
    owner = {}
    for ax in fig.axes:
        for t in _labels_of(ax):
            owner[id(t)] = ax
    for a in _labels(fig):
        bb0 = a.get_window_extent(renderer=r)
        ax0 = owner.get(id(a))
        if ax0 is not None and a in (ax0.get_xticklabels() + ax0.get_yticklabels()):
            ab = axbb[id(ax0)]
            if bb0.y1 < ab.y0 - 2 or bb0.y0 > ab.y1 + 2 or bb0.x1 < ab.x0 - 2 or bb0.x0 > ab.x1 + 2:
                continue          # a tick outside the view: matplotlib never draws it
        smallest = min(smallest, a.get_fontsize())
        try:
            rgb = mc.to_rgb(a.get_color())
        except Exception:
            rgb = (0, 0, 0)
        if sum(rgb) / 3 > 0.85:            # white type: designed to sit on a fill
            continue
        bx = a.get_bbox_patch()            # an opaque halo is a legitimate mask
        if bx is not None:
            try:
                fc = mc.to_rgba(bx.get_facecolor())
                if sum(fc[:3]) / 3 > 0.85 and fc[3] > 0.85:
                    continue
            except Exception:
                pass
        bb = a.get_window_extent(renderer=r)
        x0, x1 = max(0, int(bb.x0) - 1), min(W, int(bb.x1) + 1)
        y0, y1 = max(0, H - int(bb.y1) - 1), min(H, H - int(bb.y0) + 1)
        t = a.get_text().replace("\n", " / ")[:38]
        if x1 <= x0 or y1 <= y0 or bb.x0 < -1 or bb.x1 > W + 1 or bb.y0 < -1:
            problems.append(f"OFF CANVAS      '{t}'"); continue
        p = art[y0:y1, x0:x1]
        c = p.mean() if p.size else 0.0
        if c > cover:
            problems.append(f"INK UNDER TEXT  '{t}'   {c*100:.1f}% of its box is on the drawing")
    # ------------------------------------------------- text escaping its own panel
    # Nothing is drawn there, so the pixel test cannot see it, but it collides with the
    # neighbouring panel and it is what a reader sees.
    for ax in fig.axes:
        ab = axbb[id(ax)]
        ticks = set(map(id, ax.get_xticklabels() + ax.get_yticklabels()))
        for a in _labels_of(ax):
            if a in (ax.xaxis.label, ax.yaxis.label):
                continue
            if id(a) in ticks:                       # ticks belong outside the axes
                continue
            if a.get_text().strip() in tuple("abcdef"):
                continue
            bb0 = a.get_window_extent(renderer=r)
            if bb0.x0 < ab.x0 - 4 or bb0.x1 > ab.x1 + 4:
                problems.append(f"TEXT ESCAPES ITS PANEL  '{a.get_text()[:30]}'  "
                                f"by {max(ab.x0 - bb0.x0, bb0.x1 - ab.x1):.0f} px")
        lg = ax.get_legend()
        if lg is not None:
            lb = lg.get_window_extent(renderer=r)
            fb = fig.bbox
            if lb.y1 > fb.height + 1 or lb.y0 < -1 or lb.x1 > fb.width + 1 or lb.x0 < -1:
                over = max(lb.y1 - fb.height, -lb.y0, lb.x1 - fb.width, -lb.x0)
                problems.append(f"LEGEND CLIPPED BY THE CANVAS  by {over:.0f} px "
                                f"-- it will not be visible")
            if lb.x0 < ab.x0 - 4 or lb.x1 > ab.x1 + 4:
                problems.append(f"LEGEND ESCAPES ITS PANEL  by "
                                f"{max(ab.x0 - lb.x0, lb.x1 - ab.x1):.0f} px")

    # ------------------------------------------------------------------ text vs TEXT
    # The pixel test compares a label against the DRAWING. Two labels printed on top of each
    # other leave no drawing-ink between them, so it is blind to that. Check it separately.
    boxes = []
    for a in _labels(fig):
        bb0 = a.get_window_extent(renderer=r)
        ax0 = owner.get(id(a))
        if ax0 is not None and a in (ax0.get_xticklabels() + ax0.get_yticklabels()):
            ab = axbb[id(ax0)]
            if bb0.y1 < ab.y0 - 2 or bb0.y0 > ab.y1 + 2 or bb0.x1 < ab.x0 - 2 or bb0.x0 > ab.x1 + 2:
                continue
        if bb0.width > 0 and bb0.height > 0:
            boxes.append((a.get_text().replace("\n", " / ")[:26], bb0))
    import itertools
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        ix = max(0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
        iy = max(0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
        if ix <= 0 or iy <= 0:
            continue
        f = ix * iy / min(b1.width * b1.height, b2.width * b2.height)
        if f > 0.05:
            problems.append(f"TEXT ON TEXT    '{t1}'  and  '{t2}'   overlap {f*100:.0f}%")

    px = smallest * SCALE_IN_MS
    print(f"=== {name} ===")
    print(f"  smallest type as rendered in the manuscript: {px:.1f} pt "
          f"({'OK' if px >= 4.95 else 'BELOW THE 5 pt FLOOR'})")
    if problems:
        print(f"  {len(problems)} DEFECTS:")
        for p_ in sorted(set(problems)):
            print(f"    - {p_}")
    else:
        print("  clean: no label has drawing-ink underneath it, nothing off the canvas")
    return problems


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("bf", "build_figures.py")
    bf = importlib.util.module_from_spec(spec); sys.modules["bf"] = bf
    spec.loader.exec_module(bf)
    which = sys.argv[1] if len(sys.argv) > 1 else "fig1"
    getattr(bf, which)()
    total = sum(len(audit(f, n)) for n, f in bf.SAVED.items())
    print("\n  RESULT:", "CLEAN" if total == 0 else f"{total} DEFECTS")
    sys.exit(1 if total else 0)
