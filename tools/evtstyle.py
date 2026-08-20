# -*- coding: utf-8 -*-
"""
Monochrome publication style for journal figures.

Importing this module applies the rcParams.  Everything else is a small lookup:
given a series name it returns the grey level, marker, line style, hatch and
tick label to use, falling back to the ramp in order for names it does not know.

    from evtstyle import (FIGSIZE, FIGSIZE_WIDE, gray, hatch, dash, mark, tick,
                          series, read, save)

The point of the lookups is that a plotting script never hard-codes the list of
methods -- it reads the list from the data file and asks here for the encoding.
Adding a row to the CSV adds a series to the figure.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:                                   # read() is then unavailable
    pd = None

# --------------------------------------------------------------------- constants
PNG_DPI = 800
FIGSIZE = (20, 8)            # single panel
FIGSIZE_WIDE = (24, 9)       # two panels side by side
FONT_BASE = 26

plt.rcParams.update({
    # Times New Roman is not installed on every Linux box; Nimbus Roman is the
    # metric-compatible clone, so the figure keeps the intended proportions
    # instead of silently falling back to DejaVu Sans.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_BASE + 4,
    "xtick.labelsize": FONT_BASE,
    "ytick.labelsize": FONT_BASE,
    "legend.fontsize": FONT_BASE - 2,
})

# --------------------------------------------------------------------- encodings
# Redundant by design: grey level + marker + line style + hatch.  Any one of them
# can be lost to printing or photocopying and the series are still separable.
GRAYS = ["white", "0.75", "0.50", "0.25", "black"]
MARKS = ["o", "s", "^", "D", "v"]
DASHES = ["-", "--", "-.", (0, (3, 1, 1, 1)), ":"]
# white and black need no hatch; the mid-greys do, they are close in print
HATCHES = ["", "///", "\\\\\\", "xxx", ""]

# Names seen often enough to deserve a stable slot, so the same method keeps the
# same appearance across every figure in a manuscript.
SLOT = {
    "Standard POT": 0, "Std. POT": 0, "Conventional": 0, "Baseline": 0,
    "iForest": 1, "Isolation Forest": 1,
    "MCD": 2,
    "Autoencoder": 3, "AE": 3,
    "Deep-SVDD": 4, "Deep SVDD": 4, "DeepSVDD": 4,
}
# baseline-ish names come first when ordering a figure
ORDER = ["Standard POT", "Std. POT", "Conventional", "Baseline",
         "iForest", "Isolation Forest", "MCD",
         "Autoencoder", "AE", "Deep-SVDD", "Deep SVDD", "DeepSVDD"]

TICKS = {
    "Standard POT": "Standard\nPOT", "Std. POT": "Standard\nPOT",
    "iForest": "Isolation\nForest", "Isolation Forest": "Isolation\nForest",
    "Deep-SVDD": "Deep\nSVDD", "Deep SVDD": "Deep\nSVDD", "DeepSVDD": "Deep\nSVDD",
}

_assigned = {}


def _slot(name):
    if name in SLOT:
        return SLOT[name]
    if name not in _assigned:
        # Fallback for a name never seen before and never registered by series():
        # take the lowest slot no other assigned name is using.  This only kicks in
        # when a script asks for an encoding without going through series() first.
        used = set(_assigned.values())
        free = [i for i in range(len(GRAYS)) if i not in used]
        _assigned[name] = free[0] if free else len(_assigned) % len(GRAYS)
    return _assigned[name]


def _register(names):
    """
    Give unknown names a slot that nothing else in THIS figure is using.

    The SLOT table covers the whole ramp, so a global 'first free slot' rule would
    always collide with the baseline.  Here the occupied set is only what the
    current data actually contains, which leaves room whenever the figure has
    fewer series than the ramp has levels.
    """
    taken = {SLOT[n] for n in names if n in SLOT}
    taken |= {_assigned[n] for n in names if n in _assigned}
    for n in names:
        if n in SLOT or n in _assigned:
            continue
        free = [i for i in range(len(GRAYS)) if i not in taken]
        _assigned[n] = free[0] if free else len(taken) % len(GRAYS)
        taken.add(_assigned[n])


def gray(name):
    """Fill colour: white / 0.75 / 0.50 / 0.25 / black."""
    return GRAYS[_slot(name) % len(GRAYS)]


def mark(name):
    """Marker shape."""
    return MARKS[_slot(name) % len(MARKS)]


def dash(name):
    """Line style."""
    return DASHES[_slot(name) % len(DASHES)]


def hatch(name):
    """Bar hatch, empty for the white and black ends of the ramp."""
    return HATCHES[_slot(name) % len(HATCHES)]


def tick(name):
    """Two-line tick label; long names collide at font size 26."""
    return TICKS.get(name, name.replace(" ", "\n") if len(name) > 12 else name)


def series(df, col="variant"):
    """
    Series list taken FROM the data, baseline-ish names first, then file order.
    Never hard-code the method list -- this keeps the figure in step with the CSV.

    Call this before gray()/mark()/dash()/hatch(): it also reserves a distinct
    encoding slot for any name the SLOT table does not know, so an unfamiliar
    method cannot end up looking identical to the baseline.
    """
    present = list(dict.fromkeys(df[col].tolist()))
    known = [v for v in ORDER if v in present]
    out = known + [v for v in present if v not in known]
    _register(out)
    return out


# --------------------------------------------------------------------- plumbing
def read(argv, default=None, col="variant"):
    """First CLI argument is the data file; falls back to `default`."""
    if pd is None:
        raise ImportError("pandas is required for read()")
    p = argv[1] if len(argv) > 1 else default
    if not p:
        sys.exit(f"usage: python {os.path.basename(argv[0])} <data.csv>")
    if not os.path.exists(p):
        sys.exit(f"data file not found: {p}")
    print(f"[data] {p}")
    return pd.read_csv(p), p


def save(fig, name, src):
    """
    PNG for review, vector PDF for submission, both at 800 dpi, into a `figures/`
    directory beside the data directory.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(src)))
    out = os.path.join(root, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=PNG_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  {os.path.join(out, name)}.png / .pdf")


def top_legend(fig, handles, ncol=None, y=1.03):
    """Legend above the axes, centred. In-axes legends eat the plot at size 26."""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(handles), frameon=False)
