"""Per-epoch validation curves from SELD training logs, grouped by
training-data fraction.

Parses lines of the form
  epoch: 63, ..., ER/F/LE/LR/SELD: 0.66/0.30/38.43/0.46/0.53, best_val_epoch: ...
from each log and plots two shared panels: F-score vs. epoch (left) and
LE vs. epoch (right), with all runs overlaid. Color encodes the training
fraction (25/50/75/100%), linestyle encodes the method (solid =
Embisonics, dashed = Learn), and a star marks each run's best-val epoch.

Point RUNS at your files, or use --dir with a filename convention.

Usage:
  python plot_training_curves.py                        # uses RUNS dict
  python plot_training_curves.py --dir logs/            # auto-discover
  python plot_training_curves.py --run Embisonics 25 logs/set2_25.out \
                                 --run Learn 50 logs/set22_50.out
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------------
# 1. Where the logs live. Fill this in, or use --dir / --run instead.
# ----------------------------------------------------------------------
RUNS = {
    # (method, fraction): path
    # ("Embisonics", 25): "logs/set2_frac25.out",
    # ("Embisonics", 50): "logs/set2_frac50.out",
    # ("Embisonics", 75): "logs/set2_frac75.out",
    # ("Embisonics", 100): "logs/set2_frac100.out",
    # ("Learn", 25): "logs/set22_frac25.out",
    # ("Learn", 50): "logs/set22_frac50.out",
    # ("Learn", 75): "logs/set22_frac75.out",
    # ("Learn", 100): "logs/set22_frac100.out",
}

FRACTIONS = [25, 50, 75, 100]

# ----------------------------------------------------------------------
# 2. Parsing
# ----------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"^epoch:\s*(\d+),.*?ER/F/LE/LR/SELD:\s*"
    r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+),\s*"
    r"best_val_epoch:\s*(\d+)",
    re.M,
)

def parse_epochs(path):
    """Return dict of numpy arrays: epoch, ER, F(%), LE, LR(%), SELD,
    plus best_epoch (int, from the final epoch line)."""
    text = open(path).read()
    rows = EPOCH_RE.findall(text)
    if not rows:
        raise ValueError(f"no epoch lines found in {path}")
    arr = np.array(rows, dtype=float)
    return {
        "epoch": arr[:, 0].astype(int),
        "ER": arr[:, 1],
        "F": arr[:, 2] * 100.0,   # log stores fractions; report as %
        "LE": arr[:, 3],
        "LR": arr[:, 4] * 100.0,
        "SELD": arr[:, 5],
        "best_epoch": int(arr[-1, 6]),
    }


FNAME_RE = re.compile(r"^(25|50|75|100)_percent_(embisonics|learn)\.out$",
                      re.IGNORECASE)


def discover(directory):
    """Auto-build RUNS from files named {frac}_percent_{method}.out,
    e.g. 25_percent_embisonics.out, 100_percent_learn.out."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.out"))):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue  # e.g. gramt.out
        method = "Embisonics" if m.group(2).lower() == "embisonics" else "Learn"
        runs[(method, int(m.group(1)))] = path
    return runs


# ----------------------------------------------------------------------
# 3. Plotting
# ----------------------------------------------------------------------
LABELS = {"Embisonics": "Embisonics (ours)", "Learn": "Learnt tokens"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory of logs to auto-discover")
    ap.add_argument("--run", nargs=3, action="append", default=[],
                    metavar=("METHOD", "FRACTION", "PATH"),
                    help="explicit run, repeatable")
    ap.add_argument("--out", default="training_curves",
                    help="output basename (writes .pdf and .png)")
    ap.add_argument("--full-width", action="store_true",
                    help="size for \\textwidth (7 in) instead of column")
    args = ap.parse_args()

    runs = dict(RUNS)
    if args.dir:
        runs.update(discover(args.dir))
    for method, frac, path in args.run:
        runs[(method, int(frac))] = path
    if not runs:
        sys.exit("No runs given. Fill in RUNS, or use --dir/--run.")

    data = {}
    for key, path in runs.items():
        try:
            data[key] = parse_epochs(path)
            print(f"  parsed {key}: {len(data[key]['epoch'])} epochs "
                  f"(best @ {data[key]['best_epoch']})  <- {path}")
        except (OSError, ValueError) as e:
            print(f"  ! {key}: {e}", file=sys.stderr)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "legend.fontsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.labelsize": 7, "axes.titlesize": 7,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    width = 7.0 if args.full_width else 3.39
    fig, (axF, axLE) = plt.subplots(1, 2, sharex=True,
                                    figsize=(width, width * 0.48))

    # color encodes the training fraction; linestyle encodes the method.
    # Okabe-Ito palette: maximally distinct and colorblind-safe.
    FRAC_COLORS = {25: "#CC79A7",   # magenta
                   50: "#E69F00",   # orange
                   75: "#009E73",   # green
                   100: "#0072B2"}  # blue
    STYLES = {
        # Embisonics: heavy solid line
        "Embisonics": dict(ls="-", lw=1.6),
        # Learn: thin, coarse dashes with open circle markers every 10 epochs
        "Learn": dict(ls=(0, (4, 2)), lw=0.9, marker="o", markevery=10,
                      ms=2.6, mfc="white", mew=0.7),
    }

    for frac in FRACTIONS:
        for method in ("Embisonics", "Learn"):
            d = data.get((method, frac))
            if d is None:
                continue
            c = FRAC_COLORS[frac]
            st = STYLES[method]
            axF.plot(d["epoch"], d["F"], color=c, **st)
            axLE.plot(d["epoch"], d["LE"], color=c, **st)
            # mark the best-validation epoch on both panels
            b = d["best_epoch"]
            ib = np.where(d["epoch"] == b)[0]
            if len(ib):
                i = ib[0]
                axF.plot(b, d["F"][i], "*", color=c, ms=6, mec="none",
                         zorder=4)
                axLE.plot(b, d["LE"][i], "*", color=c, ms=6, mec="none",
                          zorder=4)

    for ax in (axF, axLE):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=1.5, width=0.5)
        ax.set_xlabel("Epoch", labelpad=1)
    axF.set_ylabel(r"val F-score (%)$\,\uparrow$", labelpad=1)
    axLE.set_ylabel(r"val LE ($^{\circ}$)$\,\downarrow$", labelpad=1)
    axLE.set_ylim(0, 185)

    # two compact legends: fractions (colors) and methods (linestyles)
    from matplotlib.lines import Line2D
    frac_handles = [Line2D([], [], color=FRAC_COLORS[f], lw=1.4,
                           label=f"{f}%") for f in FRACTIONS]
    meth_handles = [
        Line2D([], [], color="0.2", ls="-", lw=1.6,
               label=LABELS["Embisonics"]),
        Line2D([], [], color="0.2", ls=(0, (4, 2)), lw=0.9, marker="o",
               ms=2.6, mfc="white", mew=0.7, label=LABELS["Learn"]),
    ]
    leg1 = axF.legend(handles=frac_handles, loc="lower right", ncol=2,
                      frameon=False, handlelength=1.2, columnspacing=0.8,
                      borderaxespad=0.1, title="Training data",
                      title_fontsize=6.5)
    axF.add_artist(leg1)
    axLE.legend(handles=meth_handles, loc="upper right", frameon=False,
                handlelength=1.6, borderaxespad=0.1)

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()