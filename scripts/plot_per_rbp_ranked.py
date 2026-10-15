# -*- coding: utf-8 -*-
"""
plot_per_rbp_ranked.py — Figure S1: ranked per-RBP AUROC bar chart.

Reads results/per_rbp_results.json produced by src/eval_per_rbp.py and draws
a horizontal bar chart sorted by AUROC, with two reference lines:
  * macro-averaged AUROC (each RBP weighted equally)
  * sample-pooled AUROC (all test samples pooled)

Usage
-----
    python scripts/plot_per_rbp_ranked.py
    python scripts/plot_per_rbp_ranked.py --out figures/fig_per_rbp_ranked.pdf

The figure geometry is parameterised so that the same data can be rendered at
the exact size at which it will be printed.  The default values reproduce the
tall stand-alone figure; for the supplementary PDF (A4, 1-in margins) the
figure must be generated at the print size, i.e. ``--fig-width 6.36
--bar-h 0.18 --pad 1.39``, otherwise the image is taller than the text
block and the caption is pushed off the bottom of the page.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(BASE_DIR, "results", "per_rbp_results.json"))
    p.add_argument("--out", default=os.path.join(BASE_DIR, "results", "figures", "fig_per_rbp_ranked.pdf"))
    p.add_argument("--fig-width", type=float, default=7.0,
                   help="figure width in inches (default 7.0)")
    p.add_argument("--bar-h", type=float, default=0.30,
                   help="height of one RBP row in inches (default 0.30)")
    p.add_argument("--pad", type=float, default=1.6,
                   help="extra vertical space beyond the bars, in inches (default 1.6)")
    p.add_argument("--ytick-size", type=float, default=8.0,
                   help="font size of the RBP tick labels (default 8)")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        d = json.load(f)
    pr = d["per_rbp"]
    if isinstance(pr, dict):
        rows = [(k, float(v["AUROC"])) for k, v in pr.items()]
    else:
        rows = [(v["rbp"], float(v["AUROC"])) for v in pr]

    rows.sort(key=lambda x: x[1])           # ascending -> best on top
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    summary = d["summary"]
    macro = summary["macro_AUROC"]
    pooled = summary["overall"]["AUROC"]

    fig_h = args.bar_h * len(names) + args.pad
    fig, ax = plt.subplots(figsize=(args.fig_width, fig_h))
    y = range(len(names))
    ax.barh(list(y), vals, color="#4C72B0", height=0.72, edgecolor="none")
    ax.axvline(macro, color="#C44E52", ls="--", lw=1.4,
               label="Macro-averaged AUROC = %.4f" % macro)
    ax.axvline(pooled, color="#55A868", ls=":", lw=1.4,
               label="Sample-pooled AUROC = %.4f" % pooled)

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=args.ytick_size)
    ax.set_xlim(0.86, 1.005)
    ax.set_xlabel("AUROC", fontsize=10)
    ax.set_title("Per-RBP AUROC on the CircInteractome test set (%d RBPs)" % len(names), fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.95)
    ax.tick_params(axis="x", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print("Saved: %s (%d RBPs)" % (args.out, len(names)))


if __name__ == "__main__":
    main()
