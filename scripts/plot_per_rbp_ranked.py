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

    fig_h = 0.30 * len(names) + 1.6
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    y = range(len(names))
    ax.barh(list(y), vals, color="#4C72B0", height=0.72, edgecolor="none")
    ax.axvline(macro, color="#C44E52", ls="--", lw=1.4,
               label="Macro-averaged AUROC = %.4f" % macro)
    ax.axvline(pooled, color="#55A868", ls=":", lw=1.4,
               label="Sample-pooled AUROC = %.4f" % pooled)

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
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
