# -*- coding: utf-8 -*-
"""
plot_ablation.py — Four-variant ablation figure, drawn from results/ablation_summary.json.

Reproduces the paper figure:
  (a) grouped bars of AUROC / AUPRC / MCC / Accuracy for the four variants;
  (b) incremental MCC contribution along the two accumulation paths.

Usage
-----
    python scripts/plot_ablation.py                     # reads <repo>/results/ablation_summary.json
    python scripts/plot_ablation.py --reference-mcc 0.8315 --reference-auroc 0.9691
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIDTH_RATIOS = [1.3, 1]
FIGSIZE = (15, 6)


def load_summary(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    order = ["seq_only", "seq_kmer", "seq_struct", "full"]
    labels = ["Seq-only", "Seq+K-mer", "Seq+Struct", "Full"]
    return {
        "labels": labels,
        "AUROC": [raw[k]["test_auroc"] for k in order],
        "AUPRC": [raw[k]["test_auprc"] for k in order],
        "MCC": [raw[k]["test_mcc"] for k in order],
        "Accuracy": [raw[k]["test_accuracy"] for k in order],
        "params": [raw[k]["n_params"] for k in order],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=os.path.join(BASE_DIR, "results", "ablation_summary.json"))
    p.add_argument("--out", default=os.path.join(BASE_DIR, "results", "figures", "ablation_comparison.png"))
    p.add_argument("--reference-mcc", type=float, default=0.8315,
                   help="MCC of the full model at epoch 70 (drawn as a reference line)")
    args = p.parse_args()

    d = load_summary(args.summary)
    metrics = ["AUROC", "AUPRC", "MCC", "Accuracy"]
    colors = ["#90A4AE", "#42A5F5", "#FFA726", "#66BB6A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": WIDTH_RATIOS})

    # ---- (a) grouped bars ----
    x = np.arange(len(metrics))
    width = 0.18
    for i, (label, color) in enumerate(zip(d["labels"], colors)):
        vals = [d[m][i] for m in metrics]
        bars = ax1.bar(x + (i - 1.5) * width, vals, width, color=color,
                       label="%s (%s params)" % (label, format(d["params"][i], ",")),
                       edgecolor="white", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     "%.3f" % val, ha="center", va="bottom", fontsize=7)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("(a) Four-variant comparison (30 epochs)", fontsize=12.5, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.set_ylim(0.62, 1.01)
    ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---- (b) incremental MCC along two accumulation paths ----
    mcc = d["MCC"]
    d_kmer_over_seq = mcc[1] - mcc[0]
    d_struct_over_seq = mcc[2] - mcc[0]
    d_struct_over_kmer = mcc[3] - mcc[1]
    d_kmer_over_struct = mcc[3] - mcc[2]

    def annot(x_pos, bottom, height, text, color="black"):
        ax2.text(x_pos, bottom + height / 2, text, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=color)

    bw = 0.35
    ax2.bar(0, mcc[0], bw, color=colors[0], label="Seq-only (base)", edgecolor="white", zorder=3)
    ax2.bar(0, d_kmer_over_seq, bw, bottom=mcc[0], color=colors[1], label="+K-mer", edgecolor="white", zorder=3)
    ax2.bar(0, d_struct_over_kmer, bw, bottom=mcc[1], color=colors[2], label="+Structure", edgecolor="white", zorder=3)

    ax2.bar(1, mcc[0], bw, color=colors[0], edgecolor="white", zorder=3)
    ax2.bar(1, d_struct_over_seq, bw, bottom=mcc[0], color=colors[2], edgecolor="white", zorder=3)
    ax2.bar(1, d_kmer_over_struct, bw, bottom=mcc[2], color=colors[1], edgecolor="white", zorder=3)

    annot(0, 0, mcc[0], "%.4f" % mcc[0])
    annot(0, mcc[0], d_kmer_over_seq, "+%.4f" % d_kmer_over_seq, "white")
    annot(0, mcc[1], d_struct_over_kmer, "+%.4f" % d_struct_over_kmer, "white")
    annot(1, 0, mcc[0], "%.4f" % mcc[0])
    annot(1, mcc[0], d_struct_over_seq, "+%.4f" % d_struct_over_seq, "white")
    annot(1, mcc[2], d_kmer_over_struct, "+%.4f" % d_kmer_over_struct, "white")

    for i in (0, 1):
        ax2.text(i, mcc[3] + 0.008, "Total = %.4f" % mcc[3], ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.axhline(args.reference_mcc, color="#D32F2F", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
    ax2.text(1.42, args.reference_mcc, "Full ep70\nMCC=%.4f" % args.reference_mcc,
             ha="right", va="center", fontsize=8, color="#D32F2F", fontweight="bold")

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Path 1:\nSeq -> +K-mer -> +Struct", "Path 2:\nSeq -> +Struct -> +K-mer"], fontsize=9.5)
    ax2.set_ylabel("MCC", fontsize=12)
    ax2.set_title("(b) Incremental MCC contribution\n(two accumulation paths to Full)", fontsize=12.5, pad=10)
    ax2.set_ylim(0.60, 0.92)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
