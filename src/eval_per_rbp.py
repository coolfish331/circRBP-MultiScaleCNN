# -*- coding: utf-8 -*-
"""
eval_per_rbp.py — Per-RBP breakdown of the fine-tuned model on the test split.

Reproduces the per-protein table reported in the paper (35 evaluable RBPs):
  * macro-averaged AUROC / MCC over RBPs (each RBP weighted equally), and
  * sample-pooled AUROC / MCC over all test samples.

Split is identical to training: permutation with seed 42, 70 / 15 / 15
(train / validation / test) over the 260,554 CircInteractome samples.

Usage
-----
    python src/eval_per_rbp.py                       # uses <repo>/results/gpu_full_best.pt
    python src/eval_per_rbp.py --checkpoint path/to/best.pt
    RBP_BASE_DIR=/path/to/data python src/eval_per_rbp.py

Outputs
-------
    <repo>/results/per_rbp_results.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             matthews_corrcoef, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CNNGPUModel

BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Per-RBP evaluation on the CircInteractome test split")
    p.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "circ_processed"))
    p.add_argument("--checkpoint", default=os.path.join(BASE_DIR, "results", "gpu_full_best.pt"))
    p.add_argument("--out", default=os.path.join(BASE_DIR, "results", "per_rbp_results.json"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--min-samples", type=int, default=50,
                   help="RBPs with fewer test samples than this are skipped")
    return p.parse_args()


def get_test_indices(y_all, seed):
    n = len(y_all)
    idx = np.random.RandomState(seed).permutation(n)
    n_tr, n_vl = int(0.7 * n), int(0.15 * n)
    return idx[n_tr + n_vl:]


def predict(model, X, K, S, device, batch_size):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb = torch.tensor(X[start:end], dtype=torch.float32).to(device)
            kb = torch.tensor(K[start:end], dtype=torch.float32).to(device)
            sb = torch.tensor(S[start:end], dtype=torch.float32).to(device)
            probs.append(torch.sigmoid(model(xb, kb, sb)).cpu().numpy())
    return np.concatenate(probs)


def main():
    args = parse_args()
    t0 = time.time()
    data_dir = args.data_dir

    print("[1/5] rebuilding the test split (seed=%d, 70/15/15)" % args.seed)
    y_all = np.load(os.path.join(data_dir, "y_circrna.npy"), mmap_mode="r")
    ts_i = get_test_indices(y_all, args.seed)
    print("      total=%d  test=%d" % (len(y_all), len(ts_i)))

    print("[2/5] loading cached features (memory-mapped)")
    X_ts = np.array(np.load(os.path.join(data_dir, "X_circrna.npy"), mmap_mode="r")[ts_i], dtype=np.float32)
    K_ts = np.array(np.load(os.path.join(data_dir, "K_full.npy"), mmap_mode="r")[ts_i], dtype=np.float32)
    S_ts = np.array(np.load(os.path.join(data_dir, "S_full.npy"), mmap_mode="r")[ts_i], dtype=np.float32)
    y_ts = np.array(y_all[ts_i], dtype=np.int32)
    del y_all

    with open(os.path.join(data_dir, "rbp_labels.json"), encoding="utf-8") as f:
        rbp_all = json.load(f)
    rbp_ts = [rbp_all[i] for i in ts_i]

    print("[3/5] loading checkpoint")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNGPUModel(kmer_dim=K_ts.shape[1], struct_dim=S_ts.shape[2]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    print("      device=%s  params=%s" % (device, format(n_params, ",")))

    print("[4/5] inference")
    prob = predict(model, X_ts, K_ts, S_ts, device, args.batch_size)
    yhat = (prob > 0.5).astype(int)

    def pooled(y_true, p, b):
        return dict(AUROC=float(roc_auc_score(y_true, p)),
                    AUPRC=float(average_precision_score(y_true, p)),
                    MCC=float(matthews_corrcoef(y_true, b)),
                    Accuracy=float(accuracy_score(y_true, b)))

    print("[5/5] per-RBP metrics")
    per_rbp, skipped = [], []
    rbp_ts_arr = np.array(rbp_ts)
    for name in sorted(set(rbp_ts_arr)):
        m = rbp_ts_arr == name
        n = int(m.sum())
        n_pos = int(y_ts[m].sum())
        if n < args.min_samples or n_pos == 0 or n_pos == n:
            skipped.append({"rbp": name, "n": n, "n_pos": n_pos})
            continue
        per_rbp.append({"rbp": name, "n": n, "n_pos": n_pos,
                        **pooled(y_ts[m], prob[m], yhat[m])})

    aurocs = np.array([r["AUROC"] for r in per_rbp])
    mccs = np.array([r["MCC"] for r in per_rbp])
    overall = pooled(y_ts, prob, yhat)

    summary = {
        "n_rbp": len(per_rbp),
        "n_evaluable": len(per_rbp),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "macro_AUROC": float(aurocs.mean()),
        "macro_MCC": float(mccs.mean()),
        "AUROC_min": float(aurocs.min()),
        "AUROC_max": float(aurocs.max()),
        "AUROC_std": float(aurocs.std()),
        "MCC_min": float(mccs.min()),
        "MCC_max": float(mccs.max()),
        "MCC_std": float(mccs.std()),
        "overall": overall,
        "split": "70/15/15 (seed=%d)" % args.seed,
        "checkpoint": os.path.basename(args.checkpoint),
        "params": n_params,
        "device": device,
    }
    result = {"summary": summary, "per_rbp": per_rbp}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print("  evaluable RBPs      : %d" % len(per_rbp))
    print("  macro-avg AUROC     : %.4f" % summary["macro_AUROC"])
    print("  macro-avg MCC       : %.4f" % summary["macro_MCC"])
    print("  sample-pooled AUROC : %.4f" % overall["AUROC"])
    print("  sample-pooled MCC   : %.4f" % overall["MCC"])
    print("  saved: %s (%.1fs)" % (args.out, time.time() - t0))
    print("=" * 60)
    return result


if __name__ == "__main__":
    main()
