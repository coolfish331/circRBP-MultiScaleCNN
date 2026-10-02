"""Aggregate-level CI for per-RBP reporting. Bootstrap by resampling the 35 RBPs."""
BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np
from pathlib import Path
import os
RES = Path(BASE_DIR) / "results"
rows = json.load(open(RES / "per_rbp_results.json"))["per_rbp"]
mat = np.array([[r["AUROC"], r["AUPRC"], r["MCC"], r["Accuracy"]] for r in rows])
names = ["AUROC", "AUPRC", "MCC", "Accuracy"]
print(f"  n_rbps = {len(rows)}")

rng = np.random.RandomState(42)
B = 10000
boot = np.zeros((B, 4))
for i in range(B):
    idx = rng.randint(0, len(rows), len(rows))
    boot[i] = mat[idx].mean(axis=0)

pt = mat.mean(axis=0)
ci_l = np.percentile(boot, 2.5, axis=0)
ci_u = np.percentile(boot, 97.5, axis=0)
sd = boot.std(axis=0)
print(f"  {'metric':9s} {'macro':>8s} {'CIlow':>8s} {'CIhigh':>8s} {'sd':>8s}")
for j, n in enumerate(names):
    print(f"  {n:9s} {pt[j]:8.4f} {ci_l[j]:8.4f} {ci_u[j]:8.4f} {sd[j]:8.4f}")

out = {
    "n_rbps": len(rows),
    "B": B,
    "macro": {
        n: {"mean": float(pt[j]), "ci_low": float(ci_l[j]),
            "ci_high": float(ci_u[j]), "sd": float(sd[j])}
        for j, n in enumerate(names)
    },
    "note": "Bootstrap CI on the macro-mean (i.e., across 35 RBPs, "
            "re-sampled with replacement B=10000). Estimates the stability "
            "of the unweighted mean across RBPs, NOT the within-sample CI.",
}
json.dump(out, open(RES / "per_rbp_macro_ci.json", "w"), indent=2)
print("  -> per_rbp_macro_ci.json")
