# circRBP-MultiScaleCNN

Reference implementation of the lightweight multi-scale CNN for **circRNA-RBP
interaction prediction** described in the manuscript submitted to
*Journal of Cheminformatics*.

The model fuses three branches over each circRNA sequence:

| Branch | Input | Encoding |
|---|---|---|
| A — sequence | one-hot `(L, 4)` | multi-scale Conv1d (k = 4 / 8 / 16), 64 channels each, global max pooling → 192-d |
| B — K-mer | 3-mer + 4-mer frequency | `(320,)` → BN → FC(64) → 64-d |
| C — structure | Nussinov dot-bracket | 5-channel `(L, 5)` → Conv1d (k = 4 / 8), 32 channels each → 64-d |

Fusion head: `concat(192 + 64 + 64) → BN → FC(128) → FC(64) → FC(1)` — **80,833 parameters** in total.

---

## 1. Repository layout

```
code_release/
├── preprocessing/
│   └── preprocess_circinteractome.py   # raw CircInteractome -> X / K / S / y arrays
├── features/
│   ├── rna_structure.py                # Nussinov DP + dot-bracket traceback
│   ├── structure_features.py           # -> S_full.npy (N, 501, 5)
│   └── kmer_features.py                # -> K_full.npy (N, 320)
├── src/
│   ├── model.py                        # CNNGPUModel (shared by train / eval)
│   ├── train_full.py                   # one epoch per invocation (checkpointed)
│   ├── train_full_runner.py            # drives train_full.py, resumes automatically
│   ├── run_ablation.py                 # 4 variants: seq / +kmer / +struct / full
│   ├── evaluate.py                     # pooled test metrics (AUROC, AUPRC, MCC, Acc)
│   └── eval_per_rbp.py                 # per-RBP breakdown (36 retained RBPs)
├── scripts/
│   ├── plot_training_curves.py         # loss + val AUROC/AUPRC curves
│   ├── plot_ablation.py                # four-variant ablation figure
│   ├── plot_per_rbp_ranked.py          # ranked per-RBP AUROC bar chart (Figure S1)
│   └── bootstrap_macro_ci.py           # 10,000-fold bootstrap CI over the 36 retained RBPs
├── requirements.txt
└── LICENSE
```

## 2. Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.8; PyTorch ≥ 1.10 (CPU is sufficient for evaluation, a CUDA GPU such as
a GTX 1070 is strongly recommended for training).

## 3. Data

The experiments use the **CircInteractome** benchmark (37 RBPs as distributed,
publicly available at <http://circinteractome.nia.nih.gov/>). One RBP is
excluded by `preprocess_circinteractome.py`: **LIN28B** (only 88 negative
sequences, below the 100-per-class minimum). The remaining **36 RBPs** yield
the 270,554 circRNA-RBP pairs used in the manuscript. For each RBP,
download the "circRNA-RBP interaction" table and export the bound / unbound
sequence files into

```
data/circinteractome/circRNA-RBP/<RBP>/positive    # bound circRNA sequences
data/circinteractome/circRNA-RBP/<RBP>/negative    # unbound circRNA sequences
```

`<RBP>` is one of the 37 CircInteractome protein names (AGO1, AGO2, ...,
ZC3H7B); 36 of them survive the filtering step described above.
Protein sequences in `data/rbp_proteins/` are only needed if you re-run the
name-mapping step; the released model consumes circRNA sequence, K-mer and
structure features only.

All intermediate arrays are written to `data/circ_processed/` (up to ~5 GB). They
are **not** distributed with this repository; regenerate them with step 1–3 below.
To keep data outside the repository, set `RBP_BASE_DIR`:

```bash
export RBP_BASE_DIR=/path/to/your/workspace   # holds data/ and results/
```

## 4. Reproducing the reported results

```bash
# 1) parse the raw tables into one-hot sequences and labels
python preprocessing/preprocess_circinteractome.py     # -> data/circ_processed/X_circrna.npy, y_circrna.npy

# 2) K-mer frequency features (~2 min)
python features/kmer_features.py                       # -> data/circ_processed/K_full.npy

# 3) Nussinov secondary-structure features (~70 min with 8 workers)
python features/structure_features.py --workers 8      # -> data/circ_processed/S_full.npy

# 4) train the full model (70 epochs, seed 42, 70/15/15 split)
python src/train_full_runner.py --epochs 70            # -> results/gpu_full_best.pt

# 5) evaluate
python src/evaluate.py                                 # -> results/test_accuracy_ep70.json
python src/eval_per_rbp.py                             # -> results/per_rbp_results.json
python scripts/bootstrap_macro_ci.py                   # macro-averaged 95% CIs
```

Ablation (30 epochs, identical split and hyper-parameters):

```bash
python src/run_ablation.py --variant seq_only
python src/run_ablation.py --variant seq_kmer
python src/run_ablation.py --variant seq_struct
python src/run_ablation.py --variant full              # -> results/ablation_summary.json
python scripts/plot_ablation.py
```

## 5. Reference numbers

Split: 70 / 15 / 15 train / validation / test, `seed = 42`, 270,554 samples.

| Setting | AUROC | AUPRC | MCC | Accuracy | Params |
|---|---|---|---|---|---|
| Seq-only | 0.9160 | 0.9237 | 0.6853 | 0.8420 | 41,025 |
| Seq + K-mer | 0.9519 | 0.9548 | 0.7750 | 0.8860 | 70,529 |
| Seq + Structure | 0.9273 | 0.9338 | 0.7088 | 0.8523 | 51,329 |
| Full (30 ep) | 0.9555 | 0.9575 | 0.7883 | 0.8929 | 80,833 |
| **Full (70 ep, reported)** | **0.9688** | **0.9694** | **0.8269** | **0.9126** | 80,833 |

Per-RBP breakdown over the 36 retained RBPs: macro-averaged AUROC 0.9739
(95% CI 0.9652–0.9818), macro-averaged MCC 0.8427 (95% CI 0.8177–0.8657);
sample-pooled AUROC 0.9688.

Small deviations (±0.002) are expected across PyTorch/CUDA versions because the
split, initialisation seed and hyper-parameters are fixed but low-level kernels
are not bit-reproducible.

## 6. Trained weights

Trained checkpoints are not distributed here; they are fully reproducible with
step 4 above. If you need the exact `gpu_full_best.pt` used in the manuscript,
please contact the first author: **Meng ZhiGang** — mzg541 [at] qq [dot] com.

## 7. License

MIT — see [LICENSE](LICENSE).
