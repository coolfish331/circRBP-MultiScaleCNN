"""
evaluate.py — 用预缓存的 K_full/S_full 评估训练好的模型（测试集整体指标）
================================================================
因 K_full.npy 和 S_full.npy 已预计算好，无需重算特征，直接做划分+推理。
"""

import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, accuracy_score,
    confusion_matrix
)
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CNNGPUModel
warnings.filterwarnings("ignore")

BASE_DIR   = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = f"{BASE_DIR}/data/circ_processed"
BEST_MODEL = f"{BASE_DIR}/results/gpu_full_best.pt"
SEED       = 42
MAX_LEN    = 501

# 模型定义见 src/model.py（训练与评估共用同一计算图）


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ep70 最佳模型 — 精确测试集评估")
    print("  使用预缓存 K_full.npy + S_full.npy")
    print("=" * 60)

    # 1. 加载全量 label 做划分（仅加载标签，节省内存）
    print("\n[1/4] 数据划分 (seed=42, 70/15/15)...")
    y_all = np.load(f"{DATA_DIR}/y_circrna.npy", mmap_mode='r')
    n = len(y_all)
    idx = np.random.RandomState(SEED).permutation(n)
    n_tr = int(0.7 * n)
    n_vl = int(0.15 * n)
    ts_i = idx[n_tr + n_vl:]
    print(f"  总样本: {n:,}  测试集: {len(ts_i):,}")
    pos_ratio = float(y_all[ts_i].mean())
    print(f"  测试集正例比: {pos_ratio:.4f}")

    # 2. 加载预缓存特征 (memory-mapped, 按测试集索引切片)
    print("\n[2/4] 加载预缓存特征...")
    t1 = time.time()

    print("  加载 X_circrna.npy (mmap)...")
    X_all = np.load(f"{DATA_DIR}/X_circrna.npy", mmap_mode='r')
    print("  加载 K_full.npy    (mmap)...")
    K_all = np.load(f"{DATA_DIR}/K_full.npy",    mmap_mode='r')
    print("  加载 S_full.npy    (mmap)...")
    S_all = np.load(f"{DATA_DIR}/S_full.npy",    mmap_mode='r')

    # 按测试集索引切片到内存
    print("  切片测试集到内存...")
    X_ts = np.array(X_all[ts_i], dtype=np.float32)
    K_ts = np.array(K_all[ts_i], dtype=np.float32)
    S_ts = np.array(S_all[ts_i], dtype=np.float32)
    y_ts = np.array(y_all[ts_i], dtype=np.float32)
    del X_all, K_all, S_all, y_all

    print(f"  X_ts: {X_ts.shape}  K_ts: {K_ts.shape}  S_ts: {S_ts.shape}")
    print(f"  特征加载耗时: {time.time()-t1:.1f}s")

    # 3. 加载模型
    print("\n[3/4] 加载 ep70 模型权重...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = CNNGPUModel(kmer_dim=320, struct_dim=5).to(device)
    state = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 4. 批量推理
    print("\n[4/4] 推理中...")
    BATCH = 1024
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for start in range(0, len(y_ts), BATCH):
            end = min(start + BATCH, len(y_ts))
            xb = torch.tensor(X_ts[start:end]).to(device)
            kb = torch.tensor(K_ts[start:end]).to(device)
            sb = torch.tensor(S_ts[start:end]).to(device)
            logits = model(xb, kb, sb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(y_ts[start:end].tolist())
            if start % 10240 == 0:
                print(f"  {end}/{len(y_ts)} ...", end='\r')

    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    binary     = (preds_arr > 0.5).astype(int)

    # 计算指标
    auc   = roc_auc_score(labels_arr, preds_arr)
    auprc = average_precision_score(labels_arr, preds_arr)
    mcc   = matthews_corrcoef(labels_arr, binary)
    acc   = accuracy_score(labels_arr, binary)

    cm  = confusion_matrix(labels_arr, binary)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1          = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  === ep70 测试集最终指标 ===")
    print(f"  AUROC:       {auc:.6f}")
    print(f"  AUPRC:       {auprc:.6f}")
    print(f"  MCC:         {mcc:.6f}")
    print(f"  Accuracy:    {acc:.6f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  ---")
    print(f"  TP={TP}  FP={FP}")
    print(f"  FN={FN}  TN={TN}")
    print(f"  样本数: {len(labels_arr):,}  正例比: {pos_ratio:.4f}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"{'='*60}")

    # 保存结果
    result = {
        'method': 'CNN(multi-scale)+K-mer(3+4-mer)+Structure(Nussinov) Fusion',
        'dataset': 'CircInteractome (37 RBP, 260,554 samples)',
        'model': 'gpu_full_best.pt (ep70)',
        'train/val/test': '70/15/15 (seed=42)',
        'test_samples': int(len(labels_arr)),
        'positive_ratio': float(pos_ratio),
        'AUROC': float(auc),
        'AUPRC': float(auprc),
        'MCC': float(mcc),
        'Accuracy': float(acc),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'Precision': float(precision),
        'F1': float(f1),
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'params': n_params,
        'device': device,
        'eval_time_s': round(total_time, 1),
    }

    out_path = f"{BASE_DIR}/results/test_accuracy_ep70.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  已保存: {out_path}")
    return result


if __name__ == '__main__':
    main()
