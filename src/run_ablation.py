"""
================================================================
run_ablation.py — CircInteractome 消融实验训练器 (GPU)
================================================================

四种模型变体:
  seq_only   : Branch-A only (多尺度 Conv1D，仅序列)
  seq_kmer   : Branch-A + Branch-B (序列 + K-mer 频率)
  seq_struct : Branch-A + Branch-C (序列 + Nussinov 二级结构)
  full       : Branch-A + Branch-B + Branch-C (完整三路融合)

所有变体使用:
  - 相同数据划分 (seed=42, 70/15/15)
  - 相同超参数 (LR=5e-4, weight_decay=1e-5, dropout=0.3)
  - 相同训练轮数 (30 epochs)
  - 混合精度加速 (AMP)
  - 独立保存模型权重 + 训练历史 + 测试集指标

用法:
  python run_ablation.py --variant seq_only
  python run_ablation.py --variant seq_kmer
  python run_ablation.py --variant seq_struct
  python run_ablation.py --variant full

输出文件 (results/):
  ablation_seq_only_best.pt     / ablation_seq_only_history.json
  ablation_seq_kmer_best.pt     / ablation_seq_kmer_history.json
  ablation_seq_struct_best.pt   / ablation_seq_struct_history.json
  ablation_full_best.pt         / ablation_full_history.json
  ablation_summary.json         (汇总对比)
================================================================
"""

import json, time, sys, os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, accuracy_score,
                             confusion_matrix)
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)   # RBP_project/
DATA_DIR    = os.path.join(BASE_DIR, "data", "circ_processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SPLIT_FILE   = os.path.join(RESULTS_DIR, "ablation_split.json")  # 统一数据划分

DEVICE       = "cuda"
BATCH_SIZE   = 512
LR           = 5e-4
TOTAL_EPOCHS = 30
SEED         = 42
MAX_LEN      = 501
USE_AMP      = True
NUM_WORKERS  = 4
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0
DROPOUT      = 0.3


# ══════════════════════════════════════════════════════════════════════
# 可配置分支的消融模型
# ══════════════════════════════════════════════════════════════════════
class AblationModel(nn.Module):
    """
    通过 branches 参数控制启用哪些分支:
      ['seq']                     → Seq-only
      ['seq', 'kmer']             → Seq + K-mer
      ['seq', 'struct']           → Seq + Struct
      ['seq', 'kmer', 'struct']   → Full
    """
    def __init__(self, branches, kmer_dim=320, struct_dim=5):
        super().__init__()
        self.branches = branches
        self.use_kmer   = 'kmer'   in branches
        self.use_struct = 'struct' in branches

        # Branch A: 序列多尺度 CNN (始终启用)
        self.conv4  = nn.Conv1d(4, 64, kernel_size=4,  padding=2)
        self.conv8  = nn.Conv1d(4, 64, kernel_size=8,  padding=4)
        self.conv16 = nn.Conv1d(4, 64, kernel_size=16, padding=8)
        self.bn_seq = nn.BatchNorm1d(192)

        # Branch B: K-mer (可选)
        if self.use_kmer:
            self.kmer_bn  = nn.BatchNorm1d(kmer_dim)
            self.kmer_fc  = nn.Linear(kmer_dim, 64)
            self.kmer_bn2 = nn.BatchNorm1d(64)

        # Branch C: 二级结构 (可选)
        if self.use_struct:
            self.conv_s4 = nn.Conv1d(struct_dim, 32, kernel_size=4, padding=2)
            self.conv_s8 = nn.Conv1d(struct_dim, 32, kernel_size=8, padding=4)
            self.bn_strc = nn.BatchNorm1d(64)

        # Fusion: 动态计算输入维度
        fusion_dim = 192
        if self.use_kmer:
            fusion_dim += 64
        if self.use_struct:
            fusion_dim += 64

        self.fc1   = nn.Linear(fusion_dim, 128)
        self.bn_f1 = nn.BatchNorm1d(128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 1)
        self.drop  = nn.Dropout(DROPOUT)
        self.relu  = nn.ReLU()

    def forward(self, seq, kmer, struct):
        # Branch A: 序列多尺度 CNN
        x = seq.permute(0, 2, 1)          # (B, 4, L)
        c4  = self.relu(self.conv4(x))[:, :, :MAX_LEN]
        c8  = self.relu(self.conv8(x))[:, :, :MAX_LEN]
        c16 = self.relu(self.conv16(x))[:, :, :MAX_LEN]
        a = torch.cat([c4.max(2).values, c8.max(2).values, c16.max(2).values], dim=1)
        a = self.drop(self.bn_seq(a))

        parts = [a]

        # Branch B: K-mer (如启用)
        if self.use_kmer:
            b = self.relu(self.kmer_fc(self.kmer_bn(kmer)))
            b = self.drop(self.kmer_bn2(b))
            parts.append(b)

        # Branch C: 二级结构 (如启用)
        if self.use_struct:
            s = struct.permute(0, 2, 1)   # (B, 5, L)
            cs4 = self.relu(self.conv_s4(s))[:, :, :MAX_LEN]
            cs8 = self.relu(self.conv_s8(s))[:, :, :MAX_LEN]
            c_out = torch.cat([cs4.max(2).values, cs8.max(2).values], dim=1)
            c_out = self.drop(self.bn_strc(c_out))
            parts.append(c_out)

        # Fusion
        x = torch.cat(parts, dim=1)
        x = self.drop(self.relu(self.bn_f1(self.fc1(x))))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
# DataLoader
# ══════════════════════════════════════════════════════════════════════
def make_loader(Xa, Ka, Sa, ya, order=None):
    if order is not None:
        Xa, Ka, Sa, ya = Xa[order], Ka[order], Sa[order], ya[order]
    ds = TensorDataset(
        torch.tensor(Xa, dtype=torch.float32),
        torch.tensor(Ka, dtype=torch.float32),
        torch.tensor(Sa, dtype=torch.float32),
        torch.tensor(ya, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      drop_last=True, num_workers=NUM_WORKERS,
                      pin_memory=True)


# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, opt, criterion, scaler):
    model.train()
    total_loss, preds_list, labels_list = 0.0, [], []
    for xb, kb, sb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        kb = kb.to(DEVICE, non_blocking=True)
        sb = sb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        opt.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(xb, kb, sb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(xb, kb, sb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
        preds_list.extend(probs.tolist())
        labels_list.extend(yb.cpu().numpy().tolist())

    tr_auc = roc_auc_score(labels_list, preds_list) if len(set(labels_list)) > 1 else 0.5
    return total_loss / len(loader), tr_auc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds_list, labels_list = 0.0, [], []
    for xb, kb, sb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        kb = kb.to(DEVICE, non_blocking=True)
        sb = sb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        logits = model(xb, kb, sb)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds_list.extend(probs.tolist())
        labels_list.extend(yb.cpu().numpy().tolist())

    labels_arr = np.array(labels_list)
    preds_arr  = np.array(preds_list)
    preds_bin  = (preds_arr > 0.5).astype(int)

    auc   = roc_auc_score(labels_arr, preds_arr) if len(set(labels_list)) > 1 else 0.5
    auprc = average_precision_score(labels_arr, preds_arr) if len(set(labels_list)) > 1 else 0.5
    mcc   = matthews_corrcoef(labels_arr, preds_bin)
    acc   = accuracy_score(labels_arr, preds_bin)
    cm    = confusion_matrix(labels_arr, preds_bin)
    tn, fp, fn, tp = cm.ravel()
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1    = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    return {
        'loss': (loss_avg := total_loss / len(loader)),
        'auroc': auc, 'auprc': auprc, 'mcc': mcc,
        'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
        'precision': prec, 'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'preds': list(preds_arr), 'labels': list(labels_arr),
    }


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True,
                        choices=['seq_only', 'seq_kmer', 'seq_struct', 'full'])
    args = parser.parse_args()

    variant     = args.variant
    branches_map = {
        'seq_only':   ['seq'],
        'seq_kmer':   ['seq', 'kmer'],
        'seq_struct': ['seq', 'struct'],
        'full':       ['seq', 'kmer', 'struct'],
    }
    branches = branches_map[variant]

    best_pt     = os.path.join(RESULTS_DIR, f"ablation_{variant}_best.pt")
    ckpt_pt     = os.path.join(RESULTS_DIR, f"ablation_{variant}_ckpt.pt")
    history_js  = os.path.join(RESULTS_DIR, f"ablation_{variant}_history.json")
    variant_name = ' / '.join(b for b in ['Sequence', 'K-mer', 'Structure']
                              if b.lower() in variant or (b.lower() == 'sequence' and 'seq' in variant))

    # ── GPU 信息 ──────────────────────────────────────────────────────
    print(f"=" * 55)
    print(f"  Ablation: {variant}  ({variant_name})")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB | AMP: {USE_AMP}")
    print(f"=" * 55, flush=True)

    # ── 加载数据 ──────────────────────────────────────────────────────
    print("[Load] data...", flush=True)
    X_all = np.load(os.path.join(DATA_DIR, "X_circrna.npy"), mmap_mode='r')
    y_all = np.load(os.path.join(DATA_DIR, "y_circrna.npy"),   mmap_mode='r')
    K_all = np.load(os.path.join(DATA_DIR, "K_full.npy"),       mmap_mode='r')
    S_all = np.load(os.path.join(DATA_DIR, "S_full.npy"),       mmap_mode='r')
    n_total = len(y_all)
    print(f"[Load] N={n_total}, pos_rate={y_all.sum()/n_total:.3f}", flush=True)

    # ── 数据划分 (复用或重新生成) ────────────────────────────────────
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            split = json.load(f)
        tr_i = np.array(split['train_idx'])
        vl_i = np.array(split['val_idx'])
        ts_i = np.array(split['test_idx'])
        print(f"[Split] Reusing saved split: Train={len(tr_i)}, Val={len(vl_i)}, Test={len(ts_i)}", flush=True)
    else:
        idx  = np.random.RandomState(SEED).permutation(n_total)
        n_tr = int(0.7 * n_total)
        n_vl = int(0.15 * n_total)
        tr_i = idx[:n_tr]
        vl_i = idx[n_tr:n_tr + n_vl]
        ts_i = idx[n_tr + n_vl:]
        with open(SPLIT_FILE, 'w') as f:
            json.dump({'train_idx': tr_i.tolist(), 'val_idx': vl_i.tolist(),
                       'test_idx': ts_i.tolist(), 'seed': SEED}, f)
        print(f"[Split] Created split: Train={len(tr_i)}, Val={len(vl_i)}, Test={len(ts_i)}", flush=True)

    # ── 加载到内存 ────────────────────────────────────────────────────
    X    = np.array(X_all[tr_i], dtype=np.float32)
    K    = np.array(K_all[tr_i], dtype=np.float32)
    S    = np.array(S_all[tr_i], dtype=np.float32)
    y    = np.array(y_all[tr_i], dtype=np.float32)
    X_vl = np.array(X_all[vl_i], dtype=np.float32)
    K_vl = np.array(K_all[vl_i], dtype=np.float32)
    S_vl = np.array(S_all[vl_i], dtype=np.float32)
    y_vl = np.array(y_all[vl_i], dtype=np.float32)
    X_ts = np.array(X_all[ts_i], dtype=np.float32)
    K_ts = np.array(K_all[ts_i], dtype=np.float32)
    S_ts = np.array(S_all[ts_i], dtype=np.float32)
    y_ts = np.array(y_all[ts_i], dtype=np.float32)
    n_tr = len(y)
    n_vl = len(y_vl)
    n_ts = len(y_ts)
    print(f"[Mem] Train={X.shape}, K={K.shape}, S={S.shape}", flush=True)
    del X_all, K_all, S_all, y_all

    # ── DataLoaders ──────────────────────────────────────────────────
    tr_ld = make_loader(X, K, S, y)
    vl_ld = make_loader(X_vl, K_vl, S_vl, y_vl)
    ts_ld = make_loader(X_ts, K_ts, S_ts, y_ts)

    # ── 模型 ─────────────────────────────────────────────────────────
    model = AblationModel(branches, kmer_dim=K.shape[1], struct_dim=S.shape[2]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler() if USE_AMP else None
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    print(f"[Model] Params: {n_params:,} | Branches: {branches} | "
          f"Fusion dim: {model.fc1.in_features}", flush=True)

    # ── 训练循环 ─────────────────────────────────────────────────────
    history = []
    best_auc, best_ep = 0.0, 0

    for ep in range(TOTAL_EPOCHS):
        t0 = time.time()

        # Train
        ep_rng = np.random.RandomState(SEED + ep * 31337)
        tr_order = ep_rng.permutation(n_tr)
        tr_ld_shuffled = make_loader(X, K, S, y, order=tr_order)
        tr_loss, tr_auc = train_one_epoch(model, tr_ld_shuffled, opt, criterion, scaler)

        # Val
        val_res = evaluate(model, vl_ld, criterion)
        ep_time = time.time() - t0

        is_best = val_res['auroc'] > best_auc
        marker  = "*" if is_best else " "
        if is_best:
            best_auc = val_res['auroc']
            best_ep  = ep + 1
            torch.save(model.state_dict(), best_pt)

        print(f"  Ep {ep+1:2d}/{TOTAL_EPOCHS}{marker} | "
              f"TL={tr_loss:.4f} VL={val_res['loss']:.4f} | "
              f"AUC={val_res['auroc']:.4f} AUPRC={val_res['auprc']:.4f} "
              f"MCC={val_res['mcc']:.4f} | {ep_time:.1f}s", flush=True)

        history.append({
            "epoch": ep + 1, "train_loss": float(tr_loss),
            "val_loss": float(val_res['loss']), "val_auc": float(val_res['auroc']),
            "val_auprc": float(val_res['auprc']), "val_mcc": float(val_res['mcc']),
            "time_s": float(ep_time),
        })

        # 保存checkpoint
        ckpt_dict = {'model': model.state_dict(), 'optimizer': opt.state_dict()}
        if scaler:
            ckpt_dict['scaler'] = scaler.state_dict()
        torch.save(ckpt_dict, ckpt_pt)

        # 每5轮保存历史
        if (ep + 1) % 5 == 0 or (ep + 1) == TOTAL_EPOCHS:
            with open(history_js, 'w') as f:
                json.dump({"variant": variant, "branches": branches,
                           "best_auc": best_auc, "best_ep": best_ep,
                           "n_params": n_params, "history": history}, f, indent=2)

    # ── 最终测试 ─────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  FINAL TEST: {variant}  (best ep={best_ep})", flush=True)
    model.load_state_dict(torch.load(best_pt, map_location=DEVICE, weights_only=True))
    test_res = evaluate(model, ts_ld, criterion)

    print(f"  AUROC:      {test_res['auroc']:.4f}")
    print(f"  AUPRC:      {test_res['auprc']:.4f}")
    print(f"  MCC:        {test_res['mcc']:.4f}")
    print(f"  Accuracy:   {test_res['accuracy']:.4f}")
    print(f"  Sensitivity:{test_res['sensitivity']:.4f}")
    print(f"  Specificity:{test_res['specificity']:.4f}")
    print(f"  Precision:  {test_res['precision']:.4f}")
    print(f"  F1:         {test_res['f1']:.4f}")
    print(f"  CM: TP={test_res['tp']}, FP={test_res['fp']}, "
          f"TN={test_res['tn']}, FN={test_res['fn']}")
    print(f"{'=' * 55}", flush=True)

    # ── 保存结果 ─────────────────────────────────────────────────────
    result = {
        "variant": variant,
        "branches": branches,
        "n_params": n_params,
        "best_val_auc": float(best_auc),
        "best_epoch": best_ep,
        "test": {k: (v if k in ('tp','fp','tn','fn') else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
                 for k, v in test_res.items() if k not in ('preds', 'labels')},
        "history": history,
    }

    result_path = os.path.join(RESULTS_DIR, f"ablation_{variant}_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Done] Saved → ablation_{variant}_results.json", flush=True)

    # ── 更新汇总 ────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    summary[variant] = {
        "name": variant_name,
        "n_params": n_params,
        "best_val_auroc": float(best_auc),
        "best_epoch": best_ep,
        "test_auroc": float(test_res['auroc']),
        "test_auprc": float(test_res['auprc']),
        "test_mcc": float(test_res['mcc']),
        "test_accuracy": float(test_res['accuracy']),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] Updated → ablation_summary.json", flush=True)


if __name__ == '__main__':
    main()
