"""
================================================================
circ_train_gpu.py — GPU 版 CNN + K-mer + Structure 三路融合训练器
================================================================

每个 epoch 由 run_gpu_training.py 在独立子进程中调用，传入 epoch 编号。
完成一个完整 epoch 的 train → val → checkpoint 后退出。

GPU 优化:
  - torch.cuda.amp 混合精度 (自动选择 fp16/fp32)
  - pin_memory=True 加速 CPU→GPU 传输
  - non_blocking=True 异步传输（重叠计算和数据搬运）
  - 梯度缩放器 (GradScaler) 防止 fp16 下溢
  - num_workers=4 多核数据加载
  - 批量大小 512 (vs CPU 128)

用法:
  python circ_train_gpu.py <epoch_num>
"""

import json, time, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CNNGPUModel
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# 配置 — 笔记本迁移时只需修改 BASE_DIR 和 DEVICE
# ══════════════════════════════════════════════════════════════════════
BASE_DIR       = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR       = f"{BASE_DIR}/data/circ_processed"
RESULTS_DIR    = f"{BASE_DIR}/results"

CHECKPOINT     = f"{RESULTS_DIR}/gpu_full_ckpt.pt"
HISTORY_FILE   = f"{RESULTS_DIR}/gpu_full_history.json"
BEST_MODEL     = f"{RESULTS_DIR}/gpu_full_best.pt"

DEVICE         = "cuda"  # GTX 1070
BATCH_SIZE     = 512     # GPU 适合更大的批大小
LR             = 5e-4
TOTAL_EPOCHS   = 70
SEED           = 42
MAX_LEN        = 501
USE_AMP        = True    # 混合精度 (GTX 1070 推荐开启)
NUM_WORKERS    = 4       # 数据加载并行线程
WEIGHT_DECAY   = 1e-5
GRAD_CLIP      = 1.0
DROPOUT        = 0.3




def make_loader(Xa, Ka, Sa, ya, order=None, shuffle=False):
    """构建 GPU 友好 DataLoader"""
    if order is not None:
        Xa, Ka, Sa, ya = Xa[order], Ka[order], Sa[order], ya[order]
    ds = TensorDataset(
        torch.tensor(Xa, dtype=torch.float32),
        torch.tensor(Ka, dtype=torch.float32),
        torch.tensor(Sa, dtype=torch.float32),
        torch.tensor(ya, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      drop_last=True, num_workers=NUM_WORKERS,
                      pin_memory=True)  # pin_memory 加速 →GPU


def train_one_epoch(model, loader, opt, criterion, scaler):
    """训练一个 epoch（支持混合精度）"""
    model.train()
    total_loss, preds_list, labels_list = 0.0, [], []
    n_batches = len(loader)

    for batch_i, (xb, kb, sb, yb) in enumerate(loader):
        # 数据 → GPU (non_blocking 异步)
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

        if (batch_i + 1) % 20 == 0:
            b_auc = roc_auc_score(labels_list[-len(probs)*10:], preds_list[-len(probs)*10:]) \
                    if len(set(labels_list[-len(probs)*10:])) > 1 else 0.5
            print(f"  batch {batch_i+1}/{n_batches} loss={loss.item():.4f} "
                  f"recent_auc={b_auc:.4f}", flush=True)

    tr_auc = roc_auc_score(labels_list, preds_list) if len(set(labels_list)) > 1 else 0.5
    return total_loss / n_batches, tr_auc


@torch.no_grad()
def evaluate(model, loader, criterion):
    """验证/测试评估"""
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

    loss_avg = total_loss / len(loader)
    labels_arr = np.array(labels_list)
    preds_arr  = np.array(preds_list)

    auc   = roc_auc_score(labels_arr, preds_arr) if len(set(labels_list)) > 1 else 0.5
    auprc = average_precision_score(labels_arr, preds_arr) if len(set(labels_list)) > 1 else 0.5
    mcc   = matthews_corrcoef(labels_arr, (preds_arr > 0.5).astype(int))
    return loss_avg, auc, auprc, mcc, preds_list, labels_list


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # ── GPU 信息 ──────────────────────────────────────────────────────────
    print(f"[GPU] {torch.cuda.get_device_name(0)} | "
          f"Mem: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB | "
          f"AMP: {USE_AMP}", flush=True)

    print(f"=== Epoch {ep+1}/{TOTAL_EPOCHS} (GPU: CNN+Kmer+Struct) ===", flush=True)
    t0 = time.time()

    # ── 加载全量数据 (mmap) ──────────────────────────────────────────────
    print("[Load] data...", flush=True)
    X_all = np.load(f"{DATA_DIR}/X_circrna.npy",   mmap_mode='r')
    y_all = np.load(f"{DATA_DIR}/y_circrna.npy",   mmap_mode='r')
    K_all = np.load(f"{DATA_DIR}/K_full.npy",       mmap_mode='r')
    S_all = np.load(f"{DATA_DIR}/S_full.npy",       mmap_mode='r')

    n = len(y_all)
    print(f"[Load] N={n}, X={X_all.shape}, K={K_all.shape}, S={S_all.shape}", flush=True)

    # ── 数据集划分 (70/15/15) ─────────────────────────────────────────────
    idx  = np.random.RandomState(SEED).permutation(n)
    n_tr = int(0.7 * n)
    n_vl = int(0.15 * n)
    tr_i = idx[:n_tr]
    vl_i = idx[n_tr:n_tr + n_vl]
    ts_i = idx[n_tr + n_vl:]

    # 拷贝到内存（非 mmap，因为 GPU 需要）
    print(f"[Split] Train={n_tr}, Val={n_vl}, Test={n - n_tr - n_vl}", flush=True)
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
    del X_all, K_all, S_all, y_all

    # ── DataLoaders ──────────────────────────────────────────────────────
    ep_rng   = np.random.RandomState(SEED + ep * 31337)
    tr_order = ep_rng.permutation(n_tr)

    tr_ld = make_loader(X, K, S, y, order=tr_order, shuffle=False)
    vl_ld = make_loader(X_vl, K_vl, S_vl, y_vl)
    ts_ld = make_loader(X_ts, K_ts, S_ts, y_ts)
    print(f"[Loader] Train={len(tr_ld)}B, Val={len(vl_ld)}B, Test={len(ts_ld)}B "
          f"(batch={BATCH_SIZE})", flush=True)

    # ── 模型 ─────────────────────────────────────────────────────────────
    model    = CNNGPUModel(kmer_dim=K.shape[1], struct_dim=S.shape[2]).to(DEVICE)
    opt      = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler   = GradScaler() if USE_AMP else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Params: {n_params:,}", flush=True)

    # ── 断点恢复 ─────────────────────────────────────────────────────────
    history, best_auc, best_ep = [], 0.0, 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            saved = json.load(f)
        history  = saved.get('history', [])
        best_auc = saved.get('best_auc', 0.0)
        best_ep  = saved.get('best_ep', 0)

    if ep > 0 and os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        if scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f"[Checkpoint] Loaded from epoch {ep}", flush=True)

    criterion = nn.BCEWithLogitsLoss()

    # ── TRAIN ────────────────────────────────────────────────────────────
    print("[Train]", flush=True)
    tr_loss_avg, tr_auc = train_one_epoch(model, tr_ld, opt, criterion, scaler)
    print(f"[Train done] loss={tr_loss_avg:.4f} AUC={tr_auc:.4f}", flush=True)

    # ── VAL ──────────────────────────────────────────────────────────────
    print("[Val]", flush=True)
    vl_loss_avg, vl_auc, vl_auprc, vl_mcc, _, _ = evaluate(model, vl_ld, criterion)
    ep_time = time.time() - t0
    marker  = "*" if vl_auc > best_auc else " "

    if vl_auc > best_auc:
        best_auc = vl_auc
        best_ep  = ep + 1
        torch.save(model.state_dict(), BEST_MODEL)
        print(f"  ** Saved best model: AUC={best_auc:.4f} **", flush=True)

    print(f"\n  Ep {ep+1:2d}/{TOTAL_EPOCHS}{marker} | "
          f"TL={tr_loss_avg:.4f} VL={vl_loss_avg:.4f} | "
          f"AUC={vl_auc:.4f}(tr={tr_auc:.4f}) AUPRC={vl_auprc:.4f} "
          f"MCC={vl_mcc:.4f} | {ep_time:.1f}s", flush=True)

    # ── 保存检查点 ──────────────────────────────────────────────────────
    ckpt_dict = {'model': model.state_dict(), 'optimizer': opt.state_dict()}
    if scaler:
        ckpt_dict['scaler'] = scaler.state_dict()
    torch.save(ckpt_dict, CHECKPOINT)

    history.append({
        'epoch': ep + 1, 'train_loss': tr_loss_avg, 'val_loss': vl_loss_avg,
        'val_auc': vl_auc, 'val_auprc': vl_auprc, 'val_mcc': vl_mcc,
        'time_s': ep_time
    })
    with open(HISTORY_FILE, 'w') as f:
        json.dump({'best_auc': best_auc, 'best_ep': best_ep, 'history': history},
                  f, indent=2)

    # ── 最终测试 ─────────────────────────────────────────────────────────
    recent     = [h['val_auc'] for h in history[-10:]]
    early_stop = len(recent) >= 10 and max(recent) < best_auc - 0.001

    if ep + 1 >= TOTAL_EPOCHS or early_stop:
        print(f"\n{'='*60}")
        print("=== FINAL TEST EVALUATION ===", flush=True)
        model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE,
                                         weights_only=True))
        ts_loss, ts_auc, ts_auprc, ts_mcc, _, _ = evaluate(model, ts_ld, criterion)
        print(f"  TEST AUROC: {ts_auc:.4f}")
        print(f"  TEST AUPRC: {ts_auprc:.4f}")
        print(f"  TEST MCC:   {ts_mcc:.4f}")

        results = {
            'method': 'GPU: CNN(multi-scale)+K-mer(3+4-mer)+Structure(Nussinov) Fusion (full 37 RBP)',
            'device': DEVICE, 'batch_size': BATCH_SIZE,
            'test_auc': float(ts_auc), 'test_auprc': float(ts_auprc),
            'test_mcc': float(ts_mcc),
            'best_val_auc': float(best_auc), 'best_epoch': best_ep,
            'n_params': n_params, 'history': history
        }
        with open(f"{RESULTS_DIR}/gpu_full_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved → gpu_full_results.json", flush=True)

    if early_stop:
        sys.exit(2)


if __name__ == '__main__':
    main()
