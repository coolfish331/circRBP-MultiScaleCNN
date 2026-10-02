"""
================================================================
precompute_kmer_full.py — 全量 K-mer 频率特征预计算
================================================================

输入:  X_circrna.npy  (260554, 501, 4) — full one-hot circRNA sequences
输出:  K_full.npy      (260554, 320)    — 3-mer(64) + 4-mer(256) 归一化频率

原理:
  从 one-hot 序列中提取 3-mer 和 4-mer 的全局频率向量，用矩阵运算避免循环。
  - k3_indices = base1*16 + base2*4 + base3 → 64 维
  - k4_indices = base1*64 + base2*16 + base3*4 + base4 → 256 维

用法:
  cd <repo-root>/gpu_deploy
  python precompute_kmer_full.py

预估:
  - 运行时间: ~2 分钟 (260K 序列)
  - 输出大小: ~318 MB
"""

import os, time
import numpy as np

BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data", "circ_processed")
X_PATH    = os.path.join(DATA_DIR, "X_circrna.npy")
OUT_PATH  = os.path.join(DATA_DIR, "K_full.npy")


def compute_kmer_features(X: np.ndarray, chunk_size: int = 20000) -> np.ndarray:
    """
    分批计算 K-mer 频率，避免内存溢出。

    Args:
        X:      (N, L, 4) float32 one-hot 矩阵
        chunk_size: 分批大小

    Returns:
        (N, 320) float32 K-mer 频率矩阵
    """
    N, L, C = X.shape
    K = np.zeros((N, 320), dtype=np.float32)  # 64(3mer) + 256(4mer)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch = X[start:end]                      # (chunk, L, 4)

        # 转为碱基索引: (chunk, L)
        b = np.argmax(batch, axis=2).astype(np.int32)

        # ── 3-mer (64 维) ──
        k3_idx = b[:, :-2] * 16 + b[:, 1:-1] * 4 + b[:, 2:]   # (chunk, L-2)
        K3 = np.zeros((end - start, 64), dtype=np.float32)
        for v in range(64):
            K3[:, v] = (k3_idx == v).sum(axis=1).astype(np.float32)
        K3 /= (L - 2)  # 归一化

        # ── 4-mer (256 维) ──
        k4_idx = (b[:, :-3] * 64 + b[:, 1:-2] * 16 +
                  b[:, 2:-1] * 4 + b[:, 3:])                     # (chunk, L-3)
        K4 = np.zeros((end - start, 256), dtype=np.float32)
        for v in range(256):
            K4[:, v] = (k4_idx == v).sum(axis=1).astype(np.float32)
        K4 /= (L - 3)

        K[start:end, :64]  = K3
        K[start:end, 64:]  = K4

        elapsed = end
        pct = end / N * 100
        print(f"\r  {elapsed}/{N} ({pct:.1f}%)", end="", flush=True)

    print()
    return K


def main():
    print("=" * 60)
    print("  K-mer 频率特征预计算 (全量 260K)")
    print("=" * 60)

    # 加载全量数据（mmap）
    print(f"\n[Load] {X_PATH}")
    X = np.load(X_PATH, mmap_mode='r')
    print(f"  Shape: {X.shape}, Dtype: {X.dtype}, Memory: {X.nbytes/1024**3:.2f} GB")

    t0 = time.time()
    K = compute_kmer_features(X)

    elapsed = time.time() - t0
    print(f"\n[Done] 计算完成: {elapsed:.1f}s ({X.shape[0]/elapsed:.0f} seq/s)")

    np.save(OUT_PATH, K)
    size_mb = os.path.getsize(OUT_PATH) / 1024**2
    print(f"[Save] {OUT_PATH}  ({size_mb:.0f} MB)  shape={K.shape}")

    # 快速统计
    print(f"\n[Stats]")
    print(f"  3-mer 范围:  [{K[:,:64].min():.6f}, {K[:,:64].max():.6f}]")
    print(f"  4-mer 范围:  [{K[:,64:].min():.6f}, {K[:,64:].max():.6f}]")
    print(f"  NaN:         {np.isnan(K).sum()}")


if __name__ == "__main__":
    main()
