"""
================================================================
precompute_structure_full.py — 全量 RNA 二级结构特征预计算
================================================================

输入:  X_circrna.npy  (260554, 501, 4) — full one-hot circRNA sequences
输出:  S_full.npy      (260554, 501, 5) — structural encoding

结构编码 (5 通道 one-hot):
  Channel 0: (  (paired, 5' 侧, stem start)
  Channel 1: )  (paired, 3' 侧, stem end)
  Channel 2: H  (hairpin / near-stem)
  Channel 3: I  (interior loop, T merged into I)
  Channel 4: F  (flat / unstructured, S → paired)

算法: Nussinov DP (纯 Python, O(N^3)), s/N 约 7 ms
预估: 260K seqs @ 8 workers → ~70 分钟, 输出 ~2.5 GB

用法:
  cd <repo-root>/gpu_deploy
  python precompute_structure_full.py [--workers 8] [--chunk 500]
"""

import os, sys, time, argparse
import numpy as np
import multiprocessing as mp

# 引用父目录中的 rna_structure 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rna_structure import nussinov_dp, traceback, get_dot_bracket

BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "circ_processed")
X_PATH   = os.path.join(DATA_DIR, "X_circrna.npy")
OUT_PATH = os.path.join(DATA_DIR, "S_full.npy")
BASES    = ['A', 'C', 'G', 'U']
MAX_LEN  = 501

# 结构化编码映射
STRUCT_MAP = {
    '(':  [1, 0, 0, 0, 0],
    ')':  [0, 1, 0, 0, 0],
    'H':  [0, 0, 1, 0, 0],
    'I':  [0, 0, 0, 1, 0],
    'T':  [0, 0, 0, 1, 0],   # T → I (same channel)
    'F':  [0, 0, 0, 0, 1],
}


def encode_structure_for_seq(one_hot_row: np.ndarray) -> np.ndarray:
    """
    对单个 one-hot 序列预测并编码二级结构。

    Args:
        one_hot_row: (MAX_LEN, 4) one-hot 编码序列（含 padding）

    Returns:
        (valid_len, 5) float32 结构编码矩阵
    """
    valid_mask = np.any(one_hot_row != 0, axis=1)
    valid_len  = int(valid_mask.sum())
    if valid_len < 4:
        return np.zeros((valid_len, 5), dtype=np.float32)

    seq = ''.join([BASES[np.argmax(one_hot_row[i])] for i in range(valid_len)])
    seq = seq.upper().replace('T', 'U')

    try:
        dp    = nussinov_dp(seq, min_loop_len=3)
        pairs = traceback(seq, dp, min_loop_len=3)
        db    = get_dot_bracket(pairs, len(seq))
    except Exception:
        db = '.' * len(seq)

    n = len(db)
    result = np.zeros((n, 5), dtype=np.float32)
    for i, c in enumerate(db):
        if c == '(':
            result[i] = STRUCT_MAP['(']
        elif c == ')':
            result[i] = STRUCT_MAP[')']
        else:
            # 上下文分类
            left_p  = sum(1 for j in range(max(0, i - 5), i)      if db[j] in '()')
            right_p = sum(1 for j in range(i + 1, min(n, i + 6))  if db[j] in '()')
            if left_p >= 2 and right_p >= 2:
                result[i] = STRUCT_MAP['I']   # interior loop
            elif left_p >= 1 and right_p >= 1:
                result[i] = STRUCT_MAP['T']   # two-loop → channel 3
            elif left_p >= 1 or right_p >= 1:
                result[i] = STRUCT_MAP['H']   # hairpin
            else:
                result[i] = STRUCT_MAP['F']   # flat
    return result


def worker_fn(args):
    """多进程 worker: 处理一批索引"""
    indices, x_path = args
    X = np.load(x_path, mmap_mode='r')
    results = []
    for idx in indices:
        enc = encode_structure_for_seq(X[idx])
        results.append((idx, enc))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6,
                        help='并行进程数 (default 6)')
    parser.add_argument('--chunk', type=int, default=500,
                        help='每个 task 处理的样本数 (default 500)')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制样本数 (调试用)')
    args = parser.parse_args()

    X = np.load(X_PATH, mmap_mode='r')
    N = args.limit if args.limit else X.shape[0]

    print("=" * 60)
    print(f"  RNA 二级结构特征预计算 (全量 {N} 序列)")
    print("=" * 60)
    print(f"  Workers: {args.workers}, Chunk: {args.chunk}")
    print(f"  输入: {X_PATH}  ({X.nbytes/1024**3:.2f} GB)")
    print(f"  输出: {OUT_PATH}  (预估 ~{N*MAX_LEN*5*4/1024**3:.1f} GB)")

    # 检测有效序列长度
    valid_lens = np.array([int(np.any(X[i] != 0, axis=1).sum()) for i in range(min(10, N))])
    seq_len = int(valid_lens.max())
    print(f"  有效序列长度: {seq_len} bp")
    print(f"  预估耗时: ~{N/(67*args.workers)*60:.0f} min (基于 67 seq/s/worker)")

    # 分配输出
    S = np.zeros((N, MAX_LEN, 5), dtype=np.float32)

    # 分块并多进程计算
    all_idx = list(range(N))
    chunks  = [all_idx[i:i+args.chunk] for i in range(0, N, args.chunk)]
    tasks   = [(chunk, X_PATH) for chunk in chunks]

    t0 = time.time()
    completed = 0
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for batch_results in pool.imap_unordered(worker_fn, tasks):
                for idx, enc in batch_results:
                    vlen = enc.shape[0]
                    S[idx, :vlen, :] = enc
                completed += len(batch_results)
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (N - completed) / rate if rate > 0 else 0
                print(f"\r  {completed}/{N} ({completed/N*100:.1f}%) "
                      f"| {rate:.0f} seq/s | ETA {eta/60:.1f} min   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving partial results...")

    elapsed = time.time() - t0
    print(f"\n\n[Done] {completed}/{N} sequences in {elapsed/60:.1f} min "
          f"({completed/elapsed:.0f} seq/s)")

    np.save(OUT_PATH, S[:completed])
    size_gb = os.path.getsize(OUT_PATH) / 1024**3
    print(f"[Save] {OUT_PATH}  ({size_gb:.1f} GB)  shape={S[:completed].shape}")


if __name__ == "__main__":
    main()
