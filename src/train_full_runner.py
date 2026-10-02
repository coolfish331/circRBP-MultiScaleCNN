"""
================================================================
run_gpu_training.py — GPU 训练 Runner (子进程隔离 + 断点续训)
================================================================

每个 epoch 运行独立子进程：
  - 进程崩溃不影响整体进度
  - 自动从历史文件恢复断点
  - 支持 early stop (exit code 2)

用法:
  cd <repo-root>/gpu_deploy
  python run_gpu_training.py [--epochs 30]
"""

import subprocess, sys, os, json, time, argparse

PYTHON       = sys.executable
GPU_DIR      = os.path.dirname(os.path.abspath(__file__))
SCRIPT       = os.path.join(GPU_DIR, "train_full.py")
BASE_DIR     = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
HISTORY_FILE = os.path.join(RESULTS_DIR, "gpu_full_history.json")


def get_start_epoch():
    """从历史文件获取下一个要执行的 epoch"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        done = len(data.get("history", []))
        if done > 0:
            print(f"[Runner] 断点续训: 已完成 {done} epochs, 从 epoch {done+1} 开始")
            return done
    return 0


def main():
    parser = argparse.ArgumentParser(description="GPU 训练 Runner")
    parser.add_argument("--epochs", type=int, default=70,
                        help="总训练轮数 (default 30)")
    args = parser.parse_args()

    gpu_name = "待检测"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except:
        pass

    start_ep = get_start_epoch()
    total_ep = args.epochs

    print("=" * 60)
    print(f"  🚀 GPU Full Training Runner")
    print(f"  GPU:       {gpu_name}")
    print(f"  Epochs:    {start_ep+1} → {total_ep}")
    print(f"  Script:    {SCRIPT}")
    print("=" * 60)

    total_start = time.time()
    epoch_times = []

    for ep in range(start_ep, total_ep):
        print(f"\n{'─'*60}")
        print(f"[Runner] Epoch {ep+1}/{total_ep} starting ...")
        ep_start = time.time()

        result = subprocess.run(
            [PYTHON, "-u", SCRIPT, str(ep)],
            capture_output=False,
            timeout=1800  # 30 min max per epoch (GPU 应该 ≤30s)
        )

        elapsed = time.time() - ep_start
        epoch_times.append(elapsed)
        avg_time = sum(epoch_times) / len(epoch_times)

        print(f"[Runner] Epoch {ep+1} done: {elapsed:.1f}s "
              f"(avg {avg_time:.1f}s, exit={result.returncode})")

        if result.returncode == 2:
            print("[Runner] Early stop signal received. Training complete.")
            break
        elif result.returncode != 0:
            print(f"[Runner] ⚠ WARNING: exit code {result.returncode}, "
                  f"continuing to next epoch...")
            # 短暂暂停让 GPU 降温
            time.sleep(2)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"[Runner] All done! Total: {total_time/60:.1f} min")

    if epoch_times:
        print(f"[Runner] Avg/epoch: {sum(epoch_times)/len(epoch_times):.1f}s | "
              f"Fastest: {min(epoch_times):.1f}s | Slowest: {max(epoch_times):.1f}s")

    # 打印最终结果
    result_file = os.path.join(RESULTS_DIR, "gpu_full_results.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            res = json.load(f)
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS — GPU Full Training")
        print(f"  Model:    CNN(multi-scale) + K-mer(3+4-mer) + Structure")
        print(f"  TEST AUROC: {res['test_auc']:.4f}")
        print(f"  TEST AUPRC: {res['test_auprc']:.4f}")
        print(f"  TEST MCC:   {res['test_mcc']:.4f}")
        print(f"  Best epoch: {res['best_epoch']}")
        print(f"  Params:     {res['n_params']:,}")
        print("=" * 60)
    else:
        print(f"\n[Runner] 训练未完成，结果文件尚未生成。")


if __name__ == "__main__":
    main()
