"""
plot_training_curves.py — 生成高质量70轮训练曲线图（用于论文）
================================================================
输出: results/figures/training_curves_ep70.png  (300 DPI, 论文级)
"""

BASE_DIR = os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- 加载数据 ----
with open('<repo-root>/results/gpu_full_history.json') as f:
    h = json.load(f)

hist = h['history']
eps        = [r['epoch']      for r in hist]
train_loss = [r['train_loss'] for r in hist]
val_loss   = [r['val_loss']   for r in hist]
val_auc    = [r['val_auc']    for r in hist]
val_auprc  = [r['val_auprc'] if 'val_auprc' in r else r.get('val_auprc', r['val_auc']) for r in hist]
val_mcc    = [r['val_mcc']   for r in hist]

best_ep  = h['best_ep']        # 70
best_auc = h['best_auc']       # 0.9691

# ---- 画图 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.patch.set_facecolor('white')

COLOR_TRAIN = '#185FA5'   # c-blue 600
COLOR_VAL   = '#D85A30'   # c-coral 600
COLOR_AUC   = '#3B6D11'   # c-green 800
COLOR_AUPRC = '#BA7517'   # c-amber 600
COLOR_MCC   = '#534AB7'   # c-purple 600

LW = 1.6

# =========== subplot 1: Loss ===========
ax1 = axes[0]
ax1.plot(eps, train_loss, color=COLOR_TRAIN, lw=LW, label='Train loss')
ax1.plot(eps, val_loss,   color=COLOR_VAL,   lw=LW, linestyle='--', label='Val loss')

# 标注最优点
best_vl = val_loss[best_ep - 1]
ax1.scatter([best_ep], [best_vl], color=COLOR_VAL, s=45, zorder=5)
ax1.annotate(f'ep{best_ep}\n{best_vl:.4f}',
             xy=(best_ep, best_vl), xytext=(best_ep - 18, best_vl + 0.03),
             fontsize=8, color=COLOR_VAL,
             arrowprops=dict(arrowstyle='->', color=COLOR_VAL, lw=0.8))

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Binary cross-entropy loss', fontsize=11)
ax1.set_title('(a) Training & validation loss', fontsize=12, fontweight='normal', pad=8)
ax1.set_xlim(1, 70)
ax1.set_ylim(0.19, 0.58)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.legend(fontsize=9, framealpha=0.8, loc='upper right')
ax1.grid(True, color='#e8e8e8', linewidth=0.5)
ax1.spines[['top', 'right']].set_visible(False)

# 标注ep30和ep70的竖线
for ep_mark, lbl in [(30, 'ep30'), (70, 'ep70')]:
    ax1.axvline(ep_mark, color='#aaa', lw=0.8, linestyle=':', alpha=0.7)
    ax1.text(ep_mark + 0.5, 0.555, lbl, fontsize=7.5, color='#888', va='top')

# =========== subplot 2: AUROC + AUPRC ===========
ax2 = axes[1]
ax2.plot(eps, val_auc,   color=COLOR_AUC,   lw=LW, label='Val AUROC')
ax2.plot(eps, val_auprc, color=COLOR_AUPRC, lw=LW, linestyle='--', label='Val AUPRC')

# 标注最优
ax2.scatter([best_ep], [best_auc], color=COLOR_AUC, s=45, zorder=5)
ax2.annotate(f'ep{best_ep}: {best_auc:.4f}',
             xy=(best_ep, best_auc), xytext=(best_ep - 22, best_auc - 0.015),
             fontsize=8, color=COLOR_AUC,
             arrowprops=dict(arrowstyle='->', color=COLOR_AUC, lw=0.8))

# ep30 AUROC
auc30 = val_auc[29]
ax2.scatter([30], [auc30], color=COLOR_AUC, s=25, zorder=5, alpha=0.6)
ax2.annotate(f'ep30: {auc30:.4f}',
             xy=(30, auc30), xytext=(33, auc30 - 0.016),
             fontsize=7.5, color='#666',
             arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.7))

ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('(b) Validation AUROC & AUPRC', fontsize=12, fontweight='normal', pad=8)
ax2.set_xlim(1, 70)
ax2.set_ylim(0.82, 0.985)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax2.legend(fontsize=9, framealpha=0.8, loc='lower right')
ax2.grid(True, color='#e8e8e8', linewidth=0.5)
ax2.spines[['top', 'right']].set_visible(False)

for ep_mark, lbl in [(30, 'ep30'), (70, 'ep70')]:
    ax2.axvline(ep_mark, color='#aaa', lw=0.8, linestyle=':', alpha=0.7)
    ax2.text(ep_mark + 0.5, 0.825, lbl, fontsize=7.5, color='#888', va='bottom')

# ---- 保存 ----
plt.tight_layout(pad=1.5)
import os
os.makedirs('<repo-root>/results/figures', exist_ok=True)
out = '<repo-root>/results/figures/training_curves_ep70.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
