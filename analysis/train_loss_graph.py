"""Training loss (not validation) of normal vs QAT fine-tuning, 3-epoch runs, 5 seeds each.
Per-seed loss is logged every 10 steps; each curve is smoothed with a 5-point moving average, then averaged over seeds."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import plots

def load(arm):
    out = []
    for s in range(5):
        rows = [json.loads(l) for l in open(ROOT / f"results/runs/{arm}/seed_{s}/train_log.jsonl") if '"train"' in l]
        out.append(([r["step"] for r in rows], [r["loss"] for r in rows]))
    return out

def smooth(y, k=5):
    y = np.asarray(y); pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")

fig, ax = plots._fig(8.2, 5.0)
ax.grid(axis="y", color=plots.GRID, lw=0.8); ax.set_axisbelow(True)
res = {}
for arm, color, name in (("bf16_full", plots.BF16_C, "Normal (BF16) training"), ("qat_full", plots.QAT_C, "QAT training")):
    runs = load(arm); steps = np.array(runs[0][0])
    sm = np.array([smooth(l) for _, l in runs])
    for y in sm: ax.plot(steps, y, color=color, lw=0.9, alpha=0.3)
    m = sm.mean(0); res[arm] = m
    ax.plot(steps, m, color=color, lw=2.8, label=name)
    ax.text(steps[-1] + 6, m[-1], f"{name}\n{m[-1]:.3f}", color=plots.INK, fontsize=9.5, va="center")
for e in (219, 438):
    ax.axvline(e, color=plots.MUTED, lw=0.9, ls=":")
for e, t in ((110, "epoch 1"), (329, "epoch 2"), (548, "epoch 3")):
    ax.text(e, ax.get_ylim()[1] * 0.995, t, ha="center", va="top", fontsize=9, color=plots.MUTED)
ax.set_xlim(0, 760); ax.set_xlabel("training step", fontsize=10); ax.set_ylabel("training loss", fontsize=10)
ax.set_title("Training loss: QAT vs normal fine-tuning (3 epochs)", loc="left", color=plots.INK, fontsize=12.5, fontweight="bold", pad=14)
fig.text(0.99, 0.01, "thick = mean of 5 seeds, thin = individual seeds; 5-point moving average", ha="right", fontsize=8, color=plots.MUTED)
fig.tight_layout()
out = ROOT / "plots_with_epochs/plots/training_loss_qat_vs_normal.png"
fig.savefig(out, facecolor=plots.SURFACE)
for arm, m in res.items(): print(arm, "last-epoch mean loss %.4f" % m[-15:].mean(), "| epoch-1 mean %.4f" % m[:22].mean())
print("wrote", out)
