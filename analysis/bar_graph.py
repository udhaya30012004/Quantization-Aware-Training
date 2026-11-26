"""Bar graph: FP8 test accuracy of BF16-trained vs QAT-trained models, for the 3-epoch and 1-epoch runs.

Bars = mean over seeds (start at 0), whiskers = ± 1 std over seeds, dots = individual seeds.
Output: plots_with_epochs/plots/fp8_accuracy_bars.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import compare  # noqa: E402
import plots    # noqa: E402  (shared colours / axis style)


def stats(out_dir, prefix, suffix, epochs):
    compare.PREFIX, compare.ARM_SUFFIX, compare.EPOCHS = prefix, suffix, epochs
    compare.set_outdir(out_dir)
    return compare.compute()


def main():
    settings = [("3 epochs", stats("more_seeds_added", "", "", 3)),
                ("1 epoch", stats("plots_with_epochs", "1ep_", "_1ep", 1))]
    fig, ax = plots._fig(7.6, 5.0)
    ax.grid(axis="y", color=plots.GRID, lw=0.8)
    ax.set_axisbelow(True)
    w, off = 0.34, 0.19
    for gi, (name, S) in enumerate(settings):
        for arm_i, (cell, color) in enumerate((("B", plots.BF16_C), ("D", plots.QAT_C))):
            x = gi + (-off if arm_i == 0 else off)
            m, sd, vals = S["acc"][cell]["mean"], S["acc"][cell]["std"], S["acc"][cell]["values"]
            ax.bar(x, m, width=w, color=color, edgecolor=plots.SURFACE, linewidth=2, zorder=2)
            ax.errorbar(x, m, yerr=sd, color=plots.INK, capsize=4, lw=1.3, zorder=4)
            n = len(vals)
            jit = [(-0.5 + k / max(1, n - 1)) * w * 0.55 for k in range(n)]
            ax.scatter([x + j for j in jit], vals, s=22, color=plots.SURFACE, edgecolor=plots.INK, linewidth=0.9, zorder=5)
            ax.text(x, m + sd + 1.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=10, color=plots.INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{n}\n({len(S['seeds'])} seeds per method)" for n, S in settings], color=plots.INK2, fontsize=10)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, 45)
    ax.set_yticks(range(0, 41, 10))
    ax.set_ylabel("Test accuracy after FP8 conversion (%)", fontsize=9.5)
    ax.set_title("FP8 accuracy: QAT-trained vs normally fine-tuned model", loc="left", color=plots.INK,
                 fontsize=12.5, fontweight="bold", pad=14)
    ax.legend([Patch(color=plots.BF16_C), Patch(color=plots.QAT_C)], ["BF16-trained, converted to FP8", "QAT-trained, converted to FP8"],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9.5, labelcolor=plots.INK2)
    ax.text(1.0, -0.30, "bars start at 0; whiskers = ± 1 std over seeds; dots = individual seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=plots.MUTED)
    fig.tight_layout()
    out = ROOT / "plots_with_epochs" / "plots" / "fp8_accuracy_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=plots.SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
