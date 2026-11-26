"""Accuracy graphs for the QAT-vs-BF16 comparison -> plots/*.png (reads results/stats.json via compare.compute)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import compare  # noqa: E402

# Reference palette (categorical slots 1 and 2: fixed order, never swapped) + light-surface chrome
BF16_C, QAT_C = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUT = ROOT / "plots"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def _fig(w=7.2, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def fig_accuracy(S):
    seeds, a = S["seeds"], S["acc"]
    fig, ax = _fig()
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for color, (lo_c, hi_c) in ((BF16_C, ("A", "B")), (QAT_C, ("C", "D"))):
        for i in range(len(seeds)):
            ax.plot([0, 1], [a[lo_c]["values"][i], a[hi_c]["values"][i]], color=color, lw=1.2, alpha=0.55,
                    marker="o", ms=5, mec=SURFACE, mew=1.2)
        m0, m1 = a[lo_c]["mean"], a[hi_c]["mean"]
        ax.plot([0, 1], [m0, m1], color=color, lw=2.6, marker="o", ms=9, mec=SURFACE, mew=2, zorder=5)
    name = {BF16_C: "BF16-trained", QAT_C: "QAT-trained"}
    def label_y(m_bf16, m_qat, gap=0.42):
        """Label heights for the two arms; pushed apart (QAT above) when the means nearly coincide."""
        yb, yq = m_bf16, m_qat
        if abs(yb - yq) < gap:
            mid = (yb + yq) / 2
            yb, yq = mid - gap / 2, mid + gap / 2
        return {BF16_C: yb, QAT_C: yq}
    left_y = label_y(a["A"]["mean"], a["C"]["mean"])
    right_y = label_y(a["B"]["mean"], a["D"]["mean"])
    for color, (lo_c, hi_c) in ((BF16_C, ("A", "B")), (QAT_C, ("C", "D"))):
        ax.text(-0.05, left_y[color], f"{name[color]}  {a[lo_c]['mean']:.1f}%", ha="right", va="center",
                color=INK, fontsize=9.5)
        ax.text(1.05, right_y[color], f"{a[hi_c]['mean']:.1f}%  {name[color]}", ha="left", va="center",
                color=INK, fontsize=9.5)
    g = S["primary"]["diff_D_minus_B"]["mean"]
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(32.5, 38)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inference in BF16", "Inference in FP8 (deployment)"], color=INK2, fontsize=10)
    ax.set_ylabel("Test accuracy (%)   [axis starts at 32.5]", fontsize=9)
    ci_lo, ci_hi = S["primary"]["ci95_seeds_and_questions"]
    if ci_lo > 0:
        title = f"After FP8 conversion, QAT-trained models score {g:+.1f} points higher"
    else:   # the interval reaches zero: do not state the gain as settled
        title = (f"After FP8 conversion, QAT-trained models average {g:+.1f} points higher\n"
                 f"(95% interval {ci_lo:+.1f} to {ci_hi:+.1f}: reaches zero, so not settled)")
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=14)
    handles = [plt.Line2D([0], [0], color=BF16_C, lw=2.6), plt.Line2D([0], [0], color=QAT_C, lw=2.6),
               plt.Line2D([0], [0], color=MUTED, lw=1.2, alpha=0.7)]
    ax.legend(handles, ["BF16-trained (mean)", "QAT-trained (mean)", f"each thin line = one seed (n={len(seeds)})"],
              loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_bf16_vs_fp8.png", facecolor=SURFACE)
    plt.close(fig)


def fig_gain(S):
    P, D = S["primary"], S["decomposition"]
    rows = [("Total gain after FP8\n(QAT→FP8 − BF16→FP8)", P["diff_D_minus_B"]["values"], P["diff_D_minus_B"]["mean"]),
            ("of which: QAT effect even\nwithout FP8 (C − A)", D["qat_effect_C_minus_A"]["values"], D["qat_effect_C_minus_A"]["mean"]),
            ("of which: smaller FP8 drop\n((A−B) − (C−D))", D["drop_reduction"]["values"], D["drop_reduction"]["mean"])]
    fig, ax = _fig(7.4, 4.2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=1.1, ls="--", zorder=1)
    lo, hi = P["ci95_seeds_and_questions"]
    ys = [2, 1, 0]
    ax.plot([lo, hi], [2, 2], color=GRID, lw=9, solid_capstyle="round", zorder=2)
    ax.text(lo, 2.32, f"95% interval [{lo:+.1f}, {hi:+.1f}]  (seeds + test questions)", fontsize=8.5, color=INK2, va="bottom", ha="left")
    for y, (_, vals, mean) in zip(ys, rows):
        ax.scatter(vals, [y] * len(vals), s=40, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.scatter([mean], [y], s=130, marker="D", color=INK, edgecolor=SURFACE, linewidth=1.6, zorder=5)
        ax.text(mean, y - 0.3, f"mean {mean:+.2f}", ha="center", va="top", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=9)
    ax.set_ylim(-0.9, 2.9)
    allv = [v for r in rows for v in r[1]]
    ax.set_xlim(min(-1.0, lo - 0.3, min(allv) - 0.35), max(3.6, hi + 0.8, max(allv) + 0.5))
    ax.set_xlabel("percentage points of test accuracy (dots = individual seeds)", fontsize=9)
    ax.set_title("QAT's accuracy gain, and where it comes from", loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "qat_gain_breakdown.png", facecolor=SURFACE)
    plt.close(fig)


def fig_val_loss(S):
    fig, ax = _fig(7.0, 4.2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for arm, color, name in (("bf16_full", BF16_C, "BF16-trained"), ("qat_full", QAT_C, "QAT-trained")):
        curves = S["val_curves"].get(arm) or []
        if not curves:
            continue
        for c in curves:
            ax.plot(range(len(c)), c, color=color, lw=1.1, alpha=0.5)
        m = np.mean(curves, axis=0)
        ax.plot(range(len(m)), m, color=color, lw=2.6, marker="o", ms=7, mec=SURFACE, mew=1.8, zorder=5)
        ax.text(len(m) - 1 + 0.06, m[-1], f"{name}  {m[-1]:.3f}", va="center", fontsize=9.5, color=INK)
    n_pts = max((len(c) for cs in S["val_curves"].values() for c in cs), default=4)
    ax.set_xticks(range(n_pts))
    ax.set_xticklabels(["start"] + [f"epoch {i}" for i in range(1, n_pts)], color=INK2, fontsize=9.5)
    ax.set_xlim(-0.15, n_pts + 0.1)
    ax.set_ylabel("Validation loss (lower is better)", fontsize=9)
    ax.set_title("Validation loss bottoms out after epoch 1, then rises (overfitting)" if n_pts > 2
                 else "Validation loss after the single epoch", loc="left",
                 color=INK, fontsize=11.5, fontweight="bold", pad=12)
    handles = [plt.Line2D([0], [0], color=BF16_C, lw=2.6), plt.Line2D([0], [0], color=QAT_C, lw=2.6),
               plt.Line2D([0], [0], color=MUTED, lw=1.1, alpha=0.7)]
    ax.legend(handles, ["BF16-trained (mean)", "QAT-trained (mean)", "each thin line = one seed"],
              loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "validation_loss_by_epoch.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="folder (relative to the repo); graphs go to <out>/plots")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--arm-suffix", default="")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    compare.PREFIX, compare.ARM_SUFFIX, compare.EPOCHS = a.prefix, a.arm_suffix, a.epochs
    if a.out:
        compare.set_outdir(a.out)
        OUT = compare.OUTDIR / "plots"
    OUT.mkdir(parents=True, exist_ok=True)
    S = compare.compute()
    fig_accuracy(S)
    fig_gain(S)
    fig_val_loss(S)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))
