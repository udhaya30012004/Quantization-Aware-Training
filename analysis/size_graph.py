"""Model-size graph: BF16 vs FP8 checkpoint size on disk (Qwen2.5-0.5B-Instruct).

Sizes are read from results/bench/*.json (field checkpoint_safetensors_gb, measured on the saved checkpoints).
Writes plots_with_epochs/plots/model_size_bf16_vs_fp8.png (light) and ..._dark.png (for slides).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "bench"
OUT = ROOT / "plots_with_epochs" / "plots"

# Precision is a different thing from "training method", so it does not reuse the blue/orange of the other graphs:
# BF16 = neutral grey, FP8 = categorical slot 3 (aqua), each stepped for its surface.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  bf16="#a9a7a0", fp8="#1baf7a", file="model_size_bf16_vs_fp8.png"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 bf16="#6f6d68", fp8="#199e70", file="model_size_bf16_vs_fp8_dark.png"),
}


def size_gb(label):
    return json.loads((BENCH / f"{label}.json").read_text(encoding="utf-8"))["checkpoint_safetensors_gb"]


def draw(theme, bf16_gb, fp8_gb):
    T = THEMES[theme]
    W_IN, H_IN, DPI = 8.2, 5.6, 200
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=T["surface"])
    left, right, bottom, top = 0.07, 0.97, 0.20, 0.84
    ax = fig.add_axes([left, bottom, right - left, top - bottom], facecolor=T["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["ink2"])
        ax.spines[s].set_linewidth(1.4)
    ax.tick_params(axis="y", colors=T["ink2"], labelsize=10, length=5, width=1.2)
    ax.tick_params(axis="x", colors=T["ink2"], length=5, width=1.2)
    ax.grid(axis="y", color=T["grid"], lw=0.9)
    ax.set_axisbelow(True)

    ymax = bf16_gb * 1.32
    x_lo, x_hi = -0.80, 1.02
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    w_px = W_IN * DPI * (right - left)
    h_px = H_IN * DPI * (top - bottom)
    aspect = (ymax / h_px) / ((x_hi - x_lo) / w_px)
    r_x, bw = 0.045, 0.38
    bars = [(-0.25, bf16_gb, T["bf16"]), (0.25, fp8_gb, T["fp8"])]
    for x, v, c in bars:
        ext = 2.2 * r_x * aspect
        ax.add_patch(FancyBboxPatch((x - bw / 2, -ext), bw, v + ext, boxstyle=f"round,pad=0,rounding_size={r_x}",
                                    mutation_aspect=aspect, fc=c, ec="none", zorder=2, clip_on=True))
        ax.text(x, v + ymax * 0.03, f"{v:.2f} GB", ha="center", va="bottom", fontsize=16, fontweight="bold", color=T["ink"])

    pct = (bf16_gb - fp8_gb) / bf16_gb * 100
    xa = 0.56
    ax.plot([-0.25 + bw / 2 + 0.02, xa + 0.04], [bf16_gb, bf16_gb], color=T["ink2"], lw=1.1, ls=(0, (2, 3)), zorder=3)
    ax.annotate("", xy=(xa, fp8_gb), xytext=(xa, bf16_gb),
                arrowprops=dict(arrowstyle="-|>", color=T["ink"], lw=1.8, mutation_scale=16), zorder=7)
    ax.text(xa + 0.075, (bf16_gb + fp8_gb) / 2, f"{pct:.0f}% smaller", ha="left", va="center", fontsize=13,
            fontweight="bold", color=T["surface"], zorder=8,
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.9", fc=T["ink"], ec="none"))

    ax.set_xticks([-0.25, 0.25])
    ax.set_xticklabels(["BF16", "FP8"], color=T["ink"], fontsize=13, fontweight="bold")
    fig.text(left, 0.935, "Model size: BF16 vs FP8", ha="left", va="center", fontsize=16, fontweight="bold", color=T["ink"])
    fig.text(left, 0.885, "Qwen2.5-0.5B-Instruct, 494M parameters", ha="left", va="center", fontsize=11, color=T["ink2"])
    fig.text(right, 0.035, "size of the saved model weights; the embedding table (136M parameters) stays in BF16, so the saving is less than half",
             ha="right", va="center", fontsize=8, color=T["muted"])
    fig.savefig(OUT / T["file"], facecolor=T["surface"])
    plt.close(fig)


def main():
    bf16, fp8 = size_gb("orig_bf16"), size_gb("orig_fp8")
    # the same two sizes hold for the fine-tuned models: assert it instead of assuming it
    for lab, ref in (("bf16_bf16_s0", bf16), ("qat_bf16_s0", bf16), ("bf16_fp8_s0", fp8), ("qat_fp8_s0", fp8)):
        assert abs(size_gb(lab) - ref) < 1e-6, lab
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        draw(theme, bf16, fp8)
    print(f"BF16 {bf16} GB, FP8 {fp8} GB, {(bf16 - fp8) / bf16 * 100:.1f}% smaller; wrote {', '.join(t['file'] for t in THEMES.values())}")


if __name__ == "__main__":
    main()
