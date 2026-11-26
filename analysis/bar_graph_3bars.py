"""Modern 3-bar comparison graph: Unquantized BF16 vs BF16->FP8 (PTQ) vs QAT->FP8,
for both the 3-epoch and 1-epoch runs (5 seeds per method).

Design: crisp white editorial surface, rounded-top bars, punchy high-contrast palette
(Teal / Electric Blue / Vibrant Orange), top-left legend, and delta callout pills.
Output: plots_with_epochs/plots/accuracy_3bar_comparison.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import compare  # noqa: E402

# Modern high-contrast palette (editorial / Stripe / Vercel inspired)
SURFACE = "#FFFFFF"
CARD_BG = "#F8FAFC"
INK = "#0F172A"       # deep slate-black
INK2 = "#334155"      # secondary slate
MUTED = "#64748B"     # caption / axis ticks
GRID = "#E2E8F0"      # crisp subtle horizontal grid
SPINE = "#CBD5E1"

C_BF16_UNQ = "#0D9488"   # vibrant teal (unquantized BF16 reference)
C_BF16_FP8 = "#2563EB"   # electric royal blue (BF16 -> FP8 PTQ)
C_QAT_FP8 = "#F97316"    # punchy coral-orange (QAT -> FP8 hero)


def stats(out_dir, prefix, suffix, epochs):
    compare.PREFIX, compare.ARM_SUFFIX, compare.EPOCHS = prefix, suffix, epochs
    compare.set_outdir(out_dir)
    return compare.compute()


def draw():
    settings = [
        ("3 Epochs", stats("more_seeds_added", "", "", 3)),
        ("1 Epoch", stats("plots_with_epochs", "1ep_", "_1ep", 1)),
    ]
    arms = [
        ("A", C_BF16_UNQ, "BF16 (Unquantized)"),
        ("B", C_BF16_FP8, "BF16 → FP8 (PTQ)"),
        ("D", C_QAT_FP8, "QAT → FP8"),
    ]

    W_IN, H_IN, DPI = 9.4, 5.8, 220
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
    left, right, bottom, top = 0.085, 0.965, 0.16, 0.74
    ax = fig.add_axes([left, bottom, right - left, top - bottom], facecolor=SURFACE)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(2.0)

    ax.tick_params(axis="y", colors=INK, labelsize=10, length=5, width=1.6)
    ax.tick_params(axis="x", colors=INK, length=0, pad=10)
    ax.grid(axis="y", color=GRID, lw=0.9, ls="-")
    ax.set_axisbelow(True)

    x_lo, x_hi = -0.62, 1.68
    y_lo, y_hi = 0.0, 45.5
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%"], fontweight="bold", color=INK)
    ax.set_ylabel("GSM8K Test Accuracy", fontsize=10.5, fontweight="bold", color=INK, labelpad=10)

    # Pixel-accurate aspect ratio for perfectly circular rounded top bar corners
    w_px = W_IN * DPI * (right - left)
    h_px = H_IN * DPI * (top - bottom)
    aspect = ((y_hi - y_lo) / h_px) / ((x_hi - x_lo) / w_px)
    r_x = 0.024
    ext = 2.4 * r_x * aspect

    w = 0.235
    gap = 0.032
    offsets = (-(w + gap), 0.0, (w + gap))
    group_centers = [0.0, 1.06]

    # Subtle modern card backdrop behind each epoch group
    for cx in group_centers:
        card_w = 3 * w + 2 * gap + 0.14
        card = FancyBboxPatch(
            (cx - card_w / 2, 0.0), card_w, 44.2,
            boxstyle="round,pad=0,rounding_size=0.045",
            mutation_aspect=aspect,
            fc=CARD_BG, ec="#E2E8F0", lw=0.9, zorder=0, clip_on=True
        )
        ax.add_patch(card)

    for cx, (name, S) in zip(group_centers, settings):
        bar_tops = {}
        for (cell, color, _), dx in zip(arms, offsets):
            x = cx + dx
            m = S["acc"][cell]["mean"]
            sd = S["acc"][cell]["std"]
            vals = S["acc"][cell]["values"]

            # Rounded-top bar clipped cleanly at y = 0
            bar = FancyBboxPatch(
                (x - w / 2, -ext), w, m + ext,
                boxstyle=f"round,pad=0,rounding_size={r_x}",
                mutation_aspect=aspect,
                fc=color, ec="none", zorder=2, clip_on=True
            )
            ax.add_patch(bar)

            # Whisker (± 1 std) + crisp individual seed dots
            ax.errorbar(
                x, m, yerr=sd, color=INK, capsize=3.5, capthick=1.2, lw=1.3, zorder=4
            )
            n = len(vals)
            jit = [(-0.5 + k / max(1, n - 1)) * w * 0.52 for k in range(n)]
            ax.scatter(
                [x + j for j in jit], vals, s=24,
                color=SURFACE, edgecolor=INK, linewidth=1.0, zorder=5
            )

            top_y = max(max(vals), m + sd)
            bar_tops[cell] = top_y
            ax.text(
                x, top_y + 0.9, f"{m:.1f}%",
                ha="center", va="bottom",
                fontsize=10.5, fontweight="bold",
                color=color if cell == "D" else INK, zorder=6
            )

        # Sleek delta callout pill above the QAT -> FP8 hero bar
        gain = S["acc"]["D"]["mean"] - S["acc"]["B"]["mean"]
        pill_x = cx + offsets[2]
        pill_y = bar_tops["D"] + 4.35
        ax.text(
            pill_x, pill_y, f"{gain:+.1f} pp vs PTQ",
            ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="#C2410C",
            bbox=dict(
                boxstyle="round,pad=0.34,rounding_size=0.6",
                fc="#FFF7ED", ec="#FDBA74", lw=1.0
            ),
            zorder=7
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [f"{n}  ·  {len(S['seeds'])} seeds/arm" for n, S in settings],
        color=INK, fontsize=10.5, fontweight="bold"
    )

    # Editorial header: bold title + takeaway subtitle + top-left legend
    fig.text(
        left, 0.935,
        "Test Accuracy: Unquantized BF16 vs BF16→FP8 vs QAT→FP8",
        ha="left", va="center", fontsize=15, fontweight="bold", color=INK
    )
    fig.text(
        left, 0.878,
        "QAT recovers the FP8 quantization drop and matches unquantized BF16 (Qwen2.5-0.5B-Instruct, GSM8K)",
        ha="left", va="center", fontsize=9.2, fontweight="bold", color=INK2
    )

    handles = [Patch(facecolor=c, edgecolor="none", label=lbl) for _, c, lbl in arms]
    leg = fig.legend(
        handles=handles,
        loc="upper left", bbox_to_anchor=(left - 0.008, 0.838),
        ncol=3, frameon=False, fontsize=9.8, labelcolor=INK,
        handlelength=1.4, handleheight=0.9, columnspacing=1.8
    )
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.text(
        right, 0.032,
        "Bars start at 0%  •  Whiskers = ±1 std over 5 seeds  •  White dots = individual seeds  •  Evaluated on H100 (vLLM FP8_DYNAMIC)",
        ha="right", va="center", fontsize=8.4, fontweight="bold", color=INK2
    )

    out = ROOT / "plots_with_epochs" / "plots" / "accuracy_3bar_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    draw()
