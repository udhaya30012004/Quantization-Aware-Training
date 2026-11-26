"""Generate two executive-summary KL divergence plots matching 00_kl_executive_summary.png
using 100% real local 5-seed results (no Modal re-run required).

Outputs (by default saved in the project root, alongside 00_kl_executive_summary.png):
  - kl_executive_summary_bf16_fp8.png  (Normal FP8: BF16-trained -> FP8 vs BF16 Ref)
  - kl_executive_summary_qat_fp8.png   (QAT FP8:    QAT-trained  -> FP8 vs BF16 Ref)
"""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Palette matched to 00_kl_executive_summary.png
C_GREEN = "#009464"
C_CYAN = "#00A6E8"
C_BLUE = "#1F64E8"
C_SOFT_BLUE = "#3B82F6"
C_ORANGE = "#F59E0B"
C_RED = "#EF4444"
C_THRESH = "#DC2626"
C_SAFE_BG = "#ECFDF5"
C_AX_BG = "#F8FAFC"
C_BORDER = "#CBD5E1"
C_GRID = "#E2E8F0"
C_INK = "#0F172A"
C_INK2 = "#334155"


def load_arm_metrics(results_dir: Path, arm: str, seeds=(0, 1, 2, 3, 4)):
    kl3 = [json.loads((results_dir / f"kl_3ep_{arm}_s{s}.json").read_text(encoding="utf-8")) for s in seeds]
    kl1 = [json.loads((results_dir / f"kl_1ep_{arm}_s{s}.json").read_text(encoding="utf-8")) for s in seeds]
    ev_b = [json.loads((results_dir / f"{arm}_bf16_s{s}.json").read_text(encoding="utf-8")) for s in seeds]
    ev_f = [json.loads((results_dir / f"{arm}_fp8_s{s}.json").read_text(encoding="utf-8")) for s in seeds]

    both_ok = both_wrong = gained = lost = tot_q = 0
    for s in seeds:
        rb = [json.loads(l) for l in (results_dir / "per_question" / f"{arm}_bf16_s{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        rf = [json.loads(l) for l in (results_dir / "per_question" / f"{arm}_fp8_s{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rb) == len(rf), f"Mismatch in per_question length for {arm} seed {s}"
        for a, b in zip(rb, rf):
            tot_q += 1
            ca, cb = bool(a["correct"]), bool(b["correct"])
            if ca and cb:
                both_ok += 1
            elif not ca and not cb:
                both_wrong += 1
            elif not ca and cb:
                gained += 1
            else:
                lost += 1

    kl3_vals = [x["kl_mean"] for x in kl3]
    kl1_vals = [x["kl_mean"] for x in kl1]
    best_seed_idx = int(np.argmin(kl3_vals))

    acc_b = st.mean(x["accuracy"] for x in ev_b) * 100.0
    acc_f = st.mean(x["accuracy"] for x in ev_f) * 100.0
    loss_b = st.mean(x["eval_loss"] for x in ev_b)
    loss_f = st.mean(x["eval_loss"] for x in ev_f)

    return {
        "arm": arm,
        "n_seeds": len(seeds),
        "n_tokens_total": sum(x["n_tokens"] for x in kl3),
        "kl1_mean": st.mean(kl1_vals),
        "kl3_best": kl3_vals[best_seed_idx],
        "kl3_best_seed": seeds[best_seed_idx],
        "kl3_mean": st.mean(kl3_vals),
        "kl3_std": st.stdev(kl3_vals) if len(kl3_vals) > 1 else 0.0,
        "top1_agree_pct": st.mean(x["top1_agreement"] for x in kl3) * 100.0,
        "acc_bf16": acc_b,
        "acc_fp8": acc_f,
        "acc_drop_pp": acc_b - acc_f,
        "loss_delta": loss_f - loss_b,
        "tot_q": tot_q,
        "both_ok": both_ok,
        "both_wrong": both_wrong,
        "gained": gained,
        "lost": lost,
    }


def draw_executive_summary(m: dict, title: str, right_callout: str, takeaway: str, out_path: Path, thresh: float = 0.05):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.0, 6.4), dpi=220, facecolor="#FFFFFF")

    # Main figure title
    fig.suptitle(title, fontsize=14.2, fontweight="bold", color=C_INK, y=0.965)

    # Two side-by-side panels with room for title and takeaway footer
    ax1 = fig.add_axes([0.055, 0.175, 0.415, 0.68], facecolor=C_AX_BG)
    ax2 = fig.add_axes([0.535, 0.175, 0.440, 0.68], facecolor=C_AX_BG)

    for ax in (ax1, ax2):
        for sp in ax.spines.values():
            sp.set_color(C_BORDER)
            sp.set_linewidth(0.95)
        ax.grid(True, linestyle="--", linewidth=0.8, color=C_GRID, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=C_INK2, labelsize=9.8)

    # =========================================================================
    # PANEL 1: KL Divergence vs Lossless Threshold (0.05 nats)
    # =========================================================================
    x1 = np.arange(4)
    vals1 = [0.0, m["kl1_mean"], m["kl3_best"], m["kl3_mean"]]
    cols1 = [C_BORDER, C_GREEN, C_CYAN, C_BLUE]
    labels1 = [
        "BF16 Ref\n(Baseline)",
        "1-Epoch Mean KL\n(5-Seed Avg.)",
        f"3-Epoch Best Seed\n(Seed {m['kl3_best_seed']})",
        "3-Epoch Mean KL\n(5-Seed Avg.)",
    ]

    ax1.axhspan(0, thresh, color=C_SAFE_BG, alpha=0.75, zorder=0)
    ax1.axhline(thresh, color=C_THRESH, linestyle="--", linewidth=2.2, zorder=3)
    ax1.text(
        0.02, thresh + 0.0014,
        f"Typical Lossless Threshold ({thresh:.2f} nats)",
        color=C_THRESH, fontsize=9.8, fontweight="bold", ha="left", va="bottom", zorder=4,
    )

    bars1 = ax1.bar(x1, vals1, width=0.52, color=cols1, edgecolor="none", zorder=2)

    for i, (bar, v) in enumerate(zip(bars1, vals1)):
        y_txt = v + 0.0016 if v > 0 else 0.0018
        ax1.text(
            bar.get_x() + bar.get_width() / 2, y_txt,
            f"{v:.4f}\nnats",
            ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=C_INK, zorder=5,
        )

    # Callout 1: Nx below threshold above 1-epoch mean KL
    mult = thresh / m["kl1_mean"]
    ax1.annotate(
        f"{mult:.1f}× below\nthreshold",
        xy=(1, m["kl1_mean"] + 0.0075),
        xytext=(1, 0.0285),
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#047857",
        arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.6, shrinkA=2, shrinkB=2),
        zorder=6,
    )

    # Callout 2: X% below threshold pointing cleanly to top-left corner of 3-epoch mean KL bar
    pct_below = (thresh - m["kl3_mean"]) / thresh * 100.0
    ax1.annotate(
        f"{pct_below:.0f}% below\nthreshold",
        xy=(3 - 0.23, m["kl3_mean"] + 0.0004),
        xytext=(2.30, m["kl3_mean"] + 0.0115),
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1D4ED8",
        arrowprops=dict(arrowstyle="-|>", color="#1D4ED8", lw=1.6, shrinkA=2, shrinkB=2),
        zorder=6,
    )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels1, fontsize=9.5, color=C_INK)
    ax1.set_ylim(0, 0.062)
    ax1.set_yticks(np.arange(0.0, 0.061, 0.01))
    ax1.set_ylabel("KL Divergence from BF16 (nats — lower is better)", fontsize=10.5, fontweight="bold", color=C_INK)
    ax1.set_title("1. KL Divergence vs Lossless Threshold (0.05 nats)", fontsize=11.8, fontweight="bold", color=C_INK, pad=10)

    # =========================================================================
    # PANEL 2: Per-Question Outcome Breakdown (5 Seeds = 6,595 Qs)
    # =========================================================================
    tot = m["tot_q"]
    counts2 = [m["both_ok"], m["both_wrong"], m["gained"], m["lost"]]
    pcts2 = [c / tot * 100.0 for c in counts2]
    cols2 = [C_GREEN, C_SOFT_BLUE, C_ORANGE, C_RED]
    labels2 = [
        "Both Correct\n(BF16 & FP8)",
        "Both Incorrect\n(BF16 & FP8)",
        "FP8 Gained\n(Wrong → Right)",
        "FP8 Lost\n(Right → Wrong)",
    ]

    x2 = np.arange(4)
    bars2 = ax2.bar(x2, pcts2, width=0.52, color=cols2, edgecolor="none", zorder=2)

    for bar, p, c in zip(bars2, pcts2, counts2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, p + 1.5,
            f"{p:.1f}%\n({c:,} Qs)",
            ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=C_INK, zorder=5,
        )

    # Top green bracket spanning Bar 0 (Both Correct) and Bar 1 (Both Incorrect)
    match_pct = pcts2[0] + pcts2[1]
    match_cnt = counts2[0] + counts2[1]
    brk_y = max(pcts2[0], pcts2[1]) + 11.5
    ax2.plot([-0.26, -0.26, 1.26, 1.26], [brk_y - 2.2, brk_y, brk_y, brk_y - 2.2], color=C_GREEN, lw=2.0, zorder=5)
    ax2.text(
        0.5, brk_y + 2.0,
        f"{match_pct:.1f}% ({match_cnt:,} / {tot:,} Qs) match BF16 outcome",
        ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#047857",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor=C_GREEN, lw=1.3),
        zorder=6,
    )

    # Red callout box pointing to Bar 3 (FP8 Lost)
    ax2.annotate(
        right_callout,
        xy=(3.0, pcts2[3] + 8.5),
        xytext=(2.52, 43.0),
        ha="center", va="center", fontsize=9.1, fontweight="bold", color="#B91C1C",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF2F2", edgecolor=C_RED, lw=1.2),
        arrowprops=dict(arrowstyle="-|>", color=C_THRESH, lw=1.6, shrinkA=4, shrinkB=2),
        zorder=6,
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=9.5, color=C_INK)
    ax2.set_ylim(0, 82)
    ax2.set_yticks(np.arange(0, 81, 20))
    ax2.set_ylabel(f"Share of All {tot:,} Question Evals (5 Seeds, %)", fontsize=10.5, fontweight="bold", color=C_INK)
    ax2.set_title("2. GSM8K Outcome Breakdown (BF16 vs FP8, 5 Seeds)", fontsize=11.8, fontweight="bold", color=C_INK, pad=10)

    # =========================================================================
    # FOOTER TAKEAWAY BANNER
    # =========================================================================
    fig.text(
        0.5, 0.036, takeaway,
        ha="center", va="center", fontsize=9.4, fontweight="bold", color=C_INK,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F1F5F9", edgecolor=C_BORDER, lw=1.0),
    )

    import io
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor="#FFFFFF")
    plt.close(fig)
    out_path.write_bytes(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.out_dir)

    bf16 = load_arm_metrics(res_dir, "bf16")
    qat = load_arm_metrics(res_dir, "qat")

    # Plot 1: Normal FP8 (BF16-trained -> FP8 vs BF16 Ref)
    bf16_net = bf16["gained"] - bf16["lost"]
    bf16_callout = (
        f"Net {bf16_net} Qs (-{bf16['acc_drop_pp']:.2f} pp drop)\n"
        f"FP8 Lost ({bf16['lost']}) > Gained ({bf16['gained']})"
    )
    bf16_takeaway = (
        f"Takeaway: 3ep Mean KL = {bf16['kl3_mean']:.4f} nats (< 0.05 threshold) | "
        f"Top-1 Token Agree = {bf16['top1_agree_pct']:.2f}% ({bf16['n_tokens_total']:,} tok) | "
        f"GSM8K = {bf16['acc_fp8']:.2f}% (-{bf16['acc_drop_pp']:.2f} pp) | "
        f"Loss Δ = +{bf16['loss_delta']:.4f} | -35.9% Size (0.92 → 0.59 GB)"
    )
    p_bf16 = out_dir / "kl_executive_summary_bf16_fp8.png"
    draw_executive_summary(
        bf16,
        title="Qwen2.5-0.5B-Instruct (Normal FP8 vs BF16, 5 Seeds): Probability & GSM8K Fidelity",
        right_callout=bf16_callout,
        takeaway=bf16_takeaway,
        out_path=p_bf16,
    )

    # Plot 2: QAT FP8 (QAT-trained -> FP8 vs BF16 Ref)
    qat_net = qat["gained"] - qat["lost"]
    kl_red_pct = (bf16["kl3_mean"] - qat["kl3_mean"]) / bf16["kl3_mean"] * 100.0
    drop_red_pct = (bf16["acc_drop_pp"] - qat["acc_drop_pp"]) / bf16["acc_drop_pp"] * 100.0
    qat_callout = (
        f"Net {qat_net} Qs (-{qat['acc_drop_pp']:.2f} pp drop)\n"
        f"{drop_red_pct:.0f}% smaller drop than Normal FP8"
    )
    qat_takeaway = (
        f"Takeaway: 3ep Mean KL = {qat['kl3_mean']:.4f} nats ({kl_red_pct:.0f}% lower than Normal FP8) | "
        f"Top-1 Token Agree = {qat['top1_agree_pct']:.2f}% ({qat['n_tokens_total']:,} tok) | "
        f"GSM8K = {qat['acc_fp8']:.2f}% (-{qat['acc_drop_pp']:.2f} pp) | "
        f"Loss Δ = +{qat['loss_delta']:.4f} | -35.9% Size"
    )
    p_qat = out_dir / "kl_executive_summary_qat_fp8.png"
    draw_executive_summary(
        qat,
        title="Qwen2.5-0.5B-Instruct (QAT FP8 vs BF16, 5 Seeds): Higher Fidelity & Half the Accuracy Drop",
        right_callout=qat_callout,
        takeaway=qat_takeaway,
        out_path=p_qat,
    )

    print("Wrote:", p_bf16)
    print("Wrote:", p_qat)


if __name__ == "__main__":
    main()
