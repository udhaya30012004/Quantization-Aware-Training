"""Bar graph + short table of KL(BF16 || simulated FP8): how much FP8 changes each method's predictions (lower is better).

Reads results/kl_<3ep|1ep>_<bf16|qat>_s<seed>.json and results/kl_orig.json (from `modal run modal_app.py::kl_all`).
Writes plots_with_epochs/plots/fp8_kl_divergence.png (light) and fp8_kl_divergence_dark.png (dark, for slides),
plus plots_with_epochs/kl_results.md.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reference palette: categorical slots 1 (blue) and 2 (orange), each stepped for its surface.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7",
                  blue="#2a78d6", orange="#eb6834", file="fp8_kl_divergence.png"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835",
                 blue="#3987e5", orange="#d95926", file="fp8_kl_divergence_dark.png"),
}


def load(src, setting, arm):
    vals, agree = {}, {}
    for p in sorted(Path(src).glob(f"kl_{setting}_{arm}_s*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        s = int(p.stem.split("_s")[-1])
        vals[s], agree[s] = d["kl_mean"], d["top1_agreement"]
    return vals, agree


def draw(theme, d3, orig, out_path):
    T = THEMES[theme]
    W_IN, H_IN, DPI = 8.2, 5.6, 200
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=T["surface"])
    left, right, bottom, top = 0.07, 0.97, 0.20, 0.84
    ax = fig.add_axes([left, bottom, right - left, top - bottom], facecolor=T["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):                          # visible x and y axes
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(T["ink2"])
        ax.spines[s].set_linewidth(1.4)
    ax.tick_params(axis="y", colors=T["ink2"], labelsize=10, length=5, width=1.2)
    ax.tick_params(axis="x", colors=T["ink2"], length=5, width=1.2)
    ax.grid(axis="y", color=T["grid"], lw=0.9)
    ax.set_axisbelow(True)

    vals = {"bf16": d3["bf16"], "qat": d3["qat"]}
    means = {k: st.mean(v) for k, v in vals.items()}
    sds = {k: (st.stdev(v) if len(v) > 1 else 0.0) for k, v in vals.items()}
    ymax = max(max(v) for v in vals.values()) * 1.32
    x_lo, x_hi = -0.80, 1.02
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, ymax)
    ax.set_yticks([t for t in [0.005 * i for i in range(0, 12)] if t < ymax * 0.97])

    # rounded-top bars: a rounded box extended below the axis and clipped at y = 0; circular corners in pixel space
    w_px = W_IN * DPI * (right - left)
    h_px = H_IN * DPI * (top - bottom)
    aspect = ((ymax - 0) / h_px) / ((x_hi - x_lo) / w_px)
    r_x = 0.045
    pos = {"bf16": -0.25, "qat": 0.25}
    col = {"bf16": T["blue"], "qat": T["orange"]}
    bw = 0.38
    for k in ("bf16", "qat"):
        x, m = pos[k], means[k]
        ext = 2.2 * r_x * aspect
        ax.add_patch(FancyBboxPatch((x - bw / 2, -ext), bw, m + ext, boxstyle=f"round,pad=0,rounding_size={r_x}",
                                    mutation_aspect=aspect, fc=col[k], ec="none", zorder=2, clip_on=True))
        n = len(vals[k])
        jit = [(-0.5 + i / max(1, n - 1)) * bw * 0.62 for i in range(n)]
        ax.errorbar(x, m, yerr=sds[k], color=T["ink"], capsize=4, lw=1.4, zorder=5)
        ax.scatter([x + j for j in jit], vals[k], s=34, color=T["surface"], edgecolor=T["ink"], linewidth=1.1, zorder=6)
        ax.text(x, max(vals[k] + [m + sds[k]]) + ymax * 0.035, f"{m:.4f}", ha="center", va="bottom", fontsize=15,
                fontweight="bold", color=T["ink"])

    # "x% lower" marker
    mb, mq = means["bf16"], means["qat"]
    pct = (mb - mq) / mb * 100
    xa = 0.66
    ax.plot([pos["bf16"] + bw / 2 + 0.02, xa + 0.04], [mb, mb], color=T["ink2"], lw=1.1, ls=(0, (2, 3)), zorder=3)
    ax.annotate("", xy=(xa, mq), xytext=(xa, mb),
                arrowprops=dict(arrowstyle="-|>", color=T["ink"], lw=1.8, mutation_scale=16), zorder=7)
    ax.text(xa + 0.075, (mb + mq) / 2, f"{pct:.0f}% lower", ha="left", va="center", fontsize=13, fontweight="bold",
            color=T["surface"], zorder=8, bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.9", fc=T["ink"], ec="none"))

    ax.set_xticks([pos["bf16"], pos["qat"]])
    ax.set_xticklabels([f"BF16-trained\n{len(vals['bf16'])} seeds", f"QAT-trained\n{len(vals['qat'])} seeds"],
                       color=T["ink"], fontsize=12, fontweight="semibold")
    fig.text(left, 0.935, "How much FP8 conversion changes the model's predictions", ha="left", va="center",
             fontsize=16, fontweight="bold", color=T["ink"])
    fig.text(right, 0.035, "bars start at 0; whiskers = ± 1 std over seeds; dots = individual seeds; FP8 = simulated (exact FP8 weights)",
             ha="right", va="center", fontsize=8, color=T["muted"])
    fig.savefig(out_path, facecolor=T["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "plots_with_epochs"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    settings = [("3ep", "3 epochs"), ("1ep", "1 epoch")]
    data = {}
    for key, _ in settings:
        b, ba = load(a.src, key, "bf16")
        q, qa = load(a.src, key, "qat")
        seeds = sorted(set(b) & set(q))
        if not seeds:
            raise SystemExit(f"no KL results for setting {key} in {a.src}")
        data[key] = {"seeds": seeds, "bf16": [b[s] for s in seeds], "qat": [q[s] for s in seeds],
                     "bf16_agree": [ba[s] for s in seeds], "qat_agree": [qa[s] for s in seeds]}
    orig_p = Path(a.src) / "kl_orig.json"
    orig = json.loads(orig_p.read_text(encoding="utf-8"))["kl_mean"] if orig_p.exists() else None

    for theme in THEMES:                                  # the graph shows the 3-epoch result only (kl_results.md keeps both)
        draw(theme, data["3ep"], orig, out / "plots" / THEMES[theme]["file"])

    L = ["# KL divergence between each model's BF16 and (simulated) FP8 predictions\n",
         "Mean KL(P_BF16 || P_FP8) per answer token on the 1,319 test questions. Lower = FP8 changes the model less.\n",
         "| Setting | Method | Mean ± std | Per seed | Top-1 token agreement |", "|---|---|---:|---|---:|"]
    for key, name in settings:
        d = data[key]
        for field, label in (("bf16", "BF16-trained"), ("qat", "QAT-trained")):
            v, ag = d[field], d[field + "_agree"]
            L.append(f"| {name} | {label} | {st.mean(v):.5f} ± {st.stdev(v):.5f} | " + ", ".join(f"{x:.5f}" for x in v)
                     + f" | {st.mean(ag) * 100:.2f}% |")
        ratio = st.mean(d["qat"]) / st.mean(d["bf16"])
        lower = sum(q < b for q, b in zip(d["qat"], d["bf16"]))
        L.append(f"\n{name}: QAT KL is {ratio * 100:.0f}% of the BF16-trained KL; lower in {lower} of {len(d['seeds'])} seeds.\n")
    if orig is not None:
        L.append(f"Original, untrained model: KL = {orig:.5f}.\n")
    (out / "kl_results.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", ", ".join(THEMES[t]["file"] for t in THEMES), "and kl_results.md")


if __name__ == "__main__":
    main()
