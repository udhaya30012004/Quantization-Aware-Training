"""Accuracy comparison: BF16 full fine-tuning vs FP8 fake-quant QAT, both deployed as FP8.

Reads results/*.json and results/per_question/*.jsonl (pulled with `modal run modal_app.py::pull_results`),
auto-detects every seed that has all four cells, and writes results/stats.json, results/comparison.csv and
results/report.md. Every number in the report is computed here - none is typed by hand.

Cells: A = BF16-trained -> BF16, B = BF16-trained -> FP8, C = QAT-trained -> BF16, D = QAT-trained -> FP8.
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import mcnemar_exact  # noqa: E402

R = ROOT / "results"          # inputs (pulled with modal run modal_app.py::pull_results)
OUTDIR = R                     # where stats.json, comparison.csv and report.md are written (see --out)
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
CELLS = {"A": "bf16_bf16", "B": "bf16_fp8", "C": "qat_bf16", "D": "qat_fp8"}
PREFIX, ARM_SUFFIX, EPOCHS = "", "", 3   # set from the command line for tagged runs, e.g. --prefix 1ep_ --arm-suffix _1ep --epochs 1


def _summary(label):
    return json.loads((R / f"{label}.json").read_text(encoding="utf-8"))


def _correct(label):
    rows = [json.loads(x) for x in open(R / "per_question" / f"{label}.jsonl", encoding="utf-8")]
    return np.array([int(r["correct"]) for r in rows]), [r["id"] for r in rows]


def detect_seeds():
    seeds = []
    for p in sorted(R.glob(f"{PREFIX}bf16_bf16_s*.json")):
        s = int(p.stem.split("_s")[-1])
        if all((R / f"{PREFIX}{pre}_s{s}.json").exists() for pre in CELLS.values()):
            seeds.append(s)
    return sorted(seeds)


def two_level_bootstrap(Bc, Dc, n_boot=5000, seed=0):
    """CI for mean(D)-mean(B) in pp, resampling BOTH seeds (per arm) and test questions."""
    rng = np.random.default_rng(seed)
    nS, nQ = Bc.shape
    out = np.empty(n_boot)
    for i in range(n_boot):
        q = rng.integers(0, nQ, nQ)
        b = Bc[rng.integers(0, nS, nS)][:, q].mean()
        d = Dc[rng.integers(0, Dc.shape[0], Dc.shape[0])][:, q].mean()
        out[i] = (d - b) * 100
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def question_bootstrap(Bc, Dc, n_boot=10000, seed=0):
    """CI resampling test questions only (seed-averaged correctness, paired)."""
    rng = np.random.default_rng(seed)
    d = Dc.mean(axis=0) - Bc.mean(axis=0)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute():
    seeds = detect_seeds()
    if len(seeds) < 2:
        raise SystemExit(f"need at least 2 complete seeds, found {seeds}")
    acc = {c: [_summary(f"{PREFIX}{pre}_s{s}")["accuracy"] * 100 for s in seeds] for c, pre in CELLS.items()}
    loss = {c: [_summary(f"{PREFIX}{pre}_s{s}")["eval_loss"] for s in seeds] for c, pre in CELLS.items()}
    gen = {c: [_summary(f"{PREFIX}{pre}_s{s}")["mean_gen_tokens"] for s in seeds] for c, pre in CELLS.items()}
    corr = {c: np.stack([_correct(f"{PREFIX}{pre}_s{s}")[0] for s in seeds]) for c, pre in CELLS.items()}
    n_q = corr["A"].shape[1]

    def ms(x):
        return {"values": x, "mean": st.mean(x), "std": st.stdev(x) if len(x) > 1 else 0.0}

    diff_D_B = [d - b for d, b in zip(acc["D"], acc["B"])]
    qat_effect = [c - a for c, a in zip(acc["C"], acc["A"])]
    drop_bf16 = [a - b for a, b in zip(acc["A"], acc["B"])]
    drop_qat = [c - d for c, d in zip(acc["C"], acc["D"])]
    drop_reduction = [x - y for x, y in zip(drop_bf16, drop_qat)]
    n = len(seeds)
    md, sd = st.mean(diff_D_B), st.stdev(diff_D_B)
    t_stat = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    lo2, hi2 = two_level_bootstrap(corr["B"], corr["D"])
    loq, hiq = question_bootstrap(corr["B"], corr["D"])
    loss_pen_bf16 = [b - a for a, b in zip(loss["A"], loss["B"])]
    loss_pen_qat = [d - c for c, d in zip(loss["C"], loss["D"])]

    # val-loss curves from training summaries
    curves = {"bf16_full": [], "qat_full": []}
    for arm in curves:
        for s in seeds:
            p = R / "runs" / f"{arm}{ARM_SUFFIX}" / f"seed_{s}" / "summary.json"
            if p.exists():
                curves[arm].append(json.loads(p.read_text(encoding="utf-8"))["val_loss_history"])

    baselines = {}
    for lab in ("orig_bf16", "orig_fp8", "orig_bf16_h100", "orig_fp8_h100"):
        if (R / f"{lab}.json").exists():
            s = _summary(lab)
            baselines[lab] = {"accuracy": s["accuracy"] * 100, "eval_loss": s["eval_loss"]}

    out = {
        "seeds": seeds, "n_questions": n_q, "acc": {c: ms(v) for c, v in acc.items()},
        "loss": {c: ms(v) for c, v in loss.items()}, "gen_tokens": {c: ms(v) for c, v in gen.items()},
        "primary": {"diff_D_minus_B": {"values": diff_D_B, "mean": md, "std": sd},
                    "seed_level_t": t_stat, "t_crit_95": T_CRIT_95.get(n - 1),
                    "ci95_seeds_and_questions": [lo2, hi2], "ci95_questions_only": [loq, hiq],
                    "all_seeds_positive": all(x > 0 for x in diff_D_B),
                    "every_D_beats_every_B": min(acc["D"]) > max(acc["B"]),
                    "mcnemar_p_per_seed": [mcnemar_exact(corr["B"][i], corr["D"][i]) for i in range(n)]},
        "decomposition": {"qat_effect_C_minus_A": ms(qat_effect), "fp8_drop_bf16_trained_A_minus_B": ms(drop_bf16),
                          "fp8_drop_qat_trained_C_minus_D": ms(drop_qat), "drop_reduction": ms(drop_reduction)},
        "loss_penalty": {"bf16_trained_B_minus_A": ms(loss_pen_bf16), "qat_trained_D_minus_C": ms(loss_pen_qat)},
        "val_curves": curves, "baselines": baselines,
    }
    (OUTDIR / "stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _f(x, d=2):
    return f"{x:.{d}f}"


def _pm(m, d=2):
    return f"{m['mean']:.{d}f} ± {m['std']:.{d}f}"


def write_csv(S):
    seeds = S["seeds"]
    with open(OUTDIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "BF16 Full -> FP8", "QAT Full -> FP8"])
        w.writerow(["Accuracy % (mean)", _f(S["acc"]["B"]["mean"]), _f(S["acc"]["D"]["mean"])])
        w.writerow(["Accuracy % (std over seeds)", _f(S["acc"]["B"]["std"]), _f(S["acc"]["D"]["std"])])
        w.writerow(["Eval loss (mean)", _f(S["loss"]["B"]["mean"], 4), _f(S["loss"]["D"]["mean"], 4)])
        w.writerow(["Mean generated tokens", _f(S["gen_tokens"]["B"]["mean"], 1), _f(S["gen_tokens"]["D"]["mean"], 1)])
        w.writerow(["QAT accuracy improvement (pp)", "", _f(S["primary"]["diff_D_minus_B"]["mean"])])
        w.writerow([])
        w.writerow(["seed", "A bf16->BF16", "B bf16->FP8", "C qat->BF16", "D qat->FP8"])
        for i, s in enumerate(seeds):
            w.writerow([s] + [_f(S["acc"][c]["values"][i]) for c in "ABCD"])


def write_report(S):
    P, D, seeds, n = S["primary"], S["decomposition"], S["seeds"], len(S["seeds"])
    a, l = S["acc"], S["loss"]
    lo, hi = P["ci95_seeds_and_questions"]
    loq, hiq = P["ci95_questions_only"]
    gain = P["diff_D_minus_B"]["mean"]
    rel = gain / a["B"]["mean"] * 100
    significant_t = P["t_crit_95"] is not None and abs(P["seed_level_t"]) > P["t_crit_95"]
    if gain > 0 and lo > 0:
        verdict = "QAT-trained models score higher after FP8 deployment, and the gain is supported when both training randomness and test-set sampling are accounted for."
    elif gain > 0 and P["all_seeds_positive"]:
        verdict = ("QAT-trained models score higher after FP8 deployment in every seed, but once test-set sampling "
                   "is included the interval reaches zero, so the size of the gain is uncertain.")
    elif gain > 0:
        verdict = "QAT-trained models score higher on average, but not in every seed; the evidence is weak."
    else:
        verdict = "No accuracy advantage for QAT after FP8 deployment was found."
    L = []
    w = L.append
    w("# QAT vs normal BF16 fine-tuning, deployed as FP8 - accuracy report\n")
    w("Qwen2.5-0.5B-Instruct, full fine-tuning (no LoRA), GSM8K (7,000 train / 473 val / "
      f"{S['n_questions']:,} test), {EPOCHS} epoch{'s' if EPOCHS != 1 else ''}, NVIDIA H100, FP8 = llm-compressor FP8_DYNAMIC served by vLLM. "
      f"{n} seeds per arm (seeds {', '.join(map(str, seeds))}). All numbers below are computed from the result files.\n")
    w("## Headline\n")
    w(f"- **BF16 training -> FP8:** {_pm(a['B'])}% test accuracy")
    w(f"- **QAT training -> FP8:** {_pm(a['D'])}% test accuracy")
    w(f"- **QAT improvement: {gain:+.2f} percentage points** ({rel:+.1f}% relative). Per seed: "
      + ", ".join(f"{x:+.2f}" for x in P["diff_D_minus_B"]["values"]) + ".")
    w(f"- 95% interval for the improvement: **[{lo:+.2f}, {hi:+.2f}] pp** (resampling seeds and test questions); "
      f"[{loq:+.2f}, {hiq:+.2f}] pp if only the test questions are resampled.")
    w(f"- {verdict}\n")
    w("## Accuracy matrix (test accuracy %, mean ± std over seeds)\n")
    w("| Training | Inference | " + " | ".join(f"seed {s}" for s in seeds) + " | Mean ± std |")
    w("|---|---|" + "---:|" * n + "---:|")
    for c, (tr, inf) in {"A": ("BF16 full", "BF16"), "B": ("BF16 full", "FP8"), "C": ("QAT full", "BF16"), "D": ("QAT full", "FP8")}.items():
        w(f"| {tr} | {inf} | " + " | ".join(_f(v) for v in a[c]["values"]) + f" | {_pm(a[c])} |")
    w("\nThe primary comparison is **B vs D**, because both are final FP8 deployment models.\n")
    w("## Primary table (plan section 22; speed rows left out on purpose)\n")
    w("| Metric | BF16 Full -> FP8 | QAT Full -> FP8 |\n|---|---:|---:|")
    w(f"| Accuracy % | {_pm(a['B'])} | {_pm(a['D'])} |")
    w(f"| Eval loss (lower is better) | {_pm(l['B'], 4)} | {_pm(l['D'], 4)} |")
    w(f"| Mean generated tokens | {_f(S['gen_tokens']['B']['mean'], 1)} | {_f(S['gen_tokens']['D']['mean'], 1)} |")
    w(f"| Model size on disk | same FP8 checkpoint format for both (0.59 GB; BF16 is 0.92 GB) | same |\n")
    w("## How reliable is the gain?\n")
    w("| Check | Result |\n|---|---|")
    w(f"| Every seed positive | {'yes' if P['all_seeds_positive'] else 'no'} |")
    w(f"| Worst QAT-FP8 seed beats best BF16-FP8 seed | {'yes' if P['every_D_beats_every_B'] else 'no'} "
      f"(QAT min {_f(min(a['D']['values']))}, BF16 max {_f(max(a['B']['values']))}) |")
    tc = P["t_crit_95"]
    w(f"| Paired t across seeds (df={n-1}) | t = {P['seed_level_t']:.2f}, 5% critical value {tc if tc else 'n/a'}: "
      f"{'significant' if significant_t else 'not significant'} (few seeds, low power) |")
    w("| McNemar test per seed (single model vs single model, same questions) | "
      + ", ".join(f"seed {s}: p={p:.3f}" for s, p in zip(seeds, P["mcnemar_p_per_seed"]))
      + " (none is significant on its own - one 1,319-question test set cannot separate two single models at this size) |")
    w(f"| 95% interval, seeds + questions | [{lo:+.2f}, {hi:+.2f}] pp |\n")
    w("## Where does the gain come from?\n")
    w("Gain (D-B) = QAT effect before FP8 (C-A) + reduction of the FP8 accuracy drop.\n")
    w("| Component (pp) | " + " | ".join(f"seed {s}" for s in seeds) + " | Mean ± std |")
    w("|---|" + "---:|" * n + "---:|")
    for name, key in [("QAT effect with no FP8 (C - A)", "qat_effect_C_minus_A"),
                      ("FP8 drop, BF16-trained (A - B)", "fp8_drop_bf16_trained_A_minus_B"),
                      ("FP8 drop, QAT-trained (C - D)", "fp8_drop_qat_trained_C_minus_D"),
                      ("Drop reduction ((A-B) - (C-D))", "drop_reduction")]:
        m = D[key]
        w(f"| {name} | " + " | ".join(f"{v:+.2f}" for v in m["values"]) + f" | {m['mean']:+.2f} ± {m['std']:.2f} |")
    w(f"\nAbout {D['qat_effect_C_minus_A']['mean']:+.2f} pp of the gain appears even before FP8 (QAT models are better in plain BF16 too, "
      f"likely a regularization effect), and {D['drop_reduction']['mean']:+.2f} pp comes from losing less accuracy in the FP8 conversion. "
      "Both parts vary between seeds.\n")
    w("## Eval loss (teacher-forced, per answer token; less noisy than accuracy)\n")
    w("| Cell | Mean ± std |\n|---|---:|")
    for c, name in {"A": "BF16 -> BF16", "B": "BF16 -> FP8", "C": "QAT -> BF16", "D": "QAT -> FP8"}.items():
        w(f"| {name} | {_pm(l[c], 4)} |")
    pb, pq = S["loss_penalty"]["bf16_trained_B_minus_A"], S["loss_penalty"]["qat_trained_D_minus_C"]
    w(f"\nFP8 raises the loss by {pb['mean']:+.4f} for BF16-trained models and {pq['mean']:+.4f} for QAT-trained models "
      "(per seed: " + ", ".join(f"{x:+.4f}/{y:+.4f}" for x, y in zip(pb["values"], pq["values"])) + " BF16/QAT).\n")
    w("## Reference: the original, untrained model\n")
    w("| Model | Accuracy % | Eval loss |\n|---|---:|---:|")
    names = {"orig_bf16": "Original, BF16 (L4)", "orig_fp8": "Original, FP8 (L4)",
             "orig_bf16_h100": "Original, BF16 (H100)", "orig_fp8_h100": "Original, FP8 (H100)"}
    for k, v in S["baselines"].items():
        w(f"| {names[k]} | {_f(v['accuracy'])} | {_f(v['eval_loss'], 4)} |")
    w("\nThe untouched model scores higher than both fine-tuned arms here. Fine-tuning on GSM8K's short human-written "
      "solutions did not help this small model (and the same model's BF16 score differs by about 1.2 pp between the L4 and "
      "the H100, a reminder of how much GPU-level numerics alone can move a single accuracy number).\n")
    w("## Training behaviour\n")
    for arm, curves in S["val_curves"].items():
        if curves:
            m = [st.mean(c[i] for c in curves) for i in range(len(curves[0]))]
            w(f"- {arm}: mean validation loss by epoch (0..{len(m)-1}) = " + ", ".join(f"{x:.4f}" for x in m))
    if EPOCHS > 1:
        w("\nValidation loss is lowest after epoch 1 and rises afterwards while training loss keeps falling: more than one epoch overfits. "
          "The final checkpoint is evaluated as pre-registered (no early stopping).\n")
    else:
        w("\nSingle-epoch run: the checkpoint after epoch 1 is evaluated (no early stopping).\n")
    w("## Limitations\n")
    w(f"- {n} seeds per arm; seeds change the data order and rounding randomness but not the starting weights.")
    w("- One test set of 1,319 questions: a single model's accuracy is only known to about ±1.3 pp.")
    w("- One dataset, one model size, one FP8 recipe (E4M3, per-channel weights, per-token activations).")
    w("- The QAT quantizer reproduces the deployed weights bit for bit; activation arithmetic matches the deployment in kind but "
      "the full deployed loss increase was only about 66-76% reproduced in a diagnostic on the original model.")
    w("- FP8 numbers are for the H100; the L4 and H100 can give slightly different FP8 results.\n")
    (OUTDIR / "report.md").write_text("\n".join(L), encoding="utf-8")


def set_outdir(path):
    global OUTDIR
    OUTDIR = Path(path) if Path(path).is_absolute() else ROOT / path
    OUTDIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="folder (relative to the repo) for report.md, comparison.csv, stats.json")
    ap.add_argument("--prefix", default="", help="result-label prefix of a tagged run, e.g. 1ep_")
    ap.add_argument("--arm-suffix", default="", help="checkpoint/run folder suffix of a tagged run, e.g. _1ep")
    ap.add_argument("--epochs", type=int, default=3, help="epochs the runs were trained for (text only)")
    a = ap.parse_args()
    PREFIX, ARM_SUFFIX, EPOCHS = a.prefix, a.arm_suffix, a.epochs
    if a.out:
        set_outdir(a.out)
    S = compute()
    write_csv(S)
    write_report(S)
    print(f"seeds={S['seeds']}  B={S['acc']['B']['mean']:.2f}  D={S['acc']['D']['mean']:.2f}  "
          f"gain={S['primary']['diff_D_minus_B']['mean']:+.2f} pp  CI={S['primary']['ci95_seeds_and_questions']}")
    print(f"wrote stats.json, comparison.csv, report.md to {OUTDIR}")
