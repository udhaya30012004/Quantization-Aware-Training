"""Build the presentable report (HTML with embedded graphs; PDF is made from it) -> final_report/.

All numbers come from more_seeds_added/stats.json (3 epochs) and plots_with_epochs/stats.json (1 epoch).
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_report"
S3 = json.loads((ROOT / "more_seeds_added" / "stats.json").read_text(encoding="utf-8"))
S1 = json.loads((ROOT / "plots_with_epochs" / "stats.json").read_text(encoding="utf-8"))

FIGS = {
    "bars": ROOT / "plots_with_epochs" / "plots" / "fp8_accuracy_bars.png",
    "slope3": ROOT / "more_seeds_added" / "plots" / "accuracy_bf16_vs_fp8.png",
    "slope1": ROOT / "plots_with_epochs" / "plots" / "accuracy_bf16_vs_fp8.png",
    "val3": ROOT / "more_seeds_added" / "plots" / "validation_loss_by_epoch.png",
}


def img(key):
    return "data:image/png;base64," + base64.b64encode(FIGS[key].read_bytes()).decode()


def pm(m, d=1):
    return f"{m['mean']:.{d}f} ± {m['std']:.{d}f}"


def facts(S):
    a, P, D, L = S["acc"], S["primary"], S["decomposition"], S["loss_penalty"]
    n = len(S["seeds"])
    vals = P["diff_D_minus_B"]["values"]
    lo, hi = P["ci95_seeds_and_questions"]
    return {
        "n": n, "B": a["B"], "D": a["D"], "A": a["A"], "C": a["C"],
        "gain": P["diff_D_minus_B"]["mean"], "per_seed": vals, "pos": sum(v > 0 for v in vals),
        "lo": lo, "hi": hi, "all_beat": P["every_D_beats_every_B"],
        "drop_b": D["fp8_drop_bf16_trained_A_minus_B"]["mean"], "drop_q": D["fp8_drop_qat_trained_C_minus_D"]["mean"],
        "qat_effect": D["qat_effect_C_minus_A"]["mean"],
        "loss_pen_b": L["bf16_trained_B_minus_A"]["mean"], "loss_pen_q": L["qat_trained_D_minus_C"]["mean"],
        "loss_smaller": sum(q < b for q, b in zip(L["qat_trained_D_minus_C"]["values"], L["bf16_trained_B_minus_A"]["values"])),
        "loss": S["loss"], "seeds": S["seeds"],
    }


f3, f1 = facts(S3), facts(S1)
base = S3["baselines"]
n_q = S3["n_questions"]


def results_table(f):
    rows = [("BF16 full fine-tuning", "BF16", f["A"]), ("BF16 full fine-tuning", "FP8", f["B"]),
            ("QAT full fine-tuning", "BF16", f["C"]), ("QAT full fine-tuning", "FP8", f["D"])]
    body = "".join(
        f"<tr class='{'key' if inf == 'FP8' else ''}'><td>{tr}</td><td>{inf}</td>"
        + "".join(f"<td>{v:.2f}</td>" for v in m["values"]) + f"<td><b>{pm(m, 2)}</b></td></tr>"
        for tr, inf, m in rows)
    head = "".join(f"<th>Seed {s}</th>" for s in f["seeds"])
    return (f"<table><thead><tr><th>Training method</th><th>Run as</th>{head}<th>Mean ± std</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def seed_list(f):
    return ", ".join(f"{v:+.1f}" for v in f["per_seed"])


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>QAT vs normal fine-tuning for FP8 deployment - results</title>
<style>
  @page {{ size: A4; margin: 15mm 14mm; }}
  :root {{ --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --blue:#2a78d6; --orange:#eb6834; --surface:#fcfcfb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#fff; color:var(--ink); font-family: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
         font-size: 10.5pt; line-height:1.5; }}
  .page {{ max-width: 820px; margin: 0 auto; padding: 24px 28px 40px; }}
  h1 {{ font-size: 21pt; line-height:1.2; margin: 0 0 4px; }}
  .sub {{ color: var(--ink2); margin: 0 0 18px; }}
  h2 {{ font-size: 13.5pt; margin: 26px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line); break-after: avoid; }}
  h3 {{ font-size: 11pt; margin: 16px 0 4px; break-after: avoid; }}
  p {{ margin: 6px 0; }}
  .summary {{ background: var(--surface); border:1px solid var(--line); border-left: 4px solid var(--orange); border-radius: 6px;
             padding: 12px 16px; margin: 10px 0 6px; }}
  .summary p {{ margin: 5px 0; }}
  .kpis {{ display:flex; gap:12px; margin: 14px 0; }}
  .kpi {{ flex:1; border:1px solid var(--line); border-radius:8px; padding:10px 12px; background: var(--surface); }}
  .kpi .v {{ font-size: 20pt; font-weight: 700; line-height:1.1; }}
  .kpi .l {{ color: var(--ink2); font-size: 9pt; }}
  .b {{ color: var(--blue); }} .o {{ color: var(--orange); }}
  table {{ border-collapse: collapse; width:100%; margin: 8px 0 10px; font-size: 9.5pt; }}
  th, td {{ border-bottom:1px solid var(--line); padding: 5px 8px; text-align:right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  thead th {{ color: var(--ink2); font-weight:600; border-bottom:1px solid var(--muted); }}
  tr.key td {{ background:#f6f5f1; }}
  figure {{ margin: 12px 0 16px; break-inside: avoid; }}
  figure img {{ display:block; width:76%; height:auto; margin:0 auto; border:1px solid var(--line); border-radius:6px; }}
  table.plain th, table.plain td {{ text-align:left !important; }}
  figcaption {{ width:76%; margin-left:auto; margin-right:auto; }}
  figcaption {{ color: var(--ink2); font-size: 9pt; margin-top: 5px; }}
  ul {{ margin: 6px 0 6px 20px; padding:0; }} li {{ margin: 3px 0; }}
  .note {{ color: var(--ink2); font-size: 9.5pt; }}
  .small {{ font-size: 9pt; color: var(--ink2); }}
  code {{ background:#f2f1ec; padding: 1px 4px; border-radius:3px; font-size: 9pt; }}
  @media print {{ .page {{ padding: 0; max-width:none; }} h2 {{ break-before: auto; }} }}
</style></head>
<body><div class="page">

<h1>Does FP8-aware (QAT) fine-tuning beat normal fine-tuning when the model is deployed in FP8?</h1>
<p class="sub">Qwen2.5-0.5B-Instruct fine-tuned on GSM8K (grade-school maths). Full-model fine-tuning, NVIDIA H100.
Accuracy on the complete {n_q:,}-question GSM8K test set.</p>

<div class="summary">
<p><b>Bottom line.</b> QAT-trained models keep more of their accuracy after being converted to FP8.
After conversion they score <b>{f3['gain']:+.1f} points higher</b> than normally fine-tuned models in the 3-epoch experiment
(better in {f3['pos']} of {f3['n']} runs) and <b>{f1['gain']:+.1f} points higher</b> in the 1-epoch experiment (better in {f1['pos']} of {f1['n']} runs).</p>
<p><b>How sure are we?</b> The direction is consistent, but with one {n_q:,}-question test set the size of the gain cannot be
pinned down: the 95% intervals ({f3['lo']:+.1f} to {f3['hi']:+.1f} and {f1['lo']:+.1f} to {f1['hi']:+.1f} points) both reach zero.
The most reliable evidence is the model's loss, where FP8 conversion hurts QAT models roughly half as much or less
(details below).</p>
</div>

<div class="kpis">
  <div class="kpi"><div class="v b">{f3['B']['mean']:.1f}%</div><div class="l">Normal fine-tuning, after FP8<br>(3 epochs, {f3['n']} runs)</div></div>
  <div class="kpi"><div class="v o">{f3['D']['mean']:.1f}%</div><div class="l">QAT fine-tuning, after FP8<br>(3 epochs, {f3['n']} runs)</div></div>
  <div class="kpi"><div class="v b">{f1['B']['mean']:.1f}%</div><div class="l">Normal fine-tuning, after FP8<br>(1 epoch, {f1['n']} runs)</div></div>
  <div class="kpi"><div class="v o">{f1['D']['mean']:.1f}%</div><div class="l">QAT fine-tuning, after FP8<br>(1 epoch, {f1['n']} runs)</div></div>
</div>

<h2>1. What was tested</h2>
<ul>
  <li><b>The question:</b> models are often shrunk to 8-bit floating point (FP8) for deployment, which usually costs some accuracy.
  Quantization-aware training (QAT) simulates that FP8 rounding <i>during</i> fine-tuning so the model can adapt to it.
  Does a QAT-trained model end up more accurate in FP8 than a normally trained one?</li>
  <li><b>Two methods, same everything else:</b> both start from the same original model, use the same data, settings and
  number of steps, and train all parameters (no LoRA). The only difference is that QAT applies FP8 rounding to the model's
  main matrix layers in the forward pass (168 layers).</li>
  <li><b>Same FP8 conversion for both:</b> each trained model is converted with the same standard tool (llm-compressor FP8) and served with vLLM.
  The QAT rounding was checked to reproduce the converted weights exactly (100% of values on all 168 layers).</li>
  <li><b>Data:</b> GSM8K: 7,000 training, 473 validation and the full {n_q:,}-question test set, split once and never changed.</li>
  <li><b>Repeats:</b> each method was trained {f3['n']} times with different random seeds (data order) for 3 epochs, and
  {f1['n']} times for 1 epoch, so results are averages rather than one lucky run.</li>
</ul>

<h2>2. Main result</h2>
<figure><img src="{img('bars')}" alt="Bar graph of FP8 accuracy">
<figcaption><b>Figure 1.</b> Accuracy after FP8 conversion, normal vs QAT fine-tuning. Bars are averages over seeds (start at zero),
whiskers show ± 1 standard deviation, dots are individual runs.</figcaption></figure>

<h3>3-epoch experiment ({f3['n']} seeds per method)</h3>
{results_table(f3)}
<p>After FP8: normal <b>{pm(f3['B'], 2)}%</b> vs QAT <b>{pm(f3['D'], 2)}%</b>, a difference of <b>{f3['gain']:+.2f} points</b>
(per run: {seed_list(f3)}). Every QAT model scored above every normal model after FP8: {'yes' if f3['all_beat'] else 'no'}.
95% interval for the difference: {f3['lo']:+.2f} to {f3['hi']:+.2f} points.</p>

<h3>1-epoch experiment ({f1['n']} seeds per method)</h3>
{results_table(f1)}
<p>After FP8: normal <b>{pm(f1['B'], 2)}%</b> vs QAT <b>{pm(f1['D'], 2)}%</b>, a difference of <b>{f1['gain']:+.2f} points</b>
(per run: {seed_list(f1)}). QAT was ahead in {f1['pos']} of {f1['n']} runs.
95% interval: {f1['lo']:+.2f} to {f1['hi']:+.2f} points, so a gain of zero cannot be ruled out here.</p>

<figure><img src="{img('slope3')}" alt="BF16 versus FP8 accuracy, 3 epochs">
<figcaption><b>Figure 2.</b> 3-epoch runs: each thin line is one training run going from BF16 to FP8 inference; thick lines are averages.
The normal models drop further than the QAT models.</figcaption></figure>

<figure><img src="{img('slope1')}" alt="BF16 versus FP8 accuracy, 1 epoch">
<figcaption><b>Figure 3.</b> 1-epoch runs, same layout. The two methods have the same average in BF16 and differ only after FP8; individual runs overlap more.</figcaption></figure>

<h2>3. How much accuracy does FP8 cost each method?</h2>
<table class="plain"><thead><tr><th>Setting</th><th>Normal: BF16 → FP8</th><th>QAT: BF16 → FP8</th></tr></thead><tbody>
<tr><td>3 epochs</td><td>{f3['drop_b']:+.2f} points lost</td><td>{f3['drop_q']:+.2f} points lost</td></tr>
<tr><td>1 epoch</td><td>{f1['drop_b']:+.2f} points lost</td><td>{f1['drop_q']:+.2f} points lost</td></tr></tbody></table>
<p>In the 3-epoch runs QAT models were also slightly better in plain BF16 ({f3['C']['mean']:.1f}% vs {f3['A']['mean']:.1f}%), probably because
QAT acts as a regularizer when the model overfits. In the 1-epoch runs that extra benefit disappears (both {f1['A']['mean']:.1f}%), so the 1-epoch
gain comes only from losing less accuracy in the FP8 conversion. Individual runs vary by a point or more, which is why the intervals are wide.</p>

<h2>4. The more reliable signal: model loss</h2>
<p>Accuracy on {n_q:,} questions is a noisy measure. The model's average loss on the correct answers (lower is better) is much more stable, and it shows a clear pattern:</p>
<table class="plain"><thead><tr><th>Setting</th><th>Loss increase from FP8, normal</th><th>Loss increase from FP8, QAT</th><th>QAT smaller in</th></tr></thead><tbody>
<tr><td>3 epochs</td><td>+{f3['loss_pen_b']:.4f}</td><td>+{f3['loss_pen_q']:.4f}</td><td>{f3['loss_smaller']} of {f3['n']} runs</td></tr>
<tr><td>1 epoch</td><td>+{f1['loss_pen_b']:.4f}</td><td>+{f1['loss_pen_q']:.4f}</td><td>{f1['loss_smaller']} of {f1['n']} runs</td></tr></tbody></table>
<p>Converting to FP8 damages QAT-trained models clearly less, in both settings. This is the strongest evidence that QAT does what it is designed to do.</p>

<h2>5. Training behaviour</h2>
<figure><img src="{img('val3')}" alt="Validation loss by epoch">
<figcaption><b>Figure 4.</b> Validation loss during the 3-epoch runs (thin lines are individual runs, thick lines are averages). It is lowest after epoch 1
and rises afterwards while training loss keeps falling, so 3 epochs overfit this small model. QAT's value is measured while its FP8 rounding is on.</figcaption></figure>

<h2>6. Reference: the original, untouched model</h2>
<table><thead><tr><th>Model</th><th>Accuracy</th></tr></thead><tbody>
<tr><td>Original, BF16 (H100)</td><td>{base['orig_bf16_h100']['accuracy']:.1f}%</td></tr>
<tr><td>Original, BF16 (L4 GPU)</td><td>{base['orig_bf16']['accuracy']:.1f}%</td></tr>
<tr><td>Original, FP8 (L4 GPU)</td><td>{base['orig_fp8']['accuracy']:.1f}%</td></tr></tbody></table>
<p>The untouched model scores <b>higher than every fine-tuned model</b> here. Fine-tuning on GSM8K's short human-written solutions did not improve this
small model on this test. That does not affect the QAT-versus-normal comparison, because both methods start from the same model,
but it means neither fine-tuned variant is better than simply using the original.</p>

<h2>7. Conclusions and caveats</h2>
<ul>
  <li><b>QAT consistently reduces the damage from FP8</b>, shown most clearly by the loss: roughly half to a third of the increase, in nearly every run.</li>
  <li><b>This translates into about +1 to +1.5 points of accuracy after FP8</b>, positive in {f3['pos'] + f1['pos']} of {f3['n'] + f1['n']} runs across both experiments,
  but with one test set of {n_q:,} questions the 95% intervals reach zero, so we cannot claim a precise or guaranteed gain.</li>
  <li><b>Limits:</b> one small model, one maths dataset, one FP8 recipe, one GPU type (H100). The seeds change data order only, not the starting weights.
  FP8 results can differ slightly between GPU types.</li>
  <li><b>Not measured here:</b> speed. For a model this small, FP8 gave no measurable speed-up in a quick test (it is only about 36% smaller on disk),
  so speed was left out of scope.</li>
</ul>
<h3>Suggested next steps</h3>
<ul>
  <li>Evaluate the same trained models on a second, larger maths test set to tighten the accuracy interval (the test-set size is now the main limit).</li>
  <li>Try harder compression such as FP4 or INT4, where plain conversion loses much more accuracy and QAT has more to recover (larger models usually lose less to FP8, not more).</li>
</ul>

<p class="small" style="margin-top:22px">Reproducibility: raw results in <code>results/</code>; analysis scripts in <code>analysis/</code>; reports in <code>more_seeds_added/</code> (3 epochs)
and <code>plots_with_epochs/</code> (1 epoch). Same data split, same starting model and same FP8 conversion for every run.</p>
</div></body></html>
"""

OUT.mkdir(exist_ok=True)
(OUT / "report.html").write_text(HTML, encoding="utf-8")
print("wrote", OUT / "report.html", f"({len(HTML)/1e6:.2f} MB)")
