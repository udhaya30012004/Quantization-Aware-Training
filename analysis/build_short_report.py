"""Short report: title, problem, architecture (two flow diagrams side by side), results table, bar graph, loss graph.

Numbers come from more_seeds_added/stats.json (3 epochs) and plots_with_epochs/stats.json (1 epoch).
Output: final_report/QAT_FP8_short_report.html (PDF is exported from it).
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_report"
S3 = json.loads((ROOT / "more_seeds_added" / "stats.json").read_text(encoding="utf-8"))
S1 = json.loads((ROOT / "plots_with_epochs" / "stats.json").read_text(encoding="utf-8"))
BARS = ROOT / "plots_with_epochs" / "plots" / "fp8_accuracy_bars.png"
LOSS = ROOT / "more_seeds_added" / "plots" / "validation_loss_by_epoch.png"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def b64(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def pm(m):
    return f"{m['mean']:.2f} ± {m['std']:.2f}"


# ---------------------------------------------------------------- flow diagrams (inline SVG, side by side)
W_PANEL, NODE_W, GAP_X = 350, 290, 20
Y = [58, 130, 202, 298, 370, 442]          # node tops
H = [46, 46, 74, 46, 46, 46]
NEUTRAL = ("#f6f5f1", "#c3c2b7", "#0b0b0b")


def node(x, i, lines, style):
    fill, stroke, ink = style
    y, h = Y[i], H[i]
    cx = x + NODE_W / 2
    n = len(lines)
    line_h = 17
    y0 = y + h / 2 - (n - 1) * line_h / 2 + 5
    t = "".join(
        f'<text x="{cx}" y="{y0 + k * line_h}" text-anchor="middle" font-size="13.5" '
        f'font-weight="{600 if k == 0 else 400}" fill="{ink}">{ln}</text>' for k, ln in enumerate(lines))
    return (f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>' + t)


def arrow(x, i):
    cx = x + NODE_W / 2
    return f'<line x1="{cx}" y1="{Y[i] + H[i]}" x2="{cx}" y2="{Y[i + 1] - 3}" stroke="#52514e" stroke-width="1.8" marker-end="url(#arr)"/>'


def panel(x0, title, train_lines, train_style):
    x = x0 + (W_PANEL - NODE_W) / 2
    nodes = [(["Qwen2.5-0.5B-Instruct", "BF16 model"], NEUTRAL), (["GSM8K dataset"], NEUTRAL),
             (train_lines, train_style), (["FP8 quantization"], NEUTRAL), (["FP8 model"], NEUTRAL), (["Evaluation"], NEUTRAL)]
    s = f'<rect x="{x0}" y="4" width="{W_PANEL}" height="{Y[5] + H[5] + 22 - 4}" rx="10" fill="#fcfcfb" stroke="#e1e0d9"/>'
    s += f'<text x="{x0 + W_PANEL / 2}" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#0b0b0b">{title}</text>'
    for i, (lines, st) in enumerate(nodes):
        s += node(x, i, lines, st)
    for i in range(5):
        s += arrow(x, i)
    return s


SVG = (f'<svg viewBox="0 0 {2 * W_PANEL + GAP_X} {Y[5] + H[5] + 28}" xmlns="http://www.w3.org/2000/svg" '
       f'font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif" role="img" '
       f'aria-label="Flow diagrams: normal BF16 fine-tuning versus QAT fine-tuning">'
       '<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>'
       + panel(0, "Without QAT: normal BF16 fine-tuning", ["BF16 training", "Updated BF16 weights"], ("#e9f1fb", BLUE, "#0b0b0b"))
       + panel(W_PANEL + GAP_X, "With QAT: BF16 fine-tuning + FP8 rounding",
               ["QAT training", "Updated BF16 weights", "+ FP8 fake quantization"], ("#fdeee7", ORANGE, "#0b0b0b"))
       + "</svg>")


def table():
    rows = [("BF16 training (no QAT)", "BF16", "A"), ("BF16 training (no QAT)", "FP8", "B"),
            ("QAT training", "BF16", "C"), ("QAT training", "FP8", "D")]
    body = ""
    for tr, inf, c in rows:
        cls = "key" if inf == "FP8" else ""
        body += (f"<tr class='{cls}'><td>{tr}</td><td>{inf}</td><td>{pm(S3['acc'][c])}</td><td>{pm(S1['acc'][c])}</td></tr>")
    return ("<table><thead><tr><th>Training method</th><th>Model run as</th>"
            f"<th>3 epochs<br><span>({len(S3['seeds'])} seeds)</span></th><th>1 epoch<br><span>({len(S1['seeds'])} seeds)</span></th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>QAT vs normal fine-tuning for FP8 deployment</title>
<style>
  @page {{ size: A4; margin: 15mm 14mm; }}
  :root {{ --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#fff; color:var(--ink); font-family: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; font-size: 11pt; line-height:1.55; }}
  .page {{ max-width: 800px; margin: 0 auto; padding: 24px 28px 40px; }}
  h1 {{ font-size: 22pt; line-height:1.2; margin: 0 0 4px; }}
  .sub {{ color: var(--ink2); margin: 0 0 6px; }}
  h2 {{ font-size: 14pt; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line); break-after: avoid; }}
  p {{ margin: 7px 0; }}
  .diagram {{ margin: 12px 0 6px; break-inside: avoid; }}
  .diagram svg {{ width:100%; height:auto; display:block; }}
  .cap {{ color: var(--ink2); font-size: 9.5pt; margin: 6px 0 0; }}
  table {{ border-collapse: collapse; width:100%; margin: 10px 0; font-size: 10.5pt; }}
  th, td {{ border-bottom:1px solid var(--line); padding: 7px 10px; text-align:right; }}
  th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align:left; }}
  thead th {{ color: var(--ink2); font-weight:600; border-bottom:1px solid var(--muted); vertical-align: bottom; }}
  thead th span {{ font-weight:400; font-size: 9pt; }}
  tr.key td {{ background:#f6f5f1; font-weight:600; }}
  figure {{ margin: 20px 0 8px; break-inside: avoid; }}
  figure img {{ display:block; width:80%; height:auto; margin:0 auto; border:1px solid var(--line); border-radius:6px; }}
  figcaption {{ width:80%; margin: 5px auto 0; color: var(--ink2); font-size: 9.5pt; }}
</style></head>
<body><div class="page">

<h1>QAT vs normal fine-tuning for FP8 deployment</h1>
<p class="sub">Qwen2.5-0.5B-Instruct fine-tuned on GSM8K (grade-school maths)</p>

<h2>1. What are we solving?</h2>
<p>Large language models are often converted to 8-bit floating point (FP8) to make them smaller for deployment, but the conversion rounds every
weight and usually costs some accuracy.</p>
<p>Quantization-aware training (QAT) applies the same FP8 rounding <i>while the model is being fine-tuned</i>, so the model can adapt to it.
We test whether a model fine-tuned with QAT is more accurate after FP8 conversion than a model fine-tuned normally in BF16.</p>

<h2>2. Architecture: how we solve it</h2>
<p>Two models are fine-tuned independently from the same starting model on the same data. The only difference is the training step:
the QAT model also sees FP8 fake quantization in its forward pass, while its weights stay in BF16. Both are then converted to FP8 with the same
tool and evaluated on the same test questions.</p>
<div class="diagram">{SVG}</div>
<p class="cap">The training step is the only difference between the two pipelines. Every other step is identical.</p>

<h2>3. Results</h2>
<p>Test accuracy (%) on all {S3['n_questions']:,} GSM8K test questions, averaged over runs with different random seeds (mean ± standard deviation).
The shaded rows are the FP8 results that matter for deployment.</p>
{table()}

<figure><img src="{b64(BARS)}" alt="Bar graph of FP8 accuracy for normal and QAT fine-tuning">
<figcaption>Accuracy after FP8 conversion. Bars are averages over seeds, whiskers show ± 1 standard deviation, dots are individual runs.</figcaption></figure>

<h2>4. Loss</h2>
<figure><img src="{b64(LOSS)}" alt="Validation loss by epoch">
<figcaption>Validation loss during the 3-epoch runs (thin lines: individual seeds, thick lines: average).</figcaption></figure>

</div></body></html>
"""

OUT.mkdir(exist_ok=True)
(OUT / "QAT_FP8_short_report.html").write_text(HTML, encoding="utf-8")
print("wrote", OUT / "QAT_FP8_short_report.html")
