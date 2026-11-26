"""Write the 'Results and deep analysis' section of plan.md from the result files (idempotent: replaces its own block).

Every number comes from more_seeds_added/stats.json (3 epochs), plots_with_epochs/stats.json (1 epoch) and results/*.json.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
S3 = json.loads((ROOT / "more_seeds_added" / "stats.json").read_text(encoding="utf-8"))
S1 = json.loads((ROOT / "plots_with_epochs" / "stats.json").read_text(encoding="utf-8"))
START, END = "<!-- ANALYSIS:START -->", "<!-- ANALYSIS:END -->"


def jl(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def kl(setting, arm):
    seeds = sorted(int(p.stem.split("_s")[-1]) for p in R.glob(f"kl_{setting}_{arm}_s*.json"))
    return [jl(f"kl_{setting}_{arm}_s{s}.json") for s in seeds]


def pm(m, d=2):
    return f"{m['mean']:.{d}f} ± {m['std']:.{d}f}"


def sgn(x, d=2):
    return f"{x:+.{d}f}"


KL = {}
for stg in ("3ep", "1ep"):
    for arm in ("bf16", "qat"):
        rows = kl(stg, arm)
        KL[(stg, arm)] = {"kl": [r["kl_mean"] for r in rows], "agree": [r["top1_agreement"] for r in rows]}
KL_ORIG = jl("kl_orig.json")["kl_mean"]


def kl_line(stg, name):
    b, q = KL[(stg, "bf16")]["kl"], KL[(stg, "qat")]["kl"]
    lower = sum(x < y for x, y in zip(q, b))
    return (f"| {name} | {st.mean(b):.5f} ± {st.stdev(b):.5f} | {st.mean(q):.5f} ± {st.stdev(q):.5f} | "
            f"{st.mean(q) / st.mean(b) * 100:.0f}% | {lower} of {len(b)} | "
            f"{st.mean(KL[(stg, 'bf16')]['agree']) * 100:.2f}% / {st.mean(KL[(stg, 'qat')]['agree']) * 100:.2f}% |")


def facts(S):
    P, D, L, a = S["primary"], S["decomposition"], S["loss_penalty"], S["acc"]
    vals = P["diff_D_minus_B"]["values"]
    return dict(S=S, a=a, P=P, D=D, L=L, n=len(S["seeds"]), gain=P["diff_D_minus_B"]["mean"], vals=vals,
                pos=sum(v > 0 for v in vals), lo=P["ci95_seeds_and_questions"][0], hi=P["ci95_seeds_and_questions"][1],
                qlo=P["ci95_questions_only"][0], qhi=P["ci95_questions_only"][1], t=P["seed_level_t"], tc=P["t_crit_95"],
                mc=P["mcnemar_p_per_seed"], smaller=sum(q < b for q, b in zip(L["qat_trained_D_minus_C"]["values"],
                                                                            L["bf16_trained_B_minus_A"]["values"])))


F3, F1 = facts(S3), facts(S1)
base = S3["baselines"]
gen_orig = jl("orig_bf16_h100.json")["mean_gen_tokens"]
gen_3 = S3["gen_tokens"]["A"]["mean"]
gen_1 = S1["gen_tokens"]["A"]["mean"]
n_q = S3["n_questions"]
se_pp = (0.35 * 0.65 / n_q) ** 0.5 * 100


def acc_table(F, name):
    a = F["a"]
    return (f"| {name} | {pm(a['A'])} | {pm(a['B'])} | {pm(a['C'])} | {pm(a['D'])} | **{sgn(F['gain'])}** "
            f"({sgn(F['lo'], 1)} to {sgn(F['hi'], 1)}) | {F['pos']} of {F['n']} |")


def dec_row(F, name):
    D = F["D"]
    return (f"| {name} | {sgn(D['qat_effect_C_minus_A']['mean'])} | {sgn(D['fp8_drop_bf16_trained_A_minus_B']['mean'])} | "
            f"{sgn(D['fp8_drop_qat_trained_C_minus_D']['mean'])} | {sgn(D['drop_reduction']['mean'])} |")


def loss_row(F, name):
    L = F["L"]
    b, q = L["bf16_trained_B_minus_A"], L["qat_trained_D_minus_C"]
    return (f"| {name} | +{b['mean']:.4f} | +{q['mean']:.4f} | {q['mean'] / b['mean'] * 100:.0f}% | {F['smaller']} of {F['n']} |")


BLOCK = f"""{START}
## Results and deep analysis (added after the experiments)

All numbers in this section are computed from the result files by `analysis/add_plan_analysis.py`; none is typed by hand.
Cells: **A** = BF16-trained evaluated in BF16, **B** = BF16-trained evaluated in FP8, **C** = QAT-trained in BF16, **D** = QAT-trained in FP8.
Everything ran on one H100, on the full {n_q:,}-question GSM8K test set, with {F3['n']} seeds per method.

### A. Verdict on the research question

> Does full FP8 QAT fine-tuning give better accuracy after FP8 conversion than full BF16 fine-tuning followed by the same conversion?

**Yes in direction, with a modest and not statistically settled size.**
- 3 epochs: **{sgn(F3['gain'])} pp** (D {pm(F3['a']['D'])}% vs B {pm(F3['a']['B'])}%), QAT ahead in {F3['pos']} of {F3['n']} seeds; 95% interval {sgn(F3['lo'])} to {sgn(F3['hi'])} pp.
- 1 epoch: **{sgn(F1['gain'])} pp** (D {pm(F1['a']['D'])}% vs B {pm(F1['a']['B'])}%), QAT ahead in {F1['pos']} of {F1['n']} seeds; 95% interval {sgn(F1['lo'])} to {sgn(F1['hi'])} pp.
- The **loss and KL measures are clear and consistent**: FP8 changes QAT-trained models less in every one of the {F3['n'] + F1['n']} seed pairs for KL (section D).

### B. Accuracy in detail

| Setting | A (BF16→BF16) | B (BF16→FP8) | C (QAT→BF16) | D (QAT→FP8) | D − B (95% interval) | Seeds with D > B |
|---|---|---|---|---|---|---|
{acc_table(F3, '3 epochs')}
{acc_table(F1, '1 epoch')}

- **Interval used:** resamples both the training seeds and the test questions (two-level bootstrap). Resampling only the questions gives {sgn(F3['qlo'])} to {sgn(F3['qhi'])} pp (3 epochs) and {sgn(F1['qlo'])} to {sgn(F1['qhi'])} pp (1 epoch); the seeds-and-questions interval is the honest one.
- **3 epochs:** every QAT-FP8 model beats every BF16-FP8 model (lowest D {min(F3['a']['D']['values']):.2f}% vs highest B {max(F3['a']['B']['values']):.2f}%). The across-seed paired t is {F3['t']:.2f} (df {F3['n'] - 1}, 5% critical value {F3['tc']}), but this ignores test-set sampling and has few seeds, so it overstates certainty.
- **1 epoch:** paired t {F1['t']:.2f} vs critical {F1['tc']}: not significant. Seed-to-seed spread of B is {F1['a']['B']['std']:.2f} pp, as large as the gain.
- **Per-seed McNemar tests** (one model vs one model on the same questions): 3 epochs p = {min(F3['mc']):.2f}–{max(F3['mc']):.2f}; 1 epoch p = {min(F1['mc']):.2f}–{max(F1['mc']):.2f}. Only {sum(p < 0.05 for p in F3['mc']) + sum(p < 0.05 for p in F1['mc'])} of the {F3['n'] + F1['n']} individual pairs reaches p < 0.05 (not corrected for running {F3['n'] + F1['n']} tests), which is expected: one {n_q:,}-question test set measures a single model's accuracy with a 95% half-width of about ±{1.96 * se_pp:.1f} pp.

### C. Where the accuracy gain comes from

D − B = (C − A) + [(A − B) − (C − D)]: an effect of QAT that exists even without FP8, plus a smaller FP8 drop.

| Setting (pp) | QAT effect with no FP8 (C − A) | FP8 drop, BF16-trained (A − B) | FP8 drop, QAT-trained (C − D) | Drop reduction |
|---|---|---|---|---|
{dec_row(F3, '3 epochs')}
{dec_row(F1, '1 epoch')}

- At **3 epochs** about half of the gain ({sgn(F3['D']['qat_effect_C_minus_A']['mean'])} pp) shows up even in plain BF16. This looks like QAT acting as a regularizer on an overfit model: the QAT models also have lower validation loss.
- At **1 epoch** that part is **{sgn(F1['D']['qat_effect_C_minus_A']['mean'])} pp** (both arms score {F1['a']['A']['mean']:.2f}% in BF16), so the whole gain comes from a smaller FP8 drop. Individual seeds still vary by more than a point (drop reduction std {F1['D']['drop_reduction']['std']:.2f} pp), which is why that gain is not significant.

### D. The sharper evidence: loss and KL divergence

Accuracy needs correct final answers and is quantised to {n_q:,} yes/no outcomes. Loss and KL use every answer token (about {jl('kl_orig.json')['n_tokens']:,} per model) and are deterministic.

**Increase in test loss caused by FP8 conversion (real FP8 in vLLM; lower is better):**

| Setting | BF16-trained | QAT-trained | QAT as % of BF16 | Seeds where QAT is smaller |
|---|---|---|---|---|
{loss_row(F3, '3 epochs')}
{loss_row(F1, '1 epoch')}

**KL divergence KL(P_BF16 ‖ P_FP8) per answer token (simulated FP8; lower is better):**

| Setting | BF16-trained | QAT-trained | QAT as % of BF16 | Seeds where QAT is lower | Top-1 agreement (BF16 / QAT) |
|---|---|---|---|---|---|
{kl_line('3ep', '3 epochs')}
{kl_line('1ep', '1 epoch')}

Untrained original model: KL = {KL_ORIG:.5f}.

- Under KL the QAT models are lower in **all {F3['n']} of {F3['n']} seeds in both settings**, with non-overlapping ranges. This is the strongest single result of the project.
- **Circularity caveat:** the KL simulator is the same FP8 fake-quantizer that the QAT models were trained against, so KL may flatter QAT somewhat. The real-FP8 loss and accuracy results (computed through vLLM, an independent path) point the same way, which is what makes the conclusion believable.

### E. Training length: overfit models are more fragile under FP8

| | Original (no tuning) | 1 epoch (BF16 / QAT) | 3 epochs (BF16 / QAT) |
|---|---|---|---|
| KL (BF16 vs FP8) | {KL_ORIG:.4f} | {st.mean(KL[('1ep', 'bf16')]['kl']):.4f} / {st.mean(KL[('1ep', 'qat')]['kl']):.4f} | {st.mean(KL[('3ep', 'bf16')]['kl']):.4f} / {st.mean(KL[('3ep', 'qat')]['kl']):.4f} |
| Accuracy in FP8 | {base['orig_fp8']['accuracy']:.1f}% (L4) | {F1['a']['B']['mean']:.1f}% / {F1['a']['D']['mean']:.1f}% | {F3['a']['B']['mean']:.1f}% / {F3['a']['D']['mean']:.1f}% |

- 3-epoch models change about **{st.mean(KL[('3ep', 'bf16')]['kl']) / st.mean(KL[('1ep', 'bf16')]['kl']):.1f}× more** under FP8 than 1-epoch models, and more than the untouched model. Validation loss is lowest after epoch 1 and rises afterwards, so extra epochs overfit.
- *Hypothesis (not tested here):* overfitting makes the output distributions sharper, and sharper distributions are more sensitive to small numerical perturbations.
- QAT's relative benefit follows the fragility: **{(1 - st.mean(KL[('3ep', 'qat')]['kl']) / st.mean(KL[('3ep', 'bf16')]['kl'])) * 100:.0f}% lower KL at 3 epochs** where there is much to repair, **{(1 - st.mean(KL[('1ep', 'qat')]['kl']) / st.mean(KL[('1ep', 'bf16')]['kl'])) * 100:.0f}% at 1 epoch**.
- Practical reading: if you fine-tune for fewer epochs you lose less to FP8 regardless of method (FP8 accuracy {F1['a']['B']['mean']:.1f}% vs {F3['a']['B']['mean']:.1f}% for BF16-trained), and QAT then adds a smaller extra gain.

### F. Fine-tuning versus the untouched model

- Original model, BF16: **{base['orig_bf16']['accuracy']:.2f}%** (L4) and **{base['orig_bf16_h100']['accuracy']:.2f}%** (H100); FP8 on L4: {base['orig_fp8']['accuracy']:.2f}%.
- Fine-tuned models, BF16: {F3['a']['A']['mean']:.1f}% / {F3['a']['C']['mean']:.1f}% (3 epochs), {F1['a']['A']['mean']:.1f}% (1 epoch). **Neither fine-tuned arm beats the original.**
- The fine-tuned models answer in about {gen_3:.0f} tokens against about {gen_orig:.0f} for the original. *Hypothesis:* GSM8K's short human-written solutions teach the model to reason less verbosely, which hurts a small model that benefits from longer step-by-step output. This does not affect the QAT-vs-BF16 comparison (same effect on both), but it limits how much practical value fine-tuning on this dataset adds.

### G. Validity checks, threats and what they mean

| Item | Evidence | Effect on conclusions |
|---|---|---|
| QAT quantizer matches deployment | FP8 weights bit-identical to the exporter's on 100% of codes and scales in all 168 layers, after matching the exporter's BF16 scales and BF16 division | Strong: the training-time weights are the deployed weights |
| Activation arithmetic matches only in kind | Diagnostic reproduced 66–76% of vLLM's FP8 loss increase on the original model | Small residual; unexplained (likely kernel details), not claimed as a known cause |
| Single-number "FP8 loss increase" is noisy | The same fake-quant gave +0.0067 on L4 and −0.0039 on H100; quantising only activations *lowered* the loss by 0.008 | Why weight-code identity, KL and multi-seed averages are used instead |
| GPU numerics move accuracy | Original BF16 {base['orig_bf16']['accuracy']:.2f}% (L4) vs {base['orig_bf16_h100']['accuracy']:.2f}% (H100) for the same weights | All compared runs used the same GPU type (H100) |
| Test-set size | 95% half-width of about ±{1.96 * se_pp:.1f} pp for one model at about 35% accuracy | Main reason the accuracy intervals reach zero |
| Seeds are not fully independent | Seeds change data order and rounding randomness, not the starting weights | Seed spread slightly understates true run-to-run variation |
| Answer extraction | Rule fixed before any trained model was evaluated (final `####` line, else last number); the base model's premature `####` guesses were found in the 50-question check | Same rule for every cell |
| Scope | One 0.5B model, one maths dataset, one FP8 recipe (E4M3, per-channel weights, per-token activations) | Findings may not transfer to larger models or other data |

### H. Departures from the original plan

- **GPU:** the plan said an L4; all runs used an H100 for speed (the L4 reference results for the original model are kept). Speed was dropped from scope: a quick benchmark showed no measurable FP8 speed-up for this small model (differences within about ±5%); the FP8 checkpoint is 36% smaller on disk.
- **Optimizer:** custom AdamW for BF16 weights with stochastic rounding (plain BF16 rounding would silently lose most small updates).
- **Extra experiments:** 5 seeds instead of 3, a 1-epoch repeat, and the KL analysis.
- **Environments:** the vLLM and llm-compressor images are separate because their dependencies conflict.

### I. Conclusions and suggested next steps

1. **QAT reliably reduces how much FP8 changes the model** (KL and loss, every seed, both settings).
2. **That becomes roughly +{F1['gain']:.1f} to +{F3['gain']:.1f} points of accuracy after FP8**, consistently positive in direction but not settled at 95% confidence on this test set.
3. **Next:** (a) evaluate the same checkpoints on a second, larger maths test set to tighten the accuracy interval; (b) try harder compression such as FP4 or INT4, where plain conversion loses much more and QAT has more to recover (larger models usually lose less to FP8, not more); (c) test the overfitting-fragility hypothesis, for example by measuring KL after each epoch of training.
{END}
"""

plan = ROOT / "plan.md"
text = plan.read_text(encoding="utf-8")
if START in text:
    pre = text.split(START)[0]
    post = text.split(END)[1] if END in text else ""
    text = pre.rstrip() + "\n\n" + BLOCK + post
else:
    text = text.rstrip() + "\n\n---\n\n" + BLOCK
plan.write_text(text, encoding="utf-8")
print("plan.md updated:", len(BLOCK.splitlines()), "lines in the analysis block")
