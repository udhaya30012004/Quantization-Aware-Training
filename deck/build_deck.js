// Builds final_report/QAT_vs_BF16_FP8_presentation.pptx (themed, with layouts + sections). Numbers are read from the result files.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(process.env.APPLY_THEME);

const ROOT = path.resolve(__dirname, "..");
const J = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const S3 = J("more_seeds_added/stats.json");   // 3 epochs
const S1 = J("plots_with_epochs/stats.json");  // 1 epoch
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const klList = (st, arm) => fs.readdirSync(path.join(ROOT, "results"))
  .filter((f) => f.startsWith("kl_" + st + "_" + arm + "_s") && f.endsWith(".json"))
  .sort()
  .map((f) => J("results/" + f));
const KL = {};
for (const st of ["3ep", "1ep"]) for (const arm of ["bf16", "qat"]) KL[st + arm] = klList(st, arm);
const klMean = (k) => mean(KL[k].map((r) => r.kl_mean));
const agree = (k) => mean(KL[k].map((r) => r.top1_agreement)) * 100;
const klLower = (st) => KL[st + "qat"].filter((r, i) => r.kl_mean < KL[st + "bf16"][i].kl_mean).length;
const sizes = { bf16: J("results/bench/orig_bf16.json").checkpoint_safetensors_gb, fp8: J("results/bench/orig_fp8.json").checkpoint_safetensors_gb };

const f1 = (x) => x.toFixed(1), f2 = (x) => x.toFixed(2), f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4);
const sg = (x, d = 1) => (x >= 0 ? "+" : "-") + Math.abs(x).toFixed(d);
const gain3 = S3.primary.diff_D_minus_B.mean, gain1 = S1.primary.diff_D_minus_B.mean;
const pos3 = S3.primary.diff_D_minus_B.values.filter((v) => v > 0).length, n3 = S3.seeds.length;
const pos1 = S1.primary.diff_D_minus_B.values.filter((v) => v > 0).length, n1 = S1.seeds.length;
const klPct = (1 - klMean("3epqat") / klMean("3epbf16")) * 100;
const lp = (S, k) => S.loss_penalty[k].mean;
const curveMean = (S, arm, i) => mean(S.val_curves[arm].map((c) => c[i]));
const savedPct = (1 - sizes.fp8 / sizes.bf16) * 100;
const nSeeds = n3;

// ---- theme --------------------------------------------------------------------------------------------------------
const THEME = {
  name: "QAT FP8 Study",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "52514E", lt2: "E1E0D9",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "898781", accent5: "6250D6", accent6: "EDA100",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
pres.title = "QAT vs normal fine-tuning for FP8 deployment";
pres.subject = "Qwen2.5-0.5B-Instruct on GSM8K";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const M = 0.6, W = 13.333 - 2 * M;       // page margin and content width

pres.defineSlideMaster({
  title: "OPEN",
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.0, w: 8.4, h: 1.9, fontSize: 40, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.5, w: W, h: 0.95, fontSize: 34, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "QAT vs normal fine-tuning for FP8 deployment  |  Qwen2.5-0.5B-Instruct", options: { x: M, y: 7.0, w: 9, h: 0.3, fontSize: 10, color: C.accent4, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.53, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

// ---- helpers ------------------------------------------------------------------------------------------------------
const png = (file) => { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
function image(slide, file, x, y, w, alt) {
  const d = png(file);
  slide.addImage({ path: file, x, y, w, h: (w * d.h) / d.w, altText: alt });
}
const T = (slide, text, o) => slide.addText(text, Object.assign({ isTextBox: true, margin: 0, valign: "top" }, o));
function bulletList(slide, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 7 } }));
  slide.addText(runs, Object.assign({ isTextBox: true, margin: 0, valign: "top", fontSize: 15, color: C.text1 }, o));
}
const hair = (slide, x, y, w) => slide.addShape(pres.ShapeType.line, { x, y, w, h: 0, line: { color: C.background2, width: 1 } });
const A = (f) => path.join(__dirname, "assets", f);

// ================================================= 1. Problem and approach ==========================================
pres.addSection({ title: "Problem and approach" });
let s = pres.addSlide({ masterName: "OPEN", sectionTitle: "Problem and approach" });
T(s, "QAT VS NORMAL FINE-TUNING FOR FP8 DEPLOYMENT", { x: M, y: 0.55, w: 8.4, h: 0.3, fontSize: 12, bold: true, color: C.text2, charSpacing: 2 });
s.addText("Can FP8-aware training help a small model hold up better?", { placeholder: "title" });
T(s, "THE PROBLEM", { x: M, y: 3.35, w: 8.4, h: 0.3, fontSize: 12, bold: true, color: C.accent4, charSpacing: 2 });
T(s, "Converting a model to 8-bit floating point (FP8) makes it smaller and cheaper to run, but the rounding costs accuracy.", { x: M, y: 3.7, w: 8.4, h: 0.95, fontSize: 17, color: C.text1 });
T(s, "THE APPROACH", { x: M, y: 4.95, w: 8.4, h: 0.3, fontSize: 12, bold: true, color: C.accent4, charSpacing: 2 });
T(s, "Quantization-aware training (QAT) applies the same FP8 rounding while the model is being fine-tuned, so it can adapt. We compare it with normal BF16 fine-tuning and change nothing else.", { x: M, y: 5.3, w: 8.4, h: 1.4, fontSize: 17, color: C.text1 });
const facts = [
  ["MODEL", "Qwen2.5-0.5B-Instruct"],
  ["PARAMETERS", "494 million"],
  ["SIZE, BF16 TO FP8", f2(sizes.bf16) + " GB to " + f2(sizes.fp8) + " GB"],
  ["DATASET", "GSM8K maths: 7,000 train, 1,319 test"],
];
facts.forEach((r, i) => {
  const y = 1.0 + i * 1.45;
  if (i > 0) hair(s, 9.6, y - 0.18, 3.13);
  T(s, r[0], { x: 9.6, y, w: 3.13, h: 0.28, fontSize: 11, bold: true, color: C.accent4, charSpacing: 2 });
  T(s, r[1], { x: 9.6, y: y + 0.32, w: 3.13, h: 0.8, fontSize: 20, bold: true, color: C.text1 });
});
s.addNotes("The problem: FP8 makes a model smaller but costs accuracy. The idea: train with the FP8 rounding switched on (QAT) so the model adapts to it. The model is Qwen2.5-0.5B-Instruct, 494 million parameters, 0.92 GB in BF16 and 0.59 GB in FP8, fine-tuned on GSM8K grade-school maths.");

// ================================================= 2. Architecture =================================================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and approach" });
s.addText("Same pipeline, one different training step", { placeholder: "title" });
image(s, A("architecture.png"), M, 1.65, 7.15, "Two flow diagrams side by side. Left, normal BF16 fine-tuning. Right, QAT fine-tuning with FP8 fake quantization. Both go from the Qwen model and GSM8K data through training, FP8 quantization, the FP8 model and evaluation.");
const blocks = [
  ["Same start", "Both begin from the same Qwen2.5-0.5B-Instruct model and the same GSM8K data."],
  ["One difference", "QAT adds FP8 rounding to 168 layers in the forward pass. The weights themselves stay in BF16."],
  ["Same finish", "Both are converted with the same FP8 tool and scored on the same 1,319 test questions."],
  ["Repeated", "Each method is trained " + nSeeds + " times with different seeds, for 3 epochs and for 1 epoch."],
];
blocks.forEach((b, i) => {
  const y = 1.7 + i * 1.3;
  T(s, b[0], { x: 8.3, y, w: 4.43, h: 0.35, fontSize: 18, bold: true, color: C.text1 });
  T(s, b[1], { x: 8.3, y: y + 0.4, w: 4.43, h: 0.8, fontSize: 15, color: C.text2 });
});
s.addNotes("Both pipelines are identical except for the training step. In QAT the forward pass sees FP8 rounding on the 168 main matrix layers, while the stored weights stay BF16. Both models are then converted with the same FP8 tool and evaluated the same way.");

// ================================================= 3. Accuracy =====================================================
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("QAT keeps more accuracy after FP8", { placeholder: "title" });
image(s, A("accuracy.png"), M, 1.65, 7.6, "Bar graph of accuracy after FP8 conversion for normal and QAT fine-tuning, at 3 epochs and at 1 epoch.");
T(s, sg(gain3), { x: 8.7, y: 1.55, w: 4.03, h: 0.95, fontSize: 54, bold: true, color: C.text1, fontFace: "+mj-lt" });
T(s, "points at 3 epochs. QAT is better in " + pos3 + " of " + n3 + " seeds.", { x: 8.7, y: 2.55, w: 4.03, h: 0.7, fontSize: 15, color: C.text2 });
T(s, sg(gain1), { x: 8.7, y: 3.4, w: 4.03, h: 0.95, fontSize: 54, bold: true, color: C.text1, fontFace: "+mj-lt" });
T(s, "points at 1 epoch. QAT is better in " + pos1 + " of " + n1 + " seeds.", { x: 8.7, y: 4.4, w: 4.03, h: 0.7, fontSize: 15, color: C.text2 });
T(s, "The direction is consistent, but one 1,319-question test set cannot pin down the size: both 95% intervals reach zero.", { x: 8.7, y: 5.3, w: 4.03, h: 1.3, fontSize: 14, color: C.text1 });
s.addNotes("Accuracy after FP8: normal " + f1(S3.acc.B.mean) + "% versus QAT " + f1(S3.acc.D.mean) + "% at 3 epochs; " + f1(S1.acc.B.mean) + "% versus " + f1(S1.acc.D.mean) + "% at 1 epoch. Bars are averages over seeds, dots are individual runs. The 95% intervals are " + sg(S3.primary.ci95_seeds_and_questions[0]) + " to " + sg(S3.primary.ci95_seeds_and_questions[1]) + " and " + sg(S1.primary.ci95_seeds_and_questions[0]) + " to " + sg(S1.primary.ci95_seeds_and_questions[1]) + " points, so the size of the gain is not settled.");

// ================================================= 4. KL divergence ================================================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("FP8 disturbs QAT models " + Math.round(klPct) + "% less", { placeholder: "title" });
image(s, A("kl.png"), M, 1.6, 7.6, "Bar graph of KL divergence between BF16 and FP8 predictions for normal and QAT fine-tuning at 3 epochs. QAT is 18 percent lower.");
T(s, Math.round(klPct) + "%", { x: 8.7, y: 1.55, w: 4.03, h: 0.95, fontSize: 54, bold: true, color: C.text1, fontFace: "+mj-lt" });
T(s, "lower KL divergence with QAT, 3 epochs", { x: 8.7, y: 2.55, w: 4.03, h: 0.4, fontSize: 15, color: C.text2 });
bulletList(s, [
  "KL measures how far FP8 moves a model's predictions from its own BF16 predictions. Lower is better.",
  "Lower in all " + klLower("3ep") + " of " + n3 + " seeds, and the ranges do not overlap.",
  "The top predicted word is unchanged more often: " + f1(agree("3epqat")) + "% (QAT) vs " + f1(agree("3epbf16")) + "% (normal).",
], { x: 8.7, y: 3.25, w: 4.03, h: 2.7, fontSize: 14 });
T(s, "FP8 is simulated here, and QAT trained against the same simulation.", { x: 8.7, y: 6.1, w: 4.03, h: 0.6, fontSize: 12, color: C.text2 });
s.addNotes("KL divergence compares each model's BF16 output distribution with its own FP8 output distribution at every answer token, about 123 thousand per model. QAT gives " + f4(klMean("3epqat")) + " against " + f4(klMean("3epbf16")) + " for normal training. At 1 epoch the difference is smaller (" + f4(klMean("1epqat")) + " vs " + f4(klMean("1epbf16")) + ") but QAT is again lower in every seed. Caveat: the FP8 here is the same simulation QAT trained against; the real-FP8 loss and accuracy results point the same way.");

// ================================================= 5. Loss ==========================================================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Validation loss is lowest after one epoch", { placeholder: "title" });
image(s, A("val_loss.png"), M, 1.75, 7.6, "Line graph of validation loss by epoch for normal and QAT fine-tuning over three epochs. It falls to about 0.54 after epoch 1, then rises.");
T(s, "Loss bottoms out near " + f2(curveMean(S3, "bf16_full", 1)) + " after epoch 1, then rises: three epochs overfit. At epoch 3 it is " + f3(curveMean(S3, "bf16_full", 3)) + " for normal training and " + f3(curveMean(S3, "qat_full", 3)) + " for QAT.", { x: 8.7, y: 1.7, w: 4.03, h: 1.5, fontSize: 15, color: C.text1 });
T(s, "Increase in test loss caused by FP8 (lower is better)", { x: 8.7, y: 3.35, w: 4.03, h: 0.6, fontSize: 14, bold: true, color: C.text1 });
const hd = { bold: true, color: "52514E", fontSize: 13, align: "right", border: [{ type: "none" }, { type: "none" }, { pt: 1, color: "898781" }, { type: "none" }] };
const cell = (x, o) => ({ text: x, options: Object.assign({ fontSize: 15, color: "0B0B0B", align: "right", border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: "E1E0D9" }, { type: "none" }] }, o || {}) });
s.addTable([
  [{ text: "", options: hd }, { text: "Normal", options: hd }, { text: "QAT", options: hd }],
  [cell("3 epochs", { align: "left" }), cell("+" + f4(lp(S3, "bf16_trained_B_minus_A"))), cell("+" + f4(lp(S3, "qat_trained_D_minus_C")), { bold: true })],
  [cell("1 epoch", { align: "left" }), cell("+" + f4(lp(S1, "bf16_trained_B_minus_A"))), cell("+" + f4(lp(S1, "qat_trained_D_minus_C")), { bold: true })],
], { x: 8.7, y: 4.05, w: 4.03, colW: [1.43, 1.3, 1.3], rowH: 0.45, fontFace: "+mn-lt", margin: [0.04, 0.05, 0.04, 0.05] });
s.addNotes("Validation loss is lowest after epoch 1 and rises afterwards while training loss keeps falling, so three epochs overfit this small model. The QAT value is measured with its FP8 rounding on. The table shows how much real FP8 conversion raises the test loss: about a third as much with QAT at 3 epochs, about half as much at 1 epoch.");

// ================================================= 6. Model size ===================================================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("FP8 makes the model " + Math.round(savedPct) + "% smaller", { placeholder: "title" });
image(s, A("size.png"), M, 1.6, 7.6, "Bar graph of model size: 0.92 GB in BF16 and 0.59 GB in FP8, which is 36 percent smaller.");
T(s, "Qwen2.5-0.5B-Instruct", { x: 8.7, y: 1.65, w: 4.03, h: 0.45, fontSize: 22, bold: true, color: C.text1 });
T(s, "494 million parameters", { x: 8.7, y: 2.15, w: 4.03, h: 0.4, fontSize: 16, color: C.text2 });
T(s, f2(sizes.bf16) + " GB to " + f2(sizes.fp8) + " GB", { x: 8.7, y: 2.95, w: 4.03, h: 0.75, fontSize: 26, bold: true, color: C.text1, fontFace: "+mj-lt" });
bulletList(s, [
  "Identical for the normal and the QAT checkpoints.",
  "The embedding table (136 million parameters) stays in BF16, so the saving is less than half.",
  "No speed-up was measured at this model size.",
], { x: 8.7, y: 4.0, w: 4.03, h: 2.7, fontSize: 14 });
s.addNotes("Sizes are measured from the saved checkpoints: " + sizes.bf16 + " GB in BF16 and " + sizes.fp8 + " GB in FP8. A quick benchmark showed no measurable FP8 speed-up for a model this small, so speed was left out of scope.");

// ================================================= 7. Conclusion ===================================================
pres.addSection({ title: "Wrap-up" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
s.addText("Conclusion", { placeholder: "title" });
const klPairs = klLower("3ep") + klLower("1ep");
const cols = [
  ["QAT models change less under FP8", "KL divergence is " + Math.round(klPct) + "% lower at 3 epochs, and lower in all " + klPairs + " seed pairs. The loss increase from FP8 is a third to a half as large."],
  ["About " + sg(gain1, 1) + " to " + sg(gain3, 1) + " points of accuracy", "After FP8: " + sg(gain3) + " at 3 epochs and " + sg(gain1) + " at 1 epoch. The direction is consistent, but not settled at 95% confidence."],
  ["A model " + Math.round(savedPct) + "% smaller", "Qwen2.5-0.5B-Instruct (494M parameters) goes from " + f2(sizes.bf16) + " GB to " + f2(sizes.fp8) + " GB. No speed-up was measured at this size."],
];
cols.forEach((c, i) => {
  const x = M + i * 4.1;
  T(s, String(i + 1), { x, y: 1.75, w: 3.7, h: 0.95, fontSize: 54, bold: true, color: C.accent4, fontFace: "+mj-lt" });
  T(s, c[0], { x, y: 2.85, w: 3.7, h: 0.95, fontSize: 20, bold: true, color: C.text1 });
  T(s, c[1], { x, y: 3.9, w: 3.7, h: 1.9, fontSize: 15, color: C.text2 });
});
T(s, "The untouched original model still scores higher (" + f1(S3.baselines.orig_bf16.accuracy) + "% to " + f1(S3.baselines.orig_bf16_h100.accuracy) + "%) than either fine-tuned version: fine-tuning on GSM8K did not improve this small model.", { x: M, y: 6.1, w: W, h: 0.6, fontSize: 14, color: C.text2 });
s.addNotes("Three takeaways: QAT reliably reduces how much FP8 changes the model; that shows up as roughly one to one and a half points of accuracy, consistent in direction but not statistically settled on one 1,319-question test set; and FP8 makes the model 36 percent smaller. Honest caveat: the untouched original model still scores higher than either fine-tuned variant.");

// ================================================= 8. Next steps ===================================================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
s.addText("Next steps", { placeholder: "title" });
const nxt = [
  ["A larger test set", "Evaluate on a second, larger maths set to tighten the accuracy interval."],
  ["Harder compression", "Try lower precision such as FP4 or INT4, where plain conversion loses much more accuracy and QAT has more to recover."],
  ["Fewer epochs", "Try early stopping, since validation loss was lowest after one epoch."],
  ["Real FP8 kernels", "Measure KL divergence with the deployed FP8 kernels instead of the simulation."],
];
nxt.forEach((n, i) => {
  const x = M + (i % 2) * 6.2, y = 1.9 + Math.floor(i / 2) * 2.35;
  T(s, n[0], { x, y, w: 5.6, h: 0.5, fontSize: 22, bold: true, color: C.text1 });
  T(s, n[1], { x, y: y + 0.6, w: 5.6, h: 1.3, fontSize: 16, color: C.text2 });
});
s.addNotes("The main limit now is the size of the test set, so a second maths test set is the most useful next step. Lower precision such as FP4 or INT4 should show a much bigger penalty; larger models usually lose less to FP8, not more. Validation loss suggests one epoch may be enough. KL divergence should be re-measured with the real FP8 kernels.");

const OUT = path.join(ROOT, "final_report", "QAT_vs_BF16_FP8_presentation.pptx");
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
