# QAT vs normal BF16 fine-tuning, deployed as FP8 - accuracy report

Qwen2.5-0.5B-Instruct, full fine-tuning (no LoRA), GSM8K (7,000 train / 473 val / 1,319 test), 1 epoch, NVIDIA H100, FP8 = llm-compressor FP8_DYNAMIC served by vLLM. 5 seeds per arm (seeds 0, 1, 2, 3, 4). All numbers below are computed from the result files.

## Headline

- **BF16 training -> FP8:** 35.12 ± 1.05% test accuracy
- **QAT training -> FP8:** 35.98 ± 0.57% test accuracy
- **QAT improvement: +0.86 percentage points** (+2.5% relative). Per seed: +1.82, -0.99, -0.45, +1.59, +2.35.
- 95% interval for the improvement: **[-0.91, +2.59] pp** (resampling seeds and test questions); [-0.21, +1.93] pp if only the test questions are resampled.
- QAT-trained models score higher on average, but not in every seed; the evidence is weak.

## Accuracy matrix (test accuracy %, mean ± std over seeds)

| Training | Inference | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Mean ± std |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 full | BF16 | 36.92 | 35.41 | 36.32 | 36.01 | 36.01 | 36.13 ± 0.55 |
| BF16 full | FP8 | 34.19 | 36.47 | 36.01 | 34.34 | 34.57 | 35.12 ± 1.05 |
| QAT full | BF16 | 37.07 | 36.54 | 36.54 | 35.78 | 34.72 | 36.13 ± 0.91 |
| QAT full | FP8 | 36.01 | 35.48 | 35.56 | 35.94 | 36.92 | 35.98 ± 0.57 |

The primary comparison is **B vs D**, because both are final FP8 deployment models.

## Primary table (plan section 22; speed rows left out on purpose)

| Metric | BF16 Full -> FP8 | QAT Full -> FP8 |
|---|---:|---:|
| Accuracy % | 35.12 ± 1.05 | 35.98 ± 0.57 |
| Eval loss (lower is better) | 0.6204 ± 0.0007 | 0.6177 ± 0.0005 |
| Mean generated tokens | 100.6 | 99.3 |
| Model size on disk | same FP8 checkpoint format for both (0.59 GB; BF16 is 0.92 GB) | same |

## How reliable is the gain?

| Check | Result |
|---|---|
| Every seed positive | no |
| Worst QAT-FP8 seed beats best BF16-FP8 seed | no (QAT min 35.48, BF16 max 36.47) |
| Paired t across seeds (df=4) | t = 1.30, 5% critical value 2.776: not significant (few seeds, low power) |
| McNemar test per seed (single model vs single model, same questions) | seed 0: p=0.129, seed 1: p=0.445, seed 2: p=0.751, seed 3: p=0.199, seed 4: p=0.042 (none is significant on its own - one 1,319-question test set cannot separate two single models at this size) |
| 95% interval, seeds + questions | [-0.91, +2.59] pp |

## Where does the gain come from?

Gain (D-B) = QAT effect before FP8 (C-A) + reduction of the FP8 accuracy drop.

| Component (pp) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| QAT effect with no FP8 (C - A) | +0.15 | +1.14 | +0.23 | -0.23 | -1.29 | -0.00 ± 0.88 |
| FP8 drop, BF16-trained (A - B) | +2.73 | -1.06 | +0.30 | +1.67 | +1.44 | +1.02 ± 1.45 |
| FP8 drop, QAT-trained (C - D) | +1.06 | +1.06 | +0.99 | -0.15 | -2.20 | +0.15 ± 1.41 |
| Drop reduction ((A-B) - (C-D)) | +1.67 | -2.12 | -0.68 | +1.82 | +3.64 | +0.86 ± 2.27 |

About -0.00 pp of the gain appears even before FP8 (QAT models are better in plain BF16 too, likely a regularization effect), and +0.86 pp comes from losing less accuracy in the FP8 conversion. Both parts vary between seeds.

## Eval loss (teacher-forced, per answer token; less noisy than accuracy)

| Cell | Mean ± std |
|---|---:|
| BF16 -> BF16 | 0.6147 ± 0.0006 |
| BF16 -> FP8 | 0.6204 ± 0.0007 |
| QAT -> BF16 | 0.6148 ± 0.0012 |
| QAT -> FP8 | 0.6177 ± 0.0005 |

FP8 raises the loss by +0.0056 for BF16-trained models and +0.0029 for QAT-trained models (per seed: +0.0058/+0.0033, +0.0050/+0.0021, +0.0057/+0.0020, +0.0062/+0.0039, +0.0053/+0.0034 BF16/QAT).

## Reference: the original, untrained model

| Model | Accuracy % | Eval loss |
|---|---:|---:|
| Original, BF16 (L4) | 38.74 | 0.8156 |
| Original, FP8 (L4) | 37.60 | 0.8258 |
| Original, BF16 (H100) | 39.95 | 0.8165 |

The untouched model scores higher than both fine-tuned arms here. Fine-tuning on GSM8K's short human-written solutions did not help this small model (and the same model's BF16 score differs by about 1.2 pp between the L4 and the H100, a reminder of how much GPU-level numerics alone can move a single accuracy number).

## Training behaviour

- bf16_full: mean validation loss by epoch (0..1) = 0.7121, 0.5216
- qat_full: mean validation loss by epoch (0..1) = 0.7118, 0.5294

Single-epoch run: the checkpoint after epoch 1 is evaluated (no early stopping).

## Limitations

- 5 seeds per arm; seeds change the data order and rounding randomness but not the starting weights.
- One test set of 1,319 questions: a single model's accuracy is only known to about ±1.3 pp.
- One dataset, one model size, one FP8 recipe (E4M3, per-channel weights, per-token activations).
- The QAT quantizer reproduces the deployed weights bit for bit; activation arithmetic matches the deployment in kind but the full deployed loss increase was only about 66-76% reproduced in a diagnostic on the original model.
- FP8 numbers are for the H100; the L4 and H100 can give slightly different FP8 results.
