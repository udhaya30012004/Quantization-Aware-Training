# QAT vs normal BF16 fine-tuning, deployed as FP8 - accuracy report

Qwen2.5-0.5B-Instruct, full fine-tuning (no LoRA), GSM8K (7,000 train / 473 val / 1,319 test), 3 epochs, NVIDIA H100, FP8 = llm-compressor FP8_DYNAMIC served by vLLM. 5 seeds per arm (seeds 0, 1, 2, 3, 4). All numbers below are computed from the result files.

## Headline

- **BF16 training -> FP8:** 33.98 ± 0.27% test accuracy
- **QAT training -> FP8:** 35.45 ± 0.42% test accuracy
- **QAT improvement: +1.47 percentage points** (+4.3% relative). Per seed: +1.67, +1.44, +1.82, +1.97, +0.45.
- 95% interval for the improvement: **[-0.15, +3.03] pp** (resampling seeds and test questions); [+0.29, +2.62] pp if only the test questions are resampled.
- QAT-trained models score higher after FP8 deployment in every seed, but once test-set sampling is included the interval reaches zero, so the size of the gain is uncertain.

## Accuracy matrix (test accuracy %, mean ± std over seeds)

| Training | Inference | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Mean ± std |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 full | BF16 | 35.78 | 35.18 | 35.33 | 34.12 | 36.09 | 35.30 ± 0.75 |
| BF16 full | FP8 | 33.74 | 33.89 | 33.81 | 34.04 | 34.42 | 33.98 ± 0.27 |
| QAT full | BF16 | 36.47 | 37.00 | 35.18 | 35.78 | 36.01 | 36.09 ± 0.69 |
| QAT full | FP8 | 35.41 | 35.33 | 35.63 | 36.01 | 34.87 | 35.45 ± 0.42 |

The primary comparison is **B vs D**, because both are final FP8 deployment models.

## Primary table (plan section 22; speed rows left out on purpose)

| Metric | BF16 Full -> FP8 | QAT Full -> FP8 |
|---|---:|---:|
| Accuracy % | 33.98 ± 0.27 | 35.45 ± 0.42 |
| Eval loss (lower is better) | 0.7558 ± 0.0046 | 0.7226 ± 0.0024 |
| Mean generated tokens | 93.6 | 92.9 |
| Model size on disk | same FP8 checkpoint format for both (0.59 GB; BF16 is 0.92 GB) | same |

## How reliable is the gain?

| Check | Result |
|---|---|
| Every seed positive | yes |
| Worst QAT-FP8 seed beats best BF16-FP8 seed | yes (QAT min 34.87, BF16 max 34.42) |
| Paired t across seeds (df=4) | t = 5.47, 5% critical value 2.776: significant (few seeds, low power) |
| McNemar test per seed (single model vs single model, same questions) | seed 0: p=0.211, seed 1: p=0.283, seed 2: p=0.141, seed 3: p=0.136, seed 4: p=0.769 (none is significant on its own - one 1,319-question test set cannot separate two single models at this size) |
| 95% interval, seeds + questions | [-0.15, +3.03] pp |

## Where does the gain come from?

Gain (D-B) = QAT effect before FP8 (C-A) + reduction of the FP8 accuracy drop.

| Component (pp) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| QAT effect with no FP8 (C - A) | +0.68 | +1.82 | -0.15 | +1.67 | -0.08 | +0.79 ± 0.93 |
| FP8 drop, BF16-trained (A - B) | +2.05 | +1.29 | +1.52 | +0.08 | +1.67 | +1.32 ± 0.75 |
| FP8 drop, QAT-trained (C - D) | +1.06 | +1.67 | -0.45 | -0.23 | +1.14 | +0.64 ± 0.93 |
| Drop reduction ((A-B) - (C-D)) | +0.99 | -0.38 | +1.97 | +0.30 | +0.53 | +0.68 ± 0.87 |

About +0.79 pp of the gain appears even before FP8 (QAT models are better in plain BF16 too, likely a regularization effect), and +0.68 pp comes from losing less accuracy in the FP8 conversion. Both parts vary between seeds.

## Eval loss (teacher-forced, per answer token; less noisy than accuracy)

| Cell | Mean ± std |
|---|---:|
| BF16 -> BF16 | 0.7462 ± 0.0014 |
| BF16 -> FP8 | 0.7558 ± 0.0046 |
| QAT -> BF16 | 0.7192 ± 0.0035 |
| QAT -> FP8 | 0.7226 ± 0.0024 |

FP8 raises the loss by +0.0097 for BF16-trained models and +0.0033 for QAT-trained models (per seed: +0.0144/+0.0079, +0.0090/+0.0009, +0.0037/+0.0042, +0.0143/+0.0017, +0.0070/+0.0018 BF16/QAT).

## Reference: the original, untrained model

| Model | Accuracy % | Eval loss |
|---|---:|---:|
| Original, BF16 (L4) | 38.74 | 0.8156 |
| Original, FP8 (L4) | 37.60 | 0.8258 |
| Original, BF16 (H100) | 39.95 | 0.8165 |

The untouched model scores higher than both fine-tuned arms here. Fine-tuning on GSM8K's short human-written solutions did not help this small model (and the same model's BF16 score differs by about 1.2 pp between the L4 and the H100, a reminder of how much GPU-level numerics alone can move a single accuracy number).

## Training behaviour

- bf16_full: mean validation loss by epoch (0..3) = 0.7121, 0.5362, 0.5490, 0.6368
- qat_full: mean validation loss by epoch (0..3) = 0.7118, 0.5426, 0.5489, 0.6160

Validation loss is lowest after epoch 1 and rises afterwards while training loss keeps falling: three epochs overfit. The final checkpoint is evaluated as pre-registered (no early stopping).

## Limitations

- 5 seeds per arm; seeds change the data order and rounding randomness but not the starting weights.
- One test set of 1,319 questions: a single model's accuracy is only known to about ±1.3 pp.
- One dataset, one model size, one FP8 recipe (E4M3, per-channel weights, per-token activations).
- The QAT quantizer reproduces the deployed weights bit for bit; activation arithmetic matches the deployment in kind but the full deployed loss increase was only about 66-76% reproduced in a diagnostic on the original model.
- FP8 numbers are for the H100; the L4 and H100 can give slightly different FP8 results.
