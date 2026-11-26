# Full BF16 fine-tuning vs full FP8 QAT fine-tuning — Qwen2.5-0.5B-Instruct on GSM8K (NVIDIA L4, Modal)

## 0. Research question

> Using the same Qwen2.5-0.5B-Instruct model and the same GSM8K data, does full FP8 QAT fine-tuning give higher GSM8K accuracy **after FP8 deployment** than full BF16 fine-tuning followed by the **same** FP8 deployment?

**Primary result:** `BF16_FULL → FP8` (cell B) vs `QAT_FULL → FP8` (cell D), as mean ± std over 3 seeds, with a paired significance test.

---

## 1. Problems in the original draft and how this plan fixes them

| # | Problem in the original plan | Why it matters | Fix in this plan |
|---|---|---|---|
| 1 | **The "FP8 deployment method" is never defined.** | QAT only helps when the quantizer used in training matches the one used at deployment: same format, scale granularity and scaling mode. If the deployment method is left open, Run B trains against the wrong target. | The deployment method is fixed **before** any training: llm-compressor `FP8_DYNAMIC` (E4M3, static per-output-channel weight scales, dynamic per-token activation scales, `lm_head` ignored), served by vLLM. QAT copies exactly this quantizer. |
| 2 | **Pure BF16 master weights with a full-fine-tuning LR (~1e-5).** | BF16 has an 8-bit mantissa. For a weight of 0.03 the step between representable values is about 1.2e-4. An AdamW step of about 1e-5 rounds to zero, so most weights barely move and training silently stalls. This hits both arms. | Keep the weights in BF16 (user decision), but use **AdamW with BF16 stochastic rounding** (torchao `AdamW(..., bf16_stochastic_round=True)`; check the exact class name in the installed version). Log the fraction of weights changed per step as a smoke-test gate. Both arms get the same optimizer. |
| 3 | **Transformer Engine `fp8_autocast` is not QAT.** | TE uses per-tensor delayed amax scaling and also quantizes **gradients** to E5M2. Neither matches deployment, and the gradient noise is an extra confound. | Primary Run B: **fake-quant QAT**. The forward pass applies E4M3 quantize→dequantize with the deployment's exact granularity, the backward pass uses a straight-through estimator (STE) in BF16, and the loss is in FP32. TE FP8 training is an optional Run G. |
| 4 | **GSM8K has no validation split.** The draft also mixes a 2k-example subset with "full data". | Tuning on the test set leaks information. The 2k subset doesn't answer the real question. | Hold out a fixed 473 examples from train as validation, giving **7,000 train / 473 val / 1,319 test**. All 1,319 test questions are used. The 2k subset is only for the smoke test. |
| 5 | **No measure of headroom.** | If an untouched model barely loses accuracy in FP8, QAT has nothing to recover and a null result says nothing. | Add cells **E/F**: the original model evaluated in BF16 and in FP8. Gate in §9: if the FP8 drop is under 1 pp, add a stricter per-tensor FP8 scheme as a second deployment target. |
| 6 | **Only one run per arm.** | The test-set standard error is about ±1.4 pp, about the same size as the expected effect. | 3 seeds (0, 1, 2) per arm. Report mean ± std, a paired McNemar test per seed pair, and a bootstrap 95% CI on the B−D difference. |
| 7 | **"Accuracy" isn't defined for a generation task.** | Answer-extraction rules can change accuracy by several points. | One shared `extract_answer()`: if the last non-empty line contains `#### <number>` use it, otherwise take the last number in the output (an earlier `####`, such as the base model's premature guess or a markdown header, is ignored). Strip `,` and `$`, compare numerically. Greedy decoding, `max_tokens=512`. The same code is used for every cell. |
| 8 | **B vs D speed is identical by design.** | Same architecture plus the same FP8 kernels gives the same latency. Comparing them only reflects differences in how long the generated answers are. | The kernel benchmark uses fixed input/output lengths (`ignore_eos=True`). The useful speed comparison is **FP8 vs BF16**; B and D are reported side by side as a sanity check. End-to-end eval wall time and mean generated length are reported separately. |
| 9 | **"Peak VRAM" is meaningless under vLLM.** | vLLM preallocates `gpu_memory_utilization` of GPU memory for the KV cache. | Report **model weight memory** from the vLLM loader log and `torch.cuda.memory_allocated` after load, with `gpu_memory_utilization` fixed. Training peak VRAM comes from `torch.cuda.max_memory_allocated`. |
| 10 | **Separate `train_bf16.py` and `train_qat.py`.** | Two training loops drift apart (data order, scheduler, logging, bugs). | **One `train.py`**. The only difference between arms is the config key `quant: none \| fp8_fakequant \| te_fp8`. |
| 11 | **The model ID is ambiguous** ("Qwen2.5-0.5B" vs "-Instruct"). | "Same original checkpoint" needs an exact ID. | Use `Qwen/Qwen2.5-0.5B-Instruct`, pinned to an exact HF **revision hash** in `configs/base.yaml`, with the SHA256 of the downloaded safetensors recorded. |
| 12 | **The original plan has no eval loss on FP8.** | It's hard to compute teacher-forced loss on an FP8 vLLM engine. | Use vLLM `prompt_logprobs` on `prompt + reference answer`, summing the NLL over answer tokens only. This works the same way for BF16 and FP8 engines. |
| 13 | **Tied embeddings and `lm_head`.** | Qwen2.5-0.5B ties `lm_head` to `embed_tokens` (~136M of the 494M parameters). | `lm_head` and the embeddings stay BF16 in every quantized path (`ignore=["lm_head"]`), and QAT does not wrap them. |
| 14 | **"Model size" doesn't exist for online FP8.** | Without an exported FP8 checkpoint there is no size on disk. | llm-compressor writes a real FP8 checkpoint. Report the on-disk size of the BF16 and FP8 checkpoints. |
| 15 | **Training and serving are mixed in one environment.** | TE, vLLM and llm-compressor have conflicting dependency pins. | Separate Modal images: `train_image` (torch + transformers + torchao, plus the NGC/TE image for the optional Run G) and `serve_image` (vLLM only) and `export_image` (llm-compressor only); putting both in one image failed to resolve (pip backtracked to vllm 0.1.3). One shared Modal Volume. |

---

## 2. Experiment matrix

| Cell | Training | Inference | Purpose |
|---|---|---|---|
| E | none (original) | BF16 | zero-shot baseline |
| F | none (original) | FP8 | headroom: the PTQ gap before fine-tuning |
| A | BF16 full FT | BF16 | Run A reference |
| **B** | BF16 full FT | **FP8** | **primary baseline** |
| C | QAT full FT | BF16 | effect of QAT itself |
| **D** | QAT full FT | **FP8** | **primary treatment** |
| G/H (optional) | TE FP8 training | BF16 / FP8 | does plain FP8 training help? |

Cells A–D (and G/H) are run for seeds 0, 1 and 2.

Derived metrics:

- Quantization gap for BF16 training: `A − B`
- Quantization gap for QAT: `C − D`
- **QAT gain:** `D − B` (pp)
- Cost of QAT before quantization: `A − C`

---

## 3. Fixed components (identical across all arms)

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision. It is downloaded once to `/vol/checkpoints/original/` and never written to again.
- **Data:** GSM8K `main` config from the HF hub (`openai/gsm8k`), pinned revision.
  - Split: 7,000 train / 473 val / 1,319 test. Val is a seeded shuffle (seed 1234) of the train set; test is untouched.
  - Saved once as `train.jsonl`, `val.jsonl` and `test.jsonl` with SHA256 hashes in `data/manifest.json`. Every job checks the hashes and **never regenerates** the split.
- **Preprocessing:**
  - Strip the calculator annotations `<<...>>` from answers.
  - Target format: the reasoning, then `\n#### <answer>`.
  - Qwen chat template, same system prompt for every arm: `"Solve the math problem step by step. End with '#### <number>'."`
  - **Loss only on assistant tokens** (prompt tokens masked to -100).
  - `max_seq_length=512`. Log the truncation count; it is expected to be 0.
- **Training hyperparameters (both arms):**
  - Epochs: 3
  - Batch: micro-batch 8 × grad accumulation 4 = effective 32
  - LR 1e-5, cosine schedule, 3% warmup, weight decay 0.0, grad clip 1.0
  - AdamW (β = 0.9, 0.95) with BF16 stochastic rounding
  - Gradient checkpointing off (enable for **both** arms if memory requires it)
  - Same seed sets the data order, dropout and init
  - Final checkpoint is evaluated with no early stopping. Val loss is logged only.
- **Precision:**
  - Parameters, gradients and optimizer states are BF16.
  - Loss is computed in FP32 (logits upcast before cross-entropy).
  - No FP8 optimizer states and no FP8 loss.
- **FP8 deployment (frozen):**
  - Conversion: llm-compressor `oneshot` with `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])`. It needs no calibration data and is deterministic.
  - Output: a compressed-tensors FP8 checkpoint.
  - Serving: vLLM.
- **Evaluation:**
  - vLLM, `temperature=0`, `max_tokens=512`, fixed `seed`, identical engine args for BF16 and FP8 apart from the checkpoint path.
  - The full 1,319-question test set.
  - Per-question correctness is saved for paired statistics.

---

## 4. QAT design (Run B, primary)

`models/fp8_fake_quant.py` defines `FP8FakeQuantLinear(nn.Linear)`. It wraps all 168 projections: 24 layers × {q, k, v, o, gate, up, down}. **Embeddings, RMSNorm, softmax, `lm_head` and the loss are not wrapped.**

**Weights:** per-output-channel E4M3. The quantizer matches FP8_DYNAMIC's static channel-wise weight scheme.

```
s_w  = amax(|W|, dim=in) / 448
W_q  = (W / s_w).clamp(±448).to(float8_e4m3fn).to(fp32) * s_w
```

**Activations:** per-token dynamic E4M3, matching FP8_DYNAMIC's dynamic token-wise input scheme.

```
s_x = amax(|x|, dim=-1) / 448
```

then the same quantize→dequantize as for weights.

**Implementation rules:**

- Quant math runs in FP32. The result is cast back to BF16 before the matmul; the bias is added in BF16.
- STE: `x + (q(x) − x).detach()`. The scale is detached.
- **Before any training**, check the quantizer against llm-compressor: wrap the **original** model with fake-quant in eval mode and compare its outputs to the vLLM FP8 engine on the same prompts. Logits should agree closely and greedy outputs should mostly match. If they don't, the quantizer is wrong — fix it before training.
- **Saving:** unwrap back to plain `nn.Linear` and `save_pretrained` in BF16 HF format. That output goes through the same llm-compressor → vLLM path as Run A.

Optional Run G (TE): swap the projections for `te.Linear` and train under `te.fp8_autocast(recipe=DelayedScaling(fp8_format=Format.HYBRID))` on the NGC PyTorch image. Unwrap and save the same way. Report it as secondary.

---

## 5. File structure

```
qat/
├── plan.md                    # this document
├── README.md                  # how to run, results summary
├── requirements-local.txt     # local: modal, pandas, matplotlib, scipy
├── .gitignore                 # venv/, checkpoints/*, __pycache__, *.safetensors
├── modal_app.py               # Modal App: images, Volume("fp8-qat"), GPU="L4" functions, entrypoints
│
├── configs/
│   ├── base.yaml              # model id+revision, data paths+hashes, HPs, seeds, eval/bench settings
│   ├── bf16_full.yaml         # quant: none
│   ├── qat_fp8.yaml           # quant: fp8_fakequant (weight: per_channel, act: per_token, e4m3)
│   ├── te_fp8.yaml            # quant: te_fp8 (optional)
│   └── smoke.yaml             # 32 samples, 10 steps, overrides
│
├── common/
│   ├── config.py              # YAML load + merge (base ← arm ← CLI overrides)
│   ├── seed.py                # seed everything, deterministic flags
│   └── env_report.py          # GPU, CUDA, torch/transformers/vllm/llmcompressor/TE versions → results/env.json
│
├── data/
│   ├── prepare_dataset.py     # download GSM8K, strip <<>>, fixed split, write jsonl + manifest.json
│   ├── dataset.py             # chat-template formatting, tokenization, assistant-only label masking, collator
│   └── manifest.json          # split sizes + SHA256 (committed)
│
├── models/
│   ├── load_model.py          # load pinned original in BF16, verify hash, all params trainable
│   ├── fp8_fake_quant.py      # FP8FakeQuantLinear, wrap()/unwrap(), STE
│   ├── te_fp8.py              # optional TE swap/unswap
│   └── export_fp8.py          # llm-compressor FP8_DYNAMIC conversion (the ONE deployment path)
│
├── training/
│   └── train.py               # single training loop for all arms; logs loss/val-loss/VRAM/tok/s/updates-changed
│
├── evaluation/
│   ├── evaluate.py            # vLLM generation on test → per-question jsonl + summary json; eval loss via prompt_logprobs
│   ├── metrics.py             # extract_answer(), accuracy, McNemar, bootstrap CI
│   └── verify_fp8.py          # FP8 diagnostics (see §7)
│
├── benchmarks/
│   └── benchmark.py           # load time, warm-up, latency avg/P50/P95, throughput, tokens/s, weight memory
│
├── analysis/
│   ├── compare.py             # aggregate results/*.json → comparison.csv, matrix table, report.md
│   └── plots.py               # accuracy bar chart w/ CIs, gap chart, latency chart → plots/
│
├── tests/
│   ├── test_extract_answer.py
│   ├── test_fake_quant.py     # quant error bounds, STE grad = identity, wrap/unwrap round-trip
│   └── test_dataset.py        # label masking, no truncation, split hashes
│
├── checkpoints/               # local mirror only (gitignored); canonical copies live on the Modal Volume
│   ├── original/
│   ├── bf16_full/             # seed_{0,1,2}/  (+ fp8/ subdir per seed)
│   └── qat_full/              # seed_{0,1,2}/  (+ fp8/ subdir per seed)
│
├── results/
│   ├── env.json
│   ├── smoke/
│   ├── orig_bf16.json  orig_fp8.json
│   ├── bf16_bf16_s{0,1,2}.json   bf16_fp8_s{0,1,2}.json
│   ├── qat_bf16_s{0,1,2}.json    qat_fp8_s{0,1,2}.json
│   ├── per_question/          # one jsonl per cell+seed, for paired stats
│   ├── bench_*.json
│   ├── comparison.csv
│   └── report.md
│
└── plots/
```

Modal Volume layout: `/vol/hf_cache`, `/vol/data`, `/vol/checkpoints/...` (mirrors the tree above), `/vol/results`. Pull results to the local machine with `modal volume get fp8-qat results ./results`.

---

## 6. Development order (with gates)

| Step | Action | Gate to pass before moving on |
|---|---|---|
| 1 | `env_report` on a Modal L4: capability (8, 9), CUDA, library versions. Run a `torch._scaled_mm` FP8 sanity matmul. Pin versions. | FP8 GEMM runs on the L4. |
| 2 | Download and pin the original model and GSM8K, write the split and manifest. | Hashes recorded; sizes are 7000/473/1319. |
| 3 | Unit tests: answer extraction, label masking, fake-quant. | All tests pass. |
| 4 | Export the original model to FP8 (`export_fp8.py`) and run `verify_fp8.py`. | FP8 dtypes and scales present; weight memory roughly halved for the Linear layers. |
| 5 | Evaluate cells **E and F**. | Headroom known (see §9). |
| 6 | Check fake-quant against the vLLM FP8 engine (§4). | Outputs agree. |
| 7 | Smoke test, `quant=none`: 32 samples, 10 steps. | Loss decreases, gradients are finite, weights change in more than 50% of parameters per step (proves stochastic rounding works), save → reload → identical logits. |
| 8 | Smoke test, `quant=fp8_fakequant`: same checks, plus 168 modules wrapped, nonzero quant error logged per layer, wrapped and unwrapped logits differ, unwrapped save loads in vanilla HF. | All pass. |
| 9 | Full Run A (BF16), seeds 0–2. These can run as parallel Modal jobs. | Checkpoints saved; val loss logged. |
| 10 | Full Run B (QAT), seeds 0–2, **starting from `/vol/checkpoints/original`**. | Checkpoints saved; the bf16_full path is never read. |
| 11 | Export every checkpoint to FP8 with the **same** `export_fp8.py` and verify each one. | — |
| 12 | Evaluate cells A, B, C, D for each seed. | — |
| 13 | Benchmarks (§8). | — |
| 14 | (Optional) Run G/H with TE. | — |
| 15 | `compare.py` + `plots.py` → `comparison.csv` and `report.md`. | Every number traces back to a results json. |

---

## 7. FP8 verification (`verify_fp8.py`)

- **Exported checkpoint:**
  - Every target Linear weight has dtype `float8_e4m3fn` and a `weight_scale` tensor (per-channel shape `[out, 1]`).
  - `lm_head` and the embeddings are BF16.
  - `quantization_config` is present in `config.json`.
- **vLLM:**
  - The loader log names the compressed-tensors W8A8-FP8 scheme.
  - Model weight memory for FP8 is lower than for BF16.
  - Outputs differ from BF16 on a fixed probe set, which shows quantization is actually active.
- **QAT training:**
  - Per-layer wrapped-module count.
  - Per-layer weight and activation amax and scale statistics, plus relative quantization error, logged every N steps.
  - Assert that the quantize→dequantize path ran: hook counters must be greater than 0.

---

## 8. Benchmark protocol (identical for every cell)

- vLLM, fixed engine args: `gpu_memory_utilization=0.85`, `max_model_len=1024`, and the same `enforce_eager` setting for every cell.
- **Load time:** wall clock from engine construction until it is ready.
- **Warm-up:** 3 iterations, timed separately.
- **Latency and throughput:** 20 timed iterations per setting.
  - Fixed synthetic prompts, input length 256, output length 256, `ignore_eos=True`.
  - Batch sizes {1, 8, 32}.
  - `torch.cuda.synchronize()` around each timed region.
  - Report avg, P50 and P95 latency, requests/s, and generated tokens/s.
- **Memory:** model weight memory, plus peak `max_memory_allocated` for training jobs.
- **Model size:** on-disk size of the BF16 and FP8 checkpoints.
- Also report the end-to-end GSM8K eval wall time and the mean generated length for each cell.

Expect only a small FP8 speedup for a 0.5B model on an L4. Decoding is memory-bound and `lm_head` (27% of the parameters) stays in BF16. Report whatever is measured.

---

## 9. Headroom gate and extension

After step 5, compute `gap = acc(E) − acc(F)`:

- **Gap ≥ 1 pp:** proceed as planned.
- **Gap < 1 pp:** FP8_DYNAMIC is close to lossless here, and D vs B will probably be a null result. That is still a valid finding — report it. To stress-test QAT, add a second deployment scheme: llm-compressor `scheme="FP8"`, which uses static per-tensor weight and activation scales calibrated on 512 fixed train prompts, the same for every arm. Add a matching QAT config `qat_fp8_static.yaml` (per-tensor weights; per-tensor activations whose scale tracks a running amax during training and is frozen for export). Rerun B/D for that scheme.

---

## 10. Final report contents (`results/report.md`)

1. Environment and versions; model and dataset revisions with hashes.
2. Accuracy matrix: cells E, F, A, B, C, D (and G, H), mean ± std over seeds.
3. **Primary table:** BF16 Full → FP8 vs QAT Full → FP8.
   - Rows: accuracy, eval loss, avg/P50/P95 latency, throughput, tokens/s, weight memory, model size.
   - QAT gain `D − B` in pp, with a bootstrap 95% CI and McNemar p-values.
4. Quantization gaps `A − B` and `C − D`.
5. Training curves (loss, val loss), training throughput and peak VRAM per arm.
6. FP8 verification evidence.
7. Plots.

Every number is read from a results json; none are entered by hand.

---

## 11. Critical rules (amended)

1. Full fine-tuning only: no LoRA, QLoRA, PEFT or frozen layers.
2. Both arms start from the pinned original checkpoint, which is never written to.
3. One data split, generated once and verified by hash. Never tune on test.
4. One `train.py`. Arms differ only in the `quant` config key.
5. One FP8 deployment path (`export_fp8.py` + vLLM), fixed before training and identical for every cell.
6. The QAT quantizer must match the deployment quantizer, checked in step 6.
7. Master weights and optimizer states are BF16 with stochastic rounding. Loss is FP32. Nothing below 8 bits; the target is FP8 E4M3 only.
8. 3 seeds per arm; report variance and significance.
9. No fabricated metrics. Report null results as they are.

---

## Verification of this plan's execution

- `pytest tests/` passes locally (CPU-only tests) and on Modal.
- Smoke gates in steps 7 and 8 pass and their logs are saved to `results/smoke/`.
- `verify_fp8.py` passes for every exported checkpoint.
- `compare.py` regenerates `comparison.csv` and `report.md` from raw results with no manual edits.

---

## Appendix: measured environment (Step 1, passed)

Modal L4 (23 GB, compute capability 8.9, driver supports CUDA 13). A real `torch._scaled_mm` E4M3 GEMM ran in all three images, and the E4M3 quantize→dequantize round-trip had a relative error of 2.6%.

| Image | Pinned packages |
|---|---|
| `train_image` | torch 2.15.0 (cu132), transformers 5.23.0, accelerate 1.16.0, datasets 5.2.0, torchao 0.18.0 |
| `serve_image` | vllm 0.34.0 (brings torch 2.14.0 cu130, transformers 5.22.2, compressed-tensors 0.17.0) |
| `export_image` | llmcompressor 0.15.0 (brings torch 2.14.1 cu130, transformers 5.21.0, compressed-tensors 0.20.1) |

Open risk: the exporter writes with compressed-tensors 0.20.1 while vLLM reads with 0.17.0. Step 4 must confirm vLLM loads an exported checkpoint.

---

## Appendix: amendments made during implementation

- **Optimizer:** a custom `BF16SRAdamW` (`training/optim.py`) is used by both arms instead of a torchao class. BF16 parameters are updated with stochastic rounding; AdamW moments are kept in FP32 (not FP8, and they are not master weights). Unit tests show that small updates survive with stochastic rounding and are lost with round-to-nearest.
- **Batching:** micro-batch 4 x grad-accum 8 = effective batch 32 (the same effective batch as 8 x 4, with lower peak memory).
- **Answer extraction:** see section 1, row 7. The rule uses the last non-empty line, because the base model often writes a premature `#### N` guess at the top of its answer.
- **STE:** the quantizer runs under `no_grad` and the output is `x + (q - x.detach())`, so the gradient is exactly the identity.
- **Step 5 result (original model, full test set):** BF16 38.74% (loss 0.8156), FP8 37.60% (loss 0.8258). The FP8 gap is 1.14 pp, which passes the headroom gate.
- **Step 6 result (fake-quant vs vLLM FP8, original model, full test set):** the PyTorch BF16 loss matches vLLM's (0.81567 vs 0.81560). The exporter stores BF16 weight scales and computes `w / scale` in BF16; with that recipe the fake-quant weights are **bit-identical to the exported FP8 weights (100% of codes and scales on all 168 layers)**. The training default is now: BF16 per-channel weight scale, BF16 division, FP32 per-token activation scale. With exact weights the fake-quant reproduces 66-76% of vLLM's FP8 loss increase (+0.0067..+0.0077 vs +0.0102). Matmul-emulation and activation-arithmetic variants did not close the rest, so the residual (about 0.0035 loss) is unexplained; a likely but **unverified** cause is reduced-precision FP8 accumulation in the L4 tensor cores. Report it as a known limitation.
- Loss increase from FP8 is a noisy match signal (quantizing only the activations *lowered* the loss by 0.008), so the weight-code identity check is the decisive one.
- **Hardware change:** all GPU work now runs on an NVIDIA H100 (Modal, `gpu="H100!"`) for speed instead of the L4 used for Steps 1-6. Every run of the experiment (both arms, all seeds, all evaluations and benchmarks) uses the same GPU. The L4 results for the original model (BF16 38.74%, FP8 37.60%) are kept as a record; the H100 baseline is re-measured as `orig_bf16_h100` / `orig_fp8_h100`. Latency, throughput and memory figures in the report are for the H100, not the L4.
- **Step 6 on H100:** the fake-quant loss increase was -0.0039 on H100 vs +0.0067 on L4 with identical code; the metric is too noisy to certify the quantizer, which is why the bit-identical weight check is the decisive one.

---

<!-- ANALYSIS:START -->
## Results and deep analysis (added after the experiments)

All numbers in this section are computed from the result files by `analysis/add_plan_analysis.py`; none is typed by hand.
Cells: **A** = BF16-trained evaluated in BF16, **B** = BF16-trained evaluated in FP8, **C** = QAT-trained in BF16, **D** = QAT-trained in FP8.
Everything ran on one H100, on the full 1,319-question GSM8K test set, with 5 seeds per method.

### A. Verdict on the research question

> Does full FP8 QAT fine-tuning give better accuracy after FP8 conversion than full BF16 fine-tuning followed by the same conversion?

**Yes in direction, with a modest and not statistically settled size.**
- 3 epochs: **+1.47 pp** (D 35.45 ± 0.42% vs B 33.98 ± 0.27%), QAT ahead in 5 of 5 seeds; 95% interval -0.15 to +3.03 pp.
- 1 epoch: **+0.86 pp** (D 35.98 ± 0.57% vs B 35.12 ± 1.05%), QAT ahead in 3 of 5 seeds; 95% interval -0.91 to +2.59 pp.
- The **loss and KL measures are clear and consistent**: FP8 changes QAT-trained models less in every one of the 10 seed pairs for KL (section D).

### B. Accuracy in detail

| Setting | A (BF16→BF16) | B (BF16→FP8) | C (QAT→BF16) | D (QAT→FP8) | D − B (95% interval) | Seeds with D > B |
|---|---|---|---|---|---|---|
| 3 epochs | 35.30 ± 0.75 | 33.98 ± 0.27 | 36.09 ± 0.69 | 35.45 ± 0.42 | **+1.47** (-0.2 to +3.0) | 5 of 5 |
| 1 epoch | 36.13 ± 0.55 | 35.12 ± 1.05 | 36.13 ± 0.91 | 35.98 ± 0.57 | **+0.86** (-0.9 to +2.6) | 3 of 5 |

- **Interval used:** resamples both the training seeds and the test questions (two-level bootstrap). Resampling only the questions gives +0.29 to +2.62 pp (3 epochs) and -0.21 to +1.93 pp (1 epoch); the seeds-and-questions interval is the honest one.
- **3 epochs:** every QAT-FP8 model beats every BF16-FP8 model (lowest D 34.87% vs highest B 34.42%). The across-seed paired t is 5.47 (df 4, 5% critical value 2.776), but this ignores test-set sampling and has few seeds, so it overstates certainty.
- **1 epoch:** paired t 1.30 vs critical 2.776: not significant. Seed-to-seed spread of B is 1.05 pp, as large as the gain.
- **Per-seed McNemar tests** (one model vs one model on the same questions): 3 epochs p = 0.14–0.77; 1 epoch p = 0.04–0.75. Only 1 of the 10 individual pairs reaches p < 0.05 (not corrected for running 10 tests), which is expected: one 1,319-question test set measures a single model's accuracy with a 95% half-width of about ±2.6 pp.

### C. Where the accuracy gain comes from

D − B = (C − A) + [(A − B) − (C − D)]: an effect of QAT that exists even without FP8, plus a smaller FP8 drop.

| Setting (pp) | QAT effect with no FP8 (C − A) | FP8 drop, BF16-trained (A − B) | FP8 drop, QAT-trained (C − D) | Drop reduction |
|---|---|---|---|---|
| 3 epochs | +0.79 | +1.32 | +0.64 | +0.68 |
| 1 epoch | -0.00 | +1.02 | +0.15 | +0.86 |

- At **3 epochs** about half of the gain (+0.79 pp) shows up even in plain BF16. This looks like QAT acting as a regularizer on an overfit model: the QAT models also have lower validation loss.
- At **1 epoch** that part is **-0.00 pp** (both arms score 36.13% in BF16), so the whole gain comes from a smaller FP8 drop. Individual seeds still vary by more than a point (drop reduction std 2.27 pp), which is why that gain is not significant.

### D. The sharper evidence: loss and KL divergence

Accuracy needs correct final answers and is quantised to 1,319 yes/no outcomes. Loss and KL use every answer token (about 122,748 per model) and are deterministic.

**Increase in test loss caused by FP8 conversion (real FP8 in vLLM; lower is better):**

| Setting | BF16-trained | QAT-trained | QAT as % of BF16 | Seeds where QAT is smaller |
|---|---|---|---|---|
| 3 epochs | +0.0097 | +0.0033 | 34% | 4 of 5 |
| 1 epoch | +0.0056 | +0.0029 | 52% | 5 of 5 |

**KL divergence KL(P_BF16 ‖ P_FP8) per answer token (simulated FP8; lower is better):**

| Setting | BF16-trained | QAT-trained | QAT as % of BF16 | Seeds where QAT is lower | Top-1 agreement (BF16 / QAT) |
|---|---|---|---|---|---|
| 3 epochs | 0.03019 ± 0.00093 | 0.02478 ± 0.00043 | 82% | 5 of 5 | 95.82% / 96.14% |
| 1 epoch | 0.00976 ± 0.00016 | 0.00925 ± 0.00015 | 95% | 5 of 5 | 97.24% / 97.36% |

Untrained original model: KL = 0.01770.

- Under KL the QAT models are lower in **all 5 of 5 seeds in both settings**, with non-overlapping ranges. This is the strongest single result of the project.
- **Circularity caveat:** the KL simulator is the same FP8 fake-quantizer that the QAT models were trained against, so KL may flatter QAT somewhat. The real-FP8 loss and accuracy results (computed through vLLM, an independent path) point the same way, which is what makes the conclusion believable.

### E. Training length: overfit models are more fragile under FP8

| | Original (no tuning) | 1 epoch (BF16 / QAT) | 3 epochs (BF16 / QAT) |
|---|---|---|---|
| KL (BF16 vs FP8) | 0.0177 | 0.0098 / 0.0092 | 0.0302 / 0.0248 |
| Accuracy in FP8 | 37.6% (L4) | 35.1% / 36.0% | 34.0% / 35.5% |

- 3-epoch models change about **3.1× more** under FP8 than 1-epoch models, and more than the untouched model. Validation loss is lowest after epoch 1 and rises afterwards, so extra epochs overfit.
- *Hypothesis (not tested here):* overfitting makes the output distributions sharper, and sharper distributions are more sensitive to small numerical perturbations.
- QAT's relative benefit follows the fragility: **18% lower KL at 3 epochs** where there is much to repair, **5% at 1 epoch**.
- Practical reading: if you fine-tune for fewer epochs you lose less to FP8 regardless of method (FP8 accuracy 35.1% vs 34.0% for BF16-trained), and QAT then adds a smaller extra gain.

### F. Fine-tuning versus the untouched model

- Original model, BF16: **38.74%** (L4) and **39.95%** (H100); FP8 on L4: 37.60%.
- Fine-tuned models, BF16: 35.3% / 36.1% (3 epochs), 36.1% (1 epoch). **Neither fine-tuned arm beats the original.**
- The fine-tuned models answer in about 93 tokens against about 252 for the original. *Hypothesis:* GSM8K's short human-written solutions teach the model to reason less verbosely, which hurts a small model that benefits from longer step-by-step output. This does not affect the QAT-vs-BF16 comparison (same effect on both), but it limits how much practical value fine-tuning on this dataset adds.

### G. Validity checks, threats and what they mean

| Item | Evidence | Effect on conclusions |
|---|---|---|
| QAT quantizer matches deployment | FP8 weights bit-identical to the exporter's on 100% of codes and scales in all 168 layers, after matching the exporter's BF16 scales and BF16 division | Strong: the training-time weights are the deployed weights |
| Activation arithmetic matches only in kind | Diagnostic reproduced 66–76% of vLLM's FP8 loss increase on the original model | Small residual; unexplained (likely kernel details), not claimed as a known cause |
| Single-number "FP8 loss increase" is noisy | The same fake-quant gave +0.0067 on L4 and −0.0039 on H100; quantising only activations *lowered* the loss by 0.008 | Why weight-code identity, KL and multi-seed averages are used instead |
| GPU numerics move accuracy | Original BF16 38.74% (L4) vs 39.95% (H100) for the same weights | All compared runs used the same GPU type (H100) |
| Test-set size | 95% half-width of about ±2.6 pp for one model at about 35% accuracy | Main reason the accuracy intervals reach zero |
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
2. **That becomes roughly +0.9 to +1.5 points of accuracy after FP8**, consistently positive in direction but not settled at 95% confidence on this test set.
3. **Next:** (a) evaluate the same checkpoints on a second, larger maths test set to tighten the accuracy interval; (b) try harder compression such as FP4 or INT4, where plain conversion loses much more and QAT has more to recover (larger models usually lose less to FP8, not more); (c) test the overfitting-fragility hypothesis, for example by measuring KL after each epoch of training.
<!-- ANALYSIS:END -->


