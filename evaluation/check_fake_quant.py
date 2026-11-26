"""Step 6 gate: does the training-time fake-quant reproduce deployment (vLLM FP8) numerics?

Teacher-forced loss on the full test set, computed in HF/PyTorch on the ORIGINAL model:
    plain BF16            should match vLLM BF16 loss   (validates the loss pipeline itself)
    fake-quant FP8        should match vLLM FP8 loss    (validates the quantizer)
The vLLM numbers come from results/orig_bf16.json and results/orig_fp8.json (cells E and F).
"""
import json
from pathlib import Path

import torch

from models import fp8_fake_quant as fq


def run(original, data_dir, results_dir, batch_size=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data.dataset import GSM8KDataset
    from training.train import teacher_forced_loss

    tok = AutoTokenizer.from_pretrained(original)
    model = AutoModelForCausalLM.from_pretrained(original, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    ds = GSM8KDataset(f"{data_dir}/test.jsonl", tok, 512)
    vllm_bf16 = json.loads((Path(results_dir) / "orig_bf16.json").read_text())["eval_loss"]
    vllm_fp8 = json.loads((Path(results_dir) / "orig_fp8.json").read_text())["eval_loss"]

    hf_bf16 = teacher_forced_loss(model, ds, tok.pad_token_id, batch_size)
    n = fq.wrap(model)
    hf_fq = teacher_forced_loss(model, ds, tok.pad_token_id, batch_size)
    calls = fq.total_calls(model)

    d_vllm = vllm_fp8 - vllm_bf16
    d_fq = hf_fq - hf_bf16
    return {
        "n_wrapped": n, "fq_calls": calls, "test_examples": len(ds),
        "vllm_bf16_loss": vllm_bf16, "hf_bf16_loss": hf_bf16,
        "bf16_pipeline_abs_diff": abs(hf_bf16 - vllm_bf16),
        "vllm_fp8_loss": vllm_fp8, "hf_fakequant_loss": hf_fq,
        "fp8_abs_diff_vs_vllm": abs(hf_fq - vllm_fp8),
        "fp8_loss_increase_vllm": d_vllm, "fp8_loss_increase_fakequant": d_fq,
        "increase_ratio_fakequant_over_vllm": d_fq / d_vllm if d_vllm else None,
        "verdict_hint": "PASS if bf16_pipeline_abs_diff < 0.005 and the fake-quant loss increase is within ~30% of vLLM's",
    }
