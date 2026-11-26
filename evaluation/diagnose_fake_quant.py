"""Why is the fake-quant loss increase (+0.0035) smaller than vLLM FP8's (+0.0102)?

Part 1 - weights: compare the exported FP8 checkpoint with what the fake-quant would produce from the same
         BF16 weights (stored scales, stored FP8 codes).
Part 2 - activations/matmul: teacher-forced test loss with one detail changed at a time.
"""
import json

import torch

from models import fp8_fake_quant as fq


WEIGHT_VARIANTS = {
    "fp32_scale (old)": {"w_scale_bf16": False, "w_div_bf16": False},
    "bf16_scale": {"w_scale_bf16": True, "w_div_bf16": False},
    "bf16_scale + bf16_div": {"w_scale_bf16": True, "w_div_bf16": True},
}


def compare_weights(original, fp8_dir):
    """For each candidate weight-quantization arithmetic: fraction of FP8 codes identical to the export."""
    import os
    from safetensors.torch import load_file
    orig = load_file(os.path.join(original, "model.safetensors"))
    fp8 = {}
    for fn in os.listdir(fp8_dir):
        if fn.endswith(".safetensors"):
            fp8.update(load_file(os.path.join(fp8_dir, fn)))
    keys = [k for k in fp8 if k.endswith(".weight") and fp8[k].dtype == torch.float8_e4m3fn]
    saved = {k: fq.FQ_CONFIG[k] for k in ("w_scale_bf16", "w_div_bf16")}   # restored below (new defaults)
    out = {"n_layers": len(keys)}
    for name, cfg in WEIGHT_VARIANTS.items():
        fq.FQ_CONFIG.update(cfg)
        same, scale_eq, gap = [], [], []
        for k in keys:
            w = orig[k].cuda()
            codes, s = fq.weight_quant_parts(w)
            st_codes = fp8[k].cuda().float()
            st_scale = fp8[k.replace(".weight", ".weight_scale")].cuda().float()
            same.append((codes.float() == st_codes).float().mean().item())
            scale_eq.append((s == st_scale).float().mean().item())
            gap.append((((st_codes * st_scale) - fq.fake_quant_weight(w)).norm() / w.float().norm()).item())
        out[name] = {"frac_identical_codes_mean": sum(same) / len(same), "frac_identical_codes_min": min(same),
                     "frac_identical_scales_mean": sum(scale_eq) / len(scale_eq),
                     "deq_gap_rel_mean": sum(gap) / len(gap)}
    fq.FQ_CONFIG.update(saved)
    return out


def run(original, fp8_dir, data_dir, results_dir, batch_size=16):
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data.dataset import GSM8KDataset
    from training.train import teacher_forced_loss

    report = {"weights": compare_weights(original, fp8_dir)}
    print("weights:", json.dumps(report["weights"], indent=2), flush=True)

    tok = AutoTokenizer.from_pretrained(original)
    model = AutoModelForCausalLM.from_pretrained(original, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    ds = GSM8KDataset(f"{data_dir}/test.jsonl", tok, 512)
    vb = json.loads((Path(results_dir) / "orig_bf16.json").read_text())["eval_loss"]
    vf = json.loads((Path(results_dir) / "orig_fp8.json").read_text())["eval_loss"]
    base = teacher_forced_loss(model, ds, tok.pad_token_id, batch_size)
    fq.wrap(model)
    target = vf - base
    report["target_loss_increase_vllm"] = target
    variants = {
        "exact weights (new default)": {},
        "+ scaled_mm_emul": {"scaled_mm_emul": True},
        "+ a_recip": {"a_recip": True},
        "+ a_div_bf16": {"a_div_bf16": True},
        "+ a_scale_bf16": {"a_scale_bf16": True},
        "+ scaled_mm_emul + a_recip": {"scaled_mm_emul": True, "a_recip": True},
    }
    default_cfg = dict(fq.FQ_CONFIG)
    out = {}
    for name, change in variants.items():
        fq.FQ_CONFIG.update(default_cfg)
        fq.FQ_CONFIG.update(change)
        loss = teacher_forced_loss(model, ds, tok.pad_token_id, batch_size)
        out[name] = {"loss": loss, "increase_over_bf16": loss - base, "fraction_of_vllm_increase": (loss - base) / target}
        print(name, out[name], flush=True)
    fq.FQ_CONFIG.update(default_cfg)
    report["hf_bf16_loss"] = base
    report["variants"] = out
    return report
