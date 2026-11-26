"""FP8 fake-quant QAT for the major projections - mirrors the deployment quantizer (FP8_DYNAMIC).

Deployment (llm-compressor FP8_DYNAMIC, served by vLLM):
    weights     : E4M3, static per-OUTPUT-CHANNEL scale  s_w = amax(|W|, over input dim) / 448
    activations : E4M3, dynamic per-TOKEN scale          s_x = amax(|x|, over hidden dim) / 448
QAT forward uses exactly these quantize->dequantize operations; backward is a straight-through estimator
(identity gradient, scales detached). Parameters stay BF16 (or whatever dtype the model has).

Wrapped: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in every decoder layer (168 for Qwen2.5-0.5B).
NOT wrapped: embeddings, lm_head, norms, softmax, loss.

Wrapping swaps the class in place, so the nn.Parameter objects (and optimizer state) are untouched and
unwrap() restores the plain nn.Linear exactly.
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_MAX = 448.0  # largest finite float8_e4m3fn value
TARGET = re.compile(r"(^|\.)layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")


# Switches used by evaluation/diagnose_fake_quant.py to find which arithmetic matches the exporter exactly.
# The defaults are the training behaviour.
FQ_CONFIG = {"weight": True, "act": True,
             # Verified bit-identical to the exporter (100% of FP8 codes and scales on all 168 layers):
             "w_scale_bf16": True,    # per-channel WEIGHT scale is stored in BF16
             "w_div_bf16": True,      # w / scale is computed in BF16 before the FP8 cast
             # Activation arithmetic (vLLM computes per-token scales in FP32):
             "a_scale_bf16": False, "a_div_bf16": False, "a_recip": False,
             "scaled_mm_emul": False,  # diagnostic only: matmul on FP8 codes, scales applied after
             "fp32_matmul": False, "act_gran": "token"}


def _scale(x: torch.Tensor, dim, bf16: bool) -> torch.Tensor:
    if dim is None:                       # per-tensor
        s = x.detach().abs().amax().clamp(min=1e-12) / FP8_MAX
    else:
        s = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / FP8_MAX
    return s.to(torch.bfloat16).to(torch.float32) if bf16 else s


def weight_quant_parts(w: torch.Tensor):
    """Per-output-channel E4M3 codes and scale for w [out, in]. Returns (fp8 codes, fp32 scale [out,1])."""
    wf = w.float()
    s = _scale(wf, 1, FQ_CONFIG["w_scale_bf16"])
    if FQ_CONFIG["w_div_bf16"]:
        r = (wf.to(torch.bfloat16) / s.to(torch.bfloat16)).float()
    else:
        r = wf / s
    return r.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def fake_quant_weight(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel E4M3 quantize->dequantize; fp32 math."""
    codes, s = weight_quant_parts(w)
    return codes.to(torch.float32) * s


def act_quant_parts(x: torch.Tensor):
    """Per-token E4M3 codes and fp32 scale for activations x [..., hidden]."""
    xf = x.float()
    sc = _scale(xf, -1 if FQ_CONFIG["act_gran"] == "token" else None, FQ_CONFIG["a_scale_bf16"])
    if FQ_CONFIG["a_recip"]:
        r = xf * (1.0 / sc)
    elif FQ_CONFIG["a_div_bf16"]:
        r = (xf.to(torch.bfloat16) / sc.to(torch.bfloat16)).float()
    else:
        r = xf / sc
    return r.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), sc


def fake_quant_act(x: torch.Tensor) -> torch.Tensor:
    """Per-token E4M3 quantize->dequantize over the hidden (last) dim; fp32 math."""
    codes, sc = act_quant_parts(x)
    return codes.to(torch.float32) * sc


def ste(x: torch.Tensor, quantizer) -> torch.Tensor:
    """Forward = quantizer(x); backward = identity."""
    xf = x.float()
    with torch.no_grad():
        q = quantizer(x)
    return xf + (q - xf.detach())


class FP8FakeQuantLinear(nn.Linear):
    """nn.Linear whose forward sees E4M3 weight + per-token activation quantization."""

    fq_enabled = True
    fq_calls = 0

    def forward(self, x):
        if not self.fq_enabled:
            return F.linear(x, self.weight, self.bias)
        self.fq_calls += 1
        if FQ_CONFIG["scaled_mm_emul"]:       # diagnostic (no grad): FP8-code matmul, scales applied afterwards
            with torch.no_grad():
                xc, sx = act_quant_parts(x)
                wc, sw = weight_quant_parts(self.weight)
                y = F.linear(xc.to(x.dtype), wc.to(x.dtype)).float() * sx * sw.t()
                if self.bias is not None:
                    y = y + self.bias.float()
                return y.to(x.dtype)
        xq = ste(x, fake_quant_act).to(x.dtype) if FQ_CONFIG["act"] else x
        wq = ste(self.weight, fake_quant_weight).to(x.dtype) if FQ_CONFIG["weight"] else self.weight
        if FQ_CONFIG["fp32_matmul"]:
            b = self.bias.float() if self.bias is not None else None
            return F.linear(xq.float(), wq.float(), b).to(x.dtype)
        return F.linear(xq, wq, self.bias)


def wrap(model: nn.Module) -> int:
    """Swap target nn.Linear modules to FP8FakeQuantLinear in place. Returns how many were wrapped."""
    n = 0
    for name, mod in model.named_modules():
        if type(mod) is nn.Linear and TARGET.search(name):
            mod.__class__ = FP8FakeQuantLinear
            mod.fq_calls = 0
            n += 1
    return n


def unwrap(model: nn.Module) -> int:
    """Restore plain nn.Linear (e.g. before saving a standard HF checkpoint)."""
    n = 0
    for mod in model.modules():
        if type(mod) is FP8FakeQuantLinear:
            mod.__class__ = nn.Linear
            n += 1
    return n


def set_enabled(model: nn.Module, enabled: bool):
    for mod in model.modules():
        if type(mod) is FP8FakeQuantLinear:
            mod.fq_enabled = enabled


def total_calls(model: nn.Module) -> int:
    return sum(m.fq_calls for m in model.modules() if type(m) is FP8FakeQuantLinear)
