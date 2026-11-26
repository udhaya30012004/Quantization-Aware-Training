"""How much does FP8 change a model's predictions? KL(P_BF16 || P_FP8) on the test answers.

For a trained checkpoint, run the SAME model twice on every test question (prompt + reference answer, teacher-forced):
  P_BF16 : plain BF16 forward
  P_FP8  : forward with simulated FP8 (exact FP8 weights + per-token FP8 activation rounding; models/fp8_fake_quant.py)
and compare the full next-token distributions at every answer position. Lower KL = FP8 changes the model less.

Simulation note: the FP8 weights are bit-identical to the deployed ones; activation rounding matches the deployment in
kind but not in every kernel detail, so this is "simulated FP8", not vLLM's real FP8 kernels.
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models import fp8_fake_quant as fq


def kl_from_logits(la, lb):
    """KL(P_a || P_b) for each row of two [N, V] logit matrices (fp32 math)."""
    lpa = F.log_softmax(la.float(), dim=-1)
    lpb = F.log_softmax(lb.float(), dim=-1)
    return (lpa.exp() * (lpa - lpb)).sum(dim=-1)


def js_from_logits(la, lb):
    """Jensen-Shannon divergence JS(P_a || P_b) in nats for each row of [N, V] logits (fp32 math)."""
    lpa = F.log_softmax(la.float(), dim=-1)
    lpb = F.log_softmax(lb.float(), dim=-1)
    log_m = torch.logaddexp(lpa, lpb) - math.log(2.0)
    kl_am = (lpa.exp() * (lpa - log_m)).sum(dim=-1)
    kl_bm = (lpb.exp() * (lpb - log_m)).sum(dim=-1)
    return 0.5 * (kl_am + kl_bm)


def kl_bucket_counts(kl_tensor):
    """Count answer token positions falling into each KL divergence bucket."""
    kl = kl_tensor.float().clamp_min(0.0)
    return {
        "lt_001": int((kl < 0.01).sum().item()),
        "001_005": int(((kl >= 0.01) & (kl < 0.05)).sum().item()),
        "005_010": int(((kl >= 0.05) & (kl < 0.10)).sum().item()),
        "ge_010": int((kl >= 0.10).sum().item()),
    }


@torch.no_grad()
def batch_kl(model, input_ids, attention_mask, labels, return_js=False):
    """Returns (sum of KL over answer positions, number of positions, number where the top-1 token agrees, kl[, js])."""
    fq.set_enabled(model, False)
    la = model(input_ids=input_ids, attention_mask=attention_mask).logits
    fq.set_enabled(model, True)
    lb = model(input_ids=input_ids, attention_mask=attention_mask).logits
    mask = labels[:, 1:] != -100                       # answer tokens only (same positions as the eval loss)
    a, b = la[:, :-1][mask], lb[:, :-1][mask]
    kl = kl_from_logits(a, b)
    agree = (a.argmax(-1) == b.argmax(-1)).sum().item()
    if return_js:
        js = js_from_logits(a, b)
        return kl.sum().item(), int(mask.sum()), int(agree), kl, js
    return kl.sum().item(), int(mask.sum()), int(agree), kl


def run(ckpt, label, data_dir, out_dir, split="test", batch_size=16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data.dataset import GSM8KDataset, make_collator

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    n_wrapped = fq.wrap(model)
    ds = GSM8KDataset(f"{data_dir}/{split}.jsonl", tok, 512)
    collate = make_collator(tok.pad_token_id)
    total, js_total, n_tok, agree = 0.0, 0.0, 0, 0
    per_tok_max = 0.0
    all_kl = []
    for i in range(0, len(ds), batch_size):
        b = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        b = {k: v.to(device) for k, v in b.items()}
        s, n, a, kl, js = batch_kl(model, b["input_ids"], b["attention_mask"], b["labels"], return_js=True)
        total, js_total, n_tok, agree = total + s, js_total + js.sum().item(), n_tok + n, agree + a
        per_tok_max = max(per_tok_max, kl.max().item())
        all_kl.append(kl.float().cpu())
    kl_all = torch.cat(all_kl).clamp_min(0.0)
    summary = {"label": label, "checkpoint": ckpt, "split": split, "n_examples": len(ds), "n_tokens": n_tok,
               "n_wrapped": n_wrapped, "fq_calls": fq.total_calls(model),
               "kl_mean": total / n_tok, "kl_median": float(kl_all.median().item()),
               "js_mean": js_total / n_tok, "top1_agreement": agree / n_tok,
               "kl_p95": float(torch.quantile(kl_all, 0.95).item()),
               "kl_p99": float(torch.quantile(kl_all, 0.99).item()),
               "kl_max_token": per_tok_max, "kl_buckets": kl_bucket_counts(kl_all),
               "fp8_simulation": "exact FP8 weights + per-token FP8 activation rounding (fake quant)",
               "seconds": round(time.time() - t0, 1)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
