"""ONE training loop for BOTH arms. The only difference between runs is cfg['quant']:
    none           -> plain BF16 full fine-tuning                      (Run A)
    fp8_fakequant  -> same, with FP8 fake-quant on the 168 projections (Run B)

Full fine-tuning: every parameter trainable, no LoRA/PEFT/frozen layers. Parameters, gradients are BF16;
AdamW moments FP32; weight updates use stochastic rounding; loss is computed in FP32.
Both runs start from the pinned ORIGINAL checkpoint, which is never written to.
"""
import json
import math
import os
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.seed import seed_everything
from data.dataset import GSM8KDataset, make_collator
from models import fp8_fake_quant as fq
from training.optim import BF16SRAdamW

IGNORE = -100


def ce_sum(logits, labels):
    """Summed next-token cross-entropy in FP32 over non-ignored positions. Returns (loss_sum, n_tokens)."""
    lg = logits[:, :-1].float().reshape(-1, logits.size(-1))
    lb = labels[:, 1:].reshape(-1)
    return F.cross_entropy(lg, lb, ignore_index=IGNORE, reduction="sum"), (lb != IGNORE).sum()


@torch.no_grad()
def teacher_forced_loss(model, ds, pad_id, batch_size=16, device="cuda"):
    """Mean NLL per answer token over a dataset (same quantity vLLM's prompt_logprobs eval reports)."""
    was_training = model.training
    model.eval()
    collate = make_collator(pad_id)
    total, n = 0.0, 0
    for i in range(0, len(ds), batch_size):
        b = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        b = {k: v.to(device) for k, v in b.items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        s, c = ce_sum(out.logits, b["labels"])
        total += s.item()
        n += c.item()
    model.train(was_training)
    return total / n


def quant_stats(model):
    """Diagnostics proving fake-quant is active: wrapped count, calls, relative weight quantization error."""
    errs, calls, n = [], 0, 0
    with torch.no_grad():
        for m in model.modules():
            if type(m) is fq.FP8FakeQuantLinear:
                n += 1
                calls += m.fq_calls
                w = m.weight.float()
                errs.append(((fq.fake_quant_weight(w) - w).norm() / w.norm()).item())
    return {"n_wrapped": n, "total_fq_calls": calls,
            "weight_rel_err_mean": sum(errs) / len(errs) if errs else None,
            "weight_rel_err_max": max(errs) if errs else None}


def _lr_at(step, total, base_lr, warmup_ratio):
    warm = max(1, int(round(total * warmup_ratio)))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * prog))


def train(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tag = f"[{cfg['out_name']}]"

    def say(*args, **kw):                      # prefix every line so parallel runs stay readable
        kw.setdefault("flush", True)
        print(tag, *args, **kw)

    t, P = cfg["train"], cfg["paths"]
    dev = "cuda"
    seed_everything(cfg["seed"])
    out_dir = Path(P["ckpt_root"]) / cfg["out_name"]
    run_dir = Path(P["runs_root"]) / cfg["out_name"]
    if os.path.abspath(out_dir) == os.path.abspath(P["original"]):
        raise RuntimeError("refusing to write into the original checkpoint")
    if (out_dir / "model.safetensors").exists() and not t["overwrite"]:
        raise RuntimeError(f"{out_dir} already has a checkpoint - not overwriting (set train.overwrite=true)")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "train_log.jsonl", "w")

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    say(f"=== run {cfg['out_name']}  quant={cfg['quant']}  seed={cfg['seed']} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(P["original"])
    model = AutoModelForCausalLM.from_pretrained(P["original"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters()]
    n_params = sum(p.numel() for p in params)
    assert all(p.requires_grad for p in params) and all(p.dtype == torch.bfloat16 for p in params)

    if cfg["quant"] == "fp8_fakequant":
        n_wrapped = fq.wrap(model)
        assert n_wrapped == 168, n_wrapped
        say(f"fake-quant FP8 wrapped {n_wrapped} projection layers", flush=True)
    elif cfg["quant"] != "none":
        raise ValueError(f"unknown quant {cfg['quant']}")
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    max_len = cfg["data"]["max_seq_length"]
    train_ds = GSM8KDataset(f"{P['data']}/train.jsonl", tok, max_len, limit=t["limit_train"],
                            system_prompt=cfg["data"]["system_prompt"])
    val_ds = GSM8KDataset(f"{P['data']}/val.jsonl", tok, max_len, limit=t["limit_val"],
                          system_prompt=cfg["data"]["system_prompt"])
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=t["micro_batch_size"], shuffle=True, generator=gen,
                        collate_fn=make_collator(tok.pad_token_id))
    steps_per_epoch = math.ceil(len(loader) / t["grad_accum"])
    total_steps = t["epochs"] * steps_per_epoch
    if t["max_steps"]:
        total_steps = min(total_steps, t["max_steps"])
    say(f"params={n_params/1e6:.1f}M  train={len(train_ds)} (truncated {train_ds.n_truncated})  "
          f"val={len(val_ds)}  micro_batch={t['micro_batch_size']} x accum={t['grad_accum']}  "
          f"steps/epoch={steps_per_epoch}  total_steps={total_steps}", flush=True)

    opt = BF16SRAdamW(params, lr=t["lr"], betas=t["betas"], eps=t["eps"], weight_decay=t["weight_decay"],
                      stochastic_round=t["stochastic_rounding"])
    torch.cuda.reset_peak_memory_stats()
    val0 = teacher_forced_loss(model, val_ds, tok.pad_token_id, device=dev)
    say(f"[val] step 0  loss={val0:.4f}", flush=True)
    log({"event": "val", "step": 0, "val_loss": val0})
    if cfg["quant"] != "none":
        say("[fq] start:", quant_stats(model), flush=True)

    snap = [p.detach().clone() for p in params] if t["track_weight_change"] else None
    first_step_check = None
    step, t0, tok_count, loss_hist, val_hist = 0, time.time(), 0, [], [val0]
    done = False
    for epoch in range(t["epochs"]):
        it = iter(loader)
        while True:
            window = list(islice(it, t["grad_accum"]))
            if not window or step >= total_steps:
                done = step >= total_steps
                break
            n_tok = sum(int((b["labels"][:, 1:] != IGNORE).sum()) for b in window)
            lr = _lr_at(step, total_steps, t["lr"], t["warmup_ratio"])
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for b in window:
                b = {k: v.to(dev) for k, v in b.items()}
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                s, _ = ce_sum(out.logits, b["labels"])
                (s / n_tok).backward()
                loss_sum += s.item()
                tok_count += int(b["attention_mask"].sum())
            gnorm = torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"]).item()
            if not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite gradient norm at step {step}")
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss = loss_sum / n_tok
            loss_hist.append(loss)
            if snap is not None and first_step_check is None:
                changed = sum((p != s0).sum().item() for p, s0 in zip(params, snap))
                maxdiff = max((p.float() - s0.float()).abs().max().item() for p, s0 in zip(params, snap))
                first_step_check = {"weights_changed_frac": changed / n_params, "max_abs_change": maxdiff}
                say(f"[smoke] after 1 optimizer step: {first_step_check}", flush=True)
            if step % t["log_every"] == 0 or step == total_steps:
                el = time.time() - t0
                eta = el / step * (total_steps - step)
                rec = {"event": "train", "step": step, "epoch": epoch + 1, "loss": loss, "lr": lr,
                       "grad_norm": gnorm, "tok_per_s": tok_count / el,
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "elapsed_s": el}
                log(rec)
                say(f"step {step:>4}/{total_steps}  ep {epoch+1}  loss {loss:.4f}  lr {lr:.2e}  "
                      f"gnorm {gnorm:.2f}  {tok_count/el:,.0f} tok/s  peak {rec['peak_mem_gb']:.1f} GB  "
                      f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
            if step >= total_steps:
                done = True
                break
        v = teacher_forced_loss(model, val_ds, tok.pad_token_id, device=dev)
        val_hist.append(v)
        say(f"[val] end of epoch {epoch+1}  step {step}  loss={v:.4f}", flush=True)
        log({"event": "val", "step": step, "epoch": epoch + 1, "val_loss": v})
        if done:
            break

    train_seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    qstats = quant_stats(model) if cfg["quant"] != "none" else None
    if qstats:
        say("[fq] end:", qstats, flush=True)
        assert qstats["total_fq_calls"] > 0, "fake-quant never ran"
        fq.unwrap(model)                               # save a plain HF checkpoint (no wrapper classes)
    val_plain = teacher_forced_loss(model, val_ds, tok.pad_token_id, device=dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # reload check: saved checkpoint must reproduce the in-memory model's logits exactly
    model.eval()
    probe = make_collator(tok.pad_token_id)([val_ds[i] for i in range(min(4, len(val_ds)))])
    probe = {k: v.to(dev) for k, v in probe.items()}
    with torch.no_grad():
        ref = model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]).logits.float()
        re = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16).to(dev).eval()
        reload_diff = (re(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"])
                       .logits.float() - ref).abs().max().item()
    del re

    summary = {
        "run": cfg["out_name"], "arm": cfg["arm"], "quant": cfg["quant"], "seed": cfg["seed"],
        "n_params": n_params, "all_params_trainable": True, "steps": step, "epochs_run": epoch + 1,
        "train_examples": len(train_ds), "truncated": train_ds.n_truncated,
        "first_loss": loss_hist[0], "last_loss": loss_hist[-1], "val_loss_history": val_hist,
        "val_loss_final_plain_bf16": val_plain, "train_seconds": train_seconds,
        "train_tokens_per_s": tok_count / train_seconds, "peak_vram_gb": peak_gb,
        "first_step_weight_change": first_step_check, "quant_stats": qstats,
        "reload_max_abs_logit_diff": reload_diff, "checkpoint": str(out_dir), "config": cfg,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    say("=== done ===", json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2), flush=True)
    return summary
