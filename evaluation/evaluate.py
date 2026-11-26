"""Evaluate one checkpoint (BF16 or FP8) on GSM8K with vLLM. Identical code path for every cell.

Outputs (under <out_dir>):
    <label>.json                 summary: accuracy, correct/incorrect, eval loss, lengths, timings, engine args
    per_question/<label>.jsonl   one row per question (for paired McNemar / bootstrap)

Accuracy: greedy generation, max_tokens=512, answer extracted by evaluation.metrics.extract_answer.
Eval loss: teacher-forced NLL per answer token on prompt + reference answer, via vLLM prompt_logprobs,
so it is computed the same way on BF16 and FP8 engines.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from data.dataset import build_prompt_text, load_jsonl, tokenize_example
from evaluation.metrics import gold_answer, extract_answer, is_correct

ENGINE_ARGS = dict(gpu_memory_utilization=0.85, max_model_len=1024, seed=0, enforce_eager=True)
MAX_NEW_TOKENS = 512


def nll_from_prompt_logprobs(token_ids, prompt_logprobs, n_prompt):
    """Sum of -logprob over answer tokens (positions >= n_prompt). Returns (sum_nll, n_tokens)."""
    total, n = 0.0, 0
    for i in range(n_prompt, len(token_ids)):
        entry = prompt_logprobs[i]
        total += -entry[token_ids[i]].logprob
        n += 1
    return total, n


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model_path, label, data_dir, out_dir, split="test", limit=0, system_prompt=None):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    kw = {"system_prompt": system_prompt} if system_prompt else {}
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    split_file = f"{split}.jsonl"
    if _sha(data_dir / split_file) != manifest["files"][split_file]["sha256"]:
        raise RuntimeError(f"{split_file} does not match manifest hash - refusing to evaluate")
    rows = load_jsonl(data_dir / split_file)
    if limit:
        rows = rows[:limit]

    tok = AutoTokenizer.from_pretrained(model_path)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    prompts = [build_prompt_text(tok, r["question"], **kw) for r in rows]

    t0 = time.time()
    llm = LLM(model=model_path, **ENGINE_ARGS)
    load_s = time.time() - t0
    try:
        quant = llm.llm_engine.vllm_config.model_config.quantization
    except Exception:
        quant = None

    # ---- accuracy: greedy generation
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=0, stop_token_ids=[im_end])
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    per_q, n_gen = [], []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        gold = gold_answer(r["target"])
        pred = extract_answer(text)
        ok = is_correct(text, gold)
        n_gen.append(len(o.outputs[0].token_ids))
        per_q.append({"id": r["id"], "gold": gold, "pred": pred, "correct": bool(ok),
                      "n_gen_tokens": n_gen[-1], "finish": o.outputs[0].finish_reason, "output": text})

    # ---- eval loss: teacher-forced NLL on the reference answer
    toks = [tokenize_example(tok, r, max_len=512, **kw) for r in rows]
    n_prompts = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    lp_outs = llm.generate([{"prompt_token_ids": t["input_ids"]} for t in toks],
                           SamplingParams(max_tokens=1, prompt_logprobs=0))
    tot, cnt = 0.0, 0
    for t, npmt, o in zip(toks, n_prompts, lp_outs):
        s, c = nll_from_prompt_logprobs(t["input_ids"], o.prompt_logprobs, npmt)
        tot, cnt = tot + s, cnt + c
    eval_loss = tot / cnt

    n = len(per_q)
    correct = sum(q["correct"] for q in per_q)
    summary = {
        "label": label, "model_path": model_path, "split": split, "n_total": n,
        "n_correct": correct, "n_incorrect": n - correct, "accuracy": correct / n,
        "n_no_answer": sum(q["pred"] is None for q in per_q),
        "n_hit_max_tokens": sum(q["finish"] == "length" for q in per_q),
        "eval_loss": eval_loss, "eval_ppl": math.exp(eval_loss), "eval_loss_tokens": cnt,
        "mean_gen_tokens": sum(n_gen) / n, "load_seconds": round(load_s, 1),
        "generate_seconds": round(gen_s, 1), "vllm_quantization": quant,
        "engine_args": ENGINE_ARGS, "max_new_tokens": MAX_NEW_TOKENS, "vllm_version": vllm.__version__,
        "data_manifest_sha256": {k: v["sha256"] for k, v in manifest["files"].items()},
    }
    (out_dir / "per_question").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_question" / f"{label}.jsonl", "w", encoding="utf-8") as f:
        for q in per_q:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out_dir / f"{label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.model, a.label, a.data_dir, a.out_dir, a.split, a.limit), indent=2))
