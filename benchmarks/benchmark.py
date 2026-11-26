"""vLLM latency / throughput benchmark. IDENTICAL protocol for every model so results are comparable.

Protocol: fixed random prompts (input 256 tokens), 256 generated tokens forced with ignore_eos, greedy decoding,
batch sizes 1/8/32, 3 warm-up iterations then 20 timed iterations per batch size, prefix caching OFF (so repeated
prompts are not served from cache). Same engine arguments for every model; only the checkpoint path differs.

Timing note: generation runs in vLLM's engine process; llm.generate() blocks until every token is produced, so
wall-clock time around the call is the correct latency (a CUDA sync in this process would measure nothing).
"""
import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

BATCH_SIZES = [1, 8, 32]
IN_LEN, OUT_LEN, WARMUP, ITERS = 256, 256, 3, 20
ENGINE_ARGS = dict(gpu_memory_utilization=0.85, max_model_len=1024, seed=0, enable_prefix_caching=False)


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def disk_gb(path):
    return round(sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
                     if f.endswith(".safetensors")) / 2**30, 4)


def gpu_used_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().split(",")
        return round(float(out[0]) / 1024, 2), out[1].strip()
    except Exception:
        return None, None


def run(model_path, label, out_dir, enforce_eager=False):
    from vllm import LLM, SamplingParams
    import vllm

    rng = random.Random(0)
    pool = [[rng.randrange(1000, 100000) for _ in range(IN_LEN)] for _ in range(max(BATCH_SIZES))]
    sp = SamplingParams(temperature=0, max_tokens=OUT_LEN, min_tokens=OUT_LEN, ignore_eos=True)

    t0 = time.perf_counter()
    llm = LLM(model=model_path, enforce_eager=enforce_eager, **ENGINE_ARGS)
    load_s = time.perf_counter() - t0
    mem_used, gpu_name = gpu_used_gb()

    def call(bs):
        t = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": p} for p in pool[:bs]], sp, use_tqdm=False)
        dt = time.perf_counter() - t
        assert all(len(o.outputs[0].token_ids) == OUT_LEN for o in outs), "unexpected output length"
        return dt

    t0 = time.perf_counter()
    warm = [call(8) for _ in range(WARMUP)]
    warm_s = time.perf_counter() - t0

    results = {}
    for bs in BATCH_SIZES:
        lat = [call(bs) for _ in range(ITERS)]
        avg = sum(lat) / len(lat)
        results[str(bs)] = {"avg_s": avg, "p50_s": percentile(lat, 0.5), "p95_s": percentile(lat, 0.95),
                            "min_s": min(lat), "max_s": max(lat), "requests_per_s": bs / avg,
                            "tokens_per_s": bs * OUT_LEN / avg}
        print(f"[{label}] batch {bs:>2}: avg {avg*1000:7.1f} ms  p50 {results[str(bs)]['p50_s']*1000:7.1f}  "
              f"p95 {results[str(bs)]['p95_s']*1000:7.1f}  {results[str(bs)]['tokens_per_s']:8.0f} tok/s", flush=True)

    summary = {"label": label, "model_path": model_path, "gpu": gpu_name, "vllm_version": vllm.__version__,
               "enforce_eager": enforce_eager, "engine_args": ENGINE_ARGS, "input_len": IN_LEN, "output_len": OUT_LEN,
               "warmup_iters": WARMUP, "timed_iters": ITERS, "load_seconds": round(load_s, 1),
               "warmup_seconds": round(warm_s, 2), "first_warmup_call_s": round(warm[0], 2),
               "checkpoint_safetensors_gb": disk_gb(model_path), "gpu_memory_used_gb_after_load": mem_used,
               "batches": results}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results/bench")
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()
    run(a.model, a.label, a.out_dir, a.eager)
