"""FP8 diagnostics: prove a checkpoint really contains FP8 weights/scales, and that vLLM serves it as FP8."""
import json
import os
import re

TARGET = re.compile(r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def _dir_gb(path):
    return round(sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
                     if f.endswith(".safetensors")) / 2**30, 4)


def inspect_checkpoint(path: str, expect_fp8: bool = True) -> dict:
    """Static inspection of the safetensors files + config.json (no GPU needed)."""
    from safetensors import safe_open

    dtypes = {}   # tensor name -> dtype string
    shapes = {}
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(path, fn), framework="pt") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    dtypes[k], shapes[k] = sl.get_dtype(), sl.get_shape()

    targets = [k for k in dtypes if TARGET.search(k)]
    n_fp8 = sum(dtypes[k] == "F8_E4M3" for k in targets)
    scale_ok = sum((k + "_scale") in dtypes or k.replace(".weight", ".weight_scale") in dtypes for k in targets)
    chan_ok = sum(shapes.get(k.replace(".weight", ".weight_scale"), [None])[0] == shapes[k][0] for k in targets)
    non_target_fp8 = [k for k, d in dtypes.items() if d == "F8_E4M3" and k not in targets]
    cfg = json.load(open(os.path.join(path, "config.json")))

    rep = {
        "path": path,
        "n_target_linears": len(targets),
        "n_target_fp8": n_fp8,
        "n_with_weight_scale": scale_ok,
        "n_per_channel_scales": chan_ok,
        "non_target_fp8_tensors": non_target_fp8[:5],
        "embed_dtype": dtypes.get("model.embed_tokens.weight"),
        "lm_head_in_file": "lm_head.weight" in dtypes,
        "has_quantization_config": "quantization_config" in cfg,
        "safetensors_gb": _dir_gb(path),
    }
    if expect_fp8:
        rep["pass"] = (len(targets) == 168 and n_fp8 == 168 and scale_ok == 168 and chan_ok == 168
                       and not non_target_fp8 and rep["embed_dtype"] == "BF16" and rep["has_quantization_config"])
    else:
        rep["pass"] = n_fp8 == 0 and not rep["has_quantization_config"]
    return rep


def vllm_probe(path: str, prompts=None, max_tokens=64) -> dict:
    """Load the checkpoint in vLLM with fixed engine args and generate a few answers (greedy)."""
    import time
    from vllm import LLM, SamplingParams

    prompts = prompts or ["What is 17 * 23? Answer step by step."]
    t0 = time.time()
    llm = LLM(model=path, gpu_memory_utilization=0.85, max_model_len=1024, seed=0, enforce_eager=True)
    load_s = time.time() - t0
    quant = None
    try:
        quant = llm.llm_engine.vllm_config.model_config.quantization
    except Exception as e:
        quant = f"unreadable ({type(e).__name__})"
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                    SamplingParams(temperature=0, max_tokens=max_tokens))
    return {"path": path, "load_seconds": round(load_s, 1), "vllm_quantization": quant,
            "generations": [o.outputs[0].text for o in outs]}
