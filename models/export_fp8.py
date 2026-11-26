"""The ONE FP8 deployment conversion used for every checkpoint (BF16-trained, QAT-trained, original).

llm-compressor, scheme FP8_DYNAMIC: E4M3 weights with static per-output-channel scales, E4M3 activations
with dynamic per-token scales, lm_head left in BF16. Data-free and deterministic.

Never change this file's scheme between runs: both arms must be deployed identically.
"""
import json
import os
import shutil

SCHEME = "FP8_DYNAMIC"
IGNORE = ["lm_head"]


def export(src: str, dst: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("dst must differ from src - never overwrite a checkpoint")
    if os.path.exists(dst):
        shutil.rmtree(dst)  # dst is always a derived FP8 artifact, safe to regenerate

    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(src)
    recipe = QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=IGNORE)
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(dst, save_compressed=True)
    tok.save_pretrained(dst)
    meta = {"src": src, "dst": dst, "scheme": SCHEME, "ignore": IGNORE}
    with open(os.path.join(dst, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
