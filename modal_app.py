"""Modal app: images, shared volume, and entrypoints. All GPU work runs on an L4.

    modal run modal_app.py::env_check        # Step 1: versions + FP8 GEMM sanity on an L4
    modal run modal_app.py::prepare_volume   # copy the fixed data split to the volume
"""
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent
GPU = os.environ.get("QAT_GPU", "H100!")           # every GPU job (training, eval, export, benchmarks); "H100!" pins H100 (no H200 upgrade)
TRAIN_GPU = os.environ.get("QAT_TRAIN_GPU", GPU)  # same GPU unless overridden
VOL_PATH = "/vol"
MAX_CONTAINERS = 8   # Modal workspace limit is 10 containers; stay below it. Extra jobs queue and start as others finish.

app = modal.App("fp8-qat")
vol = modal.Volume.from_name("fp8-qat", create_if_missing=True)

_IGNORE = ["venv/**", ".git/**", "**/__pycache__/**", "checkpoints/**", "plots/**", "results/**",
           "**/*.safetensors"]
_ENV = {"PYTHONPATH": "/root/qat", "HF_HOME": f"{VOL_PATH}/hf_cache", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}

# Training image: no vLLM / llm-compressor (their pins conflict with the training stack).
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.15.0", "transformers==5.23.0", "accelerate==1.16.0", "datasets==5.2.0",
                 "torchao==0.18.0", "pyyaml", "huggingface_hub", "pytest", "numpy")
    .env(_ENV)
    .add_local_dir(str(ROOT), "/root/qat", ignore=_IGNORE)
)

# Serving image: vLLM only (evaluation + benchmarks). Pins its own torch.
serve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.34.0", "datasets==5.2.0", "pyyaml", "pytest", "numpy")
    .env(_ENV)
    .add_local_dir(str(ROOT), "/root/qat", ignore=_IGNORE)
)

# Export image: llm-compressor only (BF16 checkpoint -> FP8 compressed-tensors checkpoint).
export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("llmcompressor==0.15.0", "datasets==5.2.0", "pyyaml", "pytest", "numpy")
    .env(_ENV)
    .add_local_dir(str(ROOT), "/root/qat", ignore=_IGNORE)
)


def _reload():
    """Refresh the volume view; skip (with a log line) if a reused container still has files open."""
    try:
        vol.reload()
    except Exception as e:
        print(f"volume reload skipped: {e}", flush=True)


def _save(name, obj):
    import os
    os.makedirs(f"{VOL_PATH}/results", exist_ok=True)
    with open(f"{VOL_PATH}/results/{name}", "w") as f:
        json.dump(obj, f, indent=2)
    vol.commit()


@app.function(image=train_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=1200)
def env_train():
    from common.env_report import collect
    info = collect()
    _save("env_train.json", info)
    return info


@app.function(image=serve_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=1800)
def env_serve():
    from common.env_report import collect
    info = collect()
    _save("env_serve.json", info)
    return info


@app.function(image=export_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=1800)
def env_export():
    from common.env_report import collect
    info = collect()
    _save("env_export.json", info)
    return info


@app.local_entrypoint()
def env_check():
    for name, fn in [("train_image", env_train), ("serve_image", env_serve), ("export_image", env_export)]:
        print(f"\n===== {name} =====")
        print(json.dumps(fn.remote(), indent=2))


@app.function(image=train_image, volumes={VOL_PATH: vol}, timeout=600)
def _write_data(files: dict):
    import os
    os.makedirs(f"{VOL_PATH}/data", exist_ok=True)
    for name, content in files.items():
        with open(f"{VOL_PATH}/data/{name}", "wb") as f:
            f.write(content)
    vol.commit()
    return sorted(os.listdir(f"{VOL_PATH}/data"))


@app.local_entrypoint()
def prepare_volume():
    """Upload the fixed split (and manifest) to /vol/data, then verify hashes remotely."""
    files = {n: (ROOT / "data" / n).read_bytes()
             for n in ["train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"]}
    print("uploaded:", _write_data.remote(files))


@app.function(image=train_image, volumes={VOL_PATH: vol}, timeout=1800)
def _download_model(revision: str, manifest: dict):
    """Download the pinned original model to /vol/checkpoints/original, then verify data + model hashes."""
    import hashlib
    import os
    from huggingface_hub import snapshot_download

    dest = f"{VOL_PATH}/checkpoints/original"
    if os.path.exists(f"{dest}/SOURCE.json"):
        print("original checkpoint already present - not touching it")
    else:
        snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision=revision, local_dir=dest,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    report = {"revision": revision, "model_sha256": sha(f"{dest}/model.safetensors"), "data": {}}
    for name, info in manifest["files"].items():
        got = sha(f"{VOL_PATH}/data/{name}")
        report["data"][name] = {"n": info["n"], "hash_matches_manifest": got == info["sha256"]}
    cfg = json.load(open(f"{dest}/config.json"))
    report["config"] = {k: cfg.get(k) for k in ["num_hidden_layers", "hidden_size", "tie_word_embeddings",
                                                 "torch_dtype", "dtype", "vocab_size"]}
    if not os.path.exists(f"{dest}/SOURCE.json"):
        with open(f"{dest}/SOURCE.json", "w") as f:
            json.dump(report, f, indent=2)
        for fn in os.listdir(dest):  # read-only guard: the original must never be overwritten
            try:
                os.chmod(f"{dest}/{fn}", 0o444)
            except Exception:
                pass
    vol.commit()
    return {"files": sorted(os.listdir(dest)), **report}


@app.local_entrypoint()
def setup():
    """Step 2: upload the data split, download the pinned model, verify hashes. Prints a JSON report."""
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    files = {n: (ROOT / "data" / n).read_bytes()
             for n in ["train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"]}
    print("uploaded:", _write_data.remote(files))
    print(json.dumps(_download_model.remote(manifest["model_revision"], manifest), indent=2))


# ---------------------------------------------------------------- Step 4: FP8 export + verification
@app.function(image=export_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=1800, single_use_containers=True, max_containers=MAX_CONTAINERS)
def export_fp8_remote(src: str, dst: str):
    from models.export_fp8 import export
    from evaluation.verify_fp8 import inspect_checkpoint
    _reload()
    meta = export(src, dst)
    vol.commit()
    return {"export": meta, "bf16_source": inspect_checkpoint(src, expect_fp8=False),
            "fp8_result": inspect_checkpoint(dst, expect_fp8=True)}


@app.function(image=serve_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=1800, single_use_containers=True)
def vllm_probe_remote(path: str):
    from evaluation.verify_fp8 import vllm_probe
    _reload()
    return vllm_probe(path)


@app.local_entrypoint()
def export_fp8(src: str = f"{VOL_PATH}/checkpoints/original",
               dst: str = f"{VOL_PATH}/checkpoints/original_fp8"):
    """Convert a BF16 checkpoint on the volume to FP8 and statically verify it."""
    print(json.dumps(export_fp8_remote.remote(src, dst), indent=2))


@app.local_entrypoint()
def vllm_probe(path: str):
    """Load a checkpoint in vLLM and generate a few greedy answers."""
    print(json.dumps(vllm_probe_remote.remote(path), indent=2))


# ---------------------------------------------------------------- Step 5+: evaluation
@app.function(image=serve_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=3600, single_use_containers=True, max_containers=MAX_CONTAINERS)
def evaluate_remote(model_path: str, label: str, split: str = "test", limit: int = 0):
    from evaluation.evaluate import evaluate
    _reload()
    summary = evaluate(model_path, label, f"{VOL_PATH}/data", f"{VOL_PATH}/results", split, limit)
    vol.commit()
    return summary


@app.local_entrypoint()
def evaluate(specs: str, split: str = "test", limit: int = 0):
    """specs = 'label=/vol/path,label2=/vol/path2'. Each model is evaluated in its own fresh L4 container."""
    for spec in specs.split(","):
        label, path = spec.split("=", 1)
        print(f"\n===== {label} ({path}) =====")
        s = evaluate_remote.remote(path, label, split, limit)
        s.pop("data_manifest_sha256", None)
        print(json.dumps(s, indent=2))


# ---------------------------------------------------------------- Training
@app.function(image=train_image, gpu=TRAIN_GPU, volumes={VOL_PATH: vol}, timeout=3 * 3600, single_use_containers=True, max_containers=MAX_CONTAINERS)
def train_remote(arm: str, seed: int, smoke: bool, overrides: str):
    from common.config import load_config
    from training.train import train as run_training
    _reload()
    cfg = load_config(arm, seed, smoke, overrides.split(";") if overrides else [])
    summary = run_training(cfg)
    vol.commit()
    return {k: v for k, v in summary.items() if k != "config"}


@app.local_entrypoint()
def run_train(arm: str, seed: int = 0, smoke: bool = False, overrides: str = ""):
    """arm = bf16_full | qat_fp8. Logs stream live to this terminal. overrides: 'train.lr=2e-5;train.epochs=1'."""
    train_remote.remote(arm, seed, smoke, overrides)


@app.function(image=train_image, gpu=TRAIN_GPU, volumes={VOL_PATH: vol}, timeout=3600, single_use_containers=True)
def check_fake_quant_remote():
    from evaluation.check_fake_quant import run
    _reload()
    return run(f"{VOL_PATH}/checkpoints/original", f"{VOL_PATH}/data", f"{VOL_PATH}/results")


@app.local_entrypoint()
def check_fake_quant():
    """Step 6: does my fake-quant reproduce the real vLLM FP8 numerics on the original model?"""
    print(json.dumps(check_fake_quant_remote.remote(), indent=2))


@app.function(image=train_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=3600, single_use_containers=True)
def diagnose_fake_quant_remote():
    from evaluation.diagnose_fake_quant import run
    _reload()
    return run(f"{VOL_PATH}/checkpoints/original", f"{VOL_PATH}/checkpoints/original_fp8",
               f"{VOL_PATH}/data", f"{VOL_PATH}/results")


@app.local_entrypoint()
def diagnose_fake_quant():
    """Find which quantizer detail explains the gap between fake-quant and vLLM FP8."""
    print(json.dumps(diagnose_fake_quant_remote.remote(), indent=2))


@app.local_entrypoint()
def run_all(seeds: str = "0,1,2", arms: str = "bf16_full,qat_fp8", overrides: str = "", tag: str = "", epochs: int = 0):
    """Launch every (arm, seed) training run IN PARALLEL, one L4 container each. Lines are prefixed [arm/seed_N]."""
    ckpt_arm = {"bf16_full": "bf16_full", "qat_fp8": "qat_full"}

    def ov(a):  # tag -> separate checkpoint folders (e.g. bf16_full_1ep/seed_0) so nothing is overwritten
        parts = [overrides] if overrides else []
        if tag:
            parts.append(f"arm={ckpt_arm[a]}_{tag}")
        if epochs:
            parts.append(f"train.epochs={epochs}")
        return ";".join(parts)

    jobs = [(a, int(sd), False, ov(a)) for a in arms.split(",") for sd in seeds.split(",")]
    print(f"launching {len(jobs)} runs in parallel: {jobs}")
    for r in train_remote.starmap(jobs):
        print(f"FINISHED {r['run']}: steps={r['steps']} last_loss={r['last_loss']:.4f} "
              f"val_final={r['val_loss_final_plain_bf16']:.4f} train_min={r['train_seconds']/60:.1f} "
              f"peak_gb={r['peak_vram_gb']:.1f} checkpoint={r['checkpoint']}")


# ---------------------------------------------------------------- Steps 11-12: export + evaluate every trained checkpoint
_ARM_CKPT = {"bf16": "bf16_full", "qat": "qat_full"}


@app.local_entrypoint()
def run_post(seeds: str = "0,1,2", arms: str = "bf16,qat", tag: str = ""):
    """For every (arm, seed): export the trained BF16 checkpoint to FP8 with the SAME exporter, then evaluate
    BF16 and FP8 on the full test set. All jobs of a stage run in parallel. Labels: <arm>_<bf16|fp8>_s<seed>."""
    runs = [(a, int(s)) for a in arms.split(",") for s in seeds.split(",")]
    root = f"{VOL_PATH}/checkpoints"
    print(f"stage 1/2: exporting {len(runs)} checkpoints to FP8 ...")
    sfx = f"_{tag}" if tag else ""      # tag -> reads <arm>_<tag>/seed_N, labels results <tag>_<arm>_<...>_sN
    lab = f"{tag}_" if tag else ""
    exports = [(f"{root}/{_ARM_CKPT[a]}{sfx}/seed_{s}", f"{root}/{_ARM_CKPT[a]}{sfx}/seed_{s}_fp8") for a, s in runs]
    for r in export_fp8_remote.starmap(exports):
        f = r["fp8_result"]
        print(f"  exported {r['export']['dst']}  pass={f['pass']}  fp8_layers={f['n_target_fp8']}/{f['n_target_linears']}"
              f"  size_gb={f['safetensors_gb']}")
        assert f["pass"], f"FP8 export check failed for {r['export']['dst']}"
    print(f"stage 2/2: evaluating {2 * len(runs)} models on the full test set ...")
    jobs = []
    for (a, s), (bf16_dir, fp8_dir) in zip(runs, exports):
        jobs += [(bf16_dir, f"{lab}{a}_bf16_s{s}", "test", 0), (fp8_dir, f"{lab}{a}_fp8_s{s}", "test", 0)]
    for r in evaluate_remote.starmap(jobs):
        print(f"  {r['label']:<14} acc={r['accuracy']*100:.2f}% ({r['n_correct']}/{r['n_total']})  "
              f"eval_loss={r['eval_loss']:.4f}  no_answer={r['n_no_answer']}  mean_gen_tok={r['mean_gen_tokens']:.0f}")


# ---------------------------------------------------------------- Benchmarks
@app.function(image=serve_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=3 * 3600, single_use_containers=True)
def benchmark_remote(models: list, eager: bool = False):
    """Benchmark several models back to back on the SAME GPU instance (fair comparison), one clean process each."""
    import subprocess
    import sys
    _reload()
    out_dir = f"{VOL_PATH}/results/bench"
    for label, path in models:
        cmd = [sys.executable, "-m", "benchmarks.benchmark", "--model", path, "--label", label, "--out-dir", out_dir]
        if eager:
            cmd.append("--eager")
        print(f"===== benchmarking {label} =====", flush=True)
        subprocess.run(cmd, check=True, cwd="/root/qat")
    vol.commit()
    return [json.load(open(f"{out_dir}/{label}.json")) for label, _ in models]


@app.local_entrypoint()
def benchmark_all(seed: int = 0, eager: bool = False):
    """Benchmark the original model and the seed-N BF16-trained and QAT-trained models, in BF16 and FP8."""
    root = f"{VOL_PATH}/checkpoints"
    models = [("orig_bf16", f"{root}/original"), ("orig_fp8", f"{root}/original_fp8"),
              (f"bf16_bf16_s{seed}", f"{root}/bf16_full/seed_{seed}"), (f"bf16_fp8_s{seed}", f"{root}/bf16_full/seed_{seed}_fp8"),
              (f"qat_bf16_s{seed}", f"{root}/qat_full/seed_{seed}"), (f"qat_fp8_s{seed}", f"{root}/qat_full/seed_{seed}_fp8")]
    print(f"{'model':<16}{'batch':>6}{'avg ms':>10}{'p50 ms':>10}{'p95 ms':>10}{'tok/s':>10}")
    for r in benchmark_remote.remote(models, eager):
        print(f"-- {r['label']}: load {r['load_seconds']}s, warmup {r['warmup_seconds']}s, "
              f"weights on disk {r['checkpoint_safetensors_gb']} GB, GPU used {r['gpu_memory_used_gb_after_load']} GB ({r['gpu']})")
        for bs, v in r["batches"].items():
            print(f"{r['label']:<16}{bs:>6}{v['avg_s']*1000:>10.1f}{v['p50_s']*1000:>10.1f}{v['p95_s']*1000:>10.1f}{v['tokens_per_s']:>10.0f}")


# ---------------------------------------------------------------- Pull results to the local machine
@app.function(image=train_image, volumes={VOL_PATH: vol}, timeout=600, single_use_containers=True)
def collect_results_remote():
    import os
    _reload()
    out = {}
    for base in ("results", "runs"):
        for dp, _, fns in os.walk(f"{VOL_PATH}/{base}"):
            for fn in fns:
                if fn.endswith((".json", ".jsonl")):
                    full = os.path.join(dp, fn)
                    out[os.path.relpath(full, VOL_PATH).replace("\\", "/")] = open(full, encoding="utf-8").read()
    return out


@app.local_entrypoint()
def pull_results():
    """Download every results/*.json(l) and runs/**/*.json(l) from the volume into ./results/ (runs -> results/runs)."""
    files = collect_results_remote.remote()
    for rel, text in files.items():
        dest = ROOT / (rel if rel.startswith("results/") else f"results/{rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    print(f"saved {len(files)} files under {ROOT / 'results'}")


# ---------------------------------------------------------------- KL divergence (BF16 vs simulated FP8)
@app.function(image=train_image, gpu=TRAIN_GPU, volumes={VOL_PATH: vol}, timeout=3600, single_use_containers=True, max_containers=MAX_CONTAINERS)
def kl_remote(ckpt: str, label: str):
    from evaluation.kl_divergence import run
    _reload()
    summary = run(ckpt, label, f"{VOL_PATH}/data", f"{VOL_PATH}/results")
    vol.commit()
    return summary


@app.local_entrypoint()
def kl_all(seeds: str = "0,1,2,3,4", settings: str = "3ep,1ep"):
    """KL(BF16 || simulated FP8) for every trained checkpoint (both methods, every seed, both epoch settings)
    plus the untrained original model as a reference. At most 8 containers run at a time."""
    root = f"{VOL_PATH}/checkpoints"
    jobs = [(f"{root}/original", "kl_orig")]
    for st in settings.split(","):
        sfx = "" if st == "3ep" else f"_{st}"
        for arm, ck in (("bf16", "bf16_full"), ("qat", "qat_full")):
            for sd in seeds.split(","):
                jobs.append((f"{root}/{ck}{sfx}/seed_{sd}", f"kl_{st}_{arm}_s{sd}"))
    print(f"computing KL for {len(jobs)} models ...")
    for r in kl_remote.starmap(jobs):
        print(f"  {r['label']:<18} KL={r['kl_mean']:.5f}  top1_agree={r['top1_agreement']*100:.2f}%  tokens={r['n_tokens']}  ({r['seconds']}s)")
