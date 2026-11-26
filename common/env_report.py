"""Environment report + FP8 sanity checks. Imported inside Modal containers (needs torch there)."""
import importlib
import json
import platform
import subprocess

PKGS = ["torch", "transformers", "accelerate", "datasets", "torchao", "vllm", "llmcompressor",
        "compressed_tensors", "transformer_engine"]


def _version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except Exception as e:
        return f"NOT AVAILABLE ({type(e).__name__})"


def fp8_matmul_check():
    """Run a real FP8 GEMM (E4M3 x E4M3 -> BF16) on the GPU and compare to an FP32 reference."""
    import torch
    out = {"ran": False}
    try:
        dev = "cuda"
        a32 = torch.randn(64, 128, device=dev)
        w32 = torch.randn(256, 128, device=dev)
        a8, w8 = a32.to(torch.float8_e4m3fn), w32.to(torch.float8_e4m3fn)
        one = torch.tensor(1.0, device=dev)
        y = torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        ref = a8.float() @ w8.float().t()
        out.update(ran=True, max_abs_err=(y.float() - ref).abs().max().item(),
                   ref_abs_max=ref.abs().max().item(), out_dtype=str(y.dtype))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def fake_quant_roundtrip_check():
    """E4M3 quantize->dequantize works and has sane error (no GPU FP8 kernels needed)."""
    import torch
    x = torch.randn(1024, 1024, device="cuda")
    s = x.abs().amax(dim=-1, keepdim=True) / 448.0
    xq = (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float32) * s
    rel = ((xq - x).norm() / x.norm()).item()
    return {"relative_error": rel, "ok": 0.0 < rel < 0.1}


def collect(with_fp8_check=True):
    info = {"python": platform.python_version(), "versions": {p: _version(p) for p in PKGS}}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
                           "total_memory_gb": round(props.total_memory / 2**30, 2)}
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            if with_fp8_check:
                info["fp8_gemm"] = fp8_matmul_check()
                info["fake_quant_roundtrip"] = fake_quant_roundtrip_check()
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"unavailable ({e})"
    # Plain JSON only: torch types (e.g. TorchVersion) cannot be unpickled on a machine without torch.
    return json.loads(json.dumps(info, default=str))


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
