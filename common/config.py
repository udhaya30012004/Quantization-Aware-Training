"""Config loading: base.yaml <- arm yaml <- (smoke.yaml) <- 'a.b=c' overrides. Plain nested dicts."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(name):
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(arm_file: str, seed: int = 0, smoke: bool = False, overrides=()):
    """arm_file: 'bf16_full' | 'qat_fp8' (name of a file in configs/)."""
    cfg = _merge(_load("base"), _load(arm_file))
    if smoke:
        cfg = _merge(cfg, _load("smoke"))
    for item in overrides:
        if not item.strip():
            continue
        key, val = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        parsed = yaml.safe_load(val)
        if isinstance(parsed, str):          # YAML 1.1 reads '2e-5' as a string; accept it as a float
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[parts[-1]] = parsed
    cfg["seed"] = seed
    cfg["smoke"] = smoke
    cfg["out_name"] = f"smoke/{cfg['arm']}_s{seed}" if smoke else f"{cfg['arm']}/seed_{seed}"
    return cfg
