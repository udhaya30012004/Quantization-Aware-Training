import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
torch = pytest.importorskip("torch")
from training.optim import BF16SRAdamW, stochastic_round_to_bf16
from common.config import load_config


def test_stochastic_rounding_is_unbiased_and_lands_on_neighbours():
    torch.manual_seed(0)
    x = torch.full((200000,), 1.001)                     # BF16 spacing near 1.0 is 0.0078125
    y = stochastic_round_to_bf16(x)
    assert y.dtype == torch.bfloat16
    assert set(y.float().unique().tolist()) <= {1.0, 1.0078125}
    assert abs(y.float().mean().item() - 1.001) < 5e-4
    neg = stochastic_round_to_bf16(-x)
    assert abs(neg.float().mean().item() + 1.001) < 5e-4


def _drift(stochastic_round):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.ones(20000, dtype=torch.bfloat16))
    opt = BF16SRAdamW([p], lr=1e-5, weight_decay=0.0, stochastic_round=stochastic_round)
    for _ in range(2000):                                # Adam step ~ lr each time -> expected drift 0.02
        p.grad = torch.ones_like(p)
        opt.step()
    return p.float().mean().item()


def test_small_updates_survive_only_with_stochastic_rounding():
    assert _drift(True) == pytest.approx(0.98, abs=0.004)   # weights really moved
    assert _drift(False) == 1.0                              # round-to-nearest: every update lost


def test_config_merge_and_overrides():
    cfg = load_config("qat_fp8", seed=2, smoke=True, overrides=["train.lr=2e-5", "train.epochs=1"])
    assert cfg["quant"] == "fp8_fakequant" and cfg["arm"] == "qat_full"
    assert cfg["train"]["lr"] == 2e-5 and cfg["train"]["epochs"] == 1 and cfg["train"]["max_steps"] == 10
    assert cfg["out_name"] == "smoke/qat_full_s2"
    assert load_config("bf16_full", seed=1)["out_name"] == "bf16_full/seed_1"
