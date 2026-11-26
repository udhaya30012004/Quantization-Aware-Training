import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from models.fp8_fake_quant import (FP8FakeQuantLinear, fake_quant_act, fake_quant_weight, ste, wrap, unwrap,
                                   set_enabled, total_calls, FP8_MAX)


def test_weight_quant_is_per_channel_and_on_fp8_grid():
    torch.manual_seed(0)
    w = torch.randn(32, 64) * torch.rand(32, 1) * 5     # rows with very different scales
    q = fake_quant_weight(w)
    rel = ((q - w).norm() / w.norm()).item()
    assert 0 < rel < 0.08
    # each row, divided by its own scale, lands exactly on representable E4M3 values
    s = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).to(torch.bfloat16).float()   # exporter stores BF16 scales
    unit = q / s
    assert torch.allclose(unit.to(torch.float8_e4m3fn).to(torch.float32), unit, rtol=1e-5, atol=1e-6)
    assert unit.abs().max() <= FP8_MAX + 1e-3


def test_act_quant_is_per_token():
    x = torch.randn(4, 8, 64)
    x[0, 0] *= 1000                                      # one huge token must not hurt other tokens
    q = fake_quant_act(x)
    rel_other = ((q[1] - x[1]).norm() / x[1].norm()).item()
    assert rel_other < 0.08


def test_ste_gradient_is_identity():
    x = torch.randn(5, 16, requires_grad=True)
    ste(x, fake_quant_act).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def _tiny_qwen(dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wrap_unwrap_roundtrip_and_backward(dtype):
    m = _tiny_qwen(dtype).train()
    ids = torch.randint(0, 100, (2, 12))
    ref = m(ids).logits.detach().float()
    params_before = [id(p) for p in m.parameters()]

    n = wrap(m)
    assert n == 2 * 7                                    # 7 projections x 2 layers
    assert type(m.lm_head) is nn.Linear                  # lm_head untouched
    assert all(type(mod) is not FP8FakeQuantLinear for name, mod in m.named_modules() if "embed" in name)
    assert [id(p) for p in m.parameters()] == params_before

    out = m(ids, labels=ids)
    assert total_calls(m) == n                           # every wrapped layer actually ran fake-quant
    out.loss.backward()
    wrapped_w = [mod.weight for mod in m.modules() if type(mod) is FP8FakeQuantLinear]
    assert all(w.grad is not None and torch.isfinite(w.grad.float()).all() and w.grad.abs().sum() > 0
               for w in wrapped_w)
    qlogits = m(ids).logits.detach().float()
    assert not torch.equal(qlogits, ref)                 # FP8 numerics are visible...
    assert (qlogits - ref).abs().max() < 1.0             # ...but the model is still sane

    set_enabled(m, False)                                # disabled == original behaviour
    assert torch.allclose(m(ids).logits.float(), ref, atol=1e-5 if dtype == torch.float32 else 1e-2)
    assert unwrap(m) == n
    assert all(type(mod) is not FP8FakeQuantLinear for mod in m.modules())
    assert torch.equal(m(ids).logits.detach().float(), ref)
