import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F
from evaluation.kl_divergence import kl_from_logits, js_from_logits, kl_bucket_counts, batch_kl
from models import fp8_fake_quant as fq


def test_kl_matches_torch_reference_and_is_zero_for_identical():
    torch.manual_seed(0)
    a, b = torch.randn(7, 50), torch.randn(7, 50)
    ref = F.kl_div(F.log_softmax(b, -1), F.log_softmax(a, -1), log_target=True, reduction="none").sum(-1)
    assert torch.allclose(kl_from_logits(a, b), ref, atol=1e-5)
    assert torch.all(kl_from_logits(a, b) >= -1e-6)
    assert torch.allclose(kl_from_logits(a, a), torch.zeros(7), atol=1e-6)


def test_js_is_symmetric_bounded_and_zero_for_identical():
    import math
    torch.manual_seed(1)
    a, b = torch.randn(9, 50), torch.randn(9, 50)
    js_ab = js_from_logits(a, b)
    js_ba = js_from_logits(b, a)
    assert torch.allclose(js_ab, js_ba, atol=1e-6)
    assert torch.all(js_ab >= -1e-6) and torch.all(js_ab <= math.log(2.0) + 1e-6)
    assert torch.allclose(js_from_logits(a, a), torch.zeros(9), atol=1e-6)


def test_kl_bucket_counts_partition_all_tokens():
    vals = torch.tensor([-1e-7, 0.0, 0.005, 0.0099, 0.01, 0.03, 0.0499, 0.05, 0.099, 0.10, 1.5])
    b = kl_bucket_counts(vals)
    assert b == {"lt_001": 4, "001_005": 3, "005_010": 2, "ge_010": 2}
    assert sum(b.values()) == len(vals)


def test_batch_kl_only_counts_answer_positions_and_fp8_changes_outputs():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    fq.wrap(m)
    ids = torch.randint(0, 100, (3, 12))
    att = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :6] = -100                                  # first 6 tokens are prompt
    s, n, agree, kl = batch_kl(m, ids, att, labels)
    assert n == 3 * 6                                    # answer tokens are positions 6..11 -> 6 per row
    assert s > 0 and kl.shape[0] == n and 0 <= agree <= n
    # with fake-quant disabled for both passes, KL must be exactly zero
    fq.set_enabled(m, False)
    orig = fq.set_enabled
    try:
        fq.set_enabled = lambda model, enabled: orig(model, False)   # force both passes plain
        s0, n0, agree0, _ = batch_kl(m, ids, att, labels)
    finally:
        fq.set_enabled = orig
    assert s0 == 0.0 and agree0 == n0
