import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
from data.dataset import IGNORE, load_jsonl, tokenize_example, build_prompt_text
from data.prepare_dataset import clean_answer, sha256_file
from evaluation.metrics import gold_answer

DATA = ROOT / "data"
needs_data = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="run data/prepare_dataset.py first")


def test_clean_answer_strips_calculator():
    target, final = clean_answer("Tom has 2+3=<<2+3=5>>5 apples.\n#### 5")
    assert "<<" not in target and target == "Tom has 2+3=5 apples.\n#### 5" and final == "5"


@needs_data
def test_split_sizes_and_hashes():
    m = json.loads((DATA / "manifest.json").read_text())
    assert {k: v["n"] for k, v in m["files"].items()} == {"train.jsonl": 7000, "val.jsonl": 473, "test.jsonl": 1319}
    for name, info in m["files"].items():
        assert sha256_file(DATA / name) == info["sha256"], f"{name} changed since the split was created"


@needs_data
def test_no_overlap_and_clean_targets():
    sets = {n: load_jsonl(DATA / f"{n}.jsonl") for n in ("train", "val", "test")}
    q = {n: {r["question"] for r in rows} for n, rows in sets.items()}
    assert not (q["train"] & q["val"]) and not (q["train"] & q["test"]) and not (q["val"] & q["test"])
    for rows in sets.values():
        for r in rows:
            assert "<<" not in r["target"] and r["target"].rsplit("####", 1)[1].strip() == r["answer"]
            assert gold_answer(r["target"]) == float(r["answer"])


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


def test_label_masking(tok):
    ex = {"question": "What is 2+3?", "target": "2+3=5\n#### 5"}
    out = tokenize_example(tok, ex, max_len=512)
    ids, labels = out["input_ids"], out["labels"]
    assert len(ids) == len(labels) and not out["truncated"]
    n_prompt = len(tok(build_prompt_text(tok, ex["question"]), add_special_tokens=False)["input_ids"])
    assert all(l == IGNORE for l in labels[:n_prompt])           # prompt fully masked
    assert labels[n_prompt:] == ids[n_prompt:]                    # answer fully supervised
    assert tok.decode(labels[n_prompt:]) == "2+3=5\n#### 5<|im_end|>"


@needs_data
def test_no_truncation_at_512(tok):
    from data.dataset import GSM8KDataset
    for split in ("train", "val", "test"):
        ds = GSM8KDataset(DATA / f"{split}.jsonl", tok, max_len=512)
        assert ds.n_truncated == 0, f"{split}: {ds.n_truncated} examples exceed 512 tokens"
