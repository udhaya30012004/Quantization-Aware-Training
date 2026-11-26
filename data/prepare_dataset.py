"""Create the ONE fixed GSM8K split used by every run, and record hashes in manifest.json.

7473 train examples -> shuffled with a fixed seed -> 473 val + 7000 train. Test (1319) is untouched.
Refuses to overwrite an existing split unless --force. `--verify` re-checks hashes only.

Usage:
    python data/prepare_dataset.py --out data
    python data/prepare_dataset.py --out data --verify
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SPLIT_SEED = 1234
N_VAL = 473
DATASET_ID = "openai/gsm8k"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_CALC = re.compile(r"<<[^>]*>>")


def clean_answer(answer):
    """Strip calculator annotations; return (target text ending in '#### N', final number str)."""
    answer = _CALC.sub("", answer).strip()
    reasoning, final = answer.rsplit("####", 1)
    final = final.strip().replace(",", "")
    target = reasoning.rstrip() + "\n#### " + final
    return target, final


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_rows(split, rows):
    out = []
    for i, ex in enumerate(rows):
        target, final = clean_answer(ex["answer"])
        out.append({"id": f"{split}-{i}", "question": ex["question"].strip(),
                    "target": target, "answer": final})
    return out


def verify(out):
    manifest = json.loads((out / "manifest.json").read_text())
    ok = True
    for name, info in manifest["files"].items():
        got = sha256_file(out / name)
        match = got == info["sha256"]
        ok &= match
        print(f"{'OK ' if match else 'BAD'} {name} n={info['n']} sha256={got[:12]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.verify:
        sys.exit(0 if verify(out) else 1)
    if (out / "manifest.json").exists() and not args.force:
        sys.exit("manifest.json exists - the split is fixed. Use --verify, or --force to regenerate.")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    ds_sha = api.dataset_info(DATASET_ID).sha
    model_sha = api.model_info(MODEL_ID).sha
    ds = load_dataset(DATASET_ID, "main", revision=ds_sha)
    train_all, test = list(ds["train"]), list(ds["test"])
    assert len(train_all) == 7473 and len(test) == 1319, (len(train_all), len(test))

    idx = list(range(len(train_all)))
    random.Random(SPLIT_SEED).shuffle(idx)
    val = [train_all[i] for i in idx[:N_VAL]]
    train = [train_all[i] for i in idx[N_VAL:]]
    assert len(train) == 7000 and len(val) == 473

    splits = {"train.jsonl": build_rows("train", train),
              "val.jsonl": build_rows("val", val),
              "test.jsonl": build_rows("test", test)}
    files = {}
    for name, rows in splits.items():
        write_jsonl(out / name, rows)
        files[name] = {"n": len(rows), "sha256": sha256_file(out / name)}

    manifest = {"dataset": DATASET_ID, "dataset_revision": ds_sha, "model": MODEL_ID,
                "model_revision": model_sha, "split_seed": SPLIT_SEED, "n_val": N_VAL, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
