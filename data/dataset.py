"""Chat-template formatting, tokenization and assistant-only label masking (shared by all arms)."""
import json

SYSTEM_PROMPT = "Solve the math problem step by step. End with '#### <number>'."
IGNORE = -100


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_messages(question, system_prompt=SYSTEM_PROMPT):
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]


def build_prompt_text(tokenizer, question, system_prompt=SYSTEM_PROMPT):
    return tokenizer.apply_chat_template(prompt_messages(question, system_prompt), tokenize=False,
                                         add_generation_prompt=True)


def tokenize_example(tokenizer, ex, max_len=512, system_prompt=SYSTEM_PROMPT):
    """Return dict(input_ids, labels, truncated). Loss only on the answer tokens + <|im_end|>."""
    prompt_text = build_prompt_text(tokenizer, ex["question"], system_prompt)
    full_text = prompt_text + ex["target"] + "<|im_end|>"  # Qwen template minus the trailing newline
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt tokens are not a prefix of the full sequence"
    labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids):]
    truncated = len(full_ids) > max_len
    return {"input_ids": full_ids[:max_len], "labels": labels[:max_len], "truncated": truncated}


class GSM8KDataset:
    def __init__(self, path, tokenizer, max_len=512, limit=None, system_prompt=SYSTEM_PROMPT):
        rows = load_jsonl(path)
        if limit:
            rows = rows[:limit]
        self.items = [tokenize_example(tokenizer, r, max_len, system_prompt) for r in rows]
        self.n_truncated = sum(i["truncated"] for i in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_token_id):
    def collate(batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate
