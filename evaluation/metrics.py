"""Answer extraction and statistics shared by every evaluation cell."""
import math
import random
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_number(tok: str):
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def extract_answer(text):
    """Return the model's final numeric answer as a float, or None.

    Rule (identical for every cell): if the LAST non-empty line contains '####', take the first number
    after the last '####' on that line (the training format ends with '#### N'). Otherwise take the last
    number anywhere in the text. A '####' earlier in the text (e.g. a premature guess written before the
    reasoning, or a markdown header) is ignored. Commas and '$' are ignored.
    """
    if text is None:
        return None
    text = text.replace("$", "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and "####" in lines[-1]:
        m = _NUM.search(lines[-1].rsplit("####", 1)[1])
        return _to_number(m.group()) if m else None
    nums = _NUM.findall(text)
    return _to_number(nums[-1]) if nums else None


def gold_answer(answer_field: str) -> float:
    """GSM8K gold: the number after '####' in the reference answer."""
    val = extract_answer(answer_field)
    if val is None:
        raise ValueError(f"no gold answer in: {answer_field!r}")
    return val


def is_correct(pred_text, gold: float) -> bool:
    pred = extract_answer(pred_text)
    return pred is not None and math.isclose(pred, gold, rel_tol=0.0, abs_tol=1e-6)


def accuracy(correct_flags):
    flags = list(correct_flags)
    return sum(flags) / len(flags)


def mcnemar_exact(a_flags, b_flags) -> float:
    """Two-sided exact McNemar p-value for paired correctness vectors."""
    b = sum(1 for x, y in zip(a_flags, b_flags) if x and not y)
    c = sum(1 for x, y in zip(a_flags, b_flags) if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap_ci(a_flags, b_flags, n_boot=10000, seed=0, alpha=0.05):
    """95% CI (0-1 accuracy scale) of mean(b) - mean(a) via paired bootstrap."""
    diffs = [int(y) - int(x) for x, y in zip(a_flags, b_flags)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
