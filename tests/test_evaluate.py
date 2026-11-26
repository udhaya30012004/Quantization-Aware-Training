import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from evaluation.evaluate import nll_from_prompt_logprobs


def test_nll_counts_only_answer_tokens():
    ids = [10, 11, 12, 13, 14]          # first 3 are prompt
    lp = [None] + [{t: SimpleNamespace(logprob=-1.0)} for t in ids[1:]]
    lp[3] = {13: SimpleNamespace(logprob=-math.log(2))}
    lp[4] = {14: SimpleNamespace(logprob=-0.5)}
    total, n = nll_from_prompt_logprobs(ids, lp, n_prompt=3)
    assert n == 2 and abs(total - (math.log(2) + 0.5)) < 1e-9
