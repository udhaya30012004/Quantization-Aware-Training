import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from evaluation.metrics import extract_answer, gold_answer, is_correct, mcnemar_exact, paired_bootstrap_ci


@pytest.mark.parametrize("text,expected", [
    ("so 3+4=7\n#### 7", 7.0),
    ("blah #### 1,250", 1250.0),
    ("cost is $18.50 total\n#### $18.50", 18.5),
    ("the answer is 42.", 42.0),                       # no ####: last number
    ("first 5 then 6 finally 9 apples", 9.0),
    ("#### 5 and then #### 12", 12.0),                  # last ####
    ("#### -3", -3.0),
    ("#### 160\n\nThus Janet makes **$18** a day.", 18.0),   # premature #### guess is ignored
    ("#### Step 3: add\nso 5+4=9\n#### 9", 9.0),
    ("no digits here", None),
    ("#### none", None),
    ("", None),
])
def test_extract(text, expected):
    assert extract_answer(text) == expected


def test_gold_and_correct():
    assert gold_answer("reasoning\n#### 1,000") == 1000.0
    assert is_correct("... #### 72", 72.0)
    assert is_correct("it is 72.0", 72.0)
    assert not is_correct("#### 71", 72.0)
    assert not is_correct("no number", 72.0)


def test_mcnemar_and_bootstrap():
    a = [1] * 50 + [0] * 50
    assert mcnemar_exact(a, a) == 1.0
    b = [1] * 50 + [1] * 30 + [0] * 20          # b fixes 30 of a's errors, breaks none
    assert mcnemar_exact(a, b) < 1e-6
    lo, hi = paired_bootstrap_ci(a, b, n_boot=2000)
    assert 0.2 < lo < 0.3 < hi < 0.4
