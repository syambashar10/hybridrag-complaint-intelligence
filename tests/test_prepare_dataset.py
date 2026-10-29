import pandas as pd

from src.prepare_dataset import clean_text


def test_clean_text_removes_redaction_runs_and_whitespace():
    text = "  My account XXXX\n\nwas   reported incorrectly.  "
    assert clean_text(text) == "My account was reported incorrectly."


def test_clean_text_handles_missing_value():
    assert clean_text(pd.NA) == ""
