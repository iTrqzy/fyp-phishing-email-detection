import pandas as pd

from phishingdet.data.loader import load_email

def test_load_email_returns_dataframe():
    df = load_email()
    assert isinstance(df, pd.DataFrame)

def test_load_email_has_columns():
    df = load_email()
    assert {"text", "label"}.issubset(df.columns)

def test_load_email_has_correct_types():
    df = load_email()
    labels = set(df["label"].unique().tolist())
    assert labels.issubset({0, 1})

def test_load_email_text_not_empty():
    df = load_email()
    assert df["text"].isna().sum() == 0
    assert (df["text"].astype(str).str.len() > 0).all()
