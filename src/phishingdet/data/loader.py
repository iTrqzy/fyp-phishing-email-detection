from pathlib import Path
import pandas as pd

PHISHING_LABEL = 1
LEGIT_LABEL = 0

# MANUAL: change this when you switch datasets
DATASET_FILENAME = "Phishing_Email.csv"
# e.g. "sample_emails_50k.csv"


def repo_root():
    # src/phishingdet/data/loader.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[3]


def dataset_path():
    return repo_root() / "data" / "raw" / DATASET_FILENAME


def load_email():
    path = dataset_path()
    df = pd.read_csv(path)

    if {"text", "label"}.issubset(df.columns):
        df = df[["text", "label"]].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

    elif {"Email Text", "Email Type"}.issubset(df.columns):
        df = df[["Email Text", "Email Type"]].copy()
        df = df.rename(columns={"Email Text": "text", "Email Type": "label"})

        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        # Maps different label strings -> 0/1
        mapping = {
            "safe email": LEGIT_LABEL,
            "legit email": LEGIT_LABEL,
            "ham": LEGIT_LABEL,
            "phishing email": PHISHING_LABEL,
            "phishing": PHISHING_LABEL,
            "spam": PHISHING_LABEL,
        }

        df["label"] = df["label"].map(mapping)

        # If mapping failed for some rows, show what labels exist
        if df["label"].isna().any():
            unknown = (
                pd.read_csv(path)[["Email Type"]]
                .astype(str)
                .dropna()["Email Type"]
                .str.strip()
                .value_counts()
                .head(20)
            )
            raise ValueError(
                "Found label values that loader.py doesn't recognise.\n"
                "Update the mapping dict.\n\n"
                f"Top label values:\n{unknown}"
            )

        df["label"] = df["label"].astype(int)

    else:
        raise ValueError(
            "CSV schema not recognised. Expected either:\n"
            "  - columns: text, label\n"
            "  - columns: Email Text, Email Type\n\n"
            f"Got columns:\n{list(df.columns)}"
        )

    # ---------- basic cleaning ----------
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0].copy()

    # removes duplicate texts to reduce train/test leakage
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df
