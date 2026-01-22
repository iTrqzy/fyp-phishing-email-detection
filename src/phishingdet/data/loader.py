from pathlib import Path
import pandas as pd

PHISHING_LABEL = 1
LEGIT_LABEL = 0

# Change this to switch datasets
DATASET_FILENAME = "sample_emails_50k.csv"


def repo_root():
    # Cross-platform: finds repo root based on this file's location
    return Path(__file__).resolve().parents[3]


def dataset_path():
    return repo_root() / "data" / "raw" / DATASET_FILENAME


def load_email():
    csv_file_path = dataset_path()

    if not csv_file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path)

    # basic checks
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: 'text' and 'label'")

    df = df[["text", "label"]].copy()

    # formatting
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # safety check: labels should be 0/1
    bad = set(df["label"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"Labels must be 0/1 only. Found: {sorted(list(bad))}")

    return df
