from pathlib import Path
import pandas as pd

PHISHING_LABEL = 1
LEGIT_LABEL = 0

# cross-platform compatability
def repo_root():
    return Path(__file__).resolve().parents[3]


def load_email():
    csv_file_path = repo_root() / "data" / "raw" / "sample_emails_50k.csv"
    # read the csv file
    df = pd.read_csv(csv_file_path)

    # basic checks
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: 'text' and 'label'")

    df = df[["text","label"]].copy()

    # formatting
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    return df
