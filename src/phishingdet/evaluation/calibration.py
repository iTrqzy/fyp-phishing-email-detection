from pathlib import Path
import pandas as pd

# Numeric labels used across the project
PHISHING_LABEL = 1
LEGIT_LABEL = 0



DATASET_FILENAME = "Phishing_Email.csv"



def repo_root():
    return Path(__file__).resolve().parents[3]


def dataset_path():
    return repo_root() / "data" / "raw" / DATASET_FILENAME


def load_email():
    path = dataset_path()
    df = pd.read_csv(path)

    # Case 1:
    # Dataset already uses the expected schema: columns named "text" and "label"
    if {"text", "label"}.issubset(df.columns):
        # Keep only the required columns
        df = df[["text", "label"]].copy()

        # Ensure text is stored as clean strings
        df["text"] = df["text"].astype(str).str.strip()

        # Ensure label is numeric
        df["label"] = df["label"].astype(int)

    # Case 2:
    # Dataset uses alternative column names: "Email Text" and "Email Type"
    elif {"Email Text", "Email Type"}.issubset(df.columns):
        # Keep only the required columns
        df = df[["Email Text", "Email Type"]].copy()

        # Rename them into the standard schema expected by the rest of the project
        df = df.rename(columns={"Email Text": "text", "Email Type": "label"})

        # Clean text column
        df["text"] = df["text"].astype(str).str.strip()

        # Clean label column and normalize to lowercase for easier matching
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        # Maps string label values to numeric 0/1 labels
        mapping = {
            "safe email": LEGIT_LABEL,
            "legit email": LEGIT_LABEL,
            "ham": LEGIT_LABEL,
            "phishing email": PHISHING_LABEL,
            "phishing": PHISHING_LABEL,
            "spam": PHISHING_LABEL,
        }

        # Replace string labels with numeric labels
        df["label"] = df["label"].map(mapping)

        # If some labels were not recognised by the mapping,
        # show the most common original label values to help debugging
        if df["label"].isna().any():
            unknown = (
                pd.read_csv(path)[["Email Type"]]
                .astype(str) # Converts values to string
                .dropna()["Email Type"] # Removes rows where values are missing
                .str.strip() # Removes any spacing in the rows
                .value_counts() # Counts how many times each label appears
                .head(20) # Keeps the top 20 most common values
            )
            raise ValueError(
                "Found label values that loader.py doesn't recognise.\n"
                "Update the mapping dict.\n\n"
                f"Top label values:\n{unknown}"
            )

        # Convert mapped labels to integers
        df["label"] = df["label"].astype(int)

    # If the CSV does not match either supported schema, raise an error
    else:
        raise ValueError(
            "CSV schema not recognised. Expected either:\n"
            "  - columns: text, label\n"
            "  - columns: Email Text, Email Type\n\n"
            f"Got columns:\n{list(df.columns)}"
        )


    # Remove rows where text or label is missing
    df = df.dropna(subset=["text", "label"])

    # Remove rows where text is empty after cleaning
    df = df[df["text"].str.len() > 0].copy()

    # Remove duplicate email texts
    # This helps reduce train/test leakage if the same email appears multiple times
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Return the cleaned DataFrame
    return df