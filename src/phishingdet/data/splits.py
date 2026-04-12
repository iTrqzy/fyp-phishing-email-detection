import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from phishingdet.data.loader import repo_root

# This function makes sure Stage 1, Stage 2, and Stage 3
# all use the exact same train/test rows.
# That makes the evaluation fair, because every model is tested on the same emails.

# THE USE OF AI WAS USED HERE TO HELP MAKE THIS FUNCTION
def get_or_make_split_indices(labels, test_size=0.2, random_state=42, stratify=True,
                               split_name="split_1"):
    artifacts_directory = repo_root() / "artifacts" / "splits"

    # Create the folder if it does not already exist
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    split_path = artifacts_directory / f"{split_name}.json"

    # If this split file already exists, load it and reuse it
    # This prevents creating a different split every time the code runs
    if split_path.exists():
        data = json.loads(split_path.read_text(encoding="utf-8"))

        # Convert saved Python lists back into NumPy arrays
        return np.array(data["train_idx"]), np.array(data["test_idx"])

    # Create an array of row indices:
    # If labels has length 100, this becomes [0, 1, 2, ..., 99]
    index = np.arange(len(labels))

    # Convert labels into a NumPy array
    y = np.array(labels)

    # Decide whether to use stratified splitting
    # Stratified means the train and test sets keep roughly the same class balance
    # as the original dataset, e.g. similar phishing/legit proportions
    # only do stratification if:
    # 1) stratify=True was requested
    # 2) there is more than one class in y
    #
    # If there is only one class, stratified splitting would fail
    strat = y if (stratify and len(set(y)) > 1) else None

    # Split the row indices into train and test indices
    # We split the indices, not the data itself, so the same split can be reused later
    train_idx, test_idx = train_test_split(
        index,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

    # Save useful information about the split into a dictionary
    # so it can be written to JSON and reused later
    out = {
        "split_name": split_name,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist()
    }

    # Write the split information to disk as a JSON file
    split_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Return the train/test indices as NumPy arrays
    return train_idx, test_idx