import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from phishingdet.data.loader import repo_root

# Ensures that Stage 1, Stage 2, & Stage 3 are all tested on the exact rows

# THE USE OF AI WAS USED HERE TO HELP MAKE THIS FUNCTION
def get_or_make_split_indicies(labels, test_size=0.2, random_state=42,stratify=True,
                               split_name="split_1"):
    artifacts_directory = repo_root()/"artifacts"/"splits"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    split_path = artifacts_directory / f"{split_name}.json"

    if split_path.exists():
        data = json.loads(split_path.read_text(encoding="utf-8"))
        return np.array(data["train_idx"]), np.array(data["test_idx"])

    index = np.arange(len(labels))
    y = np.array(labels)

    strat = y if (stratify and len(set(y)) > 1) else None

    train_idx, test_idx = train_test_split(
        index,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

    out = {
        "split_name": split_name,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist()
    }
    split_path.write_text(json.dumps(out,indent=2), encoding="utf-8")
    return train_idx, test_idx