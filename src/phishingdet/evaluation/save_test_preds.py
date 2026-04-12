from pathlib import Path
from datetime import datetime
import pandas as pd

from phishingdet.data.loader import repo_root


def save_test_predictions_csv(
    stage_name,
    y_true,
    prob,
    pred_0_5,
    pred_best=None,
    threshold_best=None,
    extra_cols=None,
    also_save_timestamped=True,
):
    """
    Save per-sample test predictions into artifacts/eval/.

    Parameters:
    - stage_name: name of the model stage, e.g. "stage1", "stage2", "stage3"
    - y_true: true labels for each test sample
    - prob: predicted phishing probabilities for each sample
    - pred_0_5: predicted class labels using the default threshold 0.5
    - pred_best: optional predicted class labels using the best threshold found from evaluation
    - threshold_best: optional best threshold value to store for transparency
    - extra_cols: optional dictionary of extra columns to include in the CSV
    - also_save_timestamped: if True, save an extra timestamped copy as well
    """

    # Build the output folder path: repo_root()/artifacts/eval
    eval_dir = repo_root() / "artifacts" / "eval"

    # Create the folder if it does not already exist
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Number of rows/samples in the test set
    n = len(y_true)

    # Create a DataFrame where each row is one test sample
    df = pd.DataFrame({
        # Repeat the stage name for every row so the file clearly shows
        # which model stage produced the predictions
        "stage": [stage_name] * n,

        # Add a simple row id: 0, 1, 2, ...
        "id": list(range(n)),

        # Ground-truth label for each test sample
        "y_true": list(y_true),

        # Predicted probability for each sample
        "prob": list(prob),

        # Predicted class using threshold = 0.5
        "pred_0_5": list(pred_0_5),
    })

    # If predictions using the best threshold were provided, store them too
    if pred_best is not None:
        df["pred_best"] = list(pred_best)

    # Store the best threshold value in every row
    # Repeating it per row is slightly redundant, but makes the CSV self-contained
    # and easier to read later in reports/debugging
    if threshold_best is not None:
        df["threshold_best"] = [float(threshold_best)] * n
    else:
        # If no best threshold is provided, fill the column with blank strings
        df["threshold_best"] = [""] * n

    # If extra columns were provided, add each one to the DataFrame
    # extra_cols should look like:
    # {"text_prob": [...], "metadata_prob": [...]}
    if extra_cols:
        for col_name, values in extra_cols.items():
            df[col_name] = list(values)

    # Save the main "latest" file, which always overwrites the previous latest version
    latest_path = eval_dir / f"{stage_name}_test_preds.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8")

    # By default, also save a timestamped copy so older runs are preserved
    timestamped_path = None
    if also_save_timestamped:
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        timestamped_path = eval_dir / f"{stage_name}_test_preds_{stamp}.csv"

        # Save the timestamped CSV
        df.to_csv(timestamped_path, index=False, encoding="utf-8")

    return latest_path, timestamped_path