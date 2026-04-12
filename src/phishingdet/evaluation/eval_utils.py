import csv
from pathlib import Path


def ensure_eval_dir(repo_root):
    # Build the path: repo_root()/artifacts/eval
    eval_dir = Path(repo_root()) / "artifacts" / "eval"

    # Create the folder if it does not already exist
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Return the folder path so other functions can save files there
    return eval_dir


def save_stage1_test_preds_csv(repo_root, out_name, y_true, prob, pred_0_5, pred_best, threshold_best):
    """
    Save per-sample Stage 1 test predictions to a CSV file.

    This is useful because it stores the model's output for every test email,
    so PR curves, ROC curves, and other analysis can be plotted later.

    Columns saved:
    - id: row number in the saved file
    - y_true: the actual true label
    - prob: predicted phishing probability
    - pred_0_5: predicted class using threshold 0.5
    - pred_best: predicted class using the best threshold found during evaluation
    - threshold_best: the best threshold value used
    """

    # Make sure the evaluation folder exists
    eval_dir = ensure_eval_dir(repo_root)

    # Build the full output file path inside artifacts/eval
    out_path = eval_dir / out_name

    # Open the CSV file for writing
    # newline="" avoids blank lines on some systems
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write the CSV header row
        writer.writerow(["id", "y_true", "prob", "pred_0_5", "pred_best", "threshold_best"])

        # Write one row per test sample
        for i in range(len(y_true)):
            writer.writerow([
                i,                        # simple row id
                int(y_true[i]),          # true label
                float(prob[i]),          # predicted phishing probability
                int(pred_0_5[i]),        # predicted class at threshold 0.5
                int(pred_best[i]),       # predicted class at best threshold
                float(threshold_best),   # repeat best threshold on every row
            ])

    # Return the path to the saved CSV file
    return out_path