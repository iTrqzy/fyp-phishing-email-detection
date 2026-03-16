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
    Saves per-sample test predictions to artifacts/eval/.

    Adds:
      - stage column (stage1/stage2/stage3)
      - threshold_best column (repeated per row, but useful for transparency)
    """
    eval_dir = repo_root() / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    n = len(y_true)

    df = pd.DataFrame({
        "stage": [stage_name] * n,
        "id": list(range(n)),
        "y_true": list(y_true),
        "prob": list(prob),
        "pred_0_5": list(pred_0_5),
    })

    if pred_best is not None:
        df["pred_best"] = list(pred_best)

    # repeat the threshold value per row (nice for report + debugging)
    if threshold_best is not None:
        df["threshold_best"] = [float(threshold_best)] * n
    else:
        df["threshold_best"] = [""] * n

    if extra_cols:
        for col_name, values in extra_cols.items():
            df[col_name] = list(values)

    latest_path = eval_dir / f"{stage_name}_test_preds.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8")

    timestamped_path = None
    if also_save_timestamped:
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        timestamped_path = eval_dir / f"{stage_name}_test_preds_{stamp}.csv"
        df.to_csv(timestamped_path, index=False, encoding="utf-8")

    return latest_path, timestamped_path