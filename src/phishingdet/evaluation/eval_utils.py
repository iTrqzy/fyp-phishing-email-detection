import csv
from pathlib import Path


def ensure_eval_dir(repo_root):
    eval_dir = Path(repo_root()) / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def save_stage1_test_preds_csv(repo_root, out_name, y_true, prob, pred_0_5, pred_best, threshold_best):
    """
    Saves per-sample outputs for Stage 1 so you can plot PR/ROC curves later.
    Columns: id, y_true, prob, pred_0_5, pred_best, threshold_best
    """
    eval_dir = ensure_eval_dir(repo_root)
    out_path = eval_dir / out_name

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "y_true", "prob", "pred_0_5", "pred_best", "threshold_best"])
        for i in range(len(y_true)):
            writer.writerow([
                i,
                int(y_true[i]),
                float(prob[i]),
                int(pred_0_5[i]),
                int(pred_best[i]),
                float(threshold_best),
            ])

    return out_path
