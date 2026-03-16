import csv
from datetime import datetime


def make_pred_row(row_id, y_true, prob, pred_0_5, pred_best, extra=None):
    row = {
        "id": row_id,
        "y_true": int(y_true),
        "prob": float(prob),
        "pred_0_5": int(pred_0_5),
        "pred_best": int(pred_best),
    }
    if extra:
        for k, v in extra.items():
            row[k] = float(v) if v is not None else ""
    return row


def write_test_predictions_csv(output_path, rows):
    """
    rows: list of dicts (all same keys)
    """
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def timestamped_copy_path(base_path):
    """
    If base_path is: artifacts/eval/stage3_test_preds.csv
    returns:          artifacts/eval/stage3_test_preds_YYYY-MM-DDTHH-MM-SS.csv
    """
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = base_path.stem + "_" + stamp + base_path.suffix
    return base_path.with_name(name)
