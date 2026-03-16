from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)


def load_preds(csv_path: Path):
    df = pd.read_csv(csv_path)
    # expecting at least: y_true, prob
    if "y_true" not in df.columns or "prob" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: y_true, prob")
    return df["y_true"].values, df["prob"].values


def main():
    repo_root = Path(__file__).resolve().parents[3]  # .../src/phishingdet/evaluation -> repo
    eval_dir = repo_root / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stage_files = [
        ("Stage 1 (Text-only)", eval_dir / "stage1_test_preds.csv"),
        ("Stage 2 (Metadata-only)", eval_dir / "stage2_test_preds.csv"),
        ("Stage 3 (Hybrid)", eval_dir / "stage3_test_preds.csv"),
    ]

    # Line styles so overlapping curves stay visible
    styles = [
        {"linestyle": "-",  "linewidth": 2.8, "marker": None},
        {"linestyle": "--", "linewidth": 2.8, "marker": None},
        {"linestyle": "-.", "linewidth": 3.2, "marker": None},
    ]

    # -------------------------
    # Precision-Recall Curves
    # -------------------------
    plt.figure()
    pr_summary = []

    for (label, path), style in zip(stage_files, styles):
        y_true, prob = load_preds(path)

        precision, recall, _ = precision_recall_curve(y_true, prob)
        pr_auc = average_precision_score(y_true, prob)
        pr_summary.append((label, pr_auc))

        plt.plot(
            recall,
            precision,
            label=f"{label} (PR-AUC={pr_auc:.3f})",
            **style,
        )

    plt.title("Precision-Recall Curves (Test Set)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left")

    pr_out = eval_dir / "pr_curves.png"
    plt.tight_layout()
    plt.savefig(pr_out, dpi=200)
    plt.close()

    # -------------------------
    # ROC Curves
    # -------------------------
    plt.figure()
    roc_summary = []

    for (label, path), style in zip(stage_files, styles):
        y_true, prob = load_preds(path)

        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        roc_summary.append((label, auc))

        plt.plot(
            fpr,
            tpr,
            label=f"{label} (ROC-AUC={auc:.3f})",
            **style,
        )

    # No-skill baseline (random guess)
    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=2.0, label="No-skill (random)")

    plt.title("ROC Curves (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")

    roc_out = eval_dir / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(roc_out, dpi=200)
    plt.close()

    # -------------------------
    # Console summary (so you can sanity-check quickly)
    # -------------------------
    print("Saved:", pr_out)
    print("Saved:", roc_out)
    print("\nPR-AUC summary:")
    for label, val in pr_summary:
        print(f"  {label}: {val:.4f}")

    print("\nROC-AUC summary:")
    for label, val in roc_summary:
        print(f"  {label}: {val:.4f}")


if __name__ == "__main__":
    main()
