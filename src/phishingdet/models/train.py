import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)

from phishingdet.data.loader import load_email, repo_root, dataset_path
from phishingdet.features.build_features import fit_vectorizer, transform_vectorizer
from phishingdet.evaluation.eval_utils import save_stage1_test_preds_csv


def best_threshold_by_f1(y_true, prob):
    # Try thresholds from 0.00 -> 1.00
    thresholds = np.arange(0.0, 1.01, 0.01)

    best_threshold = 0.50
    best_f1_score = -1.0

    for threshold in thresholds:
        preds = (prob >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)

        if score > best_f1_score:
            best_f1_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1_score


def train_model(test_size=0.2, random_state=42, max_features=5000):
    # load the data and make it into a list form
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # if dataset is very small, there is a failsafe
    # stratify tries to keep the same class balance in train and test
    use_stratify = len(set(labels)) > 1 and len(labels) >= 6

    # split training & testing
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if use_stratify else None
    )

    # converting text to numbers
    vectorizer, X_train = fit_vectorizer(x_train_text, max_features=max_features)
    X_test = transform_vectorizer(vectorizer, x_test_text)

    # train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # outcomes (default 0/1)
    preds_0_5 = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, preds_0_5))
    precision = float(precision_score(y_test, preds_0_5, zero_division=0))
    recall = float(recall_score(y_test, preds_0_5, zero_division=0))
    f1 = float(f1_score(y_test, preds_0_5, zero_division=0))

    print("Model evaluation:")
    print("  Accuracy :", round(accuracy, 3))
    print("  Precision:", round(precision, 3))
    print("  Recall   :", round(recall, 3))
    print("  F1 Score :", round(f1, 3))
    print()

    print("Dataset info:")
    print("  Dataset path:", dataset_path())
    print("  Total rows:", len(df))
    print("  Label counts:")
    print(df["label"].value_counts())
    print()

    print("Split info:")
    print("  Train size:", len(y_train))
    print("  Test size :", len(y_test))
    print("  test_size :", test_size)
    print()

    cm = confusion_matrix(y_test, preds_0_5)
    print("Confusion matrix:")
    print(cm)
    print()

    print("Classification report:")
    print(classification_report(y_test, preds_0_5, zero_division=0))
    print()

    # probabilities + extra metrics
    roc_auc = None
    pr_auc = None
    best_t = None
    best_f1_at_t = None

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, prob))
        pr_auc = float(average_precision_score(y_test, prob))
        best_t, best_f1_at_t = best_threshold_by_f1(np.array(y_test), prob)

        print("ROC-AUC :", round(roc_auc, 3))
        print("PR-AUC  :", round(pr_auc, 3))
        print("Best threshold (F1):", best_t, "| F1:", round(best_f1_at_t, 3))
        print()

        preds_best = (prob >= best_t).astype(int)

        # save per-sample test probabilities for plots
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_name = f"stage1_test_preds_{stamp}.csv"
        out_path = save_stage1_test_preds_csv(
            repo_root,
            out_name=out_name,
            y_true=np.array(y_test),
            prob=prob,
            pred_0_5=np.array(preds_0_5),
            pred_best=np.array(preds_best),
            threshold_best=best_t,
        )
        # also overwrite a “latest”
        save_stage1_test_preds_csv(
            repo_root,
            out_name="stage1_test_preds.csv",
            y_true=np.array(y_test),
            prob=prob,
            pred_0_5=np.array(preds_0_5),
            pred_best=np.array(preds_best),
            threshold_best=best_t,
        )
        print("Saved Stage 1 test preds to:", out_path)
        print()

    # save artifacts
    artifacts_dir = repo_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    vec_path = artifacts_dir / "vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print("Saved model to:", model_path)
    print("Saved vectorizer to:", vec_path)

    # save results.json (keep it simple)
    results = {
        "stage": "stage1_text_only",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path()),
        "rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "split": {
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "test_size_ratio": float(test_size),
            "random_state": int(random_state),
            "use_stratify": bool(use_stratify),
        },
        "vectorizer": {
            "type": "tfidf",
            "max_features": int(max_features),
            "ngram_range": [1, 2],
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold_f1": best_t,
            "best_f1_at_threshold": best_f1_at_t,
        },
        "confusion_matrix": cm.tolist(),
    }

    latest_path = artifacts_dir / "results.json"
    latest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_path = artifacts_dir / f"results_{stamp}.json"
    run_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved latest results to:", latest_path)
    print("Saved run results to:", run_path)


if __name__ == "__main__":
    train_model()
