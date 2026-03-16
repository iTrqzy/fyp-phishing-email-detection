import json
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from phishingdet.data.loader import load_email, repo_root, dataset_path
from phishingdet.data.splits import get_or_make_split_indices
from phishingdet.features.build_features import fit_vectorizer, transform_vectorizer
from phishingdet.features.build_metadata_features import (
    fit_metadata_vectorizer,
    transform_metadata_vectorizer,
)
from phishingdet.evaluation.model_evaluations import (
    make_pred_row,
    timestamped_copy_path,
    write_test_predictions_csv,
)

RANDOM_STATE = 42


def best_threshold_by_f1(y_true, probabilities):
    best_threshold = 0.50
    best_f1_score = -1.0

    thresholds = np.arange(0.0, 1.01, 0.01)
    for threshold in thresholds:
        preds = (probabilities > threshold).astype(int)
        current_f1 = f1_score(y_true, preds, zero_division=0)

        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1_score)


def get_text_probabilities(train_texts, train_labels, valid_texts, max_features=5000):
    vec, X_train = fit_vectorizer(train_texts, max_features=max_features)
    X_valid = transform_vectorizer(vec, valid_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    return model.predict_proba(X_valid)[:, 1]


def get_metadata_probabilities(train_texts, train_labels, valid_texts):
    vec, X_train = fit_metadata_vectorizer(train_texts)
    X_valid = transform_metadata_vectorizer(vec, valid_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    return model.predict_proba(X_valid)[:, 1]


def train_hybrid_stack(test_size=0.2, n_folds=5, random_state=RANDOM_STATE, max_features=5000):
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    t0_total = time.perf_counter()

    # deterministic split (shared across stages)
    train_idx, test_idx = get_or_make_split_indices(
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
        split_name="phishing_email_split_1",
    )

    x_train_text = [texts[i] for i in train_idx]
    y_train = np.array([labels[i] for i in train_idx])

    x_test_text = [texts[i] for i in test_idx]
    y_test = np.array([labels[i] for i in test_idx])

    # Out-of-fold probs for the meta learner
    oof_text_prob = np.zeros(len(x_train_text), dtype=float)
    oof_meta_prob = np.zeros(len(x_train_text), dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for _, (train_rows, valid_rows) in enumerate(skf.split(x_train_text, y_train), start=1):
        fold_train_texts = [x_train_text[i] for i in train_rows]
        fold_train_labels = y_train[train_rows]
        fold_valid_texts = [x_train_text[i] for i in valid_rows]

        text_probs = get_text_probabilities(
            fold_train_texts, fold_train_labels, fold_valid_texts, max_features=max_features
        )
        meta_probs = get_metadata_probabilities(fold_train_texts, fold_train_labels, fold_valid_texts)

        for j, row_index in enumerate(valid_rows):
            oof_text_prob[row_index] = float(text_probs[j])
            oof_meta_prob[row_index] = float(meta_probs[j])

    meta_train_X = np.column_stack([oof_text_prob, oof_meta_prob])

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_train_X, y_train)

    # Train final base models on full training split
    text_vec, X_train_text = fit_vectorizer(x_train_text, max_features=max_features)
    X_test_text = transform_vectorizer(text_vec, x_test_text)

    text_model = LogisticRegression(max_iter=1000)
    text_model.fit(X_train_text, y_train)
    test_text_prob = text_model.predict_proba(X_test_text)[:, 1]

    meta_vec, X_train_meta = fit_metadata_vectorizer(x_train_text)
    X_test_meta = transform_metadata_vectorizer(meta_vec, x_test_text)

    metadata_model = LogisticRegression(max_iter=1000)
    metadata_model.fit(X_train_meta, y_train)
    test_meta_prob = metadata_model.predict_proba(X_test_meta)[:, 1]

    # Hybrid probabilities
    meta_test_X = np.column_stack([test_text_prob.astype(float), test_meta_prob.astype(float)])
    hybrid_prob = meta_model.predict_proba(meta_test_X)[:, 1]

    preds_0_5 = (hybrid_prob > 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, preds_0_5))
    precision = float(precision_score(y_test, preds_0_5, zero_division=0))
    recall = float(recall_score(y_test, preds_0_5, zero_division=0))
    f1 = float(f1_score(y_test, preds_0_5, zero_division=0))
    cm = confusion_matrix(y_test, preds_0_5)

    roc_auc = None
    pr_auc = None
    if len(set(y_test)) == 2:
        roc_auc = float(roc_auc_score(y_test, hybrid_prob))
        pr_auc = float(average_precision_score(y_test, hybrid_prob))

    best_threshold, best_f1 = best_threshold_by_f1(y_test, hybrid_prob)
    preds_best = (hybrid_prob > best_threshold).astype(int)

    runtime_total = float(time.perf_counter() - t0_total)

    print("##### STAGE 3 (HYBRID STACKING) EVALUATION #####")
    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("F1       :", round(f1, 3))
    print()
    print("ROC AUC:", None if roc_auc is None else round(roc_auc, 3))
    print("PR  AUC:", None if pr_auc is None else round(pr_auc, 3))
    print("Best threshold (F1):", best_threshold, "| F1:", round(best_f1, 3))
    print()
    print("Confusion matrix:")
    print(cm)
    print()
    print("Classification report:")
    print(classification_report(y_test, preds_0_5, zero_division=0))
    print()

    # Save stage 3 artifacts
    stage3_dir = repo_root() / "artifacts" / "stage3_hybrid"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(text_model, stage3_dir / "text_model.joblib")
    joblib.dump(text_vec, stage3_dir / "text_vectorizer.joblib")
    joblib.dump(metadata_model, stage3_dir / "metadata_model.joblib")
    joblib.dump(meta_vec, stage3_dir / "metadata_vectorizer.joblib")
    joblib.dump(meta_model, stage3_dir / "meta_model.joblib")

    # Save results JSON
    results = {
        "stage": "stage3_hybrid_stacking",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path()),
        "split_name": "phishing_email_split_1",
        "split": {"train": int(len(train_idx)), "test": int(len(test_idx)), "test_size": float(test_size)},
        "n_folds": int(n_folds),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold_f1": float(best_threshold),
            "best_f1_at_threshold": float(best_f1),
        },
        "confusion_matrix": cm.tolist(),
        "meta_weights": {
            "text_prob_weight": float(meta_model.coef_[0][0]),
            "metadata_prob_weight": float(meta_model.coef_[0][1]),
            "intercept": float(meta_model.intercept_[0]),
        },
        "runtime_seconds": {"total_s": runtime_total},
    }

    latest_json = stage3_dir / "results.json"
    latest_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_json = stage3_dir / ("results_" + stamp + ".json")
    run_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved Stage 3 artifacts to:", stage3_dir)
    print("Saved latest results to:", latest_json)
    print("Saved run results to:", run_json)

    # ===== EVAL EXPORT (CSV for PR/ROC/threshold plots) =====
    eval_dir = repo_root() / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_csv = eval_dir / "stage3_test_preds.csv"
    run_csv = timestamped_copy_path(base_csv)

    rows = []
    for i in range(len(y_test)):
        row = make_pred_row(
            row_id=i,
            y_true=y_test[i],
            prob=hybrid_prob[i],
            pred_0_5=preds_0_5[i],
            pred_best=preds_best[i],
            extra={
                "text_prob": test_text_prob[i],
                "meta_prob": test_meta_prob[i],
                "hybrid_prob": hybrid_prob[i],
            },
        )

        # tiny improvements (for consistency across stages)
        row["stage"] = "stage3"
        row["threshold_best"] = float(best_threshold)

        rows.append(row)

    write_test_predictions_csv(base_csv, rows)
    write_test_predictions_csv(run_csv, rows)

    print("Saved eval CSV:", base_csv)
    print("Saved eval CSV (run):", run_csv)


if __name__ == "__main__":
    train_hybrid_stack(test_size=0.2, n_folds=5)