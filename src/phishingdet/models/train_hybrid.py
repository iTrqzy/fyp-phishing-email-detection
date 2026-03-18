import json
import time
from datetime import datetime
import pandas as pd

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

def save_error_analysis_csv(test_texts, true_labels, predicted_labels, predicted_probabilities, output_csv_path, max_rows=200):
    # Save the misclassified test examples for easier qualitative analysis in Chapter 4.
    error_rows = []

    for row_index in range(len(true_labels)):
        true_label = int(true_labels[row_index])
        predicted_label = int(predicted_labels[row_index])

        # Skip correctly classified examples
        if true_label == predicted_label:
            continue

        if true_label == 0 and predicted_label == 1:
            error_type = "false_positive"
        else:
            error_type = "false_negative"

        phishing_probability = (
            float(predicted_probabilities[row_index])
            if predicted_probabilities is not None
            else None
        )

        text_preview = str(test_texts[row_index]).replace("\n", " ").strip()
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."

        error_rows.append(
            {
                "error_type": error_type,
                "true_label": true_label,
                "pred_label": predicted_label,
                "phishing_prob": phishing_probability,
                "text_preview": text_preview,
            }
        )

    error_dataframe = pd.DataFrame(error_rows)

    if (
        "phishing_prob" in error_dataframe.columns
        and not error_dataframe.empty
        and error_dataframe["phishing_prob"].notna().any()
    ):
        error_dataframe = error_dataframe.sort_values("phishing_prob", ascending=False)

    error_dataframe = error_dataframe.head(max_rows)
    error_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8")
    print("Saved error analysis to:", output_csv_path)

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

    # Use the same deterministic split across all stages for fair comparison
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

    # Out-of-fold probabilities used to train the stacking model
    out_of_fold_text_probabilities = np.zeros(len(x_train_text), dtype=float)
    out_of_fold_metadata_probabilities = np.zeros(len(x_train_text), dtype=float)

    stratified_kfold = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    for _, (train_rows, valid_rows) in enumerate(
        stratified_kfold.split(x_train_text, y_train),
        start=1,
    ):
        fold_train_texts = [x_train_text[i] for i in train_rows]
        fold_train_labels = y_train[train_rows]
        fold_valid_texts = [x_train_text[i] for i in valid_rows]

        text_probabilities = get_text_probabilities(
            fold_train_texts,
            fold_train_labels,
            fold_valid_texts,
            max_features=max_features,
        )
        metadata_probabilities = get_metadata_probabilities(
            fold_train_texts,
            fold_train_labels,
            fold_valid_texts,
        )

        for position_in_fold, original_row_index in enumerate(valid_rows):
            out_of_fold_text_probabilities[original_row_index] = float(
                text_probabilities[position_in_fold]
            )
            out_of_fold_metadata_probabilities[original_row_index] = float(
                metadata_probabilities[position_in_fold]
            )

    # Train the meta model on out-of-fold probabilities
    meta_train_features = np.column_stack(
        [out_of_fold_text_probabilities, out_of_fold_metadata_probabilities]
    )

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_train_features, y_train)

    # Train final text model on the full training split
    text_vectorizer, x_train_text_features = fit_vectorizer(
        x_train_text,
        max_features=max_features,
    )
    x_test_text_features = transform_vectorizer(text_vectorizer, x_test_text)

    text_model = LogisticRegression(max_iter=1000)
    text_model.fit(x_train_text_features, y_train)
    test_text_probabilities = text_model.predict_proba(x_test_text_features)[:, 1]

    # Train final metadata model on the full training split
    metadata_vectorizer, x_train_metadata_features = fit_metadata_vectorizer(x_train_text)
    x_test_metadata_features = transform_metadata_vectorizer(metadata_vectorizer, x_test_text)

    metadata_model = LogisticRegression(max_iter=1000)
    metadata_model.fit(x_train_metadata_features, y_train)
    test_metadata_probabilities = metadata_model.predict_proba(x_test_metadata_features)[:, 1]

    # Final hybrid probabilities from the stacking model
    meta_test_features = np.column_stack(
        [
            test_text_probabilities.astype(float),
            test_metadata_probabilities.astype(float),
        ]
    )
    hybrid_probabilities = meta_model.predict_proba(meta_test_features)[:, 1]

    predictions_at_default_threshold = (hybrid_probabilities > 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, predictions_at_default_threshold))
    precision = float(precision_score(y_test, predictions_at_default_threshold, zero_division=0))
    recall = float(recall_score(y_test, predictions_at_default_threshold, zero_division=0))
    f1 = float(f1_score(y_test, predictions_at_default_threshold, zero_division=0))
    confusion_matrix_result = confusion_matrix(y_test, predictions_at_default_threshold)

    roc_auc = None
    pr_auc = None
    if len(set(y_test)) == 2:
        roc_auc = float(roc_auc_score(y_test, hybrid_probabilities))
        pr_auc = float(average_precision_score(y_test, hybrid_probabilities))

    best_threshold, best_f1 = best_threshold_by_f1(y_test, hybrid_probabilities)
    predictions_at_best_threshold = (hybrid_probabilities > best_threshold).astype(int)

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
    print(confusion_matrix_result)
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions_at_default_threshold, zero_division=0))
    print()

    # Save stage 3 trained artifacts
    stage3_dir = repo_root() / "artifacts" / "stage3_hybrid"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(text_model, stage3_dir / "text_model.joblib")
    joblib.dump(text_vectorizer, stage3_dir / "text_vectorizer.joblib")
    joblib.dump(metadata_model, stage3_dir / "metadata_model.joblib")
    joblib.dump(metadata_vectorizer, stage3_dir / "metadata_vectorizer.joblib")
    joblib.dump(meta_model, stage3_dir / "meta_model.joblib")

    # Save misclassified examples for later qualitative evaluation
    error_analysis_csv_path = stage3_dir / "error_analysis.csv"
    save_error_analysis_csv(
        test_texts=x_test_text,
        true_labels=y_test,
        predicted_labels=predictions_at_default_threshold,
        predicted_probabilities=hybrid_probabilities,
        output_csv_path=error_analysis_csv_path,
    )

    # Save results JSON
    results = {
        "stage": "stage3_hybrid_stacking",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path()),
        "split_name": "phishing_email_split_1",
        "split": {
            "train": int(len(train_idx)),
            "test": int(len(test_idx)),
            "test_size": float(test_size),
        },
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
        "confusion_matrix": confusion_matrix_result.tolist(),
        "meta_weights": {
            "text_prob_weight": float(meta_model.coef_[0][0]),
            "metadata_prob_weight": float(meta_model.coef_[0][1]),
            "intercept": float(meta_model.intercept_[0]),
        },
        "runtime_seconds": {
            "total_s": runtime_total,
        },
    }

    latest_json = stage3_dir / "results.json"
    latest_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_json = stage3_dir / ("results_" + stamp + ".json")
    run_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved Stage 3 artifacts to:", stage3_dir)
    print("Saved latest results to:", latest_json)
    print("Saved run results to:", run_json)

    # Save test predictions for ROC / PR / threshold plots
    eval_dir = repo_root() / "artifacts" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_csv = eval_dir / "stage3_test_preds.csv"
    run_csv = timestamped_copy_path(base_csv)

    rows = []
    for i in range(len(y_test)):
        row = make_pred_row(
            row_id=i,
            y_true=y_test[i],
            prob=hybrid_probabilities[i],
            pred_0_5=predictions_at_default_threshold[i],
            pred_best=predictions_at_best_threshold[i],
            extra={
                "text_prob": test_text_probabilities[i],
                "meta_prob": test_metadata_probabilities[i],
                "hybrid_prob": hybrid_probabilities[i],
            },
        )

        # Small extras for reporting consistency
        row["stage"] = "stage3"
        row["threshold_best"] = float(best_threshold)

        rows.append(row)

    write_test_predictions_csv(base_csv, rows)
    write_test_predictions_csv(run_csv, rows)

    print("Saved eval CSV:", base_csv)
    print("Saved eval CSV (run):", run_csv)

if __name__ == "__main__":
    train_hybrid_stack(test_size=0.2, n_folds=5)