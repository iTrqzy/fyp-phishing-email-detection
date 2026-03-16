import json
from datetime import datetime

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)

from phishingdet.data.loader import load_email, repo_root, dataset_path
from phishingdet.features.build_metadata_features import (
    fit_metadata_vectorizer, transform_metadata_vectorizer
)

import pandas as pd
from phishingdet.evaluation.save_test_preds import save_test_predictions_csv


def save_top_features(vectorizer, model, out_path, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    sorted_indices = np.argsort(coefficients)
    legit_indices = sorted_indices[:top_n]              # most negative
    phishing_indices = sorted_indices[-top_n:][::-1]    # most positive

    lines = ["direction,feature,weight"]

    print("##### DOCUMENTATION: TOP FEATURES (METADATA) #####")
    print("Saved top features to:", out_path)
    print("Top features pushing towards PHISHING (1):")

    for idx in phishing_indices:
        feature = feature_names[idx]
        weight = float(coefficients[idx])
        lines.append(f"phishing,{feature},{weight}")

    for idx in legit_indices:
        feature = feature_names[idx]
        weight = float(coefficients[idx])
        lines.append(f"legit,{feature},{weight}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # quick preview
    for idx in phishing_indices[:10]:
        print(f"  {feature_names[idx]}: {round(float(coefficients[idx]), 4)}")

    print("Top features pushing towards LEGIT (0):")
    for idx in legit_indices[:10]:
        print(f"  {feature_names[idx]}: {round(float(coefficients[idx]), 4)}")

    print("#### END DOCUMENTATION ####")


def save_error_analysis_csv(x_test_text, y_test, preds, probs, out_path, max_rows=200):
    rows = []
    for i in range(len(y_test)):
        true_label = int(y_test[i])
        pred_label = int(preds[i])

        if true_label == pred_label:
            continue

        if true_label == 0 and pred_label == 1:
            error_type = "false_positive"
        else:
            error_type = "false_negative"

        p = float(probs[i]) if probs is not None else None

        text_preview = str(x_test_text[i]).replace("\n", " ").strip()
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."

        rows.append({
            "error_type": error_type,
            "true_label": true_label,
            "pred_label": pred_label,
            "phishing_prob": p,
            "text_preview": text_preview,
        })

    df_err = pd.DataFrame(rows)

    if "phishing_prob" in df_err.columns and df_err["phishing_prob"].notna().any():
        df_err = df_err.sort_values("phishing_prob", ascending=False)

    df_err = df_err.head(max_rows)
    df_err.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved error analysis to:", out_path)


def best_threshold_by_f1(y_true, probability):
    best_threshold = 0.50
    best_f1_score = -1.0

    thresholds = np.arange(0.0, 1.01, 0.01)

    for threshold in thresholds:
        predictions = (probability > threshold).astype(int)
        current_f1 = f1_score(y_true, predictions, zero_division=0)

        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1_score


def label_shuffle_sanity_check(X_train, y_train, X_test, y_test, random_state=42):
    shuffled_labels = list(y_train)

    rng = np.random.RandomState(random_state)
    rng.shuffle(shuffled_labels)

    shuffled_model = LogisticRegression(max_iter=2000)
    shuffled_model.fit(X_train, shuffled_labels)

    shuffled_predictions = shuffled_model.predict(X_test)
    return float(accuracy_score(y_test, shuffled_predictions))


def train_metadata_model(test_size=0.2, random_state=42, max_iter=1000):
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    use_stratify = len(set(labels)) > 1 and len(labels) >= 6

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if use_stratify else None
    )

    vectorizer, X_train = fit_metadata_vectorizer(x_train_text)
    X_test = transform_metadata_vectorizer(vectorizer, x_test_text)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, preds))
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    print("##### METADATA EVALUATION #####")
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
    print()

    print("##### DATASET INFO #####")
    print("Dataset:", dataset_path())
    print("Total rows:", len(df))
    print("Label counts:\n", df["label"].value_counts())
    print()

    print("##### SPLIT INFO #####")
    print("Train size:", len(y_train))
    print("Test size :", len(y_test))
    print("test_size :", test_size)
    print("random_state:", random_state)
    print()

    cm = confusion_matrix(y_test, preds)
    print("##### CONFUSION MATRIX #####")
    print("Confusion matrix:", cm)
    print()

    print("##### CLASSIFICATION INFO #####")
    print("classification report:\n", classification_report(y_test, preds, zero_division=0))
    print()

    roc_auc = None
    pr_auc = None
    best_t = None
    best_f1_at_t = None
    probs = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        best_t, best_f1_at_t = best_threshold_by_f1(np.array(y_test), probs)

        print("##### ROC AUC:", round(roc_auc, 3))
        print("PR AUC:", round(pr_auc, 3))
        print("Best threshold (F1):", best_t, "| F1:", round(best_f1_at_t, 3))
        print()

        # ===== export test preds for PR/ROC curves (Stage 2) =====
        pred_0_5 = (probs > 0.5).astype(int)
        pred_best = (probs > best_t).astype(int) if best_t is not None else None

        latest_csv, stamped_csv = save_test_predictions_csv(
            stage_name="stage2",
            y_true=y_test,
            prob=probs,
            pred_0_5=pred_0_5,
            pred_best=pred_best,
            threshold_best=best_t,
        )

        print("Saved Stage 2 test predictions to:", latest_csv)
        if stamped_csv:
            print("Saved timestamped copy to:", stamped_csv)
        print()

    shuffle_accuracy = label_shuffle_sanity_check(X_train, y_train, X_test, y_test, random_state=random_state)
    print("##### SHUFFLE ACCURACY:", round(shuffle_accuracy, 3))
    print()

    artifacts_directory = repo_root() / "artifacts" / "stage2_metadata"
    artifacts_directory.mkdir(parents=True, exist_ok=True)

    error_csv_path = artifacts_directory / "error_analysis.csv"
    save_error_analysis_csv(x_test_text, y_test, preds, probs, error_csv_path)

    model_path = artifacts_directory / "model.joblib"
    vectorizer_path = artifacts_directory / "vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Saved model to", model_path)
    print("Saved vectorizer to", vectorizer_path)
    print()

    top_features_path = artifacts_directory / "top_features_stage2_metadata.csv"
    save_top_features(vectorizer, model, top_features_path, top_n=15)

    print("Saved features to", top_features_path)
    print()

    results = {
        "stage": "stage2_metadata_only",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path()),
        "rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "split": {
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "test_size": float(test_size),
            "random_state": int(random_state),
            "stratified": bool(use_stratify),
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold": best_t,
            "best_f1_at_threshold": best_f1_at_t,
            "label_shuffle_accuracy": shuffle_accuracy,
        },
        "confusion_matrix": cm.tolist(),
        "number_of_features": int(len(vectorizer.get_feature_names_out())),
    }

    latest_file = artifacts_directory / "results.json"
    latest_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_file = artifacts_directory / ("results_" + stamp + ".json")
    run_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved latest results to:", latest_file)
    print("Saved run results to:", run_file)


if __name__ == "__main__":
    train_metadata_model(test_size=0.2, random_state=42, max_iter=1000)
