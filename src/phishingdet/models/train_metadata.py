import json
from datetime import datetime


import joblib
import numpy as np
import pandas as pd

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

def save_top_features(vectorizer, model, out_path, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]  # one coefficient per feature

    # Sort feature indices from most negative -> most positive
    sorted_indices = np.argsort(coefficients)

    # Most negative weights push towards class 0 (legit)
    legit_indices = sorted_indices[:top_n]

    # Most positive weights push towards class 1 (phishing)
    phishing_indices = sorted_indices[-top_n:][::-1]

    top_legit = []
    for idx in legit_indices:
        feature = feature_names[idx]
        weight = float(coefficients[idx])
        top_legit.append((feature, weight))

    top_phishing = []
    for idx in phishing_indices:
        feature = feature_names[idx]
        weight = float(coefficients[idx])
        top_phishing.append((feature, weight))

    # Save to CSV
    lines = ["direction,feature,weight"]
    for feature, weight in top_phishing:
        lines.append(f"phishing,{feature},{weight}")
    for feature, weight in top_legit:
        lines.append(f"legit,{feature},{weight}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        "##### DOCUMENTATION: TOP FEATURES (METADATA) #####\n"
        f"Saved top features to: {out_path}\n"
        "Top features pushing towards PHISHING (1):"
    )
    for feature, weight in top_phishing[:10]:
        print(f"  {feature}: {round(weight, 4)}")

    print("Top features pushing towards LEGIT (0):")
    for feature, weight in top_legit[:10]:
        print(f"  {feature}: {round(weight, 4)}")

    print("#### END DOCUMENTATION ####")

# AI GENERATED TO CAPTURE FURTHER DATA
def save_error_analysis_csv(x_test_text, y_test, preds, probs, out_path, max_rows=200):
    rows = []
    for i in range(len(y_test)):
        true_label = int(y_test[i])
        pred_label = int(preds[i])

        if true_label == pred_label:
            continue  # only keep mistakes

        if true_label == 0 and pred_label == 1:
            error_type = "false_positive"   # legit flagged as phishing
        else:
            error_type = "false_negative"   # phishing missed as legit

        p = None
        if probs is not None:
            p = float(probs[i])

        # keep text readable in the CSV
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

    # sort by probability so the most confident mistakes appear first
    if "phishing_prob" in df_err.columns and df_err["phishing_prob"].notna().any():
        df_err = df_err.sort_values("phishing_prob", ascending=False)

    # limit rows so the CSV stays small
    df_err = df_err.head(max_rows)

    df_err.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved error analysis to:", out_path)

def best_threshold_by_f1(y_true, probability):
    # Try lots of cutoffs from 0.00 to 1.00.
    # For each cutoff, turn probabilities into 0/1 predictions and measure F1.
    # We keep the cutoff that gives the best F1 (balance of precision + recall).
    # Precision: How often is it correct?
    # Recall: How many phishing did we catch?
    best_threshold = 0.50
    best_f1_score = -1.0

    thresholds = np.arange(0.0, 1.01, 0.01)  # 0.00, 0.01, ..., 1.00

    for threshold in thresholds:
        # If prob > threshold -> predict phishing (1), else legit (0)
        predictions = (probability > threshold).astype(int)

        # F1 is a single score that balances precision and recall
        current_f1 = f1_score(y_true, predictions, zero_division=0)

        # Keep the best scoring threshold
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1_score


def label_shuffle_sanity_check(X_train, y_train, X_test, y_test):
    # Creating a copy of the training labels
    shuffled_labels = list(y_train)

    # Using the same fixed seed to insure the shuffle is the same every run
    rng = np.random.RandomState(42)
    rng.shuffle(shuffled_labels)

    # Train a fresh model on the same training data but with random labels
    shuffled_model = LogisticRegression(max_iter=2000)
    shuffled_model.fit(X_train, shuffled_labels)

    # Testing
    shuffled_predictions = shuffled_model.predict(X_test)

    # If this accuracy is still high, possible leakage
    shuffled_accuracy =  accuracy_score(y_test, shuffled_predictions)

    return float(shuffled_accuracy)

def train_metadata_model(test_size=0.2):
    # loading the dataset
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Checks to see if the dataset can be split in its train/test ratio
    stratify = len(set(labels)) > 1 and len(labels) >= 6

    # splits the dataset into training/test
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels if stratify else None
    )
    # metadata features into numbers
    # For example, {"url_count": 1, "has_ip_url": 1, "char_len": 56, ...}
    vectorizer, X_train = fit_metadata_vectorizer(x_train_text)
    X_test = transform_metadata_vectorizer(vectorizer, x_test_text)

    # initiating model training
    # Learns which feature push towards phishing and which ones push towards safe

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # predictions
    preds = model.predict(X_test)

    # Calculating the metrics
    accuracy = float(accuracy_score(y_test, preds))
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"##### METADATA EVALUATION #####\n"
          f"Accuracy: {round(accuracy, 3)}\n"
          f"Precision: {round(precision, 3)}\n"
          f"Recall: {round(recall, 3)}\n"
          f"F1: {round(f1, 3)}\n")

    label_counts = df["label"].value_counts()

    print("##### DATASET INFO #####")
    print("Dataset:", dataset_path())
    print("Total rows:", len(df))
    print("Label counts:\n", label_counts)
    print()

    print("##### SPLIT INFO #####")
    print("Train size:", len(y_train))
    print("Test size :", len(y_test))
    print("test_size :", test_size)
    print()

    cm = confusion_matrix(y_test, preds)
    print(f"##### CONFUSION MATRIX #####\n"
          f"Confusion matrix: {cm}\n")

    print(f"##### CLASSIFICATION INFO #####\n"
          f"classification report: {classification_report(y_test, preds, zero_division=0)}\n")

    # Probability metrics
    roc_auc = None
    pr_auc = None
    best_t = None
    best_f1_at_t = None
    probs = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        best_t, best_f1_at_t = best_threshold_by_f1(np.array(y_test), probs)

        print(f"##### ROC AUC: {round(roc_auc, 3)}\n"
              f"PR AUC: {round(pr_auc, 3)}\n"
              f"Best threshold (F1): {best_t} | F1: {round(best_f1_at_t, 3)}")


    shuffle_accuracy = label_shuffle_sanity_check(X_train, y_train, X_test, y_test)
    print(f"##### SHUFFLE ACCURACY: {round(shuffle_accuracy,3)}\n")

    # Saving artifacts
    artifacts_directory = repo_root() / "artifacts" / "stage2_metadata"
    artifacts_directory.mkdir(parents=True, exist_ok=True)

    # Error analysis CSV (for report screenshots)
    error_csv_path = artifacts_directory / "error_analysis.csv"
    save_error_analysis_csv(x_test_text, y_test, preds, probs, error_csv_path)

    model_path = artifacts_directory / "model.joblib"
    vectorizer_path = artifacts_directory / "vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved model to {model_path}\n"
          f"Saved vectorizer to {vectorizer_path}\n")

    # DOCUMENTATION
    top_features_path = artifacts_directory / "top_features.csv"
    save_top_features(vectorizer, model, top_features_path, top_n=15)

    print(f"Saved features to {top_features_path}\n")

    # SAVING RESULTS
    results = {
        "stage": "stage2_metadata_only",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path()),
        "number of rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "split": {
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "test_size": float(test_size),
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

    # Save the "latest" copy
    latest_file = artifacts_directory / "results.json"
    latest_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save a timestamped copy
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_file = artifacts_directory / ("results_" + timestamp + ".json")
    run_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved latest results to:", latest_file)
    print("Saved run results to:", run_file)


if __name__ == "__main__":
    train_metadata_model(test_size=0.2)