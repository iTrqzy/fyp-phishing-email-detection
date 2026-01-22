import json
import csv
import random
from datetime import datetime

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from phishingdet.data.loader import load_email, repo_root, dataset_path
from phishingdet.features.build_features import fit_vectorizer, transform_vectorizer


def save_top_features(vectorizer, model, artifacts_dir, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # binary classifier -> 1 row

    top_pos_idx = coefs.argsort()[-top_n:][::-1]  # phishing (label=1)
    top_neg_idx = coefs.argsort()[:top_n]         # legit (label=0)

    out_path = artifacts_dir / "top_features.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "class_hint", "feature", "coefficient"])

        rank = 1
        for i in top_pos_idx:
            writer.writerow([rank, "phishing (label=1)", feature_names[i], float(coefs[i])])
            rank += 1

        rank = 1
        for i in top_neg_idx:
            writer.writerow([rank, "legit (label=0)", feature_names[i], float(coefs[i])])
            rank += 1

    print("\n##### DOCUMENTATION: TOP FEATURES #####")
    print(f"Saved top features to: {out_path}")
    print("Top tokens pushing towards PHISHING (label=1):")
    for i in top_pos_idx[:10]:
        print(" ", feature_names[i], "->", round(float(coefs[i]), 4))
    print("Top tokens pushing towards LEGIT (label=0):")
    for i in top_neg_idx[:10]:
        print(" ", feature_names[i], "->", round(float(coefs[i]), 4))
    print("##### END DOCUMENTATION #####\n")


def best_threshold_by_f1(y_true, probs):
    # Scan a few thresholds and pick the one with best F1 on the test set
    best_t = 0.5
    best_f1 = -1.0

    for t in [i / 100 for i in range(10, 91, 5)]:
        preds = [1 if p >= t else 0 for p in probs]
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t

    return best_t, best_f1


def label_shuffle_sanity_check(x_train, y_train, x_test, y_test, random_state=42):
    # If this gets high accuracy, something is suspicious (leakage / trivial patterns)
    rng = random.Random(random_state)
    y_shuffled = list(y_train)
    rng.shuffle(y_shuffled)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_shuffled)
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    return float(acc)


def train_model(test_size=0.2, random_state=42, max_features=5000):
    # load the data
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # where to save artifacts
    artifacts_dir = repo_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # stratify keeps class balance in train/test
    use_stratify = len(set(labels)) > 1 and len(labels) >= 6

    # split
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if use_stratify else None
    )

    # vectorize
    vectorizer, x_train = fit_vectorizer(x_train_text, max_features=max_features)
    x_test = transform_vectorizer(vectorizer, x_test_text)

    # train
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    ##### DOCUMENTATION: TOP FEATURES #####
    save_top_features(vectorizer, model, artifacts_dir, top_n=20)
    ##### END DOCUMENTATION #####

    # predict
    predictions = model.predict(x_test)

    # metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("Model evaluation:")
    print("  Accuracy :", round(accuracy, 3))
    print("  Precision:", round(precision, 3))
    print("  Recall   :", round(recall, 3))
    print("  F1 Score :", round(f1, 3))

    ##### DOCUMENTATION OUTPUT START #####
    print("\nDataset info:")
    print("  Dataset path:", str(dataset_path()))
    print("  Total rows:", len(df))
    print("  Label counts:")
    print(df["label"].value_counts())

    print("\nSplit info:")
    print("  Train size:", len(y_train))
    print("  Test size :", len(y_test))
    print("  test_size :", test_size)

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    auc = None
    pr_auc = None
    probs = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)[:, 1].tolist()

        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, probs)
            print("  ROC-AUC  :", round(auc, 3))
        except Exception as e:
            print("  ROC-AUC  : could not compute")
            print("           ", str(e))

        # PR-AUC (Average Precision)
        try:
            pr_auc = average_precision_score(y_test, probs)
            print("  PR-AUC   :", round(pr_auc, 3))
        except Exception as e:
            print("  PR-AUC   : could not compute")
            print("           ", str(e))

        # Best threshold scan
        best_t, best_f1 = best_threshold_by_f1(y_test, probs)
        print("  Best threshold (by F1 on test):", best_t, "| F1:", round(best_f1, 3))

        # Label-shuffle sanity check
        shuffle_acc = label_shuffle_sanity_check(x_train, y_train, x_test, y_test, random_state=random_state)
        print("  Label-shuffle accuracy (sanity check):", round(shuffle_acc, 3))
    else:
        best_t, best_f1 = None, None
        shuffle_acc = None
        print("  Probabilities: not available (model has no predict_proba)")

    ##### DOCUMENTATION OUTPUT END #####

    # save artifacts
    model_path = artifacts_dir / "model.joblib"
    vec_path = artifacts_dir / "vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")

    ##### DOCUMENTATION SAVE START #####
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path()),
        "dataset_rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "test_size_ratio": float(test_size),
        "random_state": int(random_state),
        "use_stratify": bool(use_stratify),
        "vectorizer": {
            "type": "tfidf",
            "max_features": int(max_features),
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
        "model": {
            "type": "logistic_regression",
            "max_iter": 1000,
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc) if auc is not None else None,
            "pr_auc": float(pr_auc) if pr_auc is not None else None,
            "best_threshold_f1": float(best_t) if best_t is not None else None,
            "label_shuffle_accuracy": float(shuffle_acc) if shuffle_acc is not None else None,
        },
        "confusion_matrix": cm.tolist(),
    }

    latest_path = artifacts_dir / "results.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ts = results["timestamp"].replace(":", "-")  # Windows-safe filename
    run_path = artifacts_dir / f"results_{ts}.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved latest results to: {latest_path}")
    print(f"Saved run results to: {run_path}")
    ##### DOCUMENTATION SAVE END #####


if __name__ == "__main__":
    train_model()
