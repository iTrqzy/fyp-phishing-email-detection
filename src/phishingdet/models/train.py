import json
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
)

from phishingdet.data.loader import load_email, repo_root
from phishingdet.features.build_features import fit_vectorizer, transform_vectorizer


def train_model():
    # load the data and make it into a list form
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # if dataset is very small, there is a failsafe
    # stratify tries to keep the same class balance in train and test
    use_stratify = len(set(labels)) > 1 and len(labels) >= 6

    # split training & testing - 20% of the data is used for testing
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if use_stratify else None
    )

    # converting text to numbers
    vectorizer, x_train = fit_vectorizer(x_train_text, max_features=5000)
    x_test = transform_vectorizer(vectorizer, x_test_text)

    # train model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # outcomes
    predictions = model.predict(x_test)

    # number correct / total
    accuracy = accuracy_score(y_test, predictions)

    # true positives / (true positives + false positives)
    precision = precision_score(y_test, predictions, zero_division=0)

    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, predictions, zero_division=0)

    # 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("Model evaluation:")
    print("  Accuracy :", round(accuracy, 3))
    print("  Precision:", round(precision, 3))
    print("  Recall   :", round(recall, 3))
    print("  F1 Score :", round(f1, 3))

    ##### DOCUMENTATION OUTPUT START #####
    print("\nDataset info:")
    print("  Total rows:", len(df))
    print("  Label counts:")
    print(df["label"].value_counts())

    print("\nSplit info:")
    print("  Train size:", len(y_train))
    print("  Test size :", len(y_test))

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # ROC-AUC is based on probabilities (how well the model ranks positives vs negatives)
    auc = None
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_test)[:, 1]
            # roc_auc_score fails if y_test contains only one class
            auc = roc_auc_score(y_test, probs)
            print("  ROC-AUC  :", round(auc, 3))
        else:
            print("  ROC-AUC  : not available (model has no predict_proba)")
    except Exception as e:
        print("  ROC-AUC  : could not compute (likely only one class in y_test)")
        print("           ", str(e))
    ##### DOCUMENTATION OUTPUT END #####

    # save artifacts (cross-platform compatible) with error checking
    artifacts_dir = repo_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # dumping the data
    model_path = artifacts_dir / "model.joblib"
    vec_path = artifacts_dir / "vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")

    ##### DOCUMENTATION SAVE START #####
    # Save a results.json file so you can compare runs later in your report
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "test_size_ratio": 0.2,
        "random_state": 42,
        "use_stratify": bool(use_stratify),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc) if auc is not None else None,
        },
        "confusion_matrix": cm.tolist(),
    }

    # 1) always overwrite latest results.json
    latest_path = artifacts_dir / "results.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    # 2) also save a timestamped copy so old runs don't get overwritten
    ts = results["timestamp"].replace(":", "-")  # Windows-safe filename
    run_path = artifacts_dir / f"results_{ts}.json"
    with open(run_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved latest results to: {latest_path}")
    print(f"Saved run results to: {run_path}")

    ##### DOCUMENTATION SAVE END #####


if __name__ == "__main__":
    train_model()
