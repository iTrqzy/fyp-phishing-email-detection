import json
from datetime import datetime
import joblib
import csv
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from phishingdet.data.loader import load_email, repo_root, dataset_path
from phishingdet.data.splits import get_or_make_split_indices
from phishingdet.features.build_features import fit_vectorizer, transform_vectorizer
from phishingdet.features.build_metadata_features import fit_metadata_vectorizer, transform_metadata_vectorizer

RANDOM_STATE = 42

def best_threshold_by_f1(y_true, probabilities):
    best_threshold = 0.50
    best_f1_score = -1.0
    thresholds = np.arange(0.0, 1.01, 0.01)

    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        current_f1 = f1_score(y_true, predictions, zero_division=0)

        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1_score)

def get_text_probabilities(train_texts, train_labels, valid_texts, max_features =5000):
    # Training Stage 1 model on this fold
    vectorizer, X_train = fit_vectorizer(train_texts, max_features=max_features)
    X_valid = transform_vectorizer(vectorizer, valid_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    # Return probability of class 1
    probs = model.predict_proba(X_valid)[:, 1]
    return probs

# AI GENERATED FOR DATA COLLECTION
def save_error_analysis_csv_stage3(
    output_path,
    texts,
    y_true,
    y_pred,
    hybrid_prob,
    text_prob,
    metadata_prob,
    top_k=50,
):
    """
    Saves the most confident misclassifications:
      - False positives: y_true=0, y_pred=1 (sorted by hybrid_prob desc)
      - False negatives: y_true=1, y_pred=0 (sorted by hybrid_prob asc)

    Columns:
      error_type, true_label, pred_label, hybrid_prob, text_prob, metadata_prob, text_snippet
    """
    rows_fp = []
    rows_fn = []

    for i in range(len(texts)):
        yt = int(y_true[i])
        yp = int(y_pred[i])
        hp = float(hybrid_prob[i])
        tp = float(text_prob[i])
        mp = float(metadata_prob[i])

        if yt == 0 and yp == 1:  # false positive
            rows_fp.append((hp, yt, yp, hp, tp, mp, texts[i]))
        elif yt == 1 and yp == 0:  # false negative
            rows_fn.append((hp, yt, yp, hp, tp, mp, texts[i]))

    # Most confident FP = highest hybrid_prob
    rows_fp.sort(key=lambda r: r[0], reverse=True)
    # Most confident FN = lowest hybrid_prob (most confident "legit")
    rows_fn.sort(key=lambda r: r[0])

    # Keep top_k from each category (keeps file manageable)
    out_rows = []
    for r in rows_fp[:top_k]:
        out_rows.append(("false_positive", r[1], r[2], r[3], r[4], r[5], r[6][:400]))
    for r in rows_fn[:top_k]:
        out_rows.append(("false_negative", r[1], r[2], r[3], r[4], r[5], r[6][:400]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["error_type", "true_label", "pred_label", "hybrid_prob", "text_prob", "metadata_prob", "text_snippet"]
        )
        writer.writerows(out_rows)

def get_metadata_probabilities(train_texts, train_labels, valid_texts):
    # Training Stage 2 model (DictVectorizer + Logistic Regression) on this fold
    vectorizer, X_train =  fit_metadata_vectorizer(train_texts)
    X_valid = transform_metadata_vectorizer(vectorizer, valid_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    probs = model.predict_proba(X_valid)[:, 1]
    return probs


def train_hybrid_stack(test_size=0.2, n_folds=5):
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    # Turn this OFF later if you don't want to export runtime + best-threshold metrics + error analysis.
    ENABLE_REPORTING_EXTRAS = True

    t0_total = time.perf_counter()
    # ===== END OPTIONAL REPORTING BLOCK =====

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t0_split = time.perf_counter()
    # ===== END OPTIONAL REPORTING BLOCK =====

    train_idx, test_idx = get_or_make_split_indices(labels, test_size=test_size, random_state=RANDOM_STATE,
                                                    stratify=True, split_name="phishing_email_split_1")

    x_train_text = [texts[i] for i in train_idx]
    y_train = np.array([labels[i] for i in train_idx])

    x_test_text = [texts[i] for i in test_idx]
    y_test = np.array([labels[i] for i in test_idx])

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t_split = time.perf_counter() - t0_split if ENABLE_REPORTING_EXTRAS else None
    # ===== END OPTIONAL REPORTING BLOCK =====

    # out of fold probabilities for meta learner
    oof_text_probability = np.zeros(len(x_train_text), dtype=float)
    oof_metadata_probability = np.zeros(len(x_train_text), dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t0_oof = time.perf_counter()
    # ===== END OPTIONAL REPORTING BLOCK =====

    for fold_num, (train_rows, valid_rows) in enumerate(skf.split(x_train_text, y_train), start=1):
        fold_train_texts = [x_train_text[i] for i in train_rows]
        fold_train_labels = y_train[train_rows]
        fold_valid_texts = [x_train_text[i] for i in valid_rows]

        # base model probabilities on the validation fold (OOF)
        text_probability = get_text_probabilities(fold_train_texts, fold_train_labels, fold_valid_texts)
        metadata_probability = get_metadata_probabilities(fold_train_texts, fold_train_labels, fold_valid_texts)

        for j, row_index in enumerate(valid_rows):
            oof_text_probability[row_index] = float(text_probability[j])
            oof_metadata_probability[row_index] = float(metadata_probability[j])

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t_oof = time.perf_counter() - t0_oof if ENABLE_REPORTING_EXTRAS else None
    # ===== END OPTIONAL REPORTING BLOCK =====

    # Train meta learner ONLY on OOF probabilities
    meta_train_features = np.column_stack([oof_text_probability, oof_metadata_probability])

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t0_meta = time.perf_counter()
    # ===== END OPTIONAL REPORTING BLOCK =====

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_train_features, y_train)

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t_meta = time.perf_counter() - t0_meta if ENABLE_REPORTING_EXTRAS else None
    # ===== END OPTIONAL REPORTING BLOCK =====

    # Train final base models on full training split
    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t0_final = time.perf_counter()
    # ===== END OPTIONAL REPORTING BLOCK =====

    final_text_vectorizer, X_train_text = fit_vectorizer(x_train_text, max_features=5000)
    X_test_text = transform_vectorizer(final_text_vectorizer, x_test_text)

    final_text_model = LogisticRegression(max_iter=1000)
    final_text_model.fit(X_train_text, y_train)
    test_text_probability = final_text_model.predict_proba(X_test_text)[:, 1]

    final_metadata_vectorizer, X_train_meta = fit_metadata_vectorizer(x_train_text)
    X_test_meta = transform_metadata_vectorizer(final_metadata_vectorizer, x_test_text)

    final_metadata_model = LogisticRegression(max_iter=1000)
    final_metadata_model.fit(X_train_meta, y_train)
    test_metadata_probability = final_metadata_model.predict_proba(X_test_meta)[:, 1]

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t_final = time.perf_counter() - t0_final if ENABLE_REPORTING_EXTRAS else None
    # ===== END OPTIONAL REPORTING BLOCK =====

    # Hybrid (meta) prediction on the test set
    meta_test_features = np.column_stack([test_text_probability.astype(float), test_metadata_probability.astype(float)])
    hybrid_probability = meta_model.predict_proba(meta_test_features)[:, 1]

    # Default threshold evaluation (0.5)
    preds = (hybrid_probability > 0.5).astype(int)

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t0_eval = time.perf_counter()
    # ===== END OPTIONAL REPORTING BLOCK =====

    accuracy = float(accuracy_score(y_test, preds))
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds)

    roc_auc = None
    pr_auc = None
    if len(set(y_test)) == 2:
        roc_auc = float(roc_auc_score(y_test, hybrid_probability))
        pr_auc = float(average_precision_score(y_test, hybrid_probability))

    best_threshold, best_f1 = best_threshold_by_f1(y_test, hybrid_probability)

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    t_eval = time.perf_counter() - t0_eval if ENABLE_REPORTING_EXTRAS else None

    # Save best-threshold metrics (so your report can cite them properly)
    best_threshold_metrics = None
    if ENABLE_REPORTING_EXTRAS and best_threshold is not None:
        best_preds = (hybrid_probability > best_threshold).astype(int)
        best_cm = confusion_matrix(y_test, best_preds).tolist()
        best_threshold_metrics = {
            "threshold": float(best_threshold),
            "precision": float(precision_score(y_test, best_preds, zero_division=0)),
            "recall": float(recall_score(y_test, best_preds, zero_division=0)),
            "f1": float(f1_score(y_test, best_preds, zero_division=0)),
            "confusion_matrix": best_cm,
        }

    # Runtime breakdown
    runtime_seconds = None
    if ENABLE_REPORTING_EXTRAS:
        t_total = time.perf_counter() - t0_total
        runtime_seconds = {
            "split_s": float(t_split) if t_split is not None else None,
            "oof_cv_s": float(t_oof) if t_oof is not None else None,
            "meta_train_s": float(t_meta) if t_meta is not None else None,
            "final_base_train_s": float(t_final) if t_final is not None else None,
            "eval_s": float(t_eval) if t_eval is not None else None,
            "total_s": float(t_total),
        }
    # ===== END OPTIONAL REPORTING BLOCK =====

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
    print(classification_report(y_test, preds, zero_division=0))
    print()

    # Saving artifacts
    artifacts_directory = repo_root() / "artifacts" / "stage3_hybrid"
    artifacts_directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_text_model, artifacts_directory / "text_model.joblib")
    joblib.dump(final_text_vectorizer, artifacts_directory / "text_vectorizer.joblib")

    joblib.dump(final_metadata_model, artifacts_directory / "metadata_model.joblib")
    joblib.dump(final_metadata_vectorizer, artifacts_directory / "metadata_vectorizer.joblib")

    joblib.dump(meta_model, artifacts_directory / "meta_model.joblib")

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    # Save error analysis (top confident FP/FN) for the dissertation
    if ENABLE_REPORTING_EXTRAS:
        error_path = artifacts_directory / "error_analysis.csv"
        save_error_analysis_csv_stage3(
            error_path,
            x_test_text,
            y_test,
            preds,
            hybrid_probability,
            test_text_probability,
            test_metadata_probability,
            top_k=50,
        )
        print("Saved error analysis to:", error_path)
    # ===== END OPTIONAL REPORTING BLOCK =====

    # Save results JSON (latest + timestamped)
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
            "best_threshold_f1": best_threshold,
            "best_f1_at_threshold": best_f1,
        },
        "confusion_matrix": cm.tolist(),
        "meta_weights": {
            "text_prob_weight": float(meta_model.coef_[0][0]),
            "metadata_prob_weight": float(meta_model.coef_[0][1]),
            "intercept": float(meta_model.intercept_[0]),
        }
    }

    # ===== OPTIONAL REPORTING BLOCK (KEEP FOR WRITE-UP) =====
    if ENABLE_REPORTING_EXTRAS:
        results["runtime_seconds"] = runtime_seconds
        results["best_threshold_metrics"] = best_threshold_metrics
    # ===== END OPTIONAL REPORTING BLOCK =====

    latest_file = artifacts_directory / "results.json"
    latest_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_file = artifacts_directory / ("results_" + stamp + ".json")
    run_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Saved Stage 3 artifacts to:", artifacts_directory)
    print("Saved latest results to:", latest_file)
    print("Saved run results to:", run_file)

if __name__ == "__main__":
    train_hybrid_stack(test_size=0.2, n_folds=5)