import json
from datetime import datetime
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from phishingdet.data.loader import load_email, repo_root, dataset_path
from phishingdet.data.splits import get_or_make_split_indicies
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

def train_hybrid_stack(test_size=0.2, n_folds=5):
    df = load_email()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_idx, test_idx = get_or_make_split_indicies(
        labels,
        test_size = test_size,
        random_state= RANDOM_STATE,
        stratify = True,
        split_name = "phishing_email_split_1",
    )

if __name__ == "__main__":
    train_hybrid_stack(test_size=0.2, n_folds=5)