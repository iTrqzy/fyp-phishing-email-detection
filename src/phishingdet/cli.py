import argparse
import csv
from pathlib import Path

from phishingdet.models.train import train_model
from phishingdet.models.predict import predict_text

from phishingdet.models.train_metadata import train_metadata_model
from phishingdet.models.predict_metadata import predict_metadata

from phishingdet.models.train_hybrid import train_hybrid_stack
from phishingdet.models.predict_hybrid import predict_hybrid

from phishingdet.features.build_metadata_features import extract_metadata_features_one
from phishingdet.data.loader import repo_root


def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def stage1_top_features_csv():
    return Path(repo_root()) / "artifacts" / "top_features.csv"


def stage2_top_features_csv():
    # you renamed it before, so we check a few common options
    return _first_existing([
        Path(repo_root()) / "artifacts" / "stage2_metadata" / "top_features.csv",
        Path(repo_root()) / "artifacts" / "stage2_metadata" / "top_features_stage2_metadata.csv",
        Path(repo_root()) / "artifacts" / "stage2_metadata" / "top_features_stage2_metadata.csv",
        Path(repo_root()) / "artifacts" / "stage2_metadata" / "top_features.csv",
        Path(repo_root()) / "artifacts" / "stage2_metadata" / "top_features_stage2_metadata.csv",
    ])


def load_feature_weights_from_csv(csv_path):
    weights = {}
    if csv_path is None or (not csv_path.exists()):
        return weights

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = (row.get("feature") or "").strip().lower()
            if not feature:
                continue

            # Stage 1 file uses "weight"; some versions might say "coefficient"
            raw = row.get("weight", row.get("coefficient", "0"))
            try:
                value = float(raw)
            except ValueError:
                value = 0.0

            weights[feature] = value

    return weights


def top_token_hits(email_text, feature_weights, max_show=5):
    text_lower = (email_text or "").lower()
    hits = []

    for feature, weight in feature_weights.items():
        if not feature:
            continue

        # crude but practical: substring check works for unigrams/bigrams
        if feature in text_lower:
            hits.append((feature, weight))

    hits.sort(key=lambda item: abs(item[1]), reverse=True)
    return hits[:max_show]


def top_metadata_hits(email_text, meta_weights, max_show=8):
    feats = extract_metadata_features_one(email_text)
    hits = []

    for key, value in feats.items():
        # only show ones that are "active" / present
        if not value:
            continue

        weight = meta_weights.get(key)
        if weight is None:
            continue

        hits.append((key, value, weight))

    hits.sort(key=lambda item: abs(item[2]), reverse=True)
    return hits[:max_show], feats


def print_explain_text(email_text, pred, prob):
    print("\n--- EXPLANATION ---")
    print("Stage 1 (text-only): TF-IDF + Logistic Regression")
    print("Probability:", round(prob, 4), "| prediction:", pred)

    stage1_weights = load_feature_weights_from_csv(stage1_top_features_csv())
    hits = top_token_hits(email_text, stage1_weights, max_show=6)

    print("\nTop text cues present (weighted):")
    if not hits:
        print("  (none from top_features.csv matched this text)")
    else:
        for token, weight in hits:
            side = "PHISHING" if weight > 0 else "LEGIT"
            print(f"  {token} ({side}) weight={round(weight, 4)}")

    print("-------------------\n")


def print_explain_metadata(email_text, pred, prob):
    print("\n--- EXPLANATION ---")
    print("Stage 2 (metadata-only): hand-crafted metadata cues + Logistic Regression")
    print("Probability:", round(prob, 4), "| prediction:", pred)

    meta_csv = stage2_top_features_csv()
    meta_weights = load_feature_weights_from_csv(meta_csv) if meta_csv else {}

    top_hits, all_feats = top_metadata_hits(email_text, meta_weights, max_show=8)

    print("\nTop metadata cues present (weighted):")
    if not top_hits:
        print("  (no matching weighted metadata features found)")
    else:
        for key, value, weight in top_hits:
            side = "PHISHING" if weight > 0 else "LEGIT"
            print(f"  {key}={value} ({side}) weight={round(weight, 4)}")

    print("\nAll triggered metadata (non-zero):")
    for key, value in all_feats.items():
        if value:
            print(f"  {key} = {value}")

    print("-------------------\n")


def print_explain_hybrid(email_text, pred, hybrid_prob, decision, text_prob, meta_prob):
    print("\n--- EXPLANATION ---")
    print("Stage 3 (hybrid stacking): combines Stage 1 + Stage 2 probabilities")
    print("Hybrid probability:", round(hybrid_prob, 4), "| decision:", decision)
    print("Base model probs  :", "text=", round(text_prob, 4), "| meta=", round(meta_prob, 4))

    # Stage 2 weighted metadata cues
    meta_csv = stage2_top_features_csv()
    meta_weights = load_feature_weights_from_csv(meta_csv) if meta_csv else {}
    top_hits, all_feats = top_metadata_hits(email_text, meta_weights, max_show=8)

    print("\nTop metadata cues present (weighted):")
    if not top_hits:
        print("  (no matching weighted metadata features found)")
    else:
        for key, value, weight in top_hits:
            side = "PHISHING" if weight > 0 else "LEGIT"
            print(f"  {key}={value} ({side}) weight={round(weight, 4)}")

    print("\nAll triggered metadata (non-zero):")
    for key, value in all_feats.items():
        if value:
            print(f"  {key} = {value}")

    # Stage 1 weighted token cues
    stage1_weights = load_feature_weights_from_csv(stage1_top_features_csv())
    token_hits = top_token_hits(email_text, stage1_weights, max_show=6)

    print("\nTop text cues present (weighted):")
    if not token_hits:
        print("  (none from top_features.csv matched this text)")
    else:
        for token, weight in token_hits:
            side = "PHISHING" if weight > 0 else "LEGIT"
            print(f"  {token} ({side}) weight={round(weight, 4)}")

    print("-------------------\n")


def main():
    parser = argparse.ArgumentParser(prog="phishingdet")
    cmd = parser.add_subparsers(dest="cmd")

    # train <text|metadata|hybrid>
    p_train = cmd.add_parser("train", help="Train a model")
    train_kind = p_train.add_subparsers(dest="kind")

    p_train_text = train_kind.add_parser("text", help="Train Stage 1 text-only model")
    p_train_text.add_argument("--test_size", type=float, default=0.2)
    p_train_text.add_argument("--random_state", type=int, default=42)
    p_train_text.add_argument("--max_features", type=int, default=5000)

    p_train_meta = train_kind.add_parser("metadata", help="Train Stage 2 metadata-only model")
    p_train_meta.add_argument("--test_size", type=float, default=0.2)
    p_train_meta.add_argument("--random_state", type=int, default=42)

    p_train_hybrid = train_kind.add_parser("hybrid", help="Train Stage 3 hybrid stacking model")
    p_train_hybrid.add_argument("--test_size", type=float, default=0.2)
    p_train_hybrid.add_argument("--random_state", type=int, default=42)

    # predict <text|metadata|hybrid> "..."
    p_pred = cmd.add_parser("predict", help="Predict on one email")
    pred_kind = p_pred.add_subparsers(dest="kind")

    p_pred_text = pred_kind.add_parser("text", help="Predict using Stage 1 text-only model")
    p_pred_text.add_argument("text", nargs="+")
    p_pred_text.add_argument("--explain", action="store_true")

    p_pred_meta = pred_kind.add_parser("metadata", help="Predict using Stage 2 metadata-only model")
    p_pred_meta.add_argument("text", nargs="+")
    p_pred_meta.add_argument("--explain", action="store_true")

    p_pred_hybrid = pred_kind.add_parser("hybrid", help="Predict using Stage 3 hybrid model")
    p_pred_hybrid.add_argument("text", nargs="+")
    p_pred_hybrid.add_argument("--threshold", type=float, default=0.5)
    p_pred_hybrid.add_argument("--explain", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        if args.kind == "text":
            train_model(test_size=args.test_size, random_state=args.random_state, max_features=args.max_features)
            return 0
        if args.kind == "metadata":
            train_metadata_model(test_size=args.test_size, random_state=args.random_state)
            return 0
        if args.kind == "hybrid":
            train_hybrid_stack(test_size=args.test_size, random_state=args.random_state)
            return 0

    if args.cmd == "predict":
        email_text = " ".join(args.text)

        if args.kind == "text":
            pred, prob, decision = predict_text(email_text)
            print("Prediction:", pred, "| phishing_prob:", prob, "| decision:", decision)
            if args.explain:
                print_explain_text(email_text, pred, prob)
            return 0

        if args.kind == "metadata":
            pred, prob, decision = predict_metadata(email_text)
            print("Prediction:", pred, "| phishing_prob:", prob, "| decision:", decision)
            if args.explain:
                print_explain_metadata(email_text, pred, prob)
            return 0

        if args.kind == "hybrid":
            pred, hybrid_prob, decision, text_prob, meta_prob = predict_hybrid(email_text, threshold=args.threshold)
            print("Prediction:", pred, "| phishing_prob:", hybrid_prob, "| decision:", decision)
            if args.explain:
                print_explain_hybrid(email_text, pred, hybrid_prob, decision, text_prob, meta_prob)
            return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())