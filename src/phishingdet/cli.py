import argparse

from phishingdet.models.train import train_model
from phishingdet.models.predict_hybrid import predict_hybrid
from phishingdet.features.build_metadata_features import extract_metadata_features_one
from phishingdet.models.predict_hybrid import stage1_top_features_csv
import csv
import re
from pathlib import Path

def load_stage1_feature_weights():
    csv_path = stage1_top_features_csv()
    weights = {}

    if csv_path is None:
        return weights

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = (row.get("feature") or "").strip().lower()
            if feature:
                weights[feature] = float(row.get("coefficient", 0.0))

    return weights
def tokens_in_text(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def top_token_hits(text, feature_weights, max_show=5):
    tokens = tokens_in_text(text)
    hits = []

    for feature, weight in feature_weights.items():
        if " " not in feature:
            if feature in tokens:
                hits.append((feature,weight))
    hits.sort(key=lambda x: abs(x[1]), reverse=True)
    return hits[:max_show]

def main():
    parser = argparse.ArgumentParser(prog="phishingdet")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train the Stage 1 text-only model")
    p_train.add_argument("--test_size", type=float, default=0.2)
    p_train.add_argument("--random_state", type=int, default=42)
    p_train.add_argument("--max_features", type=int, default=5000)

    p_pred = sub.add_parser("predict", help="Predict a single email text")
    p_pred.add_argument("--text", required=True)
    p_pred.add_argument("--explain", action="store_true", help="Show explanation output")

    args = parser.parse_args()

    if args.cmd == "train":
        train_model(
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
        )
        return 0

    if args.cmd == "predict":
        # Hybrid prediction (Stage 3)
        pred, prob, decision, text_prob, meta_prob = predict_hybrid(args.text)

        print("Prediction:", pred, "| phishing_prob:", prob, "| decision:", decision)

        if args.explain:
            print("\n--- EXPLANATION ---")
            print(f"Hybrid probability: {round(prob, 4)} | decision: {decision}")
            print(f"Base model probs  : text={round(text_prob, 4)} | meta={round(meta_prob, 4)}")
            print()

            # Stage 2 evidence: show the metadata features triggered by this email
            meta_features = extract_metadata_features_one(args.text)

            print("Triggered metadata:")
            for key, value in meta_features.items():
                # only show “active” ones (non-zero / non-empty)
                if value:
                    print(f"  {key} = {value}")

            # Stage 1 evidence: show top-weighted text tokens that appear in this email
            stage1_feature_weights = load_stage1_feature_weights()
            token_hits = top_token_hits(args.text, stage1_feature_weights, max_show=5)

            print("\nTop text cues present:")
            if token_hits:
                for token, weight in token_hits:
                    side = "PHISHING" if weight > 0 else "LEGIT"
                    print(f"  {token} ({side}) weight={round(weight, 4)}")
            else:
                print("  (no top tokens matched)")

            print("-------------------\n")

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
