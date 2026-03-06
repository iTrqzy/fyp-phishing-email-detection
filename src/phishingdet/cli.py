import argparse
# Stage 1 (text-only)
from phishingdet.models.train import train_model as train_text_model
from phishingdet.models.predict import predict_text as predict_text_model

# Stage 2 (metadata-only)
from phishingdet.models.train_metadata import train_metadata_model
from phishingdet.models.predict_metadata import predict_metadata as predict_metadata_model

# Stage 3 (hybrid stacking)
from phishingdet.models.train_hybrid import train_hybrid_stack
from phishingdet.models.predict_hybrid import predict_hybrid as predict_hybrid_model


def main() -> int:
    parser = argparse.ArgumentParser(prog="phishingdet")
    top_subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------
    # TRAIN
    # -------------------------
    train_parser = top_subparsers.add_parser("train", help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="model", required=True)

    train_text = train_subparsers.add_parser("text", help="Train Stage 1 (text-only)")
    train_text.add_argument("--test_size", type=float, default=0.2)
    train_text.add_argument("--random_state", type=int, default=42)
    train_text.add_argument("--max_features", type=int, default=5000)

    train_metadata = train_subparsers.add_parser("metadata", help="Train Stage 2 (metadata-only)")
    train_metadata.add_argument("--test_size", type=float, default=0.2)

    train_hybrid = train_subparsers.add_parser("hybrid", help="Train Stage 3 (hybrid stacking)")
    train_hybrid.add_argument("--test_size", type=float, default=0.2)
    train_hybrid.add_argument("--folds", type=int, default=5)

    # -------------------------
    # PREDICT
    # -------------------------
    predict_parser = top_subparsers.add_parser("predict", help="Predict with a model")
    predict_subparsers = predict_parser.add_subparsers(dest="model", required=True)

    pred_text = predict_subparsers.add_parser("text", help="Predict using Stage 1 (text-only)")
    pred_text.add_argument("text", type=str, help="Email text to score")
    pred_text.add_argument("--explain", action="store_true", help="Show extra explanation output")

    pred_metadata = predict_subparsers.add_parser("metadata", help="Predict using Stage 2 (metadata-only)")
    pred_metadata.add_argument("text", type=str, help="Email text to score")
    pred_metadata.add_argument("--explain", action="store_true", help="Show extra explanation output")

    pred_hybrid = predict_subparsers.add_parser("hybrid", help="Predict using Stage 3 (hybrid)")
    pred_hybrid.add_argument("text", type=str, help="Email text to score")
    pred_hybrid.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for hybrid probability")
    pred_hybrid.add_argument("--explain", action="store_true", help="Show explanation output")

    args = parser.parse_args()

    # -------------------------
    # DISPATCH: TRAIN
    # -------------------------
    if args.command == "train":
        if args.model == "text":
            train_text_model(
                test_size=args.test_size,
                random_state=args.random_state,
                max_features=args.max_features,
            )
            return 0

        if args.model == "metadata":
            train_metadata_model(test_size=args.test_size)
            return 0

        if args.model == "hybrid":
            train_hybrid_stack(test_size=args.test_size, n_folds=args.folds)
            return 0

    # -------------------------
    # DISPATCH: PREDICT
    # -------------------------
    if args.command == "predict":
        if args.model == "text":
            pred, prob, decision = predict_text_model(args.text)
            print("Prediction:", pred, "| phishing_prob:", prob, "| decision:", decision)

            if args.explain:
                print("\n--- EXPLANATION ---")
                print("This is Stage 1 (text-only): TF-IDF + Logistic Regression.")
                print("-------------------")
            return 0

        if args.model == "metadata":
            pred, prob, decision = predict_metadata_model(args.text)
            print("Prediction:", pred, "| phishing_prob:", prob, "| decision:", decision)

            if args.explain:
                print("\n--- EXPLANATION ---")
                print("This is Stage 2 (metadata-only): engineered cues -> DictVectorizer + Logistic Regression.")
                print("-------------------")
            return 0

        if args.model == "hybrid":
            # IMPORTANT: do NOT pass explain into predict_hybrid_model (it doesn't accept it)
            pred, hybrid_prob, decision, text_prob, metadata_prob = predict_hybrid_model(
                args.text,
                threshold=args.threshold,
            )
            print("Prediction:", pred, "| phishing_prob:", hybrid_prob, "| decision:", decision)

            if args.explain:
                print("\n--- EXPLANATION ---")
                print("Hybrid probability:", round(hybrid_prob, 4), "| decision:", decision)
                print("Base model probs  : text=", round(text_prob, 4), "| metadata=", round(metadata_prob, 4))
                print("-------------------")
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())