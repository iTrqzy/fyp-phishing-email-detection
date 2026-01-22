import argparse

from phishingdet.models.train import train_model
from phishingdet.models.predict import predict_text


def main():
    parser = argparse.ArgumentParser(prog="phishingdet")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train the Stage 1 text-only model")
    p_train.add_argument("--test_size", type=float, default=0.2)
    p_train.add_argument("--random_state", type=int, default=42)
    p_train.add_argument("--max_features", type=int, default=5000)

    p_pred = sub.add_parser("predict", help="Predict a single email text")
    p_pred.add_argument("--text", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train_model(
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
        )
        return 0

    if args.cmd == "predict":
        pred, prob, decision = predict_text(args.text)
        print("Prediction:", pred, "| phishing_prob:", prob, "| decision:", decision)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
