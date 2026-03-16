import argparse
from pathlib import Path
import joblib

from phishingdet.data.loader import repo_root
from phishingdet.features.build_metadata_features import transform_metadata_vectorizer

def load_stage2_artifacts():
    artifacts_dir = repo_root() / "artifacts" / "stage2_metadata"
    model_path = artifacts_dir / "model.joblib"
    vectorizer_path = artifacts_dir / "vectorizer.joblib"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Missing Stage 2 artifacts.\n"
            "Train Stage 2 first:\n"
            "  python -m phishingdet.models.train_metadata\n"
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_metadata(text, phishing_threshold=0.70, legit_threshold=0.30):
    model, vectorizer = load_stage2_artifacts()

    # Turn text into metadata feature matrix
    X = transform_metadata_vectorizer(vectorizer, [text])

    # Class prediction (0/1)
    prediction = model.predict(X)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0][1]  # probability of label=1 (phishing)

    decision = "Uncertain"
    if probability is not None:
        if probability >= phishing_threshold:
            decision = "Phishing"
        elif probability <= legit_threshold:
            decision = "Legit"

    return int(prediction), probability, decision


def main():
    parser = argparse.ArgumentParser(prog="predict_metadata")
    parser.add_argument("--text", type=str, default=None, help="Email text to score")
    args = parser.parse_args()

    if args.text:
        prediction, probability, decision = predict_metadata(args.text)
        print("Text:", args.text)
        print(f"Prediction: {prediction} | Probability: {probability} | Decision: {decision}")
        return

    sample_1 = "Win a free iPhone now!!! Click here http://1.2.3.4/login"
    sample_2 = "Hi, are we still on for the meeting tomorrow?"

    prediction1, probability1, decision1 = predict_metadata(sample_1)
    print("Text:", sample_1)
    print(f"Prediction: {prediction1} | Probability: {probability1} | Decision: {decision1}")

    prediction2, probability2, decision2 = predict_metadata(sample_2)
    print("Text:", sample_2)
    print(f"Prediction: {prediction2} | Probability: {probability2} | Decision: {decision2}")


if __name__ == "__main__":
    main()
