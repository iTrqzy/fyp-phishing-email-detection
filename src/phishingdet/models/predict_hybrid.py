import argparse
from pathlib import Path
import joblib

from phishingdet.data.loader import repo_root
from phishingdet.features.build_features import transform_vectorizer
from phishingdet.features.build_metadata_features import transform_metadata_vectorizer


def load_stage3_artifacts():
    artifacts_dir = Path(repo_root()) / "artifacts" / "stage3_hybrid"

    text_model_path = artifacts_dir / "text_model.joblib"
    text_vectorizer_path = artifacts_dir / "text_vectorizer.joblib"

    metadata_model_path = artifacts_dir / "metadata_model.joblib"
    metadata_vectorizer_path = artifacts_dir / "metadata_vectorizer.joblib"  # FIXED

    meta_model_path = artifacts_dir / "meta_model.joblib"

    if not (
        text_model_path.exists()
        and text_vectorizer_path.exists()
        and metadata_model_path.exists()
        and metadata_vectorizer_path.exists()
        and meta_model_path.exists()
    ):
        raise FileNotFoundError(
            "Missing Stage 3 artifacts.\n"
            "Train Stage 3 first:\n"
            "  python -m phishingdet.models.train_hybrid\n"
        )

    text_model = joblib.load(text_model_path)
    text_vectorizer = joblib.load(text_vectorizer_path)

    metadata_model = joblib.load(metadata_model_path)
    metadata_vectorizer = joblib.load(metadata_vectorizer_path)

    meta_model = joblib.load(meta_model_path)

    return text_model, text_vectorizer, metadata_model, metadata_vectorizer, meta_model


def predict_hybrid(text, threshold=0.5):
    text_model, text_vectorizer, metadata_model, metadata_vectorizer, meta_model = load_stage3_artifacts()

    # Stage 1 probability (text)
    X_text = transform_vectorizer(text_vectorizer, [text])
    text_prob = float(text_model.predict_proba(X_text)[0][1])

    # Stage 2 probability (metadata)
    X_meta = transform_metadata_vectorizer(metadata_vectorizer, [text])
    metadata_prob = float(metadata_model.predict_proba(X_meta)[0][1])

    # Stage 3 stacking: meta model combines the two probabilities
    hybrid_prob = float(meta_model.predict_proba([[text_prob, metadata_prob]])[0][1])
    pred = 1 if hybrid_prob >= threshold else 0

    decision = "phishing" if pred == 1 else "legit"
    if 0.30 < hybrid_prob < 0.70:
        decision = "uncertain"

    return pred, hybrid_prob, decision, text_prob, metadata_prob


def main():
    parser = argparse.ArgumentParser(prog="predict_hybrid")
    parser.add_argument("--text", type=str, default=None, help="Email text to score")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for hybrid prob")
    args = parser.parse_args()

    if args.text:
        pred, hybrid_prob, decision, text_prob, metadata_prob = predict_hybrid(args.text, threshold=args.threshold)
        print("Text:", args.text)
        print("Prediction:", pred, "| hybrid_prob:", hybrid_prob, "| decision:", decision)
        print("  text_prob    :", text_prob)
        print("  metadata_prob:", metadata_prob)
        return

    sample_1 = "Win a free iPhone now!!! Click here http://1.2.3.4/login"
    sample_2 = "Hi, are we still on for the meeting tomorrow?"

    for s in [sample_1, sample_2]:
        pred, hybrid_prob, decision, text_prob, metadata_prob = predict_hybrid(s, threshold=args.threshold)
        print("\nText:", s)
        print("Prediction:", pred, "| hybrid_prob:", hybrid_prob, "| decision:", decision)
        print("  text_prob    :", text_prob)
        print("  metadata_prob:", metadata_prob)


if __name__ == "__main__":
    main()