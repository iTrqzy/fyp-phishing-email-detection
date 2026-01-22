from pathlib import Path
import joblib

from phishingdet.data.loader import repo_root


def load_artifacts():
    artifacts_path = Path(repo_root()) / "artifacts"
    model_path = artifacts_path / "model.joblib"
    vectorizer_path = artifacts_path / "vectorizer.joblib"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Train model first:\n"
            "python -m phishingdet.models.train"
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_text(text):
    model, vectorizer = load_artifacts()
    x = vectorizer.transform([text])  # must be list of strings

    pred = int(model.predict(x)[0])  # 0 or 1
    phishing_prob = None

    if hasattr(model, "predict_proba"):
        phishing_prob = float(model.predict_proba(x)[0][1])

    # decision band
    decision = "phishing" if pred == 1 else "legit"
    if phishing_prob is not None and 0.30 < phishing_prob < 0.70:
        decision = "uncertain"

    return pred, phishing_prob, decision


if __name__ == "__main__":
    sample_1 = "Win a free iPhone now!!! Click here"
    sample_2 = "Hi, are we still on for the meeting tomorrow?"

    pred1, prob1, dec1 = predict_text(sample_1)
    print("Sample 1:", sample_1)
    print("Prediction:", pred1, "| phishing_prob:", prob1, "| decision:", dec1)

    pred2, prob2, dec2 = predict_text(sample_2)
    print("\nSample 2:", sample_2)
    print("Prediction:", pred2, "| phishing_prob:", prob2, "| decision:", dec2)
