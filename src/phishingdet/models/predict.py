from pathlib import Path
import joblib

from phishingdet.data.loader import repo_root


def load_artifacts():
    # load the paths for the joblib files
    artifacts_path = Path(repo_root()) / "artifacts"
    model_path = artifacts_path / "model.joblib"
    vectorizer_path = artifacts_path / "vectorizer.joblib"

    # error handling
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Train model first:\n"
            "python -m phishingdet.models.train"
        )

    # load the data within the files
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_text(text):
    model, vectorizer = load_artifacts()

    # convert text to numbers using the saved vectorizer
    x = vectorizer.transform([text])  # must be a list of strings

    # 0 or 1 prediction
    pred = model.predict(x)[0]

    # confidence (probability of phishing = 1)
    phishing_prob = None
    if hasattr(model, "predict_proba"):
        phishing_prob = model.predict_proba(x)[0][1]

    # simple decision rule
    # >= 0.70 -> phishing
    # <= 0.30 -> legit
    # otherwise -> uncertain
    # default decision is whatever the model predicted (0.5 threshold)
    decision = "phishing" if pred == 1 else "legit"

    # only label "uncertain" in the middle band
    if phishing_prob is not None and 0.30 < phishing_prob < 0.70:
        decision = "uncertain"

    return int(pred), phishing_prob, decision


if __name__ == "__main__":
    sample_1 = "Win a free iPhone now!!! Click here"
    sample_2 = "Hi, are we still on for the meeting tomorrow?"

    pred1, prob1, dec1 = predict_text(sample_1)
    print("Sample 1:", sample_1)
    print("Prediction:", pred1, "| phishing_prob:", prob1, "| decision:", dec1)

    pred2, prob2, dec2 = predict_text(sample_2)
    print("\nSample 2:", sample_2)
    print("Prediction:", pred2, "| phishing_prob:", prob2, "| decision:", dec2)
