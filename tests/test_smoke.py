from pathlib import Path
import pytest

from phishingdet.data.loader import repo_root
from phishingdet.models.predict import predict_text


def test_smoke_stage1_predict_if_artifacts_exist():
    # Checks if stage 1 artifacts exist, load them and run one prediction.
    # Avoiding re-training and still proves end-to-end prediction.

    artifacts_dir = repo_root() / "artifacts"
    model_path = artifacts_dir / "model.joblib"
    vec_path = artifacts_dir / "vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        pytest.skip("Missing artifacts. Train model first:\n"
                    "python -m phishingdet.models.train")

    pred, prob, decision = predict_text("Win a free iPhone now!!! Click here http://1.2.3.4/login")

    assert pred in (0,1)
    assert 0.0 <= float(prob) <= 1.0
    assert decision in ("phishing", "legit", "uncertain")