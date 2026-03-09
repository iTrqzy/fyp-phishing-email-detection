from phishingdet.data.loader import repo_root

def test_stage1_artifacts_exist_after_integration_run():
    base = repo_root() / "artifacts"
    assert (base / "model.joblib").exists()
    assert (base / "vectorizer.joblib").exists()
    assert (base / "results.json").exists()

def test_stage2_artifacts_exist_after_integration_run():
    base = repo_root() / "artifacts" / "stage2_metadata"
    assert (base / "results.json").exists()
    assert (base / "model.joblib").exists()
    assert (base / "vectorizer.joblib").exists()

    # Stage 2 outputs
    assert (base / "top_features_stage2_metadata.csv").exists()
    assert (base / "error_analysis.csv").exists()

def test_stage3_artifacts_exist_after_integration_run():
    base = repo_root() / "artifacts" / "stage3_hybrid"
    assert (base / "results.json").exists()
    assert (base / "meta_model.joblib").exists()
    assert (base / "text_model.joblib").exists()
    assert (base / "metadata_model.joblib").exists()
    assert (base / "metadata_vectorizer.joblib").exists()
    assert (base / "text_vectorizer.joblib").exists()