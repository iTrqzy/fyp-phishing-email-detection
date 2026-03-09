import json
import pytest
from phishingdet.data.loader import repo_root


def test_stage1_f1_above_threshold_if_results_exist():
    results_path = repo_root() / "artifacts" / "results.json"
    if not results_path.exists():
        pytest.skip("No Stage 1 results.json found; run training first.")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    f1 = float(data["metrics"]["f1"])


    assert f1 >= 0.90