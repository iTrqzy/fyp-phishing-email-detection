import numpy as np

from phishingdet.models.train_metadata import best_threshold_by_f1


def test_best_threshold_by_f1_returns_reasonable_values():
    # y_true: 1s should align with high probs
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, f1 = best_threshold_by_f1(y_true, probs)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.5