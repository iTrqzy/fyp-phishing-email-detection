import numpy as np
import pytest

import phishingdet.data.splits as splits

def test_same_split_twice_is_identical():
    # Tiny deterministic label list (balanced)
    labels = np.array([0, 1] * 50)

    a_train, a_test = splits.get_or_make_split_indices(labels, test_size=0.2, random_state=42)
    b_train, b_test = splits.get_or_make_split_indices(labels, test_size=0.2, random_state=42)

    assert list(a_train) == list(b_train)
    assert list(a_test) == list(b_test)