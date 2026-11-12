import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from wdbc.evaluate import choose_threshold, classification_metrics
from wdbc.train import select_model


def test_metrics_on_a_hand_checked_example():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.1, 0.05]
    m = classification_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)


def test_threshold_meets_recall_target_and_is_highest_possible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 500), 0, 1)
    t = choose_threshold(y, p, 0.95)
    assert recall_score(y, p >= t) >= 0.95
    higher = np.unique(p)[np.unique(p) > t]
    if len(higher):
        assert recall_score(y, p >= higher.min()) < 0.95


def test_model_selection_prefers_simpler_model_on_ties():
    s = pd.DataFrame(
        {"roc_auc_mean": [0.9950, 0.9955, 0.9900]},
        index=["logreg_scratch", "random_forest", "decision_tree"],
    )
    assert select_model(s, "roc_auc", "auto", tie_tolerance=0.002) == "logreg_scratch"
    assert select_model(s, "roc_auc", "auto", tie_tolerance=0.0) == "random_forest"
    assert select_model(s, "roc_auc", "decision_tree") == "decision_tree"
