import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from wdbc.logistic_scratch import (
    ScratchLogisticRegression,
    cross_entropy,
    numerical_gradient,
    sigmoid,
)
from wdbc.models import MatchedLogisticRegression
from wdbc.preprocessing import ZScoreScaler


@pytest.fixture(scope="module")
def zdata(dataset):
    X_tr, _, y_tr, _ = dataset
    return ZScoreScaler().fit_transform(X_tr.values), y_tr.values


def test_sigmoid_is_stable_for_extreme_inputs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        out = sigmoid(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]))
    assert out[0] == 0.0 and out[-1] == 1.0 and out[2] == 0.5
    assert np.all(np.diff(out) >= 0)


def test_cross_entropy_matches_sklearn_log_loss():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(float)
    z = rng.normal(0, 3, 200)
    assert cross_entropy(y, z) == pytest.approx(log_loss(y, sigmoid(z)), rel=1e-9)


def test_analytic_gradient_matches_finite_differences(zdata):
    """The key test: proves the hand-derived gradient formula is correct."""
    X, y = zdata
    model = ScratchLogisticRegression(l2=0.05)
    rng = np.random.default_rng(1)
    w, b = rng.normal(0, 0.5, X.shape[1]), 0.3
    g_w, g_b = model._gradient(X, y.astype(float), w, b)
    n_w, n_b = numerical_gradient(model, X, y, w, b)
    np.testing.assert_allclose(g_w, n_w, rtol=1e-5, atol=1e-8)
    assert g_b == pytest.approx(n_b, rel=1e-5)


def test_gradient_descent_converges_and_cost_never_rises(zdata):
    X, y = zdata
    m = ScratchLogisticRegression().fit(X, y)
    assert m.converged_
    assert np.all(np.diff(m.cost_history_) <= 1e-12)  # monotone with step 1/L


def test_matches_sklearn_on_the_same_objective(zdata):
    X, y = zdata
    ours = ScratchLogisticRegression(l2=0.01, tol=1e-12, max_iter=100000).fit(X, y)
    ref = MatchedLogisticRegression(l2=0.01, tol=1e-12).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=2e-3)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-3)


def test_auto_learning_rate_is_one_over_lipschitz(zdata):
    X, y = zdata
    m = ScratchLogisticRegression(l2=0.0).fit(X, y)
    Xb = np.column_stack([np.ones(len(X)), X])
    L = np.linalg.eigvalsh(Xb.T @ Xb / len(X)).max() / 4
    assert m.learning_rate_ == pytest.approx(1 / L)


def test_too_large_learning_rate_is_reported(zdata):
    X, y = zdata
    m = ScratchLogisticRegression(learning_rate=50.0, max_iter=200, tol=0).fit(X, y)
    assert not np.all(np.diff(m.cost_history_) <= 0)  # oscillates: the reason for 1/L


def test_works_inside_sklearn_pipeline_and_cv(dataset):
    X_tr, _, y_tr, _ = dataset
    pipe = Pipeline([("scaler", ZScoreScaler()), ("clf", ScratchLogisticRegression())])
    scores = cross_val_score(pipe, X_tr.values, y_tr.values, cv=3, scoring="roc_auc")
    assert scores.mean() > 0.98


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        ScratchLogisticRegression().fit(np.eye(3), np.array([0, 1, 2]))
