import numpy as np
from sklearn.preprocessing import StandardScaler

from wdbc.preprocessing import ZScoreScaler, correlation_from_z


def test_zscore_has_zero_mean_unit_std(dataset):
    X_tr = dataset[0].values
    Z = ZScoreScaler().fit_transform(X_tr)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1, atol=1e-12)


def test_matches_sklearn_up_to_ddof(dataset):
    X_tr = dataset[0].values
    n = len(X_tr)
    ours = ZScoreScaler(ddof=0).fit_transform(X_tr)
    np.testing.assert_allclose(ours, StandardScaler().fit_transform(X_tr), atol=1e-12)
    ours_ddof1 = ZScoreScaler(ddof=1).fit_transform(X_tr)
    np.testing.assert_allclose(ours_ddof1 * np.sqrt(n / (n - 1)), ours, atol=1e-12)


def test_uses_training_statistics_only(dataset):
    X_tr, X_te = dataset[0].values, dataset[1].values
    scaler = ZScoreScaler().fit(X_tr)
    np.testing.assert_allclose(scaler.transform(X_te), (X_te - X_tr.mean(0)) / X_tr.std(0, ddof=1))


def test_constant_column_does_not_blow_up():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Z = ZScoreScaler().fit_transform(X)
    assert np.isfinite(Z).all()
    assert (Z[:, 1] == 0).all()


def test_inverse_transform_round_trip(dataset):
    X_tr = dataset[0].values
    s = ZScoreScaler().fit(X_tr)
    np.testing.assert_allclose(s.inverse_transform(s.transform(X_tr)), X_tr)


def test_correlation_from_z_equals_numpy_corrcoef(dataset):
    X_tr = dataset[0].values
    Z = ZScoreScaler().fit_transform(X_tr)
    np.testing.assert_allclose(correlation_from_z(Z), np.corrcoef(X_tr, rowvar=False), atol=1e-12)
