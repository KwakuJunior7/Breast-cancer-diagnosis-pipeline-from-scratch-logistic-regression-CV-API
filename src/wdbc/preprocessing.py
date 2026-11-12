"""Z-score standardisation written from scratch.

This is the Python version of MLA Lecture 8 (cs_5740_mla_l8_z_score_9182025.m):

    x_mean = mean(X);  x_std = std(X);
    Z = (X - ones(n,1)*x_mean) .* (1 ./ (ones(n,1)*x_std));

Two things change for a real pipeline:

1. fit / transform are separate. The mean and std are learned from the
   TRAINING rows only and then re-used on validation, test and live data.
   Computing them on all data first lets test information leak into training.

2. NumPy broadcasting replaces ones(n,1)*x_mean: `X - mean` subtracts the
   1 x p row vector from every row automatically.

It follows the scikit-learn transformer interface (fit / transform), so it can
sit inside an sklearn Pipeline and be cloned inside cross-validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class ZScoreScaler(TransformerMixin, BaseEstimator):
    """z = (x - mean) / std, column by column.

    ddof=1 matches Octave/Matlab std() (sample standard deviation).
    scikit-learn's StandardScaler uses ddof=0; the difference is a factor of
    sqrt(n/(n-1)), about 0.1% for n = 455, and does not change any model.
    """

    def __init__(self, ddof: int = 1):
        self.ddof = ddof

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=self.ddof)
        # A constant column has std 0; dividing by it would give inf/NaN.
        self.scale_ = np.where(std > 0, std, 1.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        check_is_fitted(self, ["mean_", "scale_"])
        return np.asarray(Z, dtype=float) * self.scale_ + self.mean_


def correlation_from_z(Z: np.ndarray) -> np.ndarray:
    """Correlation matrix R from standardised data: R = Z'Z / (n - 1).

    n is the number of ROWS (samples). The Octave lecture script divided by
    (N - 1) with N = size(X,2), the number of columns; it gave the right answer
    only because that toy X was 3 x 3.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    return Z.T @ Z / (n - 1)
