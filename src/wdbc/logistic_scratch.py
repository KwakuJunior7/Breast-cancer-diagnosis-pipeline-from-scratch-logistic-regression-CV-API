"""Logistic regression trained by batch gradient descent, in NumPy only.

How it connects to the MLA 5740 linear-regression GD code
---------------------------------------------------------
Linear regression (Lectures 18-24)          Logistic regression (this file)
  X_hat = [1 | X]                     ->      same augmented matrix
  y_pred = X_hat * u                  ->      p = sigmoid(X_hat * u)        (a probability)
  C = 1/(2n) * sum(e.^2)              ->      J = mean cross-entropy (+ L2 penalty)
  Grad = -(1/n) * X_hat' * e          ->      Grad = (1/n) * X_hat' * (p - y)   (+ lambda*w)
  u_new = u_old - mu * Grad           ->      identical update rule
  epochs = 5                          ->      loop until |J(k) - J(k-1)| < tol

The gradient has the same shape as in linear regression because the
derivative of the sigmoid cancels the derivative of the log in the
cross-entropy. That is the main reason cross-entropy (not squared error) is
the standard loss for classification: it keeps the problem convex, so gradient
descent reaches the single global minimum.

Notation
    X   : n x p feature matrix (already z-scored)
    y   : n vector of 0/1 labels
    w   : p weights,  b : bias (intercept)
    z   = X w + b                      (the "logit")
    p   = sigmoid(z) = 1 / (1 + e^-z)  (probability of class 1 = malignant)
    J   = -(1/n) sum[ y log p + (1-y) log(1-p) ] + (lambda/2) ||w||^2
    dJ/dw = (1/n) X'(p - y) + lambda w
    dJ/db = (1/n) sum(p - y)            (the bias is not regularised)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    1/(1+exp(-z)) overflows for large negative z (exp(800) = inf). Splitting by
    sign keeps every exp() argument <= 0, so it never overflows.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def cross_entropy(y: np.ndarray, z: np.ndarray) -> float:
    """Mean binary cross-entropy computed from logits z (not probabilities).

    Uses log(1 + e^z) = logaddexp(0, z), which avoids log(0) when p is
    exactly 0 or 1.  -[y log p + (1-y) log(1-p)] = log(1+e^z) - y z.
    """
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


class ScratchLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression fitted by full-batch gradient descent.

    Parameters
    ----------
    learning_rate : step size mu in  w <- w - mu * grad, or "auto".
                    "auto" uses mu = 1/L, where L = lambda_max(X'X)/(4n) + l2 is the
                    Lipschitz constant of the gradient (the steepest the cost's
                    slope can change). Any mu < 2/L is guaranteed not to diverge;
                    1/L is the textbook safe choice.
    l2            : lambda, strength of the L2 penalty on w (not on b).
    max_iter      : hard cap on iterations.
    tol           : stop when the cost improves by less than this.
    random_state  : seed for the small random initial weights.

    After fit():
    coef_ (1 x p), intercept_ (1,), cost_history_ (list), n_iter_, converged_,
    learning_rate_ (the step size actually used)
    """

    def __init__(self, learning_rate="auto", l2=0.01, max_iter=20000, tol=1e-9, random_state=0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ---- the math -------------------------------------------------------
    def _cost(self, X, y, w, b):
        z = X @ w + b
        return cross_entropy(y, z) + 0.5 * self.l2 * float(w @ w)

    def _gradient(self, X, y, w, b):
        n = X.shape[0]
        p = sigmoid(X @ w + b)
        error = p - y  # same role as e in linear regression
        grad_w = X.T @ error / n + self.l2 * w
        grad_b = float(error.sum() / n)
        return grad_w, grad_b

    def _step_size(self, X) -> float:
        if self.learning_rate != "auto":
            return float(self.learning_rate)
        n = X.shape[0]
        # Include the bias column, since b is updated with the same step.
        Xb = np.column_stack([np.ones(n), X])
        lam_max = float(np.linalg.eigvalsh(Xb.T @ Xb / n).max())
        return 1.0 / (lam_max / 4.0 + self.l2)

    # ---- sklearn interface ---------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("labels must be 0 and 1")
        y = y.astype(float)

        self.learning_rate_ = self._step_size(X)
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(0.0, 0.01, size=X.shape[1])  # "throw the dart" near zero
        b = 0.0

        history = [self._cost(X, y, w, b)]
        self.converged_ = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007 (n_iter is used below)
            grad_w, grad_b = self._gradient(X, y, w, b)
            w = w - self.learning_rate_ * grad_w  # gradient-descent steering
            b = b - self.learning_rate_ * grad_b
            history.append(self._cost(X, y, w, b))
            if abs(history[-2] - history[-1]) < self.tol:
                self.converged_ = True
                break
            if not np.isfinite(history[-1]):
                raise FloatingPointError(
                    "cost diverged; lower the learning rate or scale the features"
                )

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        self.cost_history_ = history
        self.n_iter_ = n_iter
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X):
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def numerical_gradient(model: ScratchLogisticRegression, X, y, w, b, eps=1e-6):
    """Central finite differences: dJ/dw_j ~ (J(w + eps e_j) - J(w - eps e_j)) / 2eps.

    Used in the tests to prove the analytic gradient in _gradient() is right.
    This is the standard way to check hand-derived backprop / GD formulas.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    g_w = np.zeros_like(w)
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = eps
        g_w[j] = (model._cost(X, y, w + step, b) - model._cost(X, y, w - step, b)) / (2 * eps)
    g_b = (model._cost(X, y, w, b + eps) - model._cost(X, y, w, b - eps)) / (2 * eps)
    return g_w, g_b
