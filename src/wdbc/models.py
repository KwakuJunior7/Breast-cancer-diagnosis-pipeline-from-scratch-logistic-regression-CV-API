"""Candidate models compared in cross-validation.

Every candidate is a Pipeline: scaler -> classifier. Putting the scaler inside
the pipeline means cross_validate() refits it on each training fold, so the
validation fold never influences the mean/std (no leakage).

Trees and forests do not need scaling, but keeping one pipeline shape for every
model keeps the code simple and the comparison fair.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .logistic_scratch import ScratchLogisticRegression
from .preprocessing import ZScoreScaler

# Order matters only for ties: simpler models are listed first and win ties.
MODEL_ORDER = [
    "logreg_scratch",
    "logreg_sklearn",
    "decision_tree",
    "random_forest",
    "svm_rbf",
]

DISPLAY_NAMES = {
    "logreg_scratch": "Logistic regression (NumPy, from scratch)",
    "logreg_sklearn": "Logistic regression (scikit-learn)",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "svm_rbf": "SVM (RBF kernel)",
}


class MatchedLogisticRegression(LogisticRegression):
    """scikit-learn's LogisticRegression set to solve the SAME problem as the
    scratch model, so the two can be compared like for like.

    sklearn minimises  C * sum(log-loss) + 0.5*||w||^2.  Dividing by C*n gives
    mean(log-loss) + (1/(2 C n))*||w||^2, so C = 1/(lambda * n) makes the two
    objectives identical. n changes with each CV fold, so C is set at fit time.
    """

    def __init__(self, l2=0.01, max_iter=10000, tol=1e-10):
        self.l2 = l2
        super().__init__(C=1.0, max_iter=max_iter, tol=tol)

    def fit(self, X, y, sample_weight=None):
        self.C = 1.0 / (self.l2 * len(X))
        return super().fit(X, y, sample_weight=sample_weight)


def build_models(cfg: dict) -> dict[str, Pipeline]:
    """Return {name: unfitted pipeline}."""
    s = cfg["scratch_logreg"]
    seed = cfg["data"]["random_state"]

    classifiers = {
        "logreg_scratch": ScratchLogisticRegression(
            learning_rate=s["learning_rate"],
            l2=s["l2"],
            max_iter=s["max_iter"],
            tol=s["tol"],
            random_state=seed,
        ),
        "logreg_sklearn": MatchedLogisticRegression(l2=s["l2"]),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        # SVMs output distances, not probabilities. Platt scaling inside
        # CalibratedClassifierCV turns those distances into probabilities.
        "svm_rbf": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale"),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
    }
    return {
        name: Pipeline([("scaler", ZScoreScaler()), ("clf", classifiers[name])])
        for name in MODEL_ORDER
    }
