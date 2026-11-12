"""Metrics, cross-validation, threshold choice and evaluation plots.

Why these metrics (positive class = malignant = 1)
--------------------------------------------------
accuracy     share of all predictions that are right. Misleading here: always
             answering "benign" already scores 357/569 = 63%.
recall       of the truly malignant tumours, the share we catch  TP/(TP+FN).
             A false negative (missed cancer) is the costliest error, so this
             is the metric we protect with the threshold.
precision    of the tumours we flag malignant, the share that really are  TP/(TP+FP).
specificity  of the truly benign tumours, the share we clear  TN/(TN+FP).
F1           harmonic mean of precision and recall.
ROC-AUC      probability that a random malignant case scores higher than a
             random benign one. Threshold-free, so it is used to rank models.
PR-AUC       area under precision-recall curve (average precision); more
             informative than ROC-AUC when the positive class is the minority.
Brier score  mean squared error of the probabilities; checks calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def cross_validate_models(models: dict, X, y, n_splits: int, n_repeats: int, random_state: int):
    """Repeated stratified k-fold CV on the TRAINING set.

    Returns
    -------
    fold_scores : DataFrame, one row per (model, repeat, fold)
    oof_proba   : {model: out-of-fold probability per training row, averaged
                   over repeats}. Every row is predicted by a model that never
                   saw it, so these probabilities are honest and can be used to
                   choose the decision threshold without touching the test set.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    oof_sum = {name: np.zeros(len(y)) for name in models}
    for split_idx, (tr, va) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(split_idx, n_splits)
        for name, model in models.items():
            fitted = clone(model).fit(X[tr], y[tr])
            p = fitted.predict_proba(X[va])[:, 1]
            oof_sum[name][va] += p
            m = classification_metrics(y[va], p)
            rows.append({"model": name, "repeat": repeat, "fold": fold, **m})
    oof_proba = {name: s / n_repeats for name, s in oof_sum.items()}
    return pd.DataFrame(rows), oof_proba


def summarise_cv(
    fold_scores: pd.DataFrame,
    metrics=("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"),
):
    """Mean and standard deviation of each metric across folds."""
    agg = fold_scores.groupby("model")[list(metrics)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg


def choose_threshold(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target_recall.

    Lowering the threshold flags more tumours as malignant: recall goes up,
    precision goes down. We want the smallest number of false alarms that still
    meets the recall target, i.e. the HIGHEST threshold that meets it.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    candidates = np.unique(proba)[::-1]  # high -> low
    for t in candidates:
        if recall_score(y_true, (proba >= t).astype(int)) >= target_recall:
            return float(t)
    return float(candidates[-1])


# ---------------------------------------------------------------- plots ---
def _plt():
    import matplotlib

    matplotlib.use("Agg")  # no display needed (CI, Docker)
    import matplotlib.pyplot as plt

    return plt


def plot_roc_pr(y_true, oof_proba: dict, display_names: dict, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in oof_proba.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        prec, rec, _ = precision_recall_curve(y_true, p)
        # The two logistic regressions coincide; dash one so both stay visible.
        style = {"ls": "--", "lw": 2.2} if name == "logreg_sklearn" else {"lw": 1.4}
        axes[0].plot(
            fpr, tpr, label=f"{display_names[name]} ({roc_auc_score(y_true, p):.3f})", **style
        )
        axes[1].plot(
            rec,
            prec,
            label=f"{display_names[name]} ({average_precision_score(y_true, p):.3f})",
            **style,
        )
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
        title="ROC, out-of-fold (AUC)",
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, out-of-fold (AP)")
    for ax in axes:
        ax.legend(fontsize=7.5, loc="lower right" if ax is axes[0] else "lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, proba, threshold, title, path):
    plt = _plt()
    cm = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["benign", "malignant"])
    ax.set_yticks([0, 1], ["benign", "malignant"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_tradeoff(y_true, proba, chosen, target_recall, path):
    plt = _plt()
    prec, rec, thr = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr, rec[:-1], label="Recall")
    ax.plot(thr, prec[:-1], label="Precision")
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"chosen threshold = {chosen:.3f}")
    ax.axhline(target_recall, color="grey", ls=":", lw=1, label=f"recall target = {target_recall}")
    ax.set(
        xlabel="Decision threshold on P(malignant)",
        ylabel="Score",
        title="Threshold trade-off (out-of-fold, deployed model)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_history(history, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(history) + 1), history)  # x starts at 1 for the log axis
    ax.set_xscale("log")
    ax.set(
        xlabel="Iteration, 1 = initial weights (log scale)",
        ylabel="Cost J(w, b)",
        title=f"Gradient descent convergence ({len(history) - 1} iterations)",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
