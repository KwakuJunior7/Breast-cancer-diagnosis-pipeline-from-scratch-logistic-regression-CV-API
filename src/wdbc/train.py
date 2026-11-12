"""End-to-end training pipeline.

Run:  python -m wdbc.train            (or: make train)

Steps
  1. Load + validate data, stratified 80/20 split. The test set is locked away.
  2. Repeated stratified 5-fold CV of every candidate on the training set.
  3. Pick the deployed model by mean CV ROC-AUC (config: model_selection).
  4. Pick its decision threshold from out-of-fold probabilities so that
     recall >= target (config: threshold.target_recall).
  5. Refit that model on the full training set.
  6. Evaluate ONCE on the untouched test set, at 0.5 and at the chosen threshold.
  7. Save the model + metadata, write reports and figures, log the run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from . import __version__
from .config import load_config
from .data import FEATURE_NAMES, load_dataset, split
from .evaluate import (
    choose_threshold,
    classification_metrics,
    cross_validate_models,
    plot_confusion,
    plot_cost_history,
    plot_roc_pr,
    plot_threshold_tradeoff,
    summarise_cv,
)
from .models import DISPLAY_NAMES, MODEL_ORDER, build_models
from .tracking import log_run


def select_model(
    cv_summary: pd.DataFrame, metric: str, deploy: str, tie_tolerance: float = 0.0
) -> str:
    """Best mean CV score wins; anything within `tie_tolerance` of the best is
    treated as a tie and the simplest model (earliest in MODEL_ORDER) is chosen.

    Why a tolerance: fold-to-fold standard deviation of ROC-AUC here is about
    0.005, so a 0.0001 gap between two models is noise, not a real difference.
    Preferring the simpler model when results are equal is Occam's razor.
    """
    if deploy != "auto":
        if deploy not in cv_summary.index:
            raise ValueError(f"unknown model '{deploy}'")
        return deploy
    scores = cv_summary[f"{metric}_mean"]
    best = scores.max()
    return next(m for m in MODEL_ORDER if m in scores.index and scores[m] >= best - tie_tolerance)


def run(cfg: dict | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    paths = cfg["paths"]
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    # 1. data
    X, y = load_dataset(cfg["data"]["raw_path"])
    X_tr, X_te, y_tr, y_te = split(X, y, cfg["data"]["test_size"], cfg["data"]["random_state"])
    say(f"train {len(y_tr)} rows ({y_tr.mean():.1%} malignant) | test {len(y_te)} rows")

    # 2. cross-validation
    models = build_models(cfg)
    folds, oof = cross_validate_models(
        models,
        X_tr,
        y_tr,
        cfg["cv"]["n_splits"],
        cfg["cv"]["n_repeats"],
        cfg["data"]["random_state"],
    )
    cv_summary = summarise_cv(folds)
    say(
        "\nCross-validation (mean over "
        f"{cfg['cv']['n_splits']}x{cfg['cv']['n_repeats']} folds, threshold 0.5):"
    )
    say(
        cv_summary[["roc_auc_mean", "roc_auc_std", "recall_mean", "precision_mean"]]
        .round(4)
        .to_string()
    )

    # 3. model choice
    chosen = select_model(
        cv_summary,
        cfg["model_selection"]["metric"],
        cfg["model_selection"]["deploy"],
        cfg["model_selection"].get("tie_tolerance", 0.0),
    )
    say(f"\nDeployed model: {chosen} ({DISPLAY_NAMES[chosen]})")

    # 4. threshold from out-of-fold probabilities (no test data used)
    target = cfg["threshold"]["target_recall"]
    threshold = choose_threshold(y_tr, oof[chosen], target)
    say(f"Threshold for recall >= {target}: {threshold:.4f}")

    # 5. refit on all training data
    final = models[chosen].fit(X_tr.values, y_tr.values)

    # 6. one-time test evaluation
    p_te = final.predict_proba(X_te.values)[:, 1]
    test_default = classification_metrics(y_te, p_te, 0.5)
    test_tuned = classification_metrics(y_te, p_te, threshold)
    say(
        f"\nTest @0.5      : recall {test_default['recall']:.3f}  precision "
        f"{test_default['precision']:.3f}  ROC-AUC {test_default['roc_auc']:.4f}  "
        f"FN {test_default['fn']}  FP {test_default['fp']}"
    )
    say(
        f"Test @{threshold:.3f}  : recall {test_tuned['recall']:.3f}  precision "
        f"{test_tuned['precision']:.3f}  FN {test_tuned['fn']}  FP {test_tuned['fp']}"
    )

    # Also score every candidate on test, for the report only (not for choosing).
    test_all = {}
    for name in MODEL_ORDER:
        m = final if name == chosen else models[name].fit(X_tr.values, y_tr.values)
        test_all[name] = classification_metrics(y_te, m.predict_proba(X_te.values)[:, 1])

    # 7. artifacts
    figs = paths["figures_dir"]
    plot_roc_pr(y_tr.values, oof, DISPLAY_NAMES, figs / "cv_roc_pr.png")
    plot_threshold_tradeoff(
        y_tr.values, oof[chosen], threshold, target, figs / "threshold_tradeoff.png"
    )
    plot_confusion(
        y_te.values,
        p_te,
        threshold,
        f"Test set, threshold {threshold:.2f}",
        figs / "test_confusion.png",
    )
    scratch = models["logreg_scratch"].fit(X_tr.values, y_tr.values)
    plot_cost_history(scratch.named_steps["clf"].cost_history_, figs / "gd_cost_history.png")
    gd = scratch.named_steps["clf"]

    metadata = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": chosen,
        "model_display_name": DISPLAY_NAMES[chosen],
        "threshold": threshold,
        "target_recall": target,
        "feature_names": FEATURE_NAMES,
        "positive_class": "malignant",
        "train_rows": len(y_tr),
        "train_feature_min": X_tr.min().round(6).to_dict(),
        "train_feature_max": X_tr.max().round(6).to_dict(),
        "cv_summary": cv_summary.round(5).to_dict(orient="index"),
        "test_metrics_default_threshold": test_default,
        "test_metrics_tuned_threshold": test_tuned,
        "scratch_gd": {
            "iterations": gd.n_iter_,
            "converged": bool(gd.converged_),
            "final_cost": gd.cost_history_[-1],
        },
    }
    joblib.dump(final, paths["models_dir"] / "model.joblib")
    (paths["models_dir"] / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    folds.to_csv(paths["reports_dir"] / "cv_fold_scores.csv", index=False)
    write_results_md(cfg, cv_summary, chosen, threshold, test_default, test_tuned, test_all, gd)

    run_id = log_run(
        params={
            "model": chosen,
            "threshold": threshold,
            **cfg["scratch_logreg"],
            **{f"cv_{k}": v for k, v in cfg["cv"].items()},
        },
        metrics={
            "cv": {k: v for k, v in cv_summary.loc[chosen].items()},
            "test_tuned": test_tuned,
            "test_default": test_default,
        },
        artifacts=[paths["models_dir"] / "metadata.json", figs / "cv_roc_pr.png"],
        runs_dir=paths["reports_dir"] / "runs",
    )
    say(f"\nSaved models/model.joblib, reports/results.md  |  run logged: {run_id}")
    return metadata


def write_results_md(cfg, cv_summary, chosen, threshold, test_default, test_tuned, test_all, gd):
    def row(name):
        s = cv_summary.loc[name]
        return (
            f"| {DISPLAY_NAMES[name]} | {s.roc_auc_mean:.4f} ± {s.roc_auc_std:.4f} | "
            f"{s.pr_auc_mean:.4f} | {s.recall_mean:.3f} | {s.precision_mean:.3f} | "
            f"{s.accuracy_mean:.3f} |"
        )

    def trow(name):
        t = test_all[name]
        return (
            f"| {DISPLAY_NAMES[name]} | {t['roc_auc']:.4f} | {t['recall']:.3f} | "
            f"{t['precision']:.3f} | {t['accuracy']:.3f} | {t['fn']} | {t['fp']} |"
        )

    k, r = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    lines = [
        "# Results (generated by `python -m wdbc.train`)",
        "",
        f"## Cross-validation on the training set ({k}-fold x {r} repeats, threshold 0.5)",
        "",
        "| Model | ROC-AUC (mean ± sd) | PR-AUC | Recall | Precision | Accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
        *[row(n) for n in MODEL_ORDER],
        "",
        (
            f"**Deployed:** {DISPLAY_NAMES[chosen]}. Rule: highest mean CV ROC-AUC; models within "
            f"{cfg['model_selection'].get('tie_tolerance', 0.0)} of the best count as tied, "
            "and the simplest tied model is chosen."
        ),
        "",
        (
            f"**Decision threshold:** {threshold:.4f}, the highest threshold whose out-of-fold "
            f"recall is at least {cfg['threshold']['target_recall']}."
        ),
        "",
        "## Held-out test set (114 samples, used once)",
        "",
        (
            "| Threshold | Recall | Precision | Specificity | F1 | ROC-AUC "
            "| Missed cancers (FN) | False alarms (FP) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {t['threshold']:.3f} | {t['recall']:.3f} | {t['precision']:.3f} | "
            f"{t['specificity']:.3f} | {t['f1']:.3f} | {t['roc_auc']:.4f} | {t['fn']} | {t['fp']} |"
            for t in (test_default, test_tuned)
        ],
        "",
        "### Every candidate on the test set (threshold 0.5, for reference only)",
        "",
        "The deployed model was chosen from cross-validation, not from this table.",
        "",
        "| Model | ROC-AUC | Recall | Precision | Accuracy | FN | FP |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[trow(n) for n in MODEL_ORDER],
        "",
        "## From-scratch gradient descent",
        "",
        (
            f"- Iterations: {gd.n_iter_} (converged: {gd.converged_}, "
            f"tol = {cfg['scratch_logreg']['tol']})"
        ),
        f"- Final cost J = {gd.cost_history_[-1]:.5f}",
        "",
        "![ROC and PR curves](figures/cv_roc_pr.png)",
        "![Threshold trade-off](figures/threshold_tradeoff.png)",
        "![Test confusion matrix](figures/test_confusion.png)",
        "![GD convergence](figures/gd_cost_history.png)",
    ]
    (cfg["paths"]["reports_dir"] / "results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
