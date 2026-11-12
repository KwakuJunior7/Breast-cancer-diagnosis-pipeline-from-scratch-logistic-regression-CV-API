"""End-to-end: train into a temp folder, then load and predict."""

import json

import pytest

from wdbc.predict import InputError, Predictor


@pytest.fixture(scope="module")
def predictor(trained_dir):
    return Predictor(trained_dir / "models")


@pytest.fixture(scope="module")
def samples(dataset):
    X_te, y_te = dataset[1], dataset[3]
    benign = X_te[y_te == 0].iloc[0].to_dict()
    malignant = X_te[y_te == 1].iloc[0].to_dict()
    return benign, malignant


def test_artifacts_written(trained_dir):
    assert (trained_dir / "models" / "model.joblib").exists()
    meta = json.loads((trained_dir / "models" / "metadata.json").read_text())
    assert 0 < meta["threshold"] < 1
    assert meta["test_metrics_default_threshold"]["roc_auc"] > 0.97
    assert (trained_dir / "reports" / "results.md").exists()
    for fig in [
        "cv_roc_pr.png",
        "threshold_tradeoff.png",
        "test_confusion.png",
        "gd_cost_history.png",
    ]:
        assert (trained_dir / "reports" / "figures" / fig).exists()


def test_predicts_known_cases(predictor, samples):
    benign, malignant = samples
    rb, rm = predictor.predict_one(benign), predictor.predict_one(malignant)
    assert rb["probability_malignant"] < rm["probability_malignant"]
    assert set(rb) == {
        "probability_malignant",
        "prediction",
        "threshold",
        "out_of_range_features",
        "model",
    }


def test_rejects_missing_and_extra_features(predictor, samples):
    s = dict(samples[0])
    s.pop("radius_mean")
    with pytest.raises(InputError, match="missing"):
        predictor.predict_one(s)
    with pytest.raises(InputError, match="unexpected"):
        predictor.predict_one({**samples[0], "patient_name": 1.0})


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "12", True])
def test_rejects_bad_values(predictor, samples, bad):
    with pytest.raises(InputError):
        predictor.predict_one({**samples[0], "area_mean": bad})


def test_flags_values_outside_training_range(predictor, samples):
    r = predictor.predict_one({**samples[0], "area_mean": 1e6})
    assert r["out_of_range_features"] == ["area_mean"]
