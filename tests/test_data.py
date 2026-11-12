import pandas as pd
import pytest

from wdbc.data import (
    COLUMNS,
    FEATURE_NAMES,
    DataValidationError,
    load_dataset,
    load_raw,
    validate,
)


def test_shape_and_labels(cfg):
    X, y = load_dataset(cfg["data"]["raw_path"])
    assert X.shape == (569, 30)
    assert list(X.columns) == FEATURE_NAMES
    assert y.value_counts().to_dict() == {0: 357, 1: 212}
    assert "id" not in X.columns  # IDs must never be used as features


def test_split_is_stratified(dataset):
    X_tr, X_te, y_tr, y_te = dataset
    assert len(X_tr) == 455 and len(X_te) == 114
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert set(X_tr.index).isdisjoint(X_te.index)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda d: d.assign(diagnosis=d["diagnosis"].replace("M", "X")), "unknown labels"),
        (lambda d: d.assign(radius_mean=-d["radius_mean"]), "negative"),
        (lambda d: d.assign(texture_mean=d["texture_mean"].where(d.index > 0)), "missing"),
        (lambda d: d.iloc[:500], "expected 569 rows"),
        (lambda d: pd.concat([d.iloc[:568], d.iloc[:1]]), "duplicate"),
    ],
)
def test_validation_catches_bad_data(cfg, corrupt, message):
    df = load_raw(cfg["data"]["raw_path"])
    validate(df)  # the real file passes
    with pytest.raises(DataValidationError, match=message):
        validate(corrupt(df.copy()))


def test_columns_constant():
    assert len(COLUMNS) == 32
