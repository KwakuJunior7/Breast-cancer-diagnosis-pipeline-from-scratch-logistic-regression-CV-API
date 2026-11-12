"""Load, validate and split the Wisconsin Diagnostic Breast Cancer (WDBC) data.

File layout (see data/raw/wdbc.names):
    column 1  : patient ID            -> dropped (an ID is not a feature)
    column 2  : diagnosis, M or B     -> label y  (M = malignant = 1, B = benign = 0)
    column 3-32: 30 real-valued features

The 30 features are 10 cell-nucleus measurements, each summarised three ways
over the cells in the image: mean, standard error (se) and "worst" (mean of
the three largest values).

Note vs. the Octave EDA script: that script kept column 1 (the ID) inside X.
IDs are arbitrary numbers, so a model could "learn" from them and leak
information about how the data was collected. Here the ID is dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
STATS = ["mean", "se", "worst"]
# Order matches the file: all 10 means, then all 10 SEs, then all 10 worsts.
FEATURE_NAMES = [f"{base}_{stat}" for stat in STATS for base in BASE_FEATURES]
COLUMNS = ["id", "diagnosis", *FEATURE_NAMES]

LABEL_MAP = {"B": 0, "M": 1}
CLASS_NAMES = {0: "benign", 1: "malignant"}

EXPECTED_ROWS = 569
EXPECTED_COUNTS = {"B": 357, "M": 212}


class DataValidationError(ValueError):
    """Raised when the raw file does not look like the WDBC dataset."""


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw comma-separated file (it has no header row)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy wdbc.data into data/raw/ "
            "(UCI ML Repository, 'Breast Cancer Wisconsin (Diagnostic)')."
        )
    return pd.read_csv(path, header=None, names=COLUMNS)


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the data is not what the rest of the pipeline expects.

    Cheap checks like these catch the most common real-world problems:
    a truncated download, a file with a header, a wrong delimiter, bad labels.
    """
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append("unexpected columns")
    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, found {len(df)}")
    if df["id"].duplicated().any():
        problems.append("duplicate patient IDs")
    bad_labels = set(df["diagnosis"].unique()) - set(LABEL_MAP)
    if bad_labels:
        problems.append(f"unknown labels {sorted(bad_labels)}")
    feats = df[FEATURE_NAMES]
    if not all(np.issubdtype(t, np.number) for t in feats.dtypes):
        problems.append("non-numeric feature values")
    elif feats.isna().any().any():
        problems.append("missing feature values")
    elif (feats < 0).any().any():
        problems.append("negative measurements (all features are sizes, ratios or counts)")
    if problems:
        raise DataValidationError("; ".join(problems))


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (569 x 30 DataFrame) and y (569 Series of 0/1)."""
    df = load_raw(path)
    validate(df)
    X = df[FEATURE_NAMES].astype(float)
    y = df["diagnosis"].map(LABEL_MAP).astype(int).rename("malignant")
    return X, y


def split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    """Stratified hold-out split.

    `stratify=y` keeps the 63% benign / 37% malignant ratio the same in train
    and test. Without it, a small test set can end up with too few malignant
    cases to measure recall reliably.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
