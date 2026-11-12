"""Load the trained model and score new samples.

The Predictor is used by the API and the CLI. It guards against the two most
common production mistakes:
  1. wrong input  - missing, extra, non-numeric, negative or infinite features
                    are rejected with a clear message;
  2. unusual input - values outside the range seen in training are allowed but
                    flagged, because a model's output is least reliable there.

CLI:  python -m wdbc.predict path/to/samples.csv   (columns = the 30 feature names)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config


class InputError(ValueError):
    pass


class Predictor:
    def __init__(self, models_dir: str | Path | None = None):
        # Priority: argument > WDBC_MODELS_DIR environment variable > config file.
        models_dir = models_dir or os.environ.get("WDBC_MODELS_DIR")
        models_dir = Path(models_dir) if models_dir else load_config()["paths"]["models_dir"]
        model_path = models_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing; run `python -m wdbc.train` first.")
        self.model = joblib.load(model_path)
        self.meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        self.features: list[str] = self.meta["feature_names"]
        self.threshold: float = float(self.meta["threshold"])
        self._lo = self.meta["train_feature_min"]
        self._hi = self.meta["train_feature_max"]

    def _check(self, sample: dict) -> np.ndarray:
        missing = [f for f in self.features if f not in sample]
        extra = [k for k in sample if k not in self.features]
        if missing or extra:
            raise InputError(f"missing features: {missing}; unexpected features: {extra}")
        row = []
        for f in self.features:
            v = sample[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InputError(f"{f} must be a number")
            if not math.isfinite(v) or v < 0:
                raise InputError(f"{f} must be a finite, non-negative number")
            row.append(float(v))
        return np.array(row)

    def predict_one(self, sample: dict) -> dict:
        x = self._check(sample)
        p = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        out_of_range = [
            f for f, v in zip(self.features, x, strict=True) if v < self._lo[f] or v > self._hi[f]
        ]
        return {
            "probability_malignant": round(p, 6),
            "prediction": "malignant" if p >= self.threshold else "benign",
            "threshold": self.threshold,
            "out_of_range_features": out_of_range,
            "model": self.meta["model_name"],
        }

    def predict_many(self, samples: list[dict]) -> list[dict]:
        return [self.predict_one(s) for s in samples]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m wdbc.predict samples.csv")
        return 2
    df = pd.read_csv(argv[0])
    pred = Predictor()
    results = pd.DataFrame(pred.predict_many(df[pred.features].to_dict(orient="records")))
    if "true_label" in df.columns:  # handy when scoring labelled examples
        results.insert(0, "true_label", df["true_label"].values)
    print(results.drop(columns=["model"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
