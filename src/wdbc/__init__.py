"""Breast-cancer (WDBC) diagnosis pipeline.

Modules
-------
config            load configs/config.yaml
data              load + validate the UCI WDBC file, train/test split
preprocessing     z-score scaler written from scratch (mirrors the MLA Lecture 8 Octave code)
logistic_scratch  logistic regression trained by gradient descent in NumPy
models            the candidate models compared in cross-validation
evaluate          metrics, cross-validation, threshold selection, plots
eda               exploratory data analysis figures and summary
tracking          optional MLflow logging (falls back to JSON files)
train             end-to-end training entry point  (python -m wdbc.train)
predict           load the saved model and score new samples
api               FastAPI service                  (uvicorn wdbc.api:app)
"""

__version__ = "1.0.0"
