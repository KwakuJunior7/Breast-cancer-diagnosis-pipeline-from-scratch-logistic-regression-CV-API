"""REST API for the trained model.

Run locally:   uvicorn wdbc.api:app --reload      (after `python -m wdbc.train`)
Interactive docs (Swagger UI):  http://127.0.0.1:8000/docs

Endpoints
  GET  /health         liveness check
  GET  /model-info     which model, threshold, and its test metrics
  POST /predict        one sample  -> probability + label
  POST /predict/batch  list of samples (max 1000)

The request body is validated twice: by Pydantic (types, all 30 fields
present, values >= 0) and by Predictor._check() (finite numbers).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from . import __version__
from .data import FEATURE_NAMES
from .predict import InputError, Predictor

# Build the 30-field request model from the feature list, so the API schema can
# never drift from the training features.
Sample = create_model(
    "Sample",
    __config__=ConfigDict(extra="forbid"),
    **{f: (float, Field(..., ge=0)) for f in FEATURE_NAMES},
)


class Batch(BaseModel):
    samples: list[Sample] = Field(..., min_length=1, max_length=1000)  # type: ignore[valid-type]


class Prediction(BaseModel):
    probability_malignant: float
    prediction: str
    threshold: float
    out_of_range_features: list[str]
    model: str


app = FastAPI(
    title="WDBC breast-cancer diagnosis API",
    version=__version__,
    description="Portfolio demo. Not a medical device; not for clinical use.",
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Opening the bare address sends you to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    m = get_predictor().meta
    return {
        "model": m["model_display_name"],
        "threshold": m["threshold"],
        "target_recall": m["target_recall"],
        "trained_at": m["trained_at"],
        "test_metrics": m["test_metrics_tuned_threshold"],
    }


@app.post("/predict", response_model=Prediction)
def predict(sample: Sample) -> dict:  # type: ignore[valid-type]
    try:
        return get_predictor().predict_one(sample.model_dump())
    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch) -> list[dict]:
    try:
        return get_predictor().predict_many([s.model_dump() for s in batch.samples])
    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
