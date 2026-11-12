"""Experiment tracking.

If MLflow is installed (`pip install mlflow`), every training run is logged to
./mlruns and can be browsed with `mlflow ui`. If it is not installed, the same
information is written to reports/runs/<timestamp>.json, so a run is always
recorded either way.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def log_run(
    params: dict, metrics: dict, artifacts: list[Path], runs_dir: Path, run_name: str = "train"
) -> str:
    params = {k: str(v) for k, v in _flatten(params).items()}
    metrics = {k: float(v) for k, v in _flatten(metrics).items() if isinstance(v, (int, float))}
    try:
        import mlflow  # optional dependency
    except ImportError:
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = runs_dir / f"{stamp}_{run_name}.json"
        path.write_text(
            json.dumps(
                {"params": params, "metrics": metrics, "artifacts": [str(a) for a in artifacts]},
                indent=2,
            ),
            encoding="utf-8",
        )
        return f"json:{path}"

    mlflow.set_experiment("wdbc-diagnosis")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for a in artifacts:
            if Path(a).exists():
                mlflow.log_artifact(str(a))
        return f"mlflow:{run.info.run_id}"
