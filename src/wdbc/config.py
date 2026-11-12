"""Load the YAML config and resolve paths relative to the project root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/wdbc/config.py -> project root is two folders up. WDBC_ROOT overrides it
# (useful if the package is installed somewhere else).
PROJECT_ROOT = Path(os.environ.get("WDBC_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the config file and turn every entry under `paths` and `data.raw_path`
    into an absolute Path, so scripts work from any working directory."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["raw_path"] = PROJECT_ROOT / cfg["data"]["raw_path"]
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    return cfg
