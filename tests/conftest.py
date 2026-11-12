import copy

import pytest

from wdbc.config import load_config
from wdbc.data import load_dataset, split


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset(cfg):
    X, y = load_dataset(cfg["data"]["raw_path"])
    return split(X, y, cfg["data"]["test_size"], cfg["data"]["random_state"])


@pytest.fixture(scope="session")
def trained_dir(cfg, tmp_path_factory):
    """Run the full training pipeline once, quickly, into a temp folder."""
    from wdbc.train import run

    fast = copy.deepcopy(cfg)
    fast["cv"] = {"n_splits": 3, "n_repeats": 1}
    root = tmp_path_factory.mktemp("run")
    fast["paths"] = {
        "models_dir": root / "models",
        "reports_dir": root / "reports",
        "figures_dir": root / "reports" / "figures",
    }
    run(fast, verbose=False)
    return root
