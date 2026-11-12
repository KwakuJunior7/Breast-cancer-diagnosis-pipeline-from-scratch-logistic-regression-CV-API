"""API tests. Skipped automatically if FastAPI is not installed."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # needed by TestClient

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_dir, monkeypatch_module):
    monkeypatch_module.setenv("WDBC_MODELS_DIR", str(trained_dir / "models"))
    from wdbc import api

    api.get_predictor.cache_clear()
    return TestClient(api.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def sample(dataset):
    return dataset[1].iloc[0].to_dict()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert 0 < body["threshold"] < 1


def test_predict(client, sample):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json()["prediction"] in {"benign", "malignant"}


def test_predict_batch(client, sample):
    r = client.post("/predict/batch", json={"samples": [sample, sample]})
    assert r.status_code == 200 and len(r.json()) == 2


def test_rejects_negative_and_missing(client, sample):
    assert client.post("/predict", json={**sample, "area_mean": -5}).status_code == 422
    partial = {k: v for k, v in sample.items() if k != "area_mean"}
    assert client.post("/predict", json=partial).status_code == 422
    assert client.post("/predict", json={**sample, "extra": 1}).status_code == 422
