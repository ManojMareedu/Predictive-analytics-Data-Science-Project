"""FastAPI smoke tests.

Skipped rather than failed when `exported_model/` is absent: a checkout without a
trained model is a valid state (run the pipeline first), and a red test there
would say "the API is broken" when it means "there is no model yet".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not Path("exported_model/MLmodel").exists(),
    reason="exported_model/ not built yet; run `python pipeline.py` first",
)

VALID = {
    "price_per_unit_no_merch": 3.49,
    "price_per_unit_any_merch": 2.99,
    "price_per_volume_no_merch": 3.49,
    "price_per_volume_any_merch": 2.99,
    "acv_distribution_no_merch": 65.0,
    "acv_distribution_any_merch": 40.0,
    "year": 2022,
    "week": 26,
    "brand": "BLUEBONNET",
    "geography": "Great Lakes - Multi Outlet + Conv",
}


@pytest.fixture(scope="module")
def client():
    from app.model_server import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "Unit Sales No Merch"
    assert body["random_seed"] == 42
    # The metrics block must carry CV alongside the holdout, not a lone number.
    metrics = body["metrics"]
    assert metrics, "no metrics recorded"
    any_model = next(iter(metrics.values()))
    assert {"r2", "cv_r2_mean", "cv_r2_std"} <= set(any_model)


def test_predict_returns_sane_value(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    # Non-negative is a guarantee the endpoint makes (it clamps); the upper bound
    # is a loose sanity check that we are in units, not dollars or nonsense.
    assert 0 <= body["predicted_unit_sales"] < 5_000_000
    assert body["model_name"]


def test_predict_rejects_invalid_input(client):
    for bad in ({**VALID, "week": 99}, {**VALID, "price_per_unit_no_merch": -1}, {**VALID, "brand": ""}):
        assert client.post("/predict", json=bad).status_code == 422
    incomplete = {k: v for k, v in VALID.items() if k != "brand"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_predict_accepts_unseen_brand(client):
    """Unseen categories must bucket, not crash -- serving sees new brands."""
    r = client.post("/predict", json={**VALID, "brand": "NEVERSEENBRAND"})
    assert r.status_code == 200
