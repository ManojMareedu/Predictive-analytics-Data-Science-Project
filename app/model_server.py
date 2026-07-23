"""FastAPI service for the tablespreads unit-sales model.

The exported artifact is a complete sklearn Pipeline (preprocessing + estimator),
so this module does no feature engineering of its own -- it validates the request,
builds a one-row DataFrame with the raw column names the pipeline expects, and
hands it over. That is deliberate: any transformation done here and not in
training is training/serving skew waiting to happen.

Run:  uvicorn app.model_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "exported_model"))

app = FastAPI(
    title="Tablespreads Unit Sales API",
    description="Predicts non-promoted unit sales at the region x brand x week grain.",
    version="1.0.0",
)

# Loaded once at import. Kept module-level and nullable rather than failing hard,
# so /health can report "model missing" instead of the container crash-looping
# with no way to ask it what is wrong.
_model: Any = None
_metadata: dict = {}
_load_error: str | None = None

try:
    _model = mlflow.sklearn.load_model(str(MODEL_DIR))
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        _metadata = json.loads(meta_path.read_text())
except Exception as exc:  # pragma: no cover - exercised only when the export is absent
    _load_error = f"{type(exc).__name__}: {exc}"


class PredictionRequest(BaseModel):
    """One region x brand x week row.

    JSON uses snake_case; the model's columns have spaces in them. The mapping is
    explicit in `to_frame()` rather than leaking IRI's column naming into the API.
    """

    price_per_unit_no_merch: float = Field(..., ge=0, description="Avg price/unit, non-promoted")
    price_per_unit_any_merch: float = Field(..., ge=0, description="Avg price/unit, promoted")
    price_per_volume_no_merch: float = Field(..., ge=0, description="Avg price/volume, non-promoted")
    price_per_volume_any_merch: float = Field(..., ge=0, description="Avg price/volume, promoted")
    acv_distribution_no_merch: float = Field(
        ..., ge=0, le=100, description="ACV-weighted distribution %, non-promoted"
    )
    acv_distribution_any_merch: float = Field(
        ..., ge=0, le=100, description="ACV-weighted distribution %, promoted"
    )
    year: int = Field(..., ge=2018, le=2035)
    week: int = Field(..., ge=1, le=53)
    brand: str = Field(..., min_length=1, description="e.g. BLUEBONNET. Unseen brands are accepted.")
    geography: str = Field(..., min_length=1, description="e.g. 'Great Lakes - Multi Outlet + Conv'")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Price per Unit No Merch": self.price_per_unit_no_merch,
                    "Price per Unit Any Merch": self.price_per_unit_any_merch,
                    "Price per Volume No Merch": self.price_per_volume_no_merch,
                    "Price per Volume Any Merch": self.price_per_volume_any_merch,
                    "ACV Weighted Distribution No Merch": self.acv_distribution_no_merch,
                    "ACV Weighted Distribution Any Merch": self.acv_distribution_any_merch,
                    "Year": self.year,
                    "Week": self.week,
                    "Brand": self.brand,
                    "Geography": self.geography,
                }
            ]
        )


class PredictionResponse(BaseModel):
    predicted_unit_sales: float
    model_name: str
    note: str


@app.get("/health")
def health() -> dict:
    """Liveness + readiness. Reports *why* it is unhealthy, not just that it is."""
    ok = _model is not None
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "model_dir": str(MODEL_DIR),
        "error": _load_error,
    }


@app.get("/model-info")
def model_info() -> dict:
    """What is deployed, how it scored, and what it should not be used for."""
    if _model is None:
        raise HTTPException(503, detail=f"Model not loaded: {_load_error}")
    return {
        "best_model": _metadata.get("best_model"),
        "target": _metadata.get("target"),
        "feature_cols": _metadata.get("feature_cols"),
        "test_year": _metadata.get("test_year"),
        "random_seed": _metadata.get("random_seed"),
        "cv_folds": _metadata.get("cv_folds"),
        "n_train": _metadata.get("n_train"),
        "n_test": _metadata.get("n_test"),
        "metrics": _metadata.get("metrics"),
        "grain": "region x brand x week, aggregated retail POS (not household-level)",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(503, detail=f"Model not loaded: {_load_error}")
    try:
        pred = float(_model.predict(req.to_frame())[0])
    except Exception as exc:
        raise HTTPException(400, detail=f"Prediction failed: {type(exc).__name__}: {exc}") from exc

    # Unit sales cannot be negative. A linear model extrapolating outside the
    # training range can produce one, so clamp -- and say so in the response
    # rather than silently returning a different number than the model gave.
    note = "ok"
    if pred < 0:
        note = f"model returned {pred:.1f}; clamped to 0 (unit sales cannot be negative)"
        pred = 0.0

    return PredictionResponse(
        predicted_unit_sales=round(pred, 2),
        model_name=str(_metadata.get("best_model", "unknown")),
        note=note,
    )
