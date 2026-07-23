"""Candidate models for predicting Unit Sales No Merch.

Every model is a full sklearn Pipeline (preprocessing + estimator) so it can be
logged to MLflow and served as a self-contained artifact -- the FastAPI/Streamlit
side passes raw feature rows and the pipeline does its own encoding/scaling.

All models are laptop-CPU sized: no GPU, modest tree depth/iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_preprocessor,
)

# Single seed for every stochastic component (model init, CV fold shuffling) so a
# rerun reproduces these numbers exactly. Logged to MLflow as a param per run.
SEED = 42
CV_FOLDS = 5


def _tree_preprocessor() -> ColumnTransformer:
    """Trees don't need scaling; just one-hot the categoricals, pass numerics."""
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2000, sparse_output=False)
    return ColumnTransformer(
        [("cat", cat, CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )


def _poly_preprocessor() -> ColumnTransformer:
    """Degree-2 interactions on the *numeric* block only (price/dist/calendar);
    squaring one-hot brand dummies is meaningless and blows the matrix to GBs."""
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2000, sparse_output=False)
    num_poly = Pipeline(
        [("scale", StandardScaler()), ("poly", PolynomialFeatures(degree=2, include_bias=False))]
    )
    return ColumnTransformer(
        [("num", num_poly, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def candidate_models() -> dict[str, Pipeline]:
    """Name -> untrained Pipeline. Linear family shares the scaled preprocessor."""
    return {
        "ridge": Pipeline([("pre", make_preprocessor()), ("m", Ridge(alpha=1.0, random_state=SEED))]),
        "lasso": Pipeline(
            [("pre", make_preprocessor()), ("m", Lasso(alpha=1.0, max_iter=5000, random_state=SEED))]
        ),
        "elasticnet": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("m", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=SEED)),
            ]
        ),
        # Polynomial interactions on the numeric block only (degree 2).
        "polynomial": Pipeline([("pre", _poly_preprocessor()), ("m", Ridge(alpha=10.0, random_state=SEED))]),
        # Gradient-boosted trees: best honest performer, handles the heavy tail.
        "hist_gbr": Pipeline(
            [
                ("pre", _tree_preprocessor()),
                (
                    "m",
                    HistGradientBoostingRegressor(
                        max_iter=300, max_depth=8, learning_rate=0.1, random_state=SEED
                    ),
                ),
            ]
        ),
    }


def cross_validate_model(pipe: Pipeline, X, y, folds: int = CV_FOLDS) -> dict[str, float]:
    """K-fold CV on the training years -> mean/std of R2, RMSE, MAE.

    This answers "is the model stable across resamples", which a single holdout
    cannot. It is deliberately *not* the headline number: shuffled folds mix weeks
    from adjacent years, so CV reads optimistically next to the 2022 temporal
    holdout. Report both -- the gap between them is itself the finding.
    """
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
    out = {}
    for key in scoring:
        # neg_* scorers come back negated; flip them so bigger-is-worse reads right.
        vals = res[f"test_{key}"] if key == "r2" else -res[f"test_{key}"]
        out[f"cv_{key}_mean"] = float(np.mean(vals))
        out[f"cv_{key}_std"] = float(np.std(vals))
    return out


@dataclass
class Metrics:
    r2: float
    rmse: float
    mae: float

    def as_dict(self) -> dict[str, float]:
        return {"r2": self.r2, "rmse": self.rmse, "mae": self.mae}


def evaluate(y_true, y_pred) -> Metrics:
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )
