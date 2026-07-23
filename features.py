"""Feature schema + preprocessing for the tablespreads unit-sales model.

The central decision here is *which columns are legitimate predictors vs. leakage*.
The original notebook's ~81% R2 leans on `Base Volume Sales` (and friends), which
is a mechanical component of the target -- see LEAKAGE_COLS below. We exclude all
of those and predict from price, distribution, brand, region and calendar signals
only. See MODEL_CARD.md for the honest before/after numbers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = "Unit Sales No Merch"

# Exogenous drivers known independently of the no-merch unit outcome:
#   - price signals (elasticity)         - distribution breadth (ACV)
#   - brand / region                     - calendar (year trend, week seasonality)
NUMERIC_FEATURES = [
    "Price per Unit No Merch",
    "Price per Unit Any Merch",
    "Price per Volume No Merch",
    "Price per Volume Any Merch",
    "ACV Weighted Distribution No Merch",
    "ACV Weighted Distribution Any Merch",
    "Year",
    "Week",
]
CATEGORICAL_FEATURES = ["Brand", "Geography"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that are the target measured a different way, or a mechanical component
# of it (Unit Sales No Merch = Base Unit Sales + Incremental Units; Volume == Units
# in this dataset; Dollar Sales = Units x Price). Feeding any of these back in is
# data leakage, not prediction.
LEAKAGE_COLS = [
    "Dollar Sales No Merch",
    "Dollar Sales Any Merch",
    "Unit Sales Any Merch",
    "Volume Sales No Merch",
    "Volume Sales Any Merch",
    "Base Unit Sales",
    "Base Volume Sales",
    "Base Dollar Sales",
    "Incremental Units",
    "Incremental Volume",
    "Incremental Dollars",
    "Price per Unit",
    "Price per Volume",  # blended-across-merch, redundant
]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into (X with FEATURE_COLS, y = target)."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()
    return X, y


def vif_report(df: pd.DataFrame, cols: list[str] | None = None, sample: int = 50_000) -> pd.DataFrame:
    """VIF for the numeric feature block, highest first.

    VIF > 10 is the usual red flag: that feature is ~90%+ explained by the others,
    so its individual coefficient is not interpretable even if the model predicts
    well. Sampled because VIF is an OLS-per-column loop and the full 270k rows buy
    no extra precision on a collinearity diagnostic.
    """
    cols = cols or NUMERIC_FEATURES
    X = df[cols].dropna()
    if len(X) > sample:
        X = X.sample(sample, random_state=42)
    # statsmodels' VIF needs an explicit intercept or every VIF is inflated by the
    # uncentered mean.
    X = X.assign(_const=1.0).astype("float64")
    vals = [variance_inflation_factor(X.values, i) for i in range(len(cols))]
    return (
        pd.DataFrame({"feature": cols, "vif": vals})
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def make_preprocessor() -> ColumnTransformer:
    """One-hot brand/region (ignore unseen at serve time), scale numerics.

    handle_unknown='ignore' means a region or brand not seen in training encodes
    to all-zeros instead of crashing the API -- the right behavior for serving.
    """
    # min_frequency buckets the long tail of ~370 rare 2-token "brands" into one
    # 'infrequent' column, so we get ~20 real brand columns + a catch-all instead
    # of a 374-wide matrix that OOMs polynomial models. Unseen brands at serve
    # time also route to that bucket rather than crashing.
    cat = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=2000,
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", cat, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
