"""Tests for the parts that would silently corrupt results if they broke.

The bar for including a test here is "if this regressed, would a wrong number
reach MODEL_CARD.md without anyone noticing?" -- schema drift handling, the
leakage exclusion, and the temporal split all clear it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data_ingestion as di
import features as ft
from EDA import year_ttests
from models import SEED, candidate_models


def _raw_frame(product_col: str = "Product Description") -> pd.DataFrame:
    """A minimal frame with the real 24-column layout, one row per case we care about."""
    rows = []
    for geo in ("Great Lakes - Multi Outlet + Conv", di.TOTAL_US):
        for time in ("Week Ending 01-14-18", "Week Ending 01-01-23"):
            rows.append(
                {
                    "Geography": geo,
                    "Time": time,
                    product_col: "BLUE BONNET SPREAD 16 OZ",
                    "UPC 13 digit": "0001234567890",
                    **{c: 1.0 for c in di.NUMERIC_COLS},
                }
            )
    df = pd.DataFrame(rows)
    df.loc[0, "Unit Sales No Merch"] = np.nan  # the ~3% null-target case
    return df


def _write_xlsx(df: pd.DataFrame, path) -> str:
    df.to_excel(path, sheet_name=di.SHEET, index=False)
    return str(path)


def test_brand_extraction():
    assert di.brand("BLUE BONNET SPREAD 16 OZ") == "BLUEBONNET"
    assert di.brand("SINGLEWORD") == "SINGLEWORD"
    assert di.brand("") == ""


@pytest.mark.parametrize("product_col", ["Product Description", "Product"])
def test_load_year_handles_2022_schema_rename(tmp_path, product_col):
    """2022 calls the column `Product`; every other year `Product Description`.

    Both must produce the same standardised frame -- this is the one real schema
    drift across the five files, so it gets a test rather than a comment.
    """
    path = _write_xlsx(_raw_frame(product_col), tmp_path / "year.xlsx")
    out = di.load_year(path)
    assert "Product Description" in out.columns
    assert "Product" not in out.columns
    assert out["Brand"].iloc[0] == "BLUEBONNET"


def test_load_year_drops_total_us_rollup(tmp_path):
    """Keeping the national roll-up double-counts every sale."""
    path = _write_xlsx(_raw_frame(), tmp_path / "year.xlsx")
    out = di.load_year(path)
    assert di.TOTAL_US not in set(out["Geography"].astype(str))
    assert len(out) == 2  # 4 rows in, 2 of them Total US


def test_load_year_fills_nulls_and_derives_calendar(tmp_path):
    path = _write_xlsx(_raw_frame(), tmp_path / "year.xlsx")
    out = di.load_year(path)
    assert out[di.NUMERIC_COLS].isna().sum().sum() == 0
    # Week ending 01-01-23 parses into calendar year 2023, not 2022 -- the reason
    # the holdout is `Year >= 2022` and not `Year == 2022`.
    assert set(out["Year"]) == {2018, 2023}
    assert out["Week"].between(1, 53).all()


def test_no_leakage_column_is_a_feature():
    """The single most important invariant in the project.

    The original ~81% R2 came from feeding the model mechanical components of its
    own target. If anything from LEAKAGE_COLS ever reappears in FEATURE_COLS, the
    headline number becomes meaningless -- so assert it directly.
    """
    overlap = set(ft.FEATURE_COLS) & set(ft.LEAKAGE_COLS)
    assert not overlap, f"leakage columns leaked into the feature set: {sorted(overlap)}"
    assert ft.TARGET not in ft.FEATURE_COLS


def test_select_features_returns_declared_schema():
    df = pd.DataFrame(
        {
            **{c: [1.0, 2.0] for c in ft.NUMERIC_FEATURES},
            "Brand": ["BLUEBONNET", "IMPERIAL"],
            "Geography": ["Great Lakes - Multi Outlet + Conv", "West - Multi Outlet + Conv"],
            ft.TARGET: [100.0, 200.0],
            "Base Volume Sales": [90.0, 180.0],
        }  # present in the frame, must not be selected
    )
    X, y = ft.select_features(df)
    assert list(X.columns) == ft.FEATURE_COLS
    assert "Base Volume Sales" not in X.columns
    assert y.tolist() == [100.0, 200.0]


def test_year_ttests_are_multiplicity_corrected():
    """Corrected p-values must exist and never be smaller than the raw one."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Year": np.repeat([2018, 2019, 2020], 200),
            ft.TARGET: np.concatenate([rng.normal(m, 1, 200) for m in (10, 10.1, 12)]),
        }
    )
    out = year_ttests(df)
    assert len(out) == 3  # 3 years -> 3 pairwise comparisons
    assert {"p_value", "p_bonferroni", "p_bh", "significant_bh"} <= set(out.columns)
    assert (out["p_bonferroni"] >= out["p_value"] - 1e-12).all()
    assert (out["p_bh"] >= out["p_value"] - 1e-12).all()


def test_vif_report_shape():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=500) for c in ft.NUMERIC_FEATURES})
    out = ft.vif_report(df)
    assert len(out) == len(ft.NUMERIC_FEATURES)
    assert out["vif"].is_monotonic_decreasing  # sorted highest-first
    assert (out["vif"] > 0).all()


def test_every_candidate_model_is_seeded():
    """Reproducibility: no candidate may carry an unset random_state."""
    for name, pipe in candidate_models().items():
        params = pipe.get_params()
        seeds = {k: v for k, v in params.items() if k.endswith("random_state")}
        assert seeds, f"{name} exposes no random_state"
        assert all(v == SEED for v in seeds.values()), f"{name} has a non-standard seed: {seeds}"
