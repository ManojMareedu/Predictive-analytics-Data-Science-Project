"""Business dashboard for the tablespreads category.

Reads the small pre-aggregated CSVs in data/dashboard/ (a few KB, committed to
git) rather than the 81 MB DVC-tracked Parquet, and calls the exported model
directly rather than the FastAPI service. Both choices exist so the app runs
standalone on Streamlit Community Cloud with no data pull and no backend to host.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DASH = Path("data/dashboard")
MODEL_DIR = Path("exported_model")
TARGET = "Unit Sales No Merch"

st.set_page_config(page_title="Tablespreads Analytics", page_icon="📊", layout="wide")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DASH / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    """Cached as a resource, not data: it is an unserialisable object loaded once."""
    try:
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(str(MODEL_DIR))
        meta_path = MODEL_DIR / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta, None
    except Exception as exc:
        return None, {}, f"{type(exc).__name__}: {exc}"


regional = load_csv("regional_by_year.csv")
brands = load_csv("brand_by_year.csv")
weekly = load_csv("weekly_trend.csv")
elasticity = load_csv("price_elasticity.csv")
merch = load_csv("merch_split.csv")
model, meta, model_err = load_model()

st.title("Tablespreads Category Analytics")
st.caption(f"IRI point-of-sale data, 2018–2022 · 1.0M region × product × week rows · target: {TARGET}")

if regional.empty:
    st.error("No dashboard data found. Run `python pipeline.py` to generate `data/dashboard/`.")
    st.stop()

tab_overview, tab_brands, tab_price, tab_predict, tab_models = st.tabs(
    ["Regional performance", "Brand drivers", "Price & promotion", "Predict", "Model comparison"]
)

# --------------------------------------------------------------------------- #
with tab_overview:
    st.subheader("Where the category sells")

    years = sorted(regional["Year"].unique())
    picked = st.multiselect("Years", years, default=years, key="ov_years")
    view = regional[regional["Year"].isin(picked)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total non-promoted units", f"{view['sum'].sum():,.0f}")
    c2.metric("Regions", f"{view['Geography'].nunique()}")
    c3.metric("Mean units / region-week", f"{view['mean'].mean():,.0f}")

    st.plotly_chart(
        px.bar(
            view.groupby("Geography", as_index=False)["sum"].sum().sort_values("sum"),
            x="sum",
            y="Geography",
            orientation="h",
            labels={"sum": "Total non-promoted units", "Geography": ""},
            title="Total units by region",
        ),
        width="stretch",
    )

    st.plotly_chart(
        px.line(
            regional.groupby("Year", as_index=False)["mean"].mean(),
            x="Year",
            y="mean",
            markers=True,
            labels={"mean": "Mean units per region-week"},
            title="Category trend, 2018–2022",
        ),
        width="stretch",
    )

    if not weekly.empty:
        st.plotly_chart(
            px.line(
                weekly,
                x="Week",
                y=TARGET,
                color="Year",
                labels={TARGET: "Mean units"},
                title="Weekly seasonality by year",
            ),
            width="stretch",
        )

# --------------------------------------------------------------------------- #
with tab_brands:
    st.subheader("Which brands carry the category")
    if brands.empty:
        st.info("No brand aggregates available.")
    else:
        totals = brands.groupby("Brand", as_index=False)["sum"].sum().sort_values("sum")
        st.plotly_chart(
            px.bar(
                totals.tail(20),
                x="sum",
                y="Brand",
                orientation="h",
                labels={"sum": "Total non-promoted units", "Brand": ""},
                title="Top 20 brands by total units",
            ),
            width="stretch",
        )
        picked_brands = st.multiselect(
            "Compare brands over time",
            totals["Brand"].tolist()[::-1],
            default=totals["Brand"].tolist()[::-1][:5],
        )
        if picked_brands:
            st.plotly_chart(
                px.line(
                    brands[brands["Brand"].isin(picked_brands)],
                    x="Year",
                    y="sum",
                    color="Brand",
                    markers=True,
                    labels={"sum": "Total units"},
                    title="Brand trajectories",
                ),
                width="stretch",
            )
        st.caption(
            "“Brand” is the first two tokens of the product description — a heuristic "
            "carried over from the original analysis, not an official brand mapping."
        )

# --------------------------------------------------------------------------- #
with tab_price:
    st.subheader("Price elasticity")
    if elasticity.empty:
        st.info("No elasticity aggregates available.")
    else:
        st.plotly_chart(
            px.line(
                elasticity,
                x="mean_price",
                y="mean_units",
                markers=True,
                labels={"mean_price": "Mean price per unit ($)", "mean_units": "Mean units sold"},
                title="Units sold vs price (deciles of non-promoted price)",
            ),
            width="stretch",
        )
        lo, hi = elasticity.iloc[0], elasticity.iloc[-1]
        st.markdown(
            f"The cheapest decile averages **${lo['mean_price']:.2f}/unit** and "
            f"**{lo['mean_units']:,.0f} units**; the most expensive averages "
            f"**${hi['mean_price']:.2f}/unit** and **{hi['mean_units']:,.0f} units** — "
            "a strong negative price–volume relationship across the category. This is a "
            "descriptive association across products, not a causal within-product elasticity: "
            "cheap and premium items differ in more ways than price."
        )

    st.subheader("Promoted vs non-promoted volume")
    if not merch.empty:
        m = merch.groupby("Geography", as_index=False)[[TARGET, "Unit Sales Any Merch"]].sum()
        m = m.melt(id_vars="Geography", var_name="Type", value_name="Units")
        m["Type"] = m["Type"].map({TARGET: "No merchandising", "Unit Sales Any Merch": "With merchandising"})
        st.plotly_chart(
            px.bar(
                m,
                x="Units",
                y="Geography",
                color="Type",
                orientation="h",
                barmode="group",
                labels={"Geography": ""},
                title="Merchandised vs non-merchandised units by region",
            ),
            width="stretch",
        )

# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Predict non-promoted unit sales")
    if model is None:
        st.error(f"Model not available: {model_err}. Run `python pipeline.py` first.")
    else:
        st.caption(f"Model in use: **{meta.get('best_model', 'unknown')}**")
        geos = sorted(regional["Geography"].unique())
        brand_opts = sorted(brands["Brand"].unique()) if not brands.empty else ["BLUEBONNET"]

        with st.form("predict"):
            c1, c2 = st.columns(2)
            with c1:
                geography = st.selectbox("Region", geos)
                brand = st.selectbox("Brand", brand_opts)
                year = st.number_input("Year", 2018, 2035, 2022)
                week = st.number_input("Week", 1, 53, 26)
                ppu_no = st.number_input("Price per unit (non-promoted)", 0.0, 50.0, 3.49, 0.01)
            with c2:
                ppu_any = st.number_input("Price per unit (promoted)", 0.0, 50.0, 2.99, 0.01)
                ppv_no = st.number_input("Price per volume (non-promoted)", 0.0, 50.0, 3.49, 0.01)
                ppv_any = st.number_input("Price per volume (promoted)", 0.0, 50.0, 2.99, 0.01)
                acv_no = st.slider("ACV distribution % (non-promoted)", 0.0, 100.0, 65.0)
                acv_any = st.slider("ACV distribution % (promoted)", 0.0, 100.0, 40.0)
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "Price per Unit No Merch": ppu_no,
                        "Price per Unit Any Merch": ppu_any,
                        "Price per Volume No Merch": ppv_no,
                        "Price per Volume Any Merch": ppv_any,
                        "ACV Weighted Distribution No Merch": acv_no,
                        "ACV Weighted Distribution Any Merch": acv_any,
                        "Year": int(year),
                        "Week": int(week),
                        "Brand": brand,
                        "Geography": geography,
                    }
                ]
            )
            pred = float(model.predict(row)[0])
            clamped = max(pred, 0.0)
            st.metric("Predicted units (no merchandising)", f"{clamped:,.0f}")
            if pred < 0:
                st.warning(
                    f"The model returned {pred:,.0f}, which is outside the feasible range; "
                    "shown clamped to zero. This usually means the inputs fall outside the "
                    "range the model was trained on."
                )
            rmse = (meta.get("metrics", {}).get(meta.get("best_model"), {}) or {}).get("rmse")
            if rmse:
                st.caption(
                    f"Typical error for this model on the unseen 2022 holdout is ±{rmse:,.0f} units "
                    "(RMSE). Treat this as a planning estimate, not a precise forecast — see the "
                    "Model comparison tab."
                )

# --------------------------------------------------------------------------- #
with tab_models:
    st.subheader("Model comparison")
    metrics = meta.get("metrics") or {}
    if not metrics:
        st.info("No metrics recorded yet. Run `python pipeline.py`.")
    else:
        rows = []
        for name, m in metrics.items():
            rows.append(
                {
                    "Model": name,
                    "Holdout R²": round(m.get("r2", float("nan")), 4),
                    "CV R² (mean)": round(m.get("cv_r2_mean", float("nan")), 4),
                    "CV R² (std)": round(m.get("cv_r2_std", float("nan")), 4),
                    "Holdout RMSE": round(m.get("rmse", float("nan")), 1),
                    "CV RMSE (mean)": round(m.get("cv_rmse_mean", float("nan")), 1),
                    "Holdout MAE": round(m.get("mae", float("nan")), 1),
                }
            )
        table = pd.DataFrame(rows).sort_values("Holdout R²", ascending=False)
        st.dataframe(table, width="stretch", hide_index=True)

        st.markdown(
            f"""
**How to read this.** *Holdout* is the honest test: trained on 2018–2021, scored on
the unseen {meta.get("test_year", 2022)} data. *CV* is {meta.get("cv_folds", 5)}-fold
cross-validation within the training years — it measures stability across resamples,
and it reads optimistically because shuffled folds mix weeks from adjacent years.
Selection was made on the holdout. Seed `{meta.get("random_seed", 42)}` is fixed and
logged, so these numbers reproduce exactly on a rerun.

The earlier version of this analysis reported ~81% accuracy. That figure came from
training on columns (`Base Volume Sales` and similar) that are arithmetic components
of the target — the model was being handed the answer. Those are excluded here, which
is why these numbers are lower and why they are trustworthy. See `MODEL_CARD.md`.
"""
        )

        for plot, caption in [
            ("Visualization Results/residuals_vs_fitted.png", "Residuals vs fitted values"),
            ("Visualization Results/residuals_qq.png", "Normal Q-Q plot of residuals"),
        ]:
            if Path(plot).exists():
                st.image(plot, caption=caption, width="stretch")
