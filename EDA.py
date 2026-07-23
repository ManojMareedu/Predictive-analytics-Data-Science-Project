"""Reusable EDA + statistical tests, callable standalone or as a ZenML step.

Recreates the notebook's business analysis (regional ANOVA, year-over-year
t-tests) and saves plots to Visualization Results/ for the report and dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in CI and ZenML steps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f_oneway, probplot, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from features import TARGET

PLOT_DIR = Path("Visualization Results")


def region_anova(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA of unit sales across regions (per the report's H0 tests)."""
    model = ols(f'Q("{TARGET}") ~ C(Geography)', data=df).fit()
    return sm.stats.anova_lm(model, typ=2)


def year_ttests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Welch t-tests of unit sales between years, multiplicity-corrected.

    Five years give 10 pairwise comparisons, so at raw alpha=0.05 we would expect
    ~0.5 false positives by chance alone. Reporting a raw p<0.05 as "significant"
    here is exactly the multiple-comparisons error. Both Bonferroni (conservative,
    controls family-wise error) and Benjamini-Hochberg (controls false discovery
    rate) are returned alongside the raw p, so the reader can see all three.
    """
    years = sorted(df["Year"].unique())
    rows = []
    for i, a in enumerate(years):
        for b in years[i + 1 :]:
            t, p = ttest_ind(
                df.loc[df.Year == a, TARGET],
                df.loc[df.Year == b, TARGET],
                equal_var=False,
            )
            rows.append({"year_a": a, "year_b": b, "t_stat": t, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = multipletests(out["p_value"], alpha=alpha, method="bonferroni")[1]
    out["p_bh"] = multipletests(out["p_value"], alpha=alpha, method="fdr_bh")[1]
    out["significant_bh"] = out["p_bh"] < alpha
    return out


def region_oneway(df: pd.DataFrame) -> tuple[float, float]:
    """f_oneway across regions -> (F, p). Lightweight alternative to region_anova."""
    groups = [g[TARGET].values for _, g in df.groupby("Geography", observed=True)]
    f, p = f_oneway(*groups)
    return float(f), float(p)


def save_plots(df: pd.DataFrame, out_dir: str | Path = PLOT_DIR) -> list[Path]:
    """Save the key business charts. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Regional mean unit sales
    fig, ax = plt.subplots(figsize=(10, 5))
    reg = df.groupby("Geography", observed=True)[TARGET].mean().sort_values()
    reg.plot.barh(ax=ax)
    ax.set_title("Mean Unit Sales (No Merch) by Region")
    ax.set_xlabel("Mean Unit Sales")
    fig.tight_layout()
    p = out_dir / "regional_unit_sales.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # Year-over-year trend
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("Year")[TARGET].mean().plot(marker="o", ax=ax)
    ax.set_title("Mean Unit Sales (No Merch) by Year")
    ax.set_ylabel("Mean Unit Sales")
    fig.tight_layout()
    p = out_dir / "yearly_trend.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # Top brands by total unit sales
    fig, ax = plt.subplots(figsize=(10, 5))
    top = df.groupby("Brand", observed=True)[TARGET].sum().sort_values().tail(12)
    top.plot.barh(ax=ax)
    ax.set_title("Top 12 Brands by Total Unit Sales (No Merch)")
    ax.set_xlabel("Total Unit Sales")
    fig.tight_layout()
    p = out_dir / "top_brands.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    return written


DASHBOARD_DIR = Path("data/dashboard")


def save_dashboard_aggregates(df: pd.DataFrame, out_dir: str | Path = DASHBOARD_DIR) -> list[Path]:
    """Pre-aggregate the dashboard's charts into small CSVs that live in git.

    The cleaned Parquet is 81 MB and DVC-tracked, so it is not present in a plain
    clone -- and Streamlit Community Cloud deploys from git alone. Every chart in
    the dashboard is a group-by over a handful of dimensions, so shipping the
    aggregates (a few KB) instead of the source rows makes the app deployable and
    instant, at the cost of not being able to slice below these dimensions.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def _write(frame: pd.DataFrame, name: str) -> None:
        path = out_dir / name
        frame.to_csv(path, index=False)
        written.append(path)

    # Regional performance, per year.
    _write(
        df.groupby(["Geography", "Year"], observed=True)[TARGET].agg(["mean", "sum", "count"]).reset_index(),
        "regional_by_year.csv",
    )

    # Brand drivers: top 20 by total non-promoted units.
    brand_totals = df.groupby("Brand", observed=True)[TARGET].sum().nlargest(20).index
    _write(
        df[df["Brand"].isin(brand_totals)]
        .groupby(["Brand", "Year"], observed=True)[TARGET]
        .agg(["mean", "sum"])
        .reset_index(),
        "brand_by_year.csv",
    )

    # Weekly seasonality.
    _write(
        df.groupby(["Year", "Week"], observed=True)[TARGET].mean().reset_index(),
        "weekly_trend.csv",
    )

    # Price elasticity: mean units per price decile. Deciles rather than raw price
    # so the curve is readable and not dominated by a few extreme price points.
    price_col = "Price per Unit No Merch"
    priced = df[df[price_col] > 0].copy()
    priced["price_decile"] = pd.qcut(priced[price_col], 10, labels=False, duplicates="drop")
    _write(
        priced.groupby("price_decile", observed=True)
        .agg(mean_price=(price_col, "mean"), mean_units=(TARGET, "mean"), n=(TARGET, "size"))
        .reset_index(),
        "price_elasticity.csv",
    )

    # Merch vs no-merch split, the core trade-spend question.
    _write(
        df.groupby(["Geography", "Year"], observed=True)[[TARGET, "Unit Sales Any Merch"]]
        .sum()
        .reset_index(),
        "merch_split.csv",
    )

    return written


def residual_diagnostics(
    y_true, y_pred, groups: pd.Series | None = None, out_dir: str | Path = PLOT_DIR
) -> dict:
    """Residuals-vs-fitted + Q-Q plot for the winning model; returns a summary.

    Two questions these answer, which R2 alone cannot:
      - homoscedasticity: does error variance grow with the prediction? If it fans
        out, a single global RMSE understates error on high-volume rows and any
        constant-width confidence interval is wrong.
      - normality: are residuals roughly Gaussian? If the Q-Q tails diverge hard,
        interval estimates that assume normal errors don't hold.
    `groups` (region or brand) adds per-group residual spread so heteroscedasticity
    can be attributed rather than just observed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    resid = y_true - y_pred

    # Plotting 55k points makes an unreadable blob and a slow PNG; sample for the
    # scatter only -- every statistic below is computed on the full residual set.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(resid), size=min(20_000, len(resid)), replace=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred[idx], resid[idx], s=4, alpha=0.25, edgecolors="none")
    ax.axhline(0, color="crimson", lw=1)
    ax.set_xlabel("Fitted values (predicted unit sales)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs Fitted")
    fig.tight_layout()
    rvf = out_dir / "residuals_vs_fitted.png"
    fig.savefig(rvf, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    probplot(resid[idx], dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot of Residuals")
    fig.tight_layout()
    qq = out_dir / "residuals_qq.png"
    fig.savefig(qq, dpi=110)
    plt.close(fig)

    # Heteroscedasticity, quantified: split fitted values into quintiles and compare
    # residual spread in the top vs bottom bin. A ratio near 1 is homoscedastic.
    bins = pd.qcut(pd.Series(y_pred), 5, labels=False, duplicates="drop")
    spread = pd.Series(resid).groupby(bins).std()
    summary = {
        "resid_mean": float(resid.mean()),
        "resid_std": float(resid.std()),
        "resid_skew": float(pd.Series(resid).skew()),
        "resid_kurtosis": float(pd.Series(resid).kurtosis()),
        "spread_ratio_top_vs_bottom_quintile": float(spread.iloc[-1] / spread.iloc[0])
        if len(spread) > 1 and spread.iloc[0] > 0
        else float("nan"),
        "plots": [str(rvf), str(qq)],
    }
    if groups is not None:
        by_group = pd.Series(resid).groupby(np.asarray(groups)).std().sort_values()
        summary["resid_std_by_group"] = {str(k): float(v) for k, v in by_group.items()}
    return summary


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/tablespreads.parquet")
    print("Region ANOVA:\n", region_anova(df.sample(min(50000, len(df)), random_state=1)))
    print("\nYear t-tests:\n", year_ttests(df).to_string(index=False))
    print("\nSaved:", [str(p) for p in save_plots(df)])
