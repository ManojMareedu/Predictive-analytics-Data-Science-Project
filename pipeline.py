"""ZenML training pipeline: ingest -> EDA -> train/select -> export best model.

MLflow (local SQLite backend) handles experiment tracking + model registry;
ZenML handles orchestration. Both are zero-cost and run entirely on a laptop.

Run:  python pipeline.py            # runs the ZenML pipeline
The core functions (_ingest/_eda/_train_and_select) are plain and importable,
so tests and CI can exercise the logic without a ZenML server.
"""

# NOTE: deliberately no `from __future__ import annotations` in this module.
# PEP 563 turns every annotation into a string, and ZenML resolves each step's
# input/output types by looking them up in the materializer registry -- which
# indexes by class. Given the string "dict" it raises
# `AttributeError: 'str' object has no attribute '__mro__'` at compile time, the
# whole pipeline drops to the fallback path below, and nothing says why. The
# other modules keep the future import; this one must not.

import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from data_ingestion import build_dataset
from EDA import (
    region_oneway,
    residual_diagnostics,
    save_dashboard_aggregates,
    save_plots,
    year_ttests,
)
from features import FEATURE_COLS, TARGET, select_features, vif_report
from models import CV_FOLDS, SEED, candidate_models, cross_validate_model, evaluate

PARQUET = "data/processed/tablespreads.parquet"
EXPORT_DIR = "exported_model"
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "tablespreads"
TEST_YEAR = 2022  # temporal holdout: train on everything before, test on this
RESULTS_JSON = "data/processed/model_results.json"
EDA_JSON = "data/processed/eda_results.json"


# --------------------------------------------------------------------------- #
# Plain, importable core logic
# --------------------------------------------------------------------------- #
def _ingest(parquet: str = PARQUET) -> str:
    """Ensure the cleaned Parquet exists; build it from Excel if missing."""
    if not Path(parquet).exists():
        build_dataset(out_path=parquet)
    return parquet


def _eda(parquet: str) -> dict:
    """Run the statistical tests + save business plots; return a small summary."""
    df = pd.read_parquet(parquet)
    f_stat, p_val = region_oneway(df)
    save_plots(df)
    save_dashboard_aggregates(df)
    tt = year_ttests(df)
    vif = vif_report(df)
    summary = {
        "region_anova_F": f_stat,
        "region_anova_p": p_val,
        "n_year_ttests": int(len(tt)),
        "min_ttest_p_raw": float(tt["p_value"].min()),
        "n_significant_raw": int((tt["p_value"] < 0.05).sum()),
        "n_significant_bonferroni": int((tt["p_bonferroni"] < 0.05).sum()),
        "n_significant_bh": int(tt["significant_bh"].sum()),
        "year_ttests": tt.to_dict(orient="records"),
        "vif": vif.to_dict(orient="records"),
        "max_vif": float(vif["vif"].max()),
    }
    Path(EDA_JSON).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _train_and_select(parquet: str) -> dict:
    """Train all candidates on a temporal split, log to MLflow, export the best."""
    df = pd.read_parquet(parquet)
    # `>=` not `==`: the 2022 workbook's final week ends 01-01-2023, so 3,564 rows
    # carry Year=2023. An `== 2022` test silently drops them from *both* sides of
    # the split. They belong with the holdout -- they are the latest weeks we have.
    train_df = df[df["Year"] < TEST_YEAR]
    test_df = df[df["Year"] >= TEST_YEAR]
    X_train, y_train = select_features(train_df)
    X_test, y_test = select_features(test_df)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    results: dict[str, dict] = {}
    best_name, best_r2, best_pipe = None, float("-inf"), None

    for name, pipe in candidate_models().items():
        with mlflow.start_run(run_name=name):
            # K-fold CV on the training years first: stability across resamples.
            cv = cross_validate_model(pipe, X_train, y_train)
            # Then the real test -- fit on <2022, score on the unseen 2022 year.
            pipe.fit(X_train, y_train)
            m = evaluate(y_test, pipe.predict(X_test))
            mlflow.log_params(
                {
                    "model": name,
                    "test_year": TEST_YEAR,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "random_seed": SEED,
                    "cv_folds": CV_FOLDS,
                }
            )
            mlflow.log_metrics({**m.as_dict(), **cv})
            results[name] = {**m.as_dict(), **cv}
            print(
                f"  {name:12s} holdout R2={m.r2:.3f} RMSE={m.rmse:.0f} MAE={m.mae:.0f} | "
                f"CV R2={cv['cv_r2_mean']:.3f}+/-{cv['cv_r2_std']:.3f}",
                flush=True,
            )
            # Select on the temporal holdout, not CV: "works on a future year" is
            # the deployment question, and CV's shuffled folds can't answer it.
            if m.r2 > best_r2:
                best_name, best_r2, best_pipe = name, m.r2, pipe

    # Residual diagnostics for the winner, on the 2022 holdout predictions.
    diagnostics = residual_diagnostics(y_test, best_pipe.predict(X_test), groups=test_df["Geography"])

    # Export the winner as a self-contained MLflow model + register it.
    if Path(EXPORT_DIR).exists():
        shutil.rmtree(EXPORT_DIR)
    signature = infer_signature(X_train, best_pipe.predict(X_train.head(5)))
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.log_params({"model": best_name, "random_seed": SEED, "test_year": TEST_YEAR})
        mlflow.log_metrics(results[best_name])
        mlflow.sklearn.log_model(
            best_pipe,
            name="model",
            signature=signature,
            registered_model_name="tablespreads_unit_sales",
        )
    mlflow.sklearn.save_model(best_pipe, EXPORT_DIR, signature=signature)

    summary = {
        "best_model": best_name,
        "target": TARGET,
        "feature_cols": FEATURE_COLS,
        "test_year": TEST_YEAR,
        "random_seed": SEED,
        "cv_folds": CV_FOLDS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": results,
        "residual_diagnostics": diagnostics,
        "sample_input": X_test.head(1).to_dict(orient="records")[0],
    }
    # Metadata for /model-info and the dashboard.
    Path(EXPORT_DIR, "metadata.json").write_text(json.dumps(summary, indent=2, default=str))
    Path(RESULTS_JSON).write_text(json.dumps(summary, indent=2, default=str))
    print(f"Best: {best_name} (holdout R2={best_r2:.3f}) -> {EXPORT_DIR}/", flush=True)
    return {"best_model": best_name, "best_r2": best_r2, "metrics": results}


# --------------------------------------------------------------------------- #
# ZenML wrappers
# --------------------------------------------------------------------------- #
try:
    from zenml import pipeline, step

    @step
    def ingest_step() -> str:
        return _ingest()

    @step
    def eda_step(parquet: str) -> dict:
        return _eda(parquet)

    @step
    def train_step(parquet: str) -> dict:
        return _train_and_select(parquet)

    @pipeline(name="tablespreads_training")
    def training_pipeline():
        parquet = ingest_step()
        eda_step(parquet)
        train_step(parquet)

    _ZENML = True
except Exception as exc:  # pragma: no cover - ZenML optional at runtime
    _ZENML = False
    _ZENML_ERR = exc


def run(use_zenml: bool = True) -> dict:
    """Run the pipeline. Falls back to plain functions if ZenML is unavailable.

    The fallback covers *runtime* failure, not just a missing import: ZenML may
    import fine but still be configured against a tracking server that isn't up.
    Training must not depend on that -- a fresh clone and CI both need to run this
    with nothing but pip install, and the modeling result is identical either way.
    """
    if use_zenml and _ZENML:
        try:
            training_pipeline()
            return {"orchestrator": "zenml"}
        except Exception as exc:
            # Print the message, not just the exception class. A bare class name
            # made a real compile-time bug in the step signatures look identical
            # to "the server is down", and it stayed hidden for several runs.
            print(
                f"ZenML orchestration unavailable ({type(exc).__name__}: {exc}); "
                "running the same steps directly.",
                flush=True,
            )
    parquet = _ingest()
    print("EDA:", json.dumps(_eda(parquet), indent=2, default=str), flush=True)
    return _train_and_select(parquet)


if __name__ == "__main__":
    run()
