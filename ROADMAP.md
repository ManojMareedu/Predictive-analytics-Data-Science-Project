# Roadmap

Production rebuild of the tablespreads predictive-analytics project. One commit
and push per phase.

## Phase A — Repo hygiene & environment ✅

- [x] `.gitignore` covering `mlruns/`, `mlflow.db`, `.dvc/cache/`, raw `*.xlsx`,
      `__pycache__/`, `.venv/`, `logs/`, `dvc-storage/`
- [x] `requirements.txt` (serving/demo) split from `requirements-dev.txt`
      (tooling: zenml, dvc, pytest, ruff)
- [x] `dvc init` with a **local folder remote** (`./dvc-storage`) — no account, no
      signup, no cloud bill
- [x] 5 raw workbooks + the cleaned Parquet tracked by DVC and pushed to the
      local remote (379 MB), `.dvc` pointer files committed to git
- [x] Local git identity set for this repo

## Phase B — Data pipeline ✅

- [x] `data_ingestion.py` — concatenate 5 years, standardise the 2022 schema
      rename, dedup, drop the `Total US` roll-up, downcast, write Parquet
- [x] `features.py` — feature schema with an explicit leakage exclusion list
- [x] `EDA.py` — ANOVA, pairwise t-tests, business plots as reusable functions
- [x] `audit_raw()` — measured data-quality report to `quality_report.json`

## Phase C — Training pipeline + analytical rigor ✅

- [x] `models.py` — 5 candidates: ridge, lasso, elasticnet, polynomial, hist_gbr
- [x] `pipeline.py` — ingest → EDA → temporal split → train → select → export →
      register in the MLflow registry
- [x] Training runs **detached** (`nohup`), so an editor crash cannot kill it
- [x] Stale `RUNNING` state from the interrupted run cleared; clean full rerun
- [x] **Rigor upgrades:**
  - [x] 5-fold CV on the pre-2022 training data alongside the temporal holdout,
        reporting mean ± std for R², RMSE and MAE
  - [x] VIF on the numeric feature block
  - [x] Residual diagnostics for the winner — residuals-vs-fitted, Q-Q,
        quintile spread ratio, per-region residual spread
  - [x] Bonferroni + Benjamini-Hochberg correction on the pairwise year t-tests,
        raw and corrected p-values both reported
  - [x] Data-quality audit written down — null rates, duplicate counts,
        year-to-year schema drift
  - [x] Single seed (`SEED = 42`) across every model, logged as an MLflow param
  - [x] Metrics reported as CV mean ± std next to the holdout number
- [x] **Bug found and fixed during the rigor pass:** the holdout was written as
      `Year == 2022`, which silently dropped the 3,564 rows whose week ends
      01-01-2023 from both train and test. Now `Year >= 2022`.

## Phase D — Serving ✅

- [x] `app/model_server.py` — FastAPI with `/health`, `/predict`, `/model-info`
- [x] `Dockerfile` + `.dockerignore`, non-root, healthcheck
- [x] Verified: container returns the same prediction as the local model
      (140,292.4 units), against an actual median of 127,941 for that brand,
      region and distribution band

## Phase D.5 — Kubernetes (local only) ✅

- [x] `k8s/` — Deployment (2 replicas, probes, non-root, 100m/256Mi), Service,
      ConfigMap
- [x] Verified on a live local cluster: both pods `1/1 Running`, `/health` and
      `/predict` served through the Service via port-forward

## Phase E — Dashboard ✅

- [x] `streamlit_app.py` — regional performance, brand drivers, price elasticity,
      merch vs no-merch, model comparison, live prediction form
- [x] Reads 17 KB of committed aggregates instead of the 81 MB Parquet, so it
      deploys to Streamlit Community Cloud with no data pull and no backend
- [x] Verified via `AppTest`: 0 exceptions, 5 tabs, real data

## Phase F — Tests & CI ✅

- [x] `tests/` — cleaning, schema drift, calendar derivation, leakage guard,
      multiplicity correction, seeding, and FastAPI smoke tests. 15 passing.
- [x] `.github/workflows/ci.yml` — lint → format → test → docker build →
      `/health` and `/predict` smoke

## Phase G — Docs & polish ✅

- [x] `README.md` rewritten
- [x] `MODEL_CARD.md` finalised with the real numbers
- [x] This file marked complete

## Done — what changed vs. the original notebook

See `MODEL_CARD.md` for the full metrics discussion.

- **The headline accuracy figure changed.** The notebook reported ~81% by
  training on columns that are mechanical components of the target
  (`Base Volume Sales` and friends). Those are excluded now. The honest
  temporal-holdout number is **R² = 0.635** (HistGradientBoosting).
- **Evaluation moved from a random split to a temporal one** (train <2022, test
  2022+), which is the question that actually matters for deployment.
- **Every reported metric carries a CV mean ± std** next to the holdout number.
- **The CV/holdout gap is reported, not hidden.** 0.908 CV vs 0.635 holdout: a
  shuffled split lets a boosted tree memorise product-region volume levels. That
  gap is the argument for the temporal split, so it belongs in the write-up.
- **Rigor checks added:** VIF (max 2.51, nothing dropped), residual diagnostics
  (33× heteroscedasticity, skew 14.7, kurtosis 390.5), Bonferroni/BH corrections
  (11 raw → 9 Bonferroni-significant), and a measured data-quality audit.
- **Two real bugs found and fixed:**
  1. The holdout `Year == 2022` silently dropped 3,564 rows dated 01-01-2023
     from both train and test. Now `Year >= 2022`.
  2. The serving container installed whatever scikit-learn was current (1.9.0)
     against a model pickled by 1.6.1, so it could not unpickle. The image and
     CI now install `exported_model/requirements.txt`, which MLflow regenerates
     on every export.
- **Infrastructure added:** DVC data versioning, MLflow tracking and registry,
  FastAPI serving, Docker, local Kubernetes, Streamlit dashboard, tests and CI.

### Known gaps

- **ZenML orchestration runs via the direct fallback.** The `@step`/`@pipeline`
  definitions exist and are used when a ZenML server is reachable, but this
  machine's ZenML install is missing server dependencies (`sqlmodel`) and its
  global config points at a server that is not running. Training deliberately
  does not depend on it — the fallback executes identical steps and produces
  identical metrics. Chasing the dependency chain would have added nothing to
  the modelling result.
- **Streamlit Community Cloud deployment needs a manual signup**, which is
  outside what can be automated here. The app is deployment-ready.
