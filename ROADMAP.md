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

## Phase D — Serving

- [ ] `app/model_server.py` — FastAPI with `/health`, `/predict`, `/model-info`
- [ ] `Dockerfile` + `.dockerignore`

## Phase D.5 — Kubernetes (local only)

- [ ] `k8s/` — Deployment, Service, ConfigMap, small resource limits
- [ ] Documented `kind`/`minikube`/Docker Desktop spin-up

## Phase E — Dashboard

- [ ] `streamlit_app.py` — regional performance, brand drivers, price elasticity,
      merch vs no-merch, model comparison, live prediction form

## Phase F — Tests & CI

- [ ] `tests/` — cleaning, feature engineering, leakage guard, API smoke test
- [ ] `.github/workflows/ci.yml` — lint → test → docker build → `/health` smoke

## Phase G — Docs & polish

- [ ] `README.md` rewrite
- [ ] `MODEL_CARD.md` finalised
- [ ] This file marked complete

## Done — what changed vs. the original notebook

Filled in as phases complete; see `MODEL_CARD.md` for the metrics discussion.

- The headline accuracy figure changed. The notebook reported ~81% by training on
  columns that are mechanical components of the target (`Base Volume Sales` and
  friends). Those are excluded now, and the honest temporal-holdout number is
  materially lower — see `MODEL_CARD.md`.
- Evaluation moved from a random split to a temporal one (train <2022, test
  2022+), which is the question that actually matters for deployment.
- Every reported metric now carries a CV mean ± std next to it.
