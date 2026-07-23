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
      multiplicity correction, seeding, and FastAPI smoke tests. 16 passing.
- [x] `.github/workflows/ci.yml` — lint → format → test → docker build →
      `/health` and `/predict` smoke

## Phase G — Docs & polish ✅

- [x] `README.md` rewritten
- [x] `MODEL_CARD.md` finalised with the real numbers
- [x] This file marked complete

## Phase H — Close-out ✅

- [x] MIT `LICENSE` added, referenced from the README footer, with the IRI data
      explicitly carved out of it
- [x] ZenML orchestration fixed for real rather than documented around — see
      *Resolved: ZenML now orchestrates* below
- [x] `requirements.txt` pinned on the model-critical libraries so a Streamlit
      Community Cloud build cannot resolve to a scikit-learn that fails to
      unpickle the model
- [x] Dashboard verified standalone with `mlflow.db` and the Parquet removed
- [x] Both Mermaid diagrams validated against the mermaid parser, not eyeballed
- [x] Streamlit Community Cloud click path documented in the README

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
- **Four real bugs found and fixed:**
  1. The holdout `Year == 2022` silently dropped 3,564 rows dated 01-01-2023
     from both train and test. Now `Year >= 2022`.
  2. The serving container installed whatever scikit-learn was current (1.9.0)
     against a model pickled by 1.6.1, so it could not unpickle. The image and
     CI now install `exported_model/requirements.txt`, which MLflow regenerates
     on every export, and `requirements.txt` pins the model-critical libraries
     directly — Streamlit Community Cloud installs only that file and has no
     second step in which to correct a bad resolution.
  3. `from __future__ import annotations` in `pipeline.py` turned every step
     annotation into a string, so ZenML could not resolve step types and every
     run fell through to the direct-call fallback. See below.
  4. The API's OpenAPI example, the smoke tests and the CI payload all used
     `"Great Lakes - Multi Outlet + Conv"`, which is not one of the eight real
     IRI market labels (they carry a ` - IRI Standard - ` infix). Requests still
     returned 200 — the label just one-hot encoded to the infrequent bucket — so
     the documented example quietly taught callers to get a worse prediction.
- **Infrastructure added:** DVC data versioning, MLflow tracking and registry,
  FastAPI serving, Docker, local Kubernetes, Streamlit dashboard, tests and CI.

### Resolved: ZenML now orchestrates

`python pipeline.py` runs `ingest_step → eda_step → train_step` through ZenML on
the default local stack — local orchestrator, local artifact store, SQLite
metadata, no server and no account.

This was previously listed as a known gap, on the assumption that it was an
environment problem. It was not. Two things were wrong, and only the first was
environmental:

1. The global ZenML config was written by a newer client (0.93.2) than the
   installed one (0.90.0) and pointed at a REST store on `127.0.0.1:8237` that
   no longer ran. Resetting the config to a local store and installing
   `zenml[server]` for `sqlmodel` fixed the client. ZenML ≥ 0.91 requires Python
   ≥ 3.10, so upgrading was not an option in this 3.9.6 environment — the
   downgrade path was the correct one, not a workaround.
2. With a healthy client, the pipeline *still* fell back. `pipeline.py` had
   `from __future__ import annotations`, which turns every annotation into a
   string; ZenML looks step I/O types up in a materializer registry keyed by
   class and raised `AttributeError: 'str' object has no attribute '__mro__'`
   while compiling. Removing that import fixed orchestration.

The fallback masked the second bug for the whole build, because it printed only
the exception *class name* — which made a compile-time defect in the code read
exactly like an unreachable server. It now prints the message, and a test asserts
the step annotations resolve to real classes. The fallback itself stays: ZenML is
a `requirements-dev.txt` dependency, and a serving-only install must still be
able to retrain.

### Known gaps

- **Streamlit Community Cloud deployment needs a manual signup**, which is
  outside what can be automated here. The app is deployment-ready and the exact
  click path is in the README.
