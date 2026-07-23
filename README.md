# Predictive Marketing Analytics — Refrigerated Tablespreads

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%2B%20registry-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-13ADC7?logo=dvc&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-local%20cluster-326CE5?logo=kubernetes&logoColor=white)
![CI](https://github.com/ManojMareedu/Predictive-analytics-Data-Science-Project/actions/workflows/ci.yml/badge.svg)

Predicting non-promoted unit sales for the refrigerated tablespreads category
from 1.0M rows of IRI point-of-sale data (2018–2022), with the full MLOps path
around it: versioned data, tracked experiments, a registered model, a served API,
a business dashboard, and CI.

**Headline result: R² = 0.635 on a held-out year the model never saw.**

---

## Table of Contents

- [The number that changed](#the-number-that-changed)
- [Results](#results)
- [Business findings](#business-findings)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Serving the model](#serving-the-model)
- [Kubernetes](#kubernetes-local-cluster)
- [Dashboard](#dashboard)
- [Analytical rigor](#analytical-rigor)
- [Repository structure](#repository-structure)
- [Tech stack](#tech-stack)
- [Documentation](#documentation)
- [Author](#author)

---

## The number that changed

An earlier version of this analysis reported **~81% accuracy**. That figure was
produced by training on `Base Volume Sales` and related columns. Those are not
predictors — they are arithmetic components of the target:

```
Unit Sales No Merch = Base Unit Sales + Incremental Units
Dollar Sales        = Units × Price
Volume ≡ Units in this dataset, on a different scale
```

The model was being handed the answer and asked to add to it. The high R²
measured that identity, not a learned relationship, and the model could never
have run in production — nobody knows a week's baseline or incremental split
before the week has happened.

All 13 such columns are excluded (`features.py::LEAKAGE_COLS`), and a test
asserts none of them can re-enter the feature set. Evaluation also moved from a
random split to a **temporal** one: train on 2018–2021, test on the unseen 2022+.

**The honest number is 0.635, not 0.81.** The drop is the leakage coming out, not
a worse model. Full reasoning in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Results

Trained on 808,635 rows (2018–2021), tested on 193,930 unseen rows (2022+).
Seed 42, fixed and logged. Every number reproduces from `python pipeline.py`.

| Model | Holdout R² | CV R² (mean ± std) | Holdout RMSE | Holdout MAE |
|---|---|---|---|---|
| **HistGradientBoosting** ✅ | **0.635** | 0.908 ± 0.002 | 10,003 | 2,270 |
| Polynomial (degree 2) | 0.430 | 0.510 ± 0.005 | 12,491 | 4,508 |
| Ridge | 0.419 | 0.402 ± 0.004 | 12,614 | 4,959 |
| Lasso | 0.419 | 0.402 ± 0.004 | 12,616 | 4,938 |
| ElasticNet | 0.311 | 0.303 ± 0.001 | 13,735 | 4,143 |

Predicting the training mean scores **R² = −0.0002**, which is the floor these
should be read against.

### Why CV says 0.908 and the holdout says 0.635

This gap is the most informative result in the project, so it is reported rather
than buried. CV standard deviation is 0.002 — the model is extremely *stable*,
just stable at a number that does not survive a new year.

K-fold shuffles rows, so week 12 of 2019 for a given product and region lands in
training while week 13 of the same product and region lands in test. A boosted
tree can memorise each product-region's volume level and score very well. The
2022 holdout removes that entirely. **0.635 is the number to quote** — 0.908 is
what this model would have reported under a random split, which is exactly the
kind of result that looks impressive and fails in production.

Selection was made on the holdout, because "does this work on a year it has never
seen" is the deployment question.

---

## Business findings

- **Price and volume move sharply against each other.** The cheapest price decile
  averages **8,829 units at $1.37**; the most expensive averages **1,092 units at
  $9.56**. This is a cross-sectional association across products, not a
  within-product elasticity — it should not be read as "cut price by X, gain Y".
- **Region matters, and it is not noise.** One-way ANOVA across the 8 regions:
  F = 549.2, p ≈ 0.
- **2020 is genuinely different from every neighbouring year** (2019 vs 2020 at
  p = 1.5×10⁻⁶³), and that difference survives every multiplicity correction.
- **Prediction reliability varies 2.5× by region.** Residual spread runs from
  5,390 (Plains) to 13,206 (South Central). Planners should not treat all regions
  as equally trustworthy.
- **The model is strongest where the volume is.** The target is severely
  right-skewed (median 396 units, mean 4,566, max 648,753), so aggregate
  conclusions are far more dependable than any single small-volume prediction.

---

## Architecture

```mermaid
flowchart LR
    A["5 IRI workbooks<br/>1.32M raw rows"] -->|DVC| B["data_ingestion.py<br/>clean, dedup, downcast"]
    B --> C[("tablespreads.parquet<br/>1.00M rows")]
    C --> D["EDA.py<br/>ANOVA, corrected t-tests,<br/>plots, diagnostics"]
    C --> E["features.py<br/>leakage exclusion, VIF"]
    E --> F["models.py<br/>5 candidates"]
    F --> G["pipeline.py<br/>temporal split<br/>CV + holdout"]
    G --> H[("mlflow.db<br/>tracking + registry")]
    G --> I["exported_model/"]
    I --> J["FastAPI"]
    I --> K["Streamlit"]
    J --> L["Docker → k8s"]
```

Full component walkthrough in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start

```bash
git clone https://github.com/ManojMareedu/Predictive-analytics-Data-Science-Project.git
cd Predictive-analytics-Data-Science-Project

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Raw workbooks and the cleaned Parquet are DVC-tracked, not in git
dvc pull
```

### Reproduce the full pipeline

```bash
mkdir -p logs
nohup python pipeline.py > logs/train.log 2>&1 &
tail -f logs/train.log
```

Run it **detached**, not in an editor's integrated terminal — an editor crash
otherwise kills the run. Roughly 40 minutes on a laptop CPU, dominated by lasso
and elasticnet coordinate descent. **No GPU is used or needed**: the largest
matrix here is ~800k × 65 dense floats.

This regenerates `mlflow.db`, `exported_model/`, the plots, the dashboard
aggregates, and the JSON result files. To start clean, `rm -rf mlflow.db mlruns/`
first — it is experiment tracking, not data.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs
pytest -q                                            # 15 tests
```

---

## Serving the model

```bash
uvicorn app.model_server:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness/readiness, and *why* it is unhealthy if it is |
| `GET /model-info` | Deployed model, metrics, feature schema, seed |
| `POST /predict` | Prediction from a single region × brand × week row |
| `GET /docs` | Interactive OpenAPI docs |

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "price_per_unit_no_merch": 3.49, "price_per_unit_any_merch": 2.99,
  "price_per_volume_no_merch": 3.49, "price_per_volume_any_merch": 2.99,
  "acv_distribution_no_merch": 65.0, "acv_distribution_any_merch": 40.0,
  "year": 2022, "week": 26, "brand": "BLUEBONNET",
  "geography": "Great Lakes - IRI Standard - Multi Outlet + Conv"}'
```

```json
{"predicted_unit_sales": 140292.4, "model_name": "hist_gbr", "note": "ok"}
```

Sanity check: actual median for that brand, region and distribution band in 2022
is 127,941 units.

### Docker

```bash
docker build -t tablespreads-api:latest .
docker run -p 8000:8000 tablespreads-api:latest
```

The image installs `exported_model/requirements.txt` **after** `requirements.txt`.
That is not redundant: a pickled sklearn `Pipeline` only loads under the version
that wrote it, and `requirements.txt` floats (`scikit-learn>=1.3`). A build that
skips this picks up whatever sklearn is current and fails to unpickle the model.
MLflow regenerates those pins on every export, so this stays correct after a
retrain.

---

## Kubernetes (local cluster)

Free local Kubernetes only — Docker Desktop, `kind`, or `minikube`. **No managed
cloud Kubernetes**, which would cost money and adds nothing here.

```bash
docker build -t tablespreads-api:latest .

# kind only — Docker Desktop shares its image store already
kind create cluster && kind load docker-image tablespreads-api:latest

kubectl apply -f k8s/
kubectl rollout status deployment/tablespreads-api
kubectl port-forward svc/tablespreads-api 8088:80

curl http://localhost:8088/health
```

```
NAME                                READY   STATUS    RESTARTS   AGE
tablespreads-api-7c454f478f-7vtmr   1/1     Running   0          3m1s
tablespreads-api-7c454f478f-fwfbp   1/1     Running   0          3m1s
```

2 replicas, readiness and liveness probes on `/health`, non-root with all
capabilities dropped, and small resource limits (100m / 256Mi) — this proves the
deployment pattern, it does not serve production traffic. Tear down with
`kubectl delete -f k8s/`.

---

## Dashboard

```bash
streamlit run streamlit_app.py
```

Five tabs: regional performance, brand drivers, price and promotion, a live
prediction form, and a model comparison view that shows holdout against CV and
explains the difference.

It reads 17 KB of pre-aggregated CSVs from `data/dashboard/` rather than the
81 MB Parquet, and calls the exported model directly rather than the API — so it
deploys to Streamlit Community Cloud free tier with no data pull and no backend.

---

## Analytical rigor

Everything below is computed on every run and written to
`data/processed/*.json`, not asserted in prose.

| Check | Result |
|---|---|
| **Temporal holdout** | Train <2022, test 2022+ — no future data in training |
| **5-fold cross-validation** | Every model reports mean ± std, not one lucky split |
| **Multicollinearity (VIF)** | Max **2.51**, well under the threshold of 10 — no feature dropped |
| **Residual diagnostics** | Residuals-vs-fitted and Q-Q plots saved; heteroscedasticity quantified |
| **Multiple-comparison correction** | Bonferroni and Benjamini-Hochberg on all 15 pairwise year t-tests |
| **Data-quality audit** | Null rates, duplicate counts, schema drift — measured, not assumed |
| **Reproducibility** | One seed (42) across every model, logged to MLflow, asserted by a test |

Findings worth stating plainly:

- **Residuals are heteroscedastic and far from normal.** Spread in the top fitted
  quintile is **33× wider** than the bottom; skew 14.7, excess kurtosis 390.5. A
  single global RMSE is therefore *not* a valid error bar for an individual
  prediction, and confidence intervals must scale with the predicted level rather
  than being constant width.
- **The multiplicity correction changes conclusions.** 11 of 15 year comparisons
  are significant at raw p < 0.05; only **9 survive Bonferroni**. 2018 vs 2021
  (raw p = 0.0053 → 0.079) and 2019 vs 2023 (raw p = 0.0131 → 0.197) should not
  be reported as significant.
- **~3% of target values are imputed.** `Unit Sales No Merch` is null in 2.6–3.2%
  of raw rows and filled with 0.0 per IRI convention. "Sold zero" and "not
  measured" are not distinguishable in this extract.
- **Zero duplicate rows** in any of the five workbooks — checked, not assumed.
- **One schema inconsistency across five years:** 2022 renames
  `Product Description` to `Product`. Verified column-by-column; no other drift.
- **A split bug found during this pass:** the holdout was written as
  `Year == 2022`, which silently dropped the 3,564 rows whose week ends
  01-01-2023 from *both* train and test. Now `Year >= 2022`.

---

## Repository structure

```text
├── data_ingestion.py          # Excel → cleaned Parquet; audit_raw() quality report
├── features.py                # Feature schema, leakage exclusion, VIF
├── EDA.py                     # Statistical tests, plots, residual diagnostics
├── models.py                  # 5 candidate pipelines, metrics, k-fold CV
├── pipeline.py                # Orchestration: ingest → EDA → train → export
├── app/model_server.py        # FastAPI serving
├── streamlit_app.py           # Business dashboard
├── k8s/                       # Deployment, Service, ConfigMap (local cluster)
├── tests/                     # 15 unit + smoke tests
├── exported_model/            # Winning model, MLflow format
├── data/dashboard/            # Small aggregates for the dashboard
├── Visualization Results/     # Generated plots
├── .github/workflows/ci.yml   # Lint → test → build → smoke
├── Dockerfile
├── PROJECT_CHARTER.md         # Purpose, constraints, definition of done
├── DATA_DICTIONARY.md         # Every column, measured null rates, decisions
├── ARCHITECTURE.md            # Component walkthrough + diagram
├── MODEL_CARD.md              # Metrics, diagnostics, limitations
├── ROADMAP.md                 # Phase checklist
└── Group16_PA_Tablespreads.ipynb   # Original notebook, kept as the record
```

---

## Tech stack

| Layer | Technology |
|---|---|
| **Data versioning** | DVC (local folder remote — no account, no cloud cost) |
| **Processing** | pandas, PyArrow / Parquet |
| **Modelling** | scikit-learn (Ridge, Lasso, ElasticNet, Polynomial, HistGradientBoosting) |
| **Statistics** | statsmodels, SciPy |
| **Tracking & registry** | MLflow (local SQLite backend) |
| **Orchestration** | ZenML steps, with a direct fallback so training never depends on a server |
| **Serving** | FastAPI, Uvicorn, Pydantic v2 |
| **Dashboard** | Streamlit, Plotly |
| **Container / orchestration** | Docker, Kubernetes (local cluster) |
| **CI** | GitHub Actions (free tier) |
| **Quality** | pytest, ruff |

Everything runs on a laptop CPU at zero cost. No paid cloud services, no GPU, no
managed Kubernetes.

---

## Documentation

| Document | Contents |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Metrics, residual diagnostics, limitations, intended use |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | Every column, measured null rates, cleaning decisions |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Component-by-component walkthrough |
| [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) | Goals, constraints, definition of done |
| [`ROADMAP.md`](ROADMAP.md) | Build phases and what changed vs. the original notebook |

---

## Author

**Manoj Mareedu** — Data Scientist / ML Engineer
[GitHub](https://github.com/ManojMareedu) · [LinkedIn](https://www.linkedin.com/in/manojmareedu/)

Originally built as a Predictive Analytics project at the University of Texas at
Dallas, then rebuilt as a production system with the leakage corrected and the
validation redone.
