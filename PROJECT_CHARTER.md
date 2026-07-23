# Project charter

Predictive marketing analytics on IRI point-of-sale data for the tablespreads
(butter/margarine) category. Originally a university group project; this repo is
its production rebuild — same business questions, real MLOps around them.

## Business goal

Predict **`Unit Sales No Merch`** — units sold *without* promotional support — at
the region × product × week grain, and identify what actually drives it. The
practical use is trade-planning: which regions and brands carry themselves on
price and distribution alone, and where promotional spend is doing the work.

The model is a decision-support tool for category planners. It is not a demand
forecaster (it takes contemporaneous price and distribution as inputs), and it is
not a household-level tool (the data is already aggregated across stores).

## Repo layout

| Path | Role |
|---|---|
| `data_ingestion.py` | Excel → cleaned Parquet; `audit_raw()` writes the data-quality report |
| `features.py` | Feature schema, the leakage exclusion list, preprocessing, VIF |
| `EDA.py` | Statistical tests (ANOVA, corrected t-tests), business plots, residual diagnostics |
| `models.py` | The 5 candidate pipelines, metrics, k-fold CV helper |
| `pipeline.py` | Orchestration: ingest → EDA → train/select → export + register |
| `app/model_server.py` | FastAPI serving the exported model |
| `streamlit_app.py` | Business dashboard |
| `k8s/` | Local-cluster manifests for the API |
| `tests/` | Unit + smoke tests |
| `exported_model/` | The winning model, MLflow `pyfunc` format (tracked in git) |
| `Visualization Results/` | Generated plots |
| `Group16_PA_Tablespreads.ipynb` | The original notebook — kept as the record of the earlier analysis |

Reference docs: `DATA_DICTIONARY.md`, `ARCHITECTURE.md`, `MODEL_CARD.md`,
`ROADMAP.md`.

## Non-negotiable constraints

**Zero cost.** No paid cloud anything. MLflow tracks to a local SQLite file
(`mlflow.db`), DVC pushes to a local folder (`dvc-storage/`), Kubernetes means a
free local cluster (kind / minikube / Docker Desktop) and never EKS/GKE/AKS, CI
is free-tier GitHub Actions on a public repo.

**CPU only, laptop-sized.** No GPU libraries, no deep learning, no RAPIDS/cuML.
scikit-learn does not use a GPU and nothing here needs one — the largest matrix
in the project is roughly 800k × 65 dense floats, which `Ridge` solves in
seconds. If a training run seems to hang, the cause is an environment problem,
not a compute-scale problem.

**Run training detached.** Always
`nohup python pipeline.py > logs/train.log 2>&1 &`, never in an editor's
integrated terminal. An earlier run was lost mid-training when the editor
crashed and took the child process with it. See `ARCHITECTURE.md`.

**Leakage discipline.** `features.py::LEAKAGE_COLS` exists because the original
notebook's ~81% R² came from feeding the model mechanical components of its own
target. Nothing from that list goes back into the feature set. If a new feature
is proposed, the test is: *would this value be knowable before the week happened?*

**Reproducibility.** One seed (`models.SEED = 42`) drives every stochastic
component and is logged to MLflow as a param on every run. Reruns must reproduce
the reported numbers exactly.

## Working conventions

- Routine commands (git, pip, dvc, mlflow, docker, running the pipeline, writing
  files, running tests) proceed without asking.
- Commits are batched — roughly one meaningful commit and push per phase, not one
  per file.
- Every number that appears in a doc must come from a run that actually
  happened, and the artifact it came from should be nameable
  (`data/processed/model_results.json`, `quality_report.json`, `mlflow.db`).

## Definition of done

- [ ] `dvc pull && python pipeline.py` reproduces the reported metrics from a
      fresh clone.
- [ ] Every candidate model reports a temporal-holdout score *and* k-fold CV
      mean ± std; no single-split number stands alone.
- [ ] VIF, residual diagnostics, and multiplicity-corrected p-values are computed
      and written down, including where they are unflattering.
- [ ] `/health`, `/predict`, `/model-info` respond correctly.
- [ ] Streamlit renders every chart against real data and the prediction form
      works without a network call.
- [ ] `kubectl apply -f k8s/` works against a local cluster.
- [ ] CI is green.
- [ ] `MODEL_CARD.md` states limitations plainly enough that a skeptical reader
      finds nothing hidden.
