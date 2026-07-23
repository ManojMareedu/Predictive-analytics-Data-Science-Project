# Model Card — Tablespreads Unit Sales

Every number here comes from a single reproducible run
(`python pipeline.py`, seed 42) and is recoverable from
`data/processed/model_results.json`, `data/processed/eda_results.json`, and
`mlflow.db`. Nothing is quoted from the earlier report.

## Summary

| | |
|---|---|
| **Model** | `HistGradientBoostingRegressor` (`max_iter=300`, `max_depth=8`, `learning_rate=0.1`) inside an sklearn `Pipeline` with one-hot encoding for brand/region |
| **Target** | `Unit Sales No Merch` — units sold without promotional support |
| **Grain** | Retail region × product × week (aggregated POS, **not** household-level) |
| **Training data** | 808,635 rows, weeks ending Jan 2018 – Dec 2021 |
| **Holdout** | 193,930 rows, weeks ending Jan 2022 – 01 Jan 2023 |
| **Headline metric** | **R² = 0.635** on the unseen 2022 holdout |
| **Seed** | 42, fixed and logged to MLflow for every run |
| **Registry** | `tablespreads_unit_sales`, version 1 |

## Intended use

Decision support for category and trade planning: comparing regions and brands,
sizing the effect of price and distribution changes on non-promoted volume, and
sanity-checking plan assumptions.

**Not** intended for: financial forecasting, household or shopper-level
targeting, or automated pricing. The model consumes contemporaneous price and
distribution as inputs, so it answers *"given this price and this distribution,
what volume would we expect?"* — it does not forecast a future week unaided.

## Results — every candidate, both evaluations

Holdout = trained on 2018–2021, scored on unseen 2022+. CV = 5-fold
cross-validation *within* the training years, reported as mean ± std.

| Model | Holdout R² | CV R² (mean ± std) | Holdout RMSE | CV RMSE | Holdout MAE | CV MAE |
|---|---|---|---|---|---|---|
| **hist_gbr** ✅ | **0.635** | 0.908 ± 0.002 | **10,003** | 4,830 ± 58 | **2,270** | 1,480 ± 8 |
| polynomial | 0.430 | 0.510 ± 0.005 | 12,491 | 11,139 ± 26 | 4,508 | 4,353 ± 9 |
| ridge | 0.419 | 0.402 ± 0.004 | 12,614 | 12,307 ± 36 | 4,959 | 4,766 ± 15 |
| lasso | 0.419 | 0.402 ± 0.004 | 12,616 | 12,307 ± 36 | 4,938 | 4,751 ± 15 |
| elasticnet | 0.311 | 0.303 ± 0.001 | 13,735 | 13,293 ± 50 | 4,143 | 3,957 ± 11 |

Predicting the training mean for every row scores **R² = -0.0002** on the
holdout, which is the floor these numbers should be read against.

Selection was made on the **holdout**, not on CV. The deployment question is
"does this work on a year it has never seen", and CV cannot answer it.

### The gap between CV and holdout is the most important number here

`hist_gbr` scores 0.908 under cross-validation and 0.635 on the temporal holdout.
That gap is not noise — the CV standard deviation is 0.002, so the model is
extremely *stable*, just stable at a number that does not survive contact with a
new year.

The cause is the fold structure. K-fold shuffles rows, so week 12 of 2019 for a
given UPC in a given region lands in the training fold while week 13 of the same
UPC in the same region lands in the test fold. A gradient-boosted tree can
effectively memorise each product-region's typical volume level and score very
well. The 2022 holdout removes that: every row is a week the model has never
seen, in a year whose price and distribution conditions have shifted.

**0.635 is the number to quote.** 0.908 is what this model would have reported if
it had been evaluated with a random split — which is exactly the kind of number
that looks impressive and fails in production. The linear models show a much
smaller gap (0.402 → 0.419 for ridge, the holdout is even slightly *higher*)
because they have no capacity to memorise product-level effects in the first
place.

### On the earlier ~81% figure

The original analysis reported ~81% accuracy using `Base Volume Sales` and
related columns as predictors. Those are not predictors:

```
Unit Sales No Merch = Base Unit Sales + Incremental Units
Dollar Sales        = Units × Price
Volume ≡ Units in this dataset, on a different scale
```

A model given `Base Volume Sales` is being handed the answer and asked to adjust
it. The high R² measures that identity, not any learned relationship, and the
model would be unusable in practice because nobody knows a week's baseline or
incremental split before that week has happened. `features.py::LEAKAGE_COLS`
lists all 13 excluded columns and `tests/test_data_and_features.py` asserts none
of them can re-enter the feature set.

**The honest number is 0.635, not 0.81.** The drop is the leakage being removed,
not a worse model.

## Features

Eight numeric plus two categorical, all knowable independently of the outcome:

`Price per Unit No Merch`, `Price per Unit Any Merch`, `Price per Volume No Merch`,
`Price per Volume Any Merch`, `ACV Weighted Distribution No Merch`,
`ACV Weighted Distribution Any Merch`, `Year`, `Week`, `Brand`, `Geography`.

Brand and region are one-hot encoded with `min_frequency=2000`, collapsing the
long tail of ~370 heuristic brand strings into a single bucket. This keeps the
design matrix at ~65 columns and means an unseen brand at serving time routes to
that bucket instead of raising.

### Multicollinearity — checked, no action needed

VIF on the numeric block (50k sample, intercept included):

| Feature | VIF |
|---|---|
| Price per Volume Any Merch | 2.51 |
| Price per Unit Any Merch | 2.50 |
| Price per Volume No Merch | 1.70 |
| Price per Unit No Merch | 1.48 |
| ACV Weighted Distribution No Merch | 1.28 |
| ACV Weighted Distribution Any Merch | 1.21 |
| Year | 1.01 |
| Week | 1.00 |

The price-per-unit and price-per-volume pairs were the expected collinearity
risk. **Maximum VIF is 2.51, well under the conventional threshold of 10**, so no
feature was dropped or combined. Linear coefficients are interpretable on this
feature set. (This matters less for the selected model — gradient boosting is
untroubled by collinearity — but it is what licenses the ridge/lasso
coefficients as a secondary read on direction of effect.)

## Residual diagnostics — where this model should not be trusted

Computed on the 193,930 holdout predictions. Plots:
`Visualization Results/residuals_vs_fitted.png`, `residuals_qq.png`.

| Statistic | Value | Reading |
|---|---|---|
| Mean residual | +592 | Slight systematic under-prediction of 2022 volume |
| Residual std | 9,986 | |
| Skew | 14.7 | Severe right skew |
| Excess kurtosis | 390.5 | Extremely heavy tails |
| Spread ratio, top vs bottom fitted quintile | **33.1×** | Severe heteroscedasticity |

**Residuals are neither homoscedastic nor normal, and this is the model's main
limitation.**

*Heteroscedasticity.* The residuals-vs-fitted plot is a textbook funnel. Error
spread in the highest fitted quintile is **33× wider** than in the lowest. Errors
on a region-week predicted to sell 100,000 units are on a completely different
scale from one predicted to sell 500. **A single global RMSE of 10,003 is
therefore not a usable error bar for any individual prediction** — it dramatically
overstates error on small rows and understates it on large ones. Any confidence
interval must be proportional to the predicted level, not constant width.

*Non-normality.* The Q-Q plot departs from the diagonal hard in both tails and
especially the upper one, with the largest residuals near +400,000 units. Any
interval estimate assuming Gaussian errors is invalid. Interval estimates for
this model should come from empirical quantiles of the residuals, not from a
normal approximation.

*Heteroscedasticity by region.* Residual standard deviation varies **2.5× across
regions**:

| Region | Residual std |
|---|---|
| Plains | 5,390 |
| California | 6,631 |
| West | 7,103 |
| Northeast | 9,312 |
| Mid-South | 10,716 |
| Great Lakes | 11,325 |
| Southeast | 12,631 |
| South Central | 13,206 |

Predictions for Plains and California are meaningfully more reliable than for
South Central and Southeast. A planner using this model should not treat all
regions as equally trustworthy.

*Scale context.* The target is severely right-skewed — median 396 units, mean
4,566, maximum 648,753. Holdout MAE of 2,270 units therefore **exceeds the median
row's entire volume**. The model is genuinely useful for the high-volume
region-product-weeks that carry the category, and weak on the long tail of small
rows. Aggregate conclusions (regional totals, brand rankings) are far more
dependable than any single small-volume prediction.

## Statistical tests

**Regional differences.** One-way ANOVA of unit sales across the 8 regions:
F = 549.2, p ≈ 0. Regional differences are real and large.

**Year-over-year differences, multiplicity-corrected.** Pairwise Welch t-tests
across the six calendar-year labels give 15 comparisons. At raw α = 0.05, 11 are
significant. Applying corrections:

| Correction | Significant at 0.05 |
|---|---|
| None (raw) | 11 / 15 |
| Benjamini-Hochberg (FDR) | 11 / 15 |
| Bonferroni (FWER) | **9 / 15** |

The correction changes conclusions, which is the reason for doing it. Two
comparisons significant at raw p < 0.05 do **not** survive Bonferroni:

- 2018 vs 2021: raw p = 0.0053 → Bonferroni p = 0.079
- 2019 vs 2023: raw p = 0.0131 → Bonferroni p = 0.197

These should not be reported as significant year-over-year differences. The
strongly significant comparisons survive every correction comfortably — 2019 vs
2020 at p = 1.5×10⁻⁶³ and 2018 vs 2020 at p = 1.7×10⁻⁴³ are the COVID-year shift,
and no correction touches them.

Full raw and corrected p-values for all 15 comparisons are in
`data/processed/eda_results.json`.

**Caveat on the 2023 comparisons.** The "2023" label is a calendar artifact: the
2022 workbook's final week ends 01-01-2023, giving 3,564 rows. All four
comparisons involving it are underpowered and none is significant after
correction. It should not be read as a sixth year of data.

## Training data and known data-quality issues

Full detail in `DATA_DICTIONARY.md`; the measured audit is
`data/processed/quality_report.json`.

- 1,316,655 raw rows across five workbooks → 1,002,565 after cleaning.
- **Zero duplicate rows** in any year — checked, not assumed.
- 314,090 `Total US` roll-up rows dropped; keeping them would double-count.
- **One schema inconsistency across five years:** 2022 renames
  `Product Description` to `Product`. Verified column-by-column against 2018; no
  other drift exists.
- **~3% of target values are imputed.** The `Unit Sales No Merch` column is null
  in 2.6–3.2% of raw rows per year and is filled with 0.0. Exactly 3.22% of the
  cleaned target is zero. The IRI convention is that an absent measure means no
  sales recorded, so zero is the defensible reading — but *"sold zero units"* and
  *"was not measured"* are not distinguishable in this extract. If the second
  interpretation is correct for some rows, the model is being trained to predict
  zero on rows where the truth is unknown.
- The `Any Merch` block is null in ~40% of rows, filled with 0.0. Here the fill is
  unambiguous: no promotion ran, so promoted units genuinely are zero.

## Limitations

1. **Not a forecaster.** Price and distribution are contemporaneous inputs. To
   forecast a future week you must first supply that week's planned price and
   distribution.
2. **Association, not causation.** The strong negative price-volume relationship
   in the data (cheapest price decile averages 8,829 units at $1.37; most
   expensive averages 1,092 units at $9.56) is a cross-sectional comparison across
   different products. It is *not* a within-product elasticity, and it should not
   be used to predict what happens if a specific product's price is cut.
3. **Aggregate grain only.** No household or shopper inference is possible.
4. **Brand is a heuristic.** First two whitespace tokens of the product
   description, carried over from the original analysis for comparability. It is
   not an official brand mapping and will mis-group some products.
5. **Confidence intervals cannot be constant-width** — see the diagnostics above.
6. **Trained through 2021, validated on 2022.** The 2020–2021 period includes
   pandemic-era shifts (the ANOVA and t-tests confirm 2020 is significantly
   different from every neighbouring year). Performance on a post-2022 year is
   untested and should be re-validated before use.
7. **Unseen brands and regions degrade silently.** They route to the infrequent
   bucket and return a prediction rather than an error. This is the right serving
   behaviour but means the caller gets no signal that the input was out of
   distribution.

## Reproducibility

- Single seed `models.SEED = 42` drives every model's `random_state` and the CV
  fold shuffling; `tests/test_data_and_features.py::test_every_candidate_model_is_seeded`
  asserts no candidate carries a different or missing seed.
- The seed is logged as an MLflow param on all six runs.
- Full reproduction from a clean clone:

  ```bash
  pip install -r requirements-dev.txt
  dvc pull
  nohup python pipeline.py > logs/train.log 2>&1 &
  ```

  Runtime is roughly 40 minutes on a laptop CPU, dominated by lasso and
  elasticnet coordinate descent. No GPU is used or needed.

- **Training interpreter vs. serving interpreter.** The exported artifact was
  written under Python 3.9.6 — `exported_model/python_env.yaml` and `conda.yaml`
  record it — while the container and CI both run Python 3.11. This is deliberate
  and safe: the pickle's compatibility constraint is the *library* versions
  (`scikit-learn==1.6.1`, `numpy==2.0.2`, `scipy==1.13.1`, `cloudpickle==3.1.2`),
  which `requirements.txt` pins exactly and which install identically on both
  interpreters. The CI job builds the image on 3.11, loads this artifact, and
  smoke-tests `/predict` on every push, so the cross-version load is verified
  rather than assumed.
