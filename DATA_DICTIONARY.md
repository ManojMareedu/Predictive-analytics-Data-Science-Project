# Data Dictionary — IRI POS Tablespreads

Source: `IRI_POS_Tablespreads_{2018..2022}.xlsx`, sheet `Tablespreads_POS`.
Every figure below was measured by reading the raw workbooks, not estimated —
see `data_ingestion.py::audit_raw()` and the generated
`data/processed/quality_report.json`.

## Grain

One row = **Geography (region) × Product (UPC 13 digit) × Week**.

This is *aggregated retail point-of-sale data*, not household or transaction
data. There is no customer identifier and no individual basket; every measure is
already summed across all stores in the region for that week.

| | |
|---|---|
| Regions (`Geography`) | 8, after dropping the national roll-up |
| Distinct UPCs | 1,928 |
| Distinct brands (derived) | 374 raw; ~20 kept as explicit model columns, rest bucketed |
| Weeks covered | Week ending 01-14-2018 → 01-01-2023 (259 weeks) |
| Raw rows across all 5 files | 1,316,655 |
| Rows after cleaning | 1,002,565 |

## Cleaning decisions and their measured effect

| Step | Effect | Why |
|---|---|---|
| Drop exact duplicate rows | **0 rows removed** (all 5 years) | Checked, not assumed — the files are already unique at the stated grain. |
| Drop `Total US - Multi Outlet + Conv` | **314,090 rows removed** (~23.9%) | This is the national roll-up of the other regions. Keeping it double-counts every sale and inflates every correlation, because the aggregate row is a deterministic function of the regional rows. |
| `fillna(0.0)` on the 20 numeric measures | see null section below | |
| Downcast numerics to `float32` | ~50% memory saving | Values are well inside float32 range; the parquet is 81 MB instead of ~160 MB. |
| `Geography`, `Brand` → `category` dtype | ~large saving on 1M rows | Only 8 and 374 distinct values respectively. |

Rows removed per year (Total US roll-up): 2018: 66,689 · 2019: 66,090 ·
2020: 63,683 · 2021: 60,644 · 2022: 56,984.

## Schema drift across years — audited

All five workbooks were compared column-by-column against 2018 as the reference
layout. **Exactly one inconsistency exists across the five years:**

| Year | Difference vs 2018 |
|---|---|
| 2019, 2020, 2021 | none — identical 24-column schema |
| 2022 | column `Product Description` is renamed to **`Product`** |

Handled in `data_ingestion.py::load_year` by renaming `Product` →
`Product Description` on load. No other drift exists: column count (24), column
order, and every other column name are identical across all five files. This was
verified programmatically rather than by eye, and the check re-runs any time
`audit_raw()` is called, so a future year's file with a new rename will be caught
rather than silently mis-parsed.

### A second inconsistency, in the data rather than the schema

The 2022 workbook's final week ends **01-01-2023**, so 3,564 rows carry
`Year = 2023` once the week-ending date is parsed. This is a calendar artifact,
not a sixth year of data. It matters because a temporal holdout written as
`Year == 2022` silently drops those rows from *both* the training and the test
set. The split in `pipeline.py` uses `Year >= 2022` for the holdout so these rows
are scored rather than discarded.

## Null handling

Nulls are **not** scattered randomly. They occur in exactly two co-occurring
blocks, which is what determines the fill decision:

| Block | Null rate (2018→2022) | Meaning | Decision |
|---|---|---|---|
| **Any-Merch block** (9 columns) | 40.73%, 39.98%, 41.94%, 38.79%, 38.45% | No promotional/merchandised activity for that UPC in that region-week | Fill **0.0** — a null here genuinely means "no promotion ran", so zero promoted units/dollars is the correct value, not a missing value. |
| **No-Merch block** (6 columns, includes the target) | 3.19%, 3.17%, 2.94%, 3.12%, 2.59% | No non-promoted sales recorded for that UPC in that region-week | Fill **0.0** — see caveat below. |
| `Base *`, `Price per Unit`, `Price per Volume`, `Incremental*` totals | **0% null in every year** | — | No action needed. |

Within a block the columns are null together on the same rows — e.g. whenever
`Unit Sales Any Merch` is null, all nine Any-Merch columns are null on that row.

**Caveat that belongs on the record:** the No-Merch block contains the model
target, `Unit Sales No Merch`. Filling it with 0.0 means roughly **3% of rows
carry an imputed target of zero** rather than an observed one. The IRI convention
is that an absent measure means no sales were recorded, so zero is the defensible
reading — but "the product genuinely sold zero units" and "the product was not
measured in that region-week" are not distinguishable in this extract. This is
carried into `MODEL_CARD.md` as a stated limitation rather than hidden.

## Columns

Raw dtype is what pandas infers from the workbook; stored dtype is what lands in
`data/processed/tablespreads.parquet`.

### Identifiers

| Column | Stored dtype | Meaning |
|---|---|---|
| `Geography` | `category` | Retail region, e.g. *Great Lakes - Multi Outlet + Conv*. The `Total US` roll-up is dropped. |
| `Time` | `object` | Week label, literally `"Week Ending MM-DD-YY"`. |
| `Product Description` | `object` | Full product string, e.g. `BLUE BONNET ... 16 OZ`. Named `Product` in the 2022 file. |
| `UPC 13 digit` | `object` (string) | 13-digit product code. Kept as a string — leading zeros are significant and numeric parsing destroys them. |

### Measures (all `float32`, all filled with 0.0 where null)

The dataset splits every measure three ways: **No Merch** (sold without
promotional support), **Any Merch** (sold with promotional support), and in some
cases a blended total across both.

| Column | Meaning |
|---|---|
| `Unit Sales No Merch` | **TARGET.** Units sold with no merchandising/promotion, per region-week. |
| `Unit Sales Any Merch` | Units sold with promotional support. |
| `Dollar Sales No Merch` / `Any Merch` | Revenue in dollars, split the same way. |
| `Volume Sales No Merch` / `Any Merch` | Volume (equivalised units) sold, split the same way. |
| `Price per Unit` | Blended average price per unit across both merch states. |
| `Price per Unit No Merch` / `Any Merch` | Average price per unit within each merch state. |
| `Price per Volume` | Blended average price per volume unit. |
| `Price per Volume No Merch` / `Any Merch` | Average price per volume unit within each merch state. |
| `ACV Weighted Distribution No Merch` / `Any Merch` | % of All-Commodity-Volume weighted retail distribution — how widely the product was actually available, weighted by store size. The standard CPG distribution-breadth measure. |
| `Base Unit Sales` | Units that would have sold without promotional lift ("baseline"). |
| `Base Volume Sales` | Baseline in volume terms. |
| `Base Dollar Sales` | Baseline in dollar terms. |
| `Incremental Units` | Units attributable to promotion (actual − baseline). |
| `Incremental Volume` | Same, in volume terms. |
| `Incremental Dollars` | Same, in dollar terms. |

### Derived columns (added by `data_ingestion.py`)

| Column | Stored dtype | Derivation |
|---|---|---|
| `Year` | `int16` | Calendar year of the week-ending date. |
| `Week` | `int16` | ISO week number of the week-ending date — the seasonality signal. |
| `Brand` | `category` | First two whitespace tokens of `Product Description`, concatenated (`"BLUE BONNET 16OZ"` → `BLUEBONNET`). Carried over unchanged from the original notebook so results stay comparable to the earlier report. It is a heuristic, not an official brand mapping — see `MODEL_CARD.md`. |

## Which columns may be used as model features

This is the single most important distinction in the project, because getting it
wrong is what produced the original ~81% accuracy claim.

**Permitted features** (`features.py::FEATURE_COLS`) — 8 numeric + 2 categorical:
`Price per Unit No Merch`, `Price per Unit Any Merch`, `Price per Volume No Merch`,
`Price per Volume Any Merch`, `ACV Weighted Distribution No Merch`,
`ACV Weighted Distribution Any Merch`, `Year`, `Week`, `Brand`, `Geography`.

**Excluded as leakage** (`features.py::LEAKAGE_COLS`): every `Base *` column,
every `Incremental *` column, all `Dollar Sales` and `Volume Sales` columns,
`Unit Sales Any Merch`, and the blended `Price per Unit` / `Price per Volume`.

The reason is arithmetic, not caution:
`Unit Sales No Merch = Base Unit Sales + Incremental Units`, and
`Dollar Sales = Units × Price`. In this dataset volume and units are the same
measure on a different scale. Any model given `Base Volume Sales` is being handed
the answer and asked to add to it — it will report a very high R² that reflects
nothing but that identity, and it will fail immediately in production because at
prediction time nobody knows the baseline or the incremental split for a week
that has not happened yet. `MODEL_CARD.md` carries the honest numbers.

## Supplementary files (not used by the model)

`Conagra_Data_Files/Data/*.xlsx` holds adjacent categories (cooking oils, cooking
sprays, panel/buyer data, product attribute files). These are out of scope: the
target category is tablespreads, and the panel data is a different grain
(household, not region-week) that cannot be joined to it without an entirely
separate modeling design.
