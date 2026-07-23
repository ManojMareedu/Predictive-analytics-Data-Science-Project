"""Load the yearly IRI POS tablespreads Excel files into one cleaned Parquet.

Business grain: Geography (region) x Product (UPC) x Week. The raw files are the
single most expensive thing to touch, so we parse each year once, downcast, and
persist a single deduplicated Parquet that everything downstream reads instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

YEARS = [2018, 2019, 2020, 2021, 2022]
SHEET = "Tablespreads_POS"
TOTAL_US = "Total US - Multi Outlet + Conv"

# 24 raw columns; numeric ones downcast to float32 (values fit comfortably).
ID_COLS = ["Geography", "Time", "Product Description", "UPC 13 digit"]
NUMERIC_COLS = [
    "Dollar Sales No Merch",
    "Dollar Sales Any Merch",
    "Unit Sales No Merch",
    "Unit Sales Any Merch",
    "Volume Sales No Merch",
    "Volume Sales Any Merch",
    "Price per Unit",
    "Price per Unit No Merch",
    "Price per Unit Any Merch",
    "Price per Volume",
    "Price per Volume No Merch",
    "Price per Volume Any Merch",
    "ACV Weighted Distribution No Merch",
    "ACV Weighted Distribution Any Merch",
    "Base Unit Sales",
    "Base Volume Sales",
    "Base Dollar Sales",
    "Incremental Units",
    "Incremental Volume",
    "Incremental Dollars",
]


def brand(product_description: str) -> str:
    """First two whitespace tokens joined, e.g. 'BLUE BONNET ...' -> 'BLUEBONNET'.

    Mirrors the notebook's brand() convention so results stay comparable.
    """
    parts = str(product_description).split()
    return "".join(parts[:2]) if len(parts) >= 2 else "".join(parts)


def load_year(path: str | Path) -> pd.DataFrame:
    """Read one year's file, standardize schema, downcast, keep only real regions."""
    df = pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")
    # 2022 names the column "Product"; every other year uses "Product Description".
    df = df.rename(columns={"Product": "Product Description"})

    df = df.drop_duplicates()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype("float32")

    df["UPC 13 digit"] = df["UPC 13 digit"].astype(str)
    # Parse the week-ending date once; both calendar features derive from it.
    week_ending = pd.to_datetime(df["Time"], format="Week Ending %m-%d-%y")
    df["Year"] = week_ending.dt.year.astype("int16")
    df["Week"] = week_ending.dt.isocalendar().week.astype("int16")
    df["Brand"] = df["Product Description"].map(brand)

    # Drop the national roll-up: we model individual regions, and keeping the
    # aggregate row inflates correlations and double-counts sales.
    df = df[df["Geography"] != TOTAL_US].copy()

    for col in ("Geography", "Brand"):
        df[col] = df[col].astype("category")
    return df


def build_dataset(
    raw_dir: str | Path = ".",
    out_path: str | Path = "data/processed/tablespreads.parquet",
) -> Path:
    """Concatenate all years into one cleaned Parquet. Returns the output path."""
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ponytail: one raw workbook open at a time (the memory spike); cleaned
    # float32 frames (~30MB/year) accumulate in a list. Switch to an incremental
    # pyarrow ParquetWriter only if total years grow past what RAM holds.
    frames = []
    for year in YEARS:
        path = raw_dir / f"IRI_POS_Tablespreads_{year}.xlsx"
        part = load_year(path)
        print(f"  {year}: {len(part):,} rows")
        frames.append(part)

    # category dtype doesn't survive a plain concat cleanly; rebuild after.
    df = pd.concat(frames, ignore_index=True)
    for col in ("Geography", "Brand"):
        df[col] = df[col].astype("category")

    df.to_parquet(out_path, index=False)
    print(
        f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)"
    )
    return out_path


def audit_raw(raw_dir: str | Path = ".", out_path: str | Path = "data/processed/quality_report.json") -> dict:
    """Read each raw workbook once and record what is actually in it.

    Deliberately reads the *raw* files, not the cleaned Parquet -- the whole point
    is to measure what we dropped/filled, which the cleaned artifact no longer
    shows. Records per year: row/column counts, exact column names (so year-to-year
    schema drift is caught, not assumed), duplicate rows before and after dedup,
    null counts per column, and the Total US roll-up rows removed.
    """
    import json

    raw_dir, out_path = Path(raw_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_year, all_cols = {}, {}

    for year in YEARS:
        df = pd.read_excel(raw_dir / f"IRI_POS_Tablespreads_{year}.xlsx", sheet_name=SHEET, engine="openpyxl")
        all_cols[year] = list(df.columns)
        n_raw = len(df)
        n_dedup = len(df.drop_duplicates())
        nulls = df.isna().sum()
        per_year[year] = {
            "rows_raw": int(n_raw),
            "rows_after_dedup": int(n_dedup),
            "duplicate_rows_removed": int(n_raw - n_dedup),
            "n_columns": int(df.shape[1]),
            "total_us_rows_dropped": int((df["Geography"] == TOTAL_US).sum()),
            "null_counts": {c: int(v) for c, v in nulls.items() if v > 0},
            "null_rate_pct": {c: round(100 * v / n_raw, 4) for c, v in nulls.items() if v > 0},
        }
        print(f"  {year}: {n_raw:,} rows, {n_raw - n_dedup:,} dups, {int(nulls.sum()):,} nulls", flush=True)

    # Schema drift: compare every year against 2018 as the reference layout.
    ref_year = YEARS[0]
    ref = set(all_cols[ref_year])
    drift = {
        str(y): {
            f"missing_vs_{ref_year}": sorted(ref - set(cols)),
            f"extra_vs_{ref_year}": sorted(set(cols) - ref),
        }
        for y, cols in all_cols.items()
        if set(cols) != ref
    }

    report = {
        "reference_year": ref_year,
        "columns_reference": all_cols[ref_year],
        "schema_drift": drift,
        "per_year": {str(k): v for k, v in per_year.items()},
        "totals": {
            "rows_raw": sum(v["rows_raw"] for v in per_year.values()),
            "duplicate_rows_removed": sum(v["duplicate_rows_removed"] for v in per_year.values()),
            "total_us_rows_dropped": sum(v["total_us_rows_dropped"] for v in per_year.values()),
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote quality report -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=".")
    ap.add_argument("--out", default="data/processed/tablespreads.parquet")
    ap.add_argument("--audit", action="store_true", help="write the data-quality report instead of building")
    args = ap.parse_args()
    if args.audit:
        audit_raw(args.raw_dir)
    else:
        build_dataset(args.raw_dir, args.out)
