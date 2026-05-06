"""
run_longitudinal.py

Runs the RAO v2.1 pipeline for each year from START_YEAR to END_YEAR
and outputs two CSVs:
  - Long format:  one row per (country, year)
  - Wide format:  countries as rows, years as columns

Missing data: KNN imputation (k=5 neighbours). Each country's missing
variables are filled using the 5 most economically similar countries.
Data confidence (HIGH / MEDIUM / LOW) is recorded per country-year so
imputation-heavy scores can be identified.

Note: scores for 2010–2014 are less reliable because the trajectory
component (Pillar D) requires at least 3 years of history — which the
CSV only starts to provide from ~2013 onwards.
"""

import sys, warnings, importlib
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
START_YEAR  = 2010
END_YEAR    = 2024
RAW_CSV     = "../Data/var_pivot_clean.csv"
RENAMED_CSV = "../Data/var_pivot_renamed.csv"
OUTPUT_LONG = f"../Data/RAO_longitudinal_{START_YEAR}_{END_YEAR}.csv"
OUTPUT_WIDE = f"../Data/RAO_longitudinal_{START_YEAR}_{END_YEAR}_wide.csv"

EXCLUDE_COUNTRIES = [
    "MAC","ABW","WSM","KNA","PLW","DMA","TON","BRB","GRD","LCA",
    "VCT","ATG","NRU","TUV","KIR","MHL","FSM","SMR","AND","HKG",
    "PRI","WBG","TWN","FJI","MDV","BHS","SYC","VUT","SLB","BRN",
    "STP","CPV","COM","GNQ",
]
# ───────────────────────────────────────────────────────────────────────────────

# Rename columns to pipeline variable names
RENAME_MAP = {
    "current_account_balance_pct_gdp": "current_account_pct_gdp",
    "fdi_net_inflows_pct_gdp":         "net_fdi_pct_gdp",
    "government_debt_pct":             "government_debt_pct_gdp",
    "inflation_cpi":                   "cpi_inflation",
    "gross_capital_formation_pct_gdp": "gross_capital_formation",
}
df = pd.read_csv(RAW_CSV)
df = df.rename(columns=RENAME_MAP)
df.to_csv(RENAMED_CSV, index=False)

import rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result, rao_ml_overlay
for mod in [rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result, rao_ml_overlay]:
    importlib.reload(mod)

from rao_pipeline import run_rao_pipeline
from rao_config import COMMERCIAL_DEFAULT
from rao_ml_overlay import apply_ml_overlay

all_countries = df["country"].unique().tolist()
countries = [c for c in all_countries if c not in EXCLUDE_COUNTRIES]

print(f"Scoring universe : {len(countries)} countries")
print(f"Year range       : {START_YEAR} – {END_YEAR}")
print(f"Missing data     : KNN imputation (k={COMMERCIAL_DEFAULT.knn_neighbors})")
print("=" * 60)

records = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n[{year}]", end=" ", flush=True)
    try:
        results, summary = run_rao_pipeline(
            countries=countries,
            target_year=year,
            config=COMMERCIAL_DEFAULT,
            csv_override_path=RENAMED_CSV,
            fetch_live=False,
        )
        summary_enhanced = apply_ml_overlay(
            summary,
            verbose=False,
            raw_data_path=RAW_CSV,
        )

        for _, row in summary_enhanced.iterrows():
            records.append({
                "country":         row["country"],
                "country_name":    row["country_name"],
                "year":            year,
                "rao_score":       round(float(row["rao_score"]), 3),
                "data_confidence": row["data_confidence"],
            })

        scores = summary_enhanced["rao_score"]
        n_low = (summary_enhanced["data_confidence"] == "LOW").sum()
        print(f"{len(results)} countries | "
              f"mean={scores.mean():.2f}  min={scores.min():.2f}  max={scores.max():.2f} | "
              f"low-confidence: {n_low}")

    except Exception as e:
        print(f"ERROR — {e}")
        continue

# ── Long format ────────────────────────────────────────────────────────────────
long_df = pd.DataFrame(records).sort_values(["country", "year"]).reset_index(drop=True)
long_df.to_csv(OUTPUT_LONG, index=False)
print(f"\nLong format saved  → {OUTPUT_LONG}  ({len(long_df)} rows)")

# ── Wide format (country × year) ───────────────────────────────────────────────
wide_df = (
    long_df.pivot(index=["country", "country_name"], columns="year", values="rao_score")
    .reset_index()
)
wide_df.columns.name = None
wide_df.to_csv(OUTPUT_WIDE, index=False)
print(f"Wide format saved  → {OUTPUT_WIDE}  ({len(wide_df)} countries × {END_YEAR - START_YEAR + 1} years)")
