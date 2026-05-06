"""
run_export_normalised.py

Exports KNN-imputed, normalised variable scores for each country.
Normalisation: z-score across countries, then sigmoid → maps each variable to (0, 1).
Higher = better for all variables (direction is already flipped for negative indicators
like inflation and debt).

Outputs:
  - normalised_latest.csv   : latest year (2024) only, one row per country
  - normalised_allyears.csv : all years 2010–2024, one row per (country, year)
"""

import sys, warnings, importlib
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
START_YEAR   = 2010
END_YEAR     = 2024
LATEST_YEAR  = 2024
RAW_CSV      = "../Data/var_pivot_clean.csv"
RENAMED_CSV  = "../Data/var_pivot_renamed.csv"
OUT_LATEST   = "../Data/normalised_latest.csv"
OUT_ALLYEARS = "../Data/normalised_allyears.csv"

EXCLUDE_COUNTRIES = [
    "MAC","ABW","WSM","KNA","PLW","DMA","TON","BRB","GRD","LCA",
    "VCT","ATG","NRU","TUV","KIR","MHL","FSM","SMR","AND","HKG",
    "PRI","WBG","TWN","FJI","MDV","BHS","SYC","VUT","SLB","BRN",
    "STP","CPV","COM","GNQ",
]
# ───────────────────────────────────────────────────────────────────────────────

RENAME_MAP = {
    "current_account_balance_pct_gdp": "current_account_pct_gdp",
    "fdi_net_inflows_pct_gdp":         "net_fdi_pct_gdp",
    "government_debt_pct":             "government_debt_pct_gdp",
    "inflation_cpi":                   "cpi_inflation",
    "gross_capital_formation_pct_gdp": "gross_capital_formation",
}
df_raw = pd.read_csv(RAW_CSV)
df_raw = df_raw.rename(columns=RENAME_MAP)
df_raw.to_csv(RENAMED_CSV, index=False)

import rao_data, rao_scoring
for mod in [rao_data, rao_scoring]:
    importlib.reload(mod)

from rao_data import (
    load_csv_override, fuse_data_sources, build_panel,
    WB_INDICATORS, TRAJECTORY_VARS,
)
from rao_scoring import normalise_panel, compute_trajectory
from rao_config import COMMERCIAL_DEFAULT

all_countries = df_raw["country"].unique().tolist()
countries = [c for c in all_countries if c not in EXCLUDE_COUNTRIES]

# Variable direction map (used by normalise_panel)
directions = {var: d for var, (_, d, _) in WB_INDICATORS.items()}

# Column labels: pipeline vars (17) + trajectory vars (5)
PIPELINE_VARS  = [v for v, (_, _, p) in WB_INDICATORS.items() if p != "AUX"]
TRAJECTORY_COLS = list(TRAJECTORY_VARS.keys())

# Country name lookup (extended from pipeline)
from rao_pipeline import COUNTRY_NAMES

def score_year(target_year: int) -> pd.DataFrame:
    """Return a DataFrame of normalised variable scores for one target year."""
    df_csv = load_csv_override(RENAMED_CSV)
    fused  = fuse_data_sources(df_csv)

    panel, history, _, confidence_map = build_panel(
        fused, countries, target_year,
        lookback_years=COMMERCIAL_DEFAULT.lookback_years,
        knn_neighbors=COMMERCIAL_DEFAULT.knn_neighbors,
    )

    normed     = normalise_panel(panel, directions)
    trajectory = compute_trajectory(history, trend_window=COMMERCIAL_DEFAULT.trend_window)

    rows = []
    for country in countries:
        row = {
            "country":         country,
            "country_name":    COUNTRY_NAMES.get(country, country),
            "year":            target_year,
            "data_confidence": confidence_map.get(country, "MEDIUM"),
        }
        for var in PIPELINE_VARS:
            row[var] = round(float(normed.loc[country, var]), 4) if var in normed.columns else None
        for var in TRAJECTORY_COLS:
            row[var] = round(float(trajectory.loc[country, var]), 4) \
                if var in trajectory.columns and country in trajectory.index else None
        rows.append(row)

    return pd.DataFrame(rows)


# ── Latest year only ───────────────────────────────────────────────────────────
print(f"Scoring {LATEST_YEAR}...", end=" ", flush=True)
df_latest = score_year(LATEST_YEAR)
df_latest.to_csv(OUT_LATEST, index=False)
print(f"saved → {OUT_LATEST}  ({len(df_latest)} countries, {len(df_latest.columns)-4} variables)")

# ── All years ──────────────────────────────────────────────────────────────────
print(f"\nScoring {START_YEAR}–{END_YEAR}:")
frames = []
for year in range(START_YEAR, END_YEAR + 1):
    print(f"  {year}...", end=" ", flush=True)
    frames.append(score_year(year))
    print("done")

df_all = pd.concat(frames, ignore_index=True).sort_values(["country", "year"]).reset_index(drop=True)
df_all.to_csv(OUT_ALLYEARS, index=False)
print(f"\nSaved → {OUT_ALLYEARS}  ({len(df_all)} rows, {len(df_all.columns)-4} variables)")

# ── Preview ────────────────────────────────────────────────────────────────────
print("\n=== Sample (latest year, first 5 countries) ===")
print(df_latest.head(5).iloc[:, :8].to_string(index=False))

print("\n=== Variable coverage (non-null) in latest year ===")
var_cols = PIPELINE_VARS + TRAJECTORY_COLS
coverage = df_latest[var_cols].notna().sum().sort_values()
print(coverage.to_string())
