"""
run_production_panel.py

Generates a single production-grade CSV with RAO scores, pillar scores, TOS components,
classifications, and KNN-imputed normalised variables for all countries across all years.

Output: Data/RAO_production_panel_YYYY_MM_DD.csv
One row per (country, year). Column groups:

  1.  Identifiers           : country, country_name, year
  2.  Final score           : rao_score, rao_score_pre_overlay, overlay_adjustment, market_size_bonus
  3.  Risk                  : cvar25, volatility_band
  4.  Pillars               : pillar_opportunity, pillar_fragility, pillar_absorptive_capacity, pillar_trajectory
  5.  TOS components        : tos_score, tos_signal_label, sis_score, dgi_score, rvs_score
  6.  Classification        : regime, regime_label, archetype, trajectory_classification
  7.  Norm vars — Opportunity       (5 vars, prefix norm_)
  8.  Norm vars — Fragility         (7 vars, prefix norm_)
  9.  Norm vars — Absorptive Cap.   (5 vars, prefix norm_)
  10. Norm vars — Trajectory        (5 vars, prefix norm_)
  11. Flags / metadata      : universe_flag, universe_flag_severity, data_confidence,
                              imputed_vars, anomaly_flag
"""

import sys, warnings, importlib
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import date

# ── Configuration ──────────────────────────────────────────────────────────────
START_YEAR  = 2010
END_YEAR    = 2024
RAW_CSV     = "../Data/var_pivot_clean.csv"
RENAMED_CSV = "../Data/var_pivot_renamed.csv"
OUTPUT_PATH = f"../Data/RAO_production_panel_{date.today().strftime('%Y_%m_%d')}.csv"

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

import rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result, rao_ml_overlay
for mod in [rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result, rao_ml_overlay]:
    importlib.reload(mod)

from rao_pipeline import run_rao_pipeline, COUNTRY_NAMES
from rao_config import COMMERCIAL_DEFAULT
from rao_ml_overlay import apply_ml_overlay
from rao_data import (
    load_csv_override, fuse_data_sources, build_panel,
    WB_INDICATORS, TRAJECTORY_VARS,
)
from rao_scoring import normalise_panel, compute_trajectory

all_countries = df_raw["country"].unique().tolist()
countries = [c for c in all_countries if c not in EXCLUDE_COUNTRIES]

# Variable groups (preserving pillar order from rao_data.py)
PILLAR_A_VARS = [v for v, (_, _, p) in WB_INDICATORS.items() if p == "A"]
PILLAR_B_VARS = [v for v, (_, _, p) in WB_INDICATORS.items() if p == "B"]
PILLAR_C_VARS = [v for v, (_, _, p) in WB_INDICATORS.items() if p == "C"]
TRAJECTORY_COLS = list(TRAJECTORY_VARS.keys())
DIRECTIONS = {var: d for var, (_, d, _) in WB_INDICATORS.items()}

print(f"Production panel: {len(countries)} countries × {END_YEAR - START_YEAR + 1} years")
print(f"Year range : {START_YEAR}–{END_YEAR}")
print(f"Output     : {OUTPUT_PATH}")
print("=" * 60)

all_frames = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n[{year}]", end=" ", flush=True)
    try:
        # ── 1. Full pipeline + overlay (scores, pillars, TOS, classification) ──
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
        score_idx = summary_enhanced.set_index("country")

        # ── 2. Normalised variables (KNN-imputed, sigmoid-normalised) ──────────
        df_csv = load_csv_override(RENAMED_CSV)
        fused  = fuse_data_sources(df_csv)
        panel, history, _, _ = build_panel(
            fused, countries, year,
            lookback_years=COMMERCIAL_DEFAULT.lookback_years,
            knn_neighbors=COMMERCIAL_DEFAULT.knn_neighbors,
        )
        normed     = normalise_panel(panel, DIRECTIONS)
        trajectory = compute_trajectory(history, trend_window=COMMERCIAL_DEFAULT.trend_window)

        # ── 3. Assemble one row per country ───────────────────────────────────
        rows = []
        for country in countries:
            if country not in score_idx.index:
                continue

            s = score_idx.loc[country]

            def _f(key, decimals=4):
                """Safely round a float field from the score row."""
                try:
                    v = float(s.get(key, np.nan))
                    return round(v, decimals) if not np.isnan(v) else None
                except (TypeError, ValueError):
                    return None

            def _s(key):
                """Return a string/categorical field from the score row."""
                v = s.get(key, None)
                return v if pd.notna(v) else None

            def _norm(df, var):
                """Return a normalised variable value, None if missing."""
                try:
                    v = float(df.loc[country, var]) if (var in df.columns and country in df.index) else np.nan
                    return round(v, 4) if not np.isnan(v) else None
                except (TypeError, ValueError):
                    return None

            row = {
                # 1. Identifiers
                "country":      country,
                "country_name": COUNTRY_NAMES.get(country, country),
                "year":         year,

                # 2. Final score
                "rao_score":             _f("rao_score", 3),
                "rao_score_pre_overlay": _f("rao_score_pre_overlay", 3),
                "overlay_adjustment":    _f("overlay_adjustment", 3),
                "market_size_bonus":     _f("market_size_bonus", 3),

                # 3. Risk
                "cvar25":         _f("cvar25"),
                "volatility_band": _s("volatility_band"),

                # 4. Pillars (full names)
                "pillar_opportunity":         _f("pillar_a"),
                "pillar_fragility":           _f("pillar_b"),
                "pillar_absorptive_capacity": _f("pillar_c"),
                "pillar_trajectory":          _f("pillar_d"),

                # 5. TOS components
                "tos_score":        _f("tos_score"),
                "tos_signal_label": _s("tos_signal_label"),
                "sis_score":        _f("sis_score"),
                "dgi_score":        _f("dgi_score"),
                "rvs_score":        _f("rvs_score"),

                # 6. Classification
                "regime":                    _s("regime"),
                "regime_label":              _s("regime_label"),
                "archetype":                 _s("archetype"),
                "trajectory_classification": _s("trajectory_classification"),
            }

            # 7–9. Normalised pipeline vars grouped by pillar
            for var in PILLAR_A_VARS + PILLAR_B_VARS + PILLAR_C_VARS:
                row[f"norm_{var}"] = _norm(normed, var)

            # 10. Normalised trajectory vars
            for var in TRAJECTORY_COLS:
                row[f"norm_{var}"] = _norm(trajectory, var)

            # 11. Flags / metadata
            row["universe_flag"]          = _s("universe_flag")
            row["universe_flag_severity"] = _s("universe_flag_severity")
            row["data_confidence"]        = _s("data_confidence")
            row["imputed_vars"]           = _s("imputed_vars")
            row["anomaly_flag"]           = _s("anomaly_flag")

            rows.append(row)

        year_df = pd.DataFrame(rows)
        all_frames.append(year_df)

        scores = summary_enhanced["rao_score"]
        print(
            f"{len(rows)} countries | "
            f"mean={scores.mean():.2f}  "
            f"range=[{scores.min():.2f}, {scores.max():.2f}]"
        )

    except Exception as e:
        import traceback
        print(f"ERROR — {e}")
        traceback.print_exc()
        continue

# ── Assemble, sort, save ───────────────────────────────────────────────────────
panel_df = (
    pd.concat(all_frames, ignore_index=True)
    .sort_values(["country", "year"])
    .reset_index(drop=True)
)

panel_df.to_csv(OUTPUT_PATH, index=False)

norm_cols = [c for c in panel_df.columns if c.startswith("norm_")]

print(f"\n{'=' * 60}")
print(f"Saved → {OUTPUT_PATH}")
print(f"  Rows    : {len(panel_df):,}  "
      f"({panel_df['country'].nunique()} countries × {panel_df['year'].nunique()} years)")
print(f"  Columns : {len(panel_df.columns)}")
print()
print("Column summary:")
print(f"  [1]  Identifiers          : country, country_name, year")
print(f"  [2]  Final score          : rao_score, rao_score_pre_overlay, overlay_adjustment, market_size_bonus")
print(f"  [3]  Risk                 : cvar25, volatility_band")
print(f"  [4]  Pillars              : pillar_opportunity, pillar_fragility, pillar_absorptive_capacity, pillar_trajectory")
print(f"  [5]  TOS                  : tos_score, tos_signal_label, sis_score, dgi_score, rvs_score")
print(f"  [6]  Classification       : regime, regime_label, archetype, trajectory_classification")
print(f"  [7]  Norm — Opportunity   : {', '.join('norm_'+v for v in PILLAR_A_VARS)}")
print(f"  [8]  Norm — Fragility     : {', '.join('norm_'+v for v in PILLAR_B_VARS)}")
print(f"  [9]  Norm — Absorptive    : {', '.join('norm_'+v for v in PILLAR_C_VARS)}")
print(f"  [10] Norm — Trajectory    : {', '.join('norm_'+v for v in TRAJECTORY_COLS)}")
print(f"  [11] Flags / metadata     : universe_flag, universe_flag_severity, data_confidence, imputed_vars, anomaly_flag")
