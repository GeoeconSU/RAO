import sys, importlib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scipy import stats

import subprocess
required = ['wbgapi', 'scikit-learn', 'scipy', 'requests']
for pkg in required:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print(" Setup complete")


CSV_PATH = "../Data/var_pivot_clean.csv"
TARGET_YEAR = 2023

df = pd.read_csv(CSV_PATH)

RENAME_MAP = {
    "current_account_balance_pct_gdp": "current_account_pct_gdp",
    "fdi_net_inflows_pct_gdp":         "net_fdi_pct_gdp",
    "government_debt_pct":             "government_debt_pct_gdp",
    "inflation_cpi":                   "cpi_inflation",
    "gross_capital_formation_pct_gdp": "gross_capital_formation",
}
df = df.rename(columns=RENAME_MAP)

PIPELINE_VARS = [
    'gdp_growth', 'export_growth', 'net_fdi_pct_gdp', 'gross_capital_formation',
    'secondary_school_enrolment', 'cpi_inflation', 'government_debt_pct_gdp',
    'current_account_pct_gdp', 'external_debt_pct_gni', 'political_stability',
    'rule_of_law', 'control_of_corruption', 'natural_resource_rents_pct_gdp',
    'gdp_per_capita_growth', 'youth_unemployment', 'vulnerable_employment',
    'regulatory_quality'
]

missing = [v for v in PIPELINE_VARS if v not in df.columns]
if missing:
    print(f" Missing variables: {missing}")
    print("   Check RENAME_MAP above  one of your CSV column names may differ")
    raise Exception("Column mismatch  fix before proceeding")

RENAMED_PATH = CSV_PATH.replace("var_pivot_clean.csv", "var_pivot_renamed.csv")
df.to_csv(RENAMED_PATH, index=False)

print(f" All 17 variables present")
print(f" Saved renamed CSV to: {RENAMED_PATH}")
print(f"   Shape: {df.shape[0]} rows  {df.shape[1]} columns")
print(f"   Years: {df['year'].min()}  {df['year'].max()}")
print(f"   Countries: {df['country'].nunique()}")


import sys, importlib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import date

df = pd.read_csv("../Data/var_pivot_clean.csv")
df = df.rename(columns={
    "current_account_balance_pct_gdp": "current_account_pct_gdp",
    "fdi_net_inflows_pct_gdp":         "net_fdi_pct_gdp",
    "government_debt_pct":             "government_debt_pct_gdp",
    "inflation_cpi":                   "cpi_inflation",
    "gross_capital_formation_pct_gdp": "gross_capital_formation",
})
df.to_csv("../Data/var_pivot_renamed.csv", index=False)

EXCLUDE_COUNTRIES = [
    "MAC","ABW","WSM","KNA","PLW","DMA","TON","BRB","GRD","LCA",
    "VCT","ATG","NRU","TUV","KIR","MHL","FSM","SMR","AND","HKG",
    "PRI","WBG","TWN","FJI","MDV","BHS","SYC","VUT","SLB","BRN",
    "STP","CPV","COM","GNQ",
]
countries = df["country"].unique().tolist()
countries_filtered = [c for c in countries if c not in EXCLUDE_COUNTRIES]

import rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result, rao_ml_overlay
for mod in [rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result, rao_ml_overlay]:
    importlib.reload(mod)
from rao_pipeline import run_rao_pipeline
from rao_config import COMMERCIAL_DEFAULT
print(" Modules reloaded")

results, summary = run_rao_pipeline(
    countries=countries_filtered,
    target_year=2023,
    config=COMMERCIAL_DEFAULT,
    csv_override_path="../Data/var_pivot_renamed.csv",
    fetch_live=False,
)
print(f" Pipeline: {len(results)} countries | "
      f"Range: {summary.rao_score.min():.2f}{summary.rao_score.max():.2f}")

from rao_ml_overlay import apply_ml_overlay
summary_enhanced = apply_ml_overlay(
    summary,
    verbose=True,
    raw_data_path="../Data/var_pivot_clean.csv",
)

output_path = f"../Data/RAO_scores_{date.today().strftime('%Y_%m_%d')}.csv"
assert 'universe_flag' in summary_enhanced.columns, "Overlay missing"
assert 'market_size_bonus' in summary_enhanced.columns, "Layer 3 missing"
summary_enhanced.to_csv(output_path, index=False)
print(f" Saved: {output_path} | {len(summary_enhanced.columns)} columns")

summary_enhanced[["country","country_name","rao_score","rao_score_pre_overlay",
                   "market_size_bonus","overlay_adjustment","regime",
                   "tos_signal_label","universe_flag"]].head(25)


from rao_ml_overlay import apply_ml_overlay, overlay_score_card

summary_enhanced = apply_ml_overlay(summary, verbose=True,
    raw_data_path="../Data/var_pivot_clean.csv")

print()
print("POST-OVERLAY TOP 20:")
print(summary_enhanced[["country","country_name","rao_score",
                         "rao_score_pre_overlay","overlay_adjustment",
                         "regime","archetype","tos_score",
                         "trajectory_classification"]].head(20).to_string(index=False))


EXCLUDE_COUNTRIES = [
    "MAC",
    "ABW",
    "WSM",
    "KNA",
    "PLW",
    "DMA",
    "TON",
    "BRB",
    "GRD",
    "LCA",
    "VCT",
    "ATG",
    "NRU",
    "TUV",
    "KIR",
    "MHL",
    "FSM",
    "SMR",
    "AND",
    "HKG",
    "PRI",
    "WBG",
    "TWN",
    "FJI",
    "MDV",
    "BHS",
    "SYC",
    "VUT",
    "SLB",
    "BRN",
    "STP",
    "CPV",
    "COM",
    "GNQ",
]

all_countries = df["country"].unique().tolist()
countries_filtered = [c for c in all_countries if c not in EXCLUDE_COUNTRIES]

print(f" Scoring universe: {len(countries_filtered)} countries")
print(f"   (Excluded {len(all_countries) - len(countries_filtered)} micro-states/territories)")


import rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result
for mod in [rao_scoring, rao_pipeline, rao_data, rao_tdm, rao_ml, rao_result]:
    importlib.reload(mod)
from rao_pipeline import run_rao_pipeline
from rao_config import COMMERCIAL_DEFAULT
print("Modules loaded  running pipeline...")

results, summary = run_rao_pipeline(
    countries=countries_filtered,
    target_year=TARGET_YEAR,
    config=COMMERCIAL_DEFAULT,
    csv_override_path=RENAMED_PATH,
    fetch_live=False,
)

print(f"\n Pipeline complete  {len(results)} countries scored")
print(f"   Score range: {summary['rao_score'].min():.2f}  {summary['rao_score'].max():.2f}")
print(f"   Mean: {summary['rao_score'].mean():.2f}  |  Std: {summary['rao_score'].std():.2f}")


print("=" * 65)
print("TOP 20  Highest RAO Scores")
print("=" * 65)
print(summary[["country", "country_name", "rao_score", "regime",
               "archetype", "pillar_a", "pillar_b",
               "pillar_c", "pillar_d"]].head(20).to_string(index=False))

print("\n" + "=" * 65)
print("BOTTOM 10  Lowest RAO Scores")
print("=" * 65)
print(summary[["country", "country_name", "rao_score",
               "regime", "archetype"]].tail(10).to_string(index=False))

print("\n" + "=" * 65)
print("SCORE DISTRIBUTION")
print("=" * 65)
print(summary["rao_score"].describe().round(3))
print()
for p in [10, 25, 40, 60, 75, 90]:
    print(f"  P{p}: {np.percentile(summary['rao_score'], p):.2f}")

print("\n" + "=" * 65)
print("REGIME BREAKDOWN")
print("=" * 65)
print(summary["regime_label"].value_counts().to_string())

print("\n" + "=" * 65)
print("ARCHETYPE BREAKDOWN")
print("=" * 65)
print(summary["archetype"].value_counts().to_string())


COUNTRY = "VNM"

match = [r for r in results if r.country == COUNTRY]
if match:
    print(match[0].score_card())
else:
    print(f"Country {COUNTRY} not found. Available codes:")
    print(summary["country"].tolist())


MANUAL_OVERRIDES = {
}

if MANUAL_OVERRIDES:
    summary_enhanced = summary.copy()
    for iso3, adjustment in MANUAL_OVERRIDES.items():
        if iso3 in summary_enhanced["country"].values:
            old = summary_enhanced.loc[summary_enhanced["country"] == iso3, "rao_score"].values[0]
            new = float(np.clip(old + adjustment, 1.0, 10.0))
            summary_enhanced.loc[summary_enhanced["country"] == iso3, "rao_score"] = new
            summary_enhanced.loc[summary_enhanced["country"] == iso3, "event_adjustment"] += adjustment
            print(f"  {iso3}: {old:.2f}  {new:.2f} (adjustment: {adjustment:+.2f})")
    summary_enhanced = summary_enhanced.sort_values("rao_score", ascending=False).reset_index(drop=True)
    print(f"\n {len(MANUAL_OVERRIDES)} manual override(s) applied")
else:
    summary_enhanced = summary.copy()
    print("No manual overrides active  using raw model scores")

