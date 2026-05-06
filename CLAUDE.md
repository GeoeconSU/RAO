# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RAO v2.1 is a quantitative country-scoring model for emerging-market investment opportunity. It produces a 1–10 RAO (Risk Adjusted Opportunity) score per country by running a 14-stage pipeline across World Bank, IMF, and CSV data sources. All model code lives in `Model V2.1/`. Scored output CSVs land in `Data/`.

## Running the Pipeline

All commands should be run from inside `Model V2.1/`:

```bash
cd "Model V2.1"
python run.py        # full pipeline: renames CSV, scores all countries, saves output
```

`run.py` is a flat script (not a module). It:
1. Renames columns in `../Data/var_pivot_clean.csv` → `../Data/var_pivot_renamed.csv`
2. Loads all modules, reloads them (Jupyter-style), runs `run_rao_pipeline()`
3. Calls `apply_ml_overlay()` and saves `../Data/RAO_scores_YYYY_MM_DD.csv`

To run the pipeline programmatically (e.g., from a notebook or REPL):

```python
import sys; sys.path.insert(0, ".")
from rao_pipeline import run_rao_pipeline
from rao_config import COMMERCIAL_DEFAULT

results, summary = run_rao_pipeline(
    countries=countries_filtered,   # list of ISO3 codes
    target_year=2023,
    config=COMMERCIAL_DEFAULT,
    csv_override_path="../Data/var_pivot_renamed.csv",
    fetch_live=False,               # True to hit WB + IMF APIs
)
```

To pull a single country score card from `results`:
```python
match = [r for r in results if r.country == "VNM"]
print(match[0].score_card())
```

## Required Packages

```
pandas numpy scipy scikit-learn wbgapi requests
```

`run.py` auto-installs missing packages at startup via pip.

## Architecture

### Pipeline flow (`rao_pipeline.py` → `run_rao_pipeline`)

| Stage | What happens | Module |
|-------|-------------|--------|
| 1 | Config validation | `rao_config` |
| 2–4 | Data ingestion: WB → IMF → CSV override, then fused | `rao_data` |
| 5 | Panel construction + KNN imputation | `rao_data` |
| 6 | Z-score + sigmoid normalisation to [0,1] | `rao_scoring` |
| 7 | Trajectory (OLS slope × R²) per variable | `rao_scoring` |
| 8 | Takeoff Detection Module: SIS + DGI + RVS → TOS | `rao_tdm` |
| 9 | Pillar scoring (A/B/C/D) | `rao_scoring` |
| 10 | K-Means archetype clustering | `rao_ml` |
| 11–12 | Bayesian Monte Carlo (1000 sims) → CVaR25 | `rao_scoring` |
| 13 | Isolation Forest anomaly detection | `rao_ml` |
| 14 | Final score assembly + regime classification | `rao_scoring`, `rao_ml` |

After the pipeline, `apply_ml_overlay()` in `rao_ml_overlay.py` adds three more layers:
- **Layer 1**: Trajectory classification (structural / cyclical / recovery_bounce / noise) with Pillar D discounts
- **Layer 2**: Universe integrity flags (conflict states, sanctions, low absorptive capacity, etc.) with score adjustments
- **Layer 3**: Market size bonus (0 to +1.5 pts, based on ln(GDP_total))

### Pillar definitions

| Pillar | Measures | Variables |
|--------|---------|-----------|
| A — Opportunity | Growth & investment inflows | `gdp_growth`, `export_growth`, `net_fdi_pct_gdp`, `gross_capital_formation`, `secondary_school_enrolment` |
| B — Fragility | Macro-financial stability & governance | `cpi_inflation`, `government_debt_pct_gdp`, `current_account_pct_gdp`, `external_debt_pct_gni`, `political_stability`, `rule_of_law`, `control_of_corruption` |
| C — Capacity | Absorptive capacity / structural constraints | `natural_resource_rents_pct_gdp`, `gdp_per_capita_growth`, `youth_unemployment`, `vulnerable_employment`, `regulatory_quality` |
| D — Trajectory | OLS slope × R² over trend window for key variables | Derived from gdp_growth, fdi, rule_of_law, debt, inflation |

`gdp_per_capita` is an AUX variable used for S-curve fitting (SIS) and a GDP size bias guard applied to ratio variables for small economies.

### Takeoff Detection Module (`rao_tdm.py`)

- **SIS** (S-curve Inflection Signal): fits a logistic curve to GDP/capita history; scores proximity to the inflection point
- **DGI** (Discovery Gap Index): measures gap between fundamental momentum (Pillars A+D) and external recognition (FDI recognition + export concentration HHI, optionally GDELT media)
- **RVS** (Reform Velocity Signal): WGI governance trajectory, optionally blended with GDELT reform news signal
- **TOS** = geometric mean of SIS, DGI, RVS using configurable weights

### Score assembly

```
base_rao  = 1 + 9 × CVaR25
blended   = (1 − tos_weight) × CVaR25 + tos_weight × TOS
rao_raw   = 1 + 9 × blended
rao_final = clip(rao_raw + event_adj − anomaly_haircut, 1, 10)
```

Then `apply_ml_overlay()` adjusts further (trajectory discounts, universe flags, market size bonus), producing the final `rao_score` in the output CSV.

### Configuration (`rao_config.py`)

`RAOConfig` controls all weights and parameters. Four preset profiles:

| Profile | opp / frag | tos_weight | Use case |
|---------|-----------|-----------|---------|
| `COMMERCIAL_DEFAULT` | 60/40 | 0.40 | Standard |
| `FRONTIER_MANDATE` | 55/45 | 0.35 | Frontier bias |
| `DEFENSIVE` | 65/35 | 0.15 | Risk-off |
| `HIGH_CONVICTION` | 52/48 | 0.40 | Pre-breakout focus |

Key toggles:
- `gdelt_active=False` — when True, activates GDELT event/media/reform signals in DGI, RVS, and event adjustments
- `fetch_live=False` in `run_rao_pipeline()` — skips live WB/IMF API calls and uses CSV only

### Data sources and priority

Source priority for data fusion: **IMF > World Bank > CSV > external**. When the same (country, year, variable) appears in multiple sources, the higher-priority source wins. CSV override is lowest priority but most commonly used (live API fetches are off by default).

Input CSV (`Data/var_pivot_clean.csv`) must have columns `country` (ISO3), `year`, plus the 17 pipeline variables. Column renaming (5 columns) happens at the top of `run.py` and produces `var_pivot_renamed.csv`.

### Output columns

The output CSV (`Data/RAO_scores_YYYY_MM_DD.csv`) contains:
- Core: `country`, `country_name`, `rao_score`, `rao_score_pre_overlay`, `overlay_adjustment`
- Pillars: `pillar_a/b/c/d`, `cvar25`, `volatility_band`
- TOS: `tos_score`, `tos_signal_label`, `sis_score`, `dgi_score`, `rvs_score`
- Classification: `regime`, `regime_label`, `archetype`, `trajectory_classification`
- Flags: `universe_flag`, `universe_flag_severity`, `requires_disclosure`
- Metadata: `data_confidence`, `imputed_vars`, `anomaly_flag`, `market_size_bonus`

### Country exclusion list

`run.py` hard-codes `EXCLUDE_COUNTRIES` — micro-states, territories, and statistical anomalies (Macao, Hong Kong, Puerto Rico, etc.) — applied before the pipeline runs.
