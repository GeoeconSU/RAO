import logging
import sys
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from rao_config import RAOConfig, COMMERCIAL_DEFAULT
from rao_data import (
    fetch_worldbank_data_wbgapi,
    fetch_imf_data,
    load_csv_override,
    fuse_data_sources,
    build_panel,
    WB_INDICATORS,
    TRAJECTORY_VARS,
)
from rao_scoring import (
    normalise_panel,
    compute_trajectory,
    score_pillar_a,
    score_pillar_b,
    score_pillar_c,
    score_pillar_d,
    run_bayesian_simulation,
    assemble_final_score,
    PILLAR_A_PRIORS,
    PILLAR_B_PRIORS,
)
from rao_tdm import (
    compute_sis,
    compute_dgi,
    compute_rvs,
    compute_tos,
    tos_label,
)
from rao_ml import (
    run_anomaly_detection,
    compute_archetypes,
    classify_regime,
    compute_event_adjustment,
    compute_fred_adjustment,
)
from rao_result import RAOResult, results_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

COUNTRY_NAMES = {
    "VNM": "Vietnam", "BGD": "Bangladesh", "ETH": "Ethiopia", "KEN": "Kenya",
    "GHA": "Ghana", "MAR": "Morocco", "COL": "Colombia", "PER": "Peru",
    "NGA": "Nigeria", "TZA": "Tanzania", "RWA": "Rwanda", "SEN": "Senegal",
    "CIV": "Cte d'Ivoire", "MOZ": "Mozambique", "TUN": "Tunisia",
    "EGY": "Egypt", "PAK": "Pakistan", "BGD": "Bangladesh", "LKA": "Sri Lanka",
    "PHL": "Philippines", "IDN": "Indonesia", "MYS": "Malaysia",
    "THA": "Thailand", "IND": "India", "CHN": "China", "BRA": "Brazil",
    "MEX": "Mexico", "ARG": "Argentina", "TUR": "Trkiye", "ZAF": "South Africa",
    "UKR": "Ukraine", "UZB": "Uzbekistan", "KAZ": "Kazakhstan",
    "MNG": "Mongolia", "GTM": "Guatemala", "HND": "Honduras",
}

def run_rao_pipeline(
    countries: list[str],
    target_year: Optional[int] = None,
    config: RAOConfig = COMMERCIAL_DEFAULT,
    csv_override_path: Optional[str] = None,
    gdelt_events: Optional[dict] = None,
    gdelt_media_coverage: Optional[dict] = None,
    gdelt_reform_signal: Optional[dict] = None,
    hhi_export: Optional[dict] = None,
    fred_conditions: Optional[dict] = None,
    fetch_live: bool = True,
) -> tuple[list[RAOResult], pd.DataFrame]:

    logger.info("Stage 1: Configuration validation")
    config.validate()
    logger.info(f"  Profile: {config.profile_label}  "
                f"Opp={config.opp_weight:.0%} / Frag={config.frag_weight:.0%} / "
                f"TOS={config.tos_weight:.0%}  GDELT={config.gdelt_active}")

    if target_year is None:
        target_year = date.today().year - 1
        logger.info(f"  Target year defaulted to {target_year}")

    start_year = target_year - config.lookback_years + 1

    logger.info(f"Stage 2: Data ingestion ({start_year}{target_year})")
    df_wb = pd.DataFrame()
    df_imf = pd.DataFrame()

    if fetch_live:
        logger.info("  Fetching World Bank data (WDI + WGI)...")
        df_wb = fetch_worldbank_data_wbgapi(countries, start_year, target_year)
        logger.info(f"  WB: {len(df_wb)} records")

        logger.info("  Fetching IMF DataMapper data...")
        df_imf = fetch_imf_data(countries, start_year, target_year)
        logger.info(f"  IMF: {len(df_imf)} records")
    else:
        logger.info("  fetch_live=False: skipping live API calls")

    df_csv = pd.DataFrame()
    if csv_override_path:
        logger.info(f"Stage 3: Loading CSV override: {csv_override_path}")
        df_csv = load_csv_override(csv_override_path)

    logger.info("Stage 4: Data fusion (recency + source priority)")
    fused = fuse_data_sources(df_wb, df_imf, df_csv)

    if fused.empty and not fetch_live:
        logger.warning("No data available (fetch_live=False and no CSV). "
                       "Pipeline will produce neutral placeholder scores.")

    logger.info("Stage 5: Panel construction + missing data handling")
    panel, history, imputation_log, confidence_map = build_panel(
        fused, countries, target_year,
        lookback_years=config.lookback_years,
        knn_neighbors=config.knn_neighbors,
    )

    provenance = {}
    if not fused.empty:
        prov_pivot = (
            fused[fused["year"] == target_year]
            .groupby(["country", "variable"])["source"]
            .first()
            .unstack(fill_value="imputed")
        )
        for country in countries:
            if country in prov_pivot.index:
                provenance[country] = prov_pivot.loc[country].to_dict()
            else:
                provenance[country] = {v: "imputed" for v in WB_INDICATORS.keys()}

    logger.info("Stage 6: Z-score + sigmoid normalisation (10-year baseline)")
    directions = {var: d for var, (_, d, _) in WB_INDICATORS.items()}

    baseline_rows = []
    for var, hist_df in history.items():
        for yr in hist_df.columns:
            col_vals = hist_df[yr].dropna().values
            for v in col_vals:
                baseline_rows.append({"variable": var, "value": v})
    if baseline_rows:
        baseline_df_long = pd.DataFrame(baseline_rows)
        baseline_panel = baseline_df_long.groupby("variable")["value"].apply(
            lambda x: x.values
        )
        baseline_wide = pd.DataFrame(
            {var: history[var].values.flatten() for var in history.keys()
             if var in WB_INDICATORS}
        )
    else:
        baseline_wide = None

    normed = normalise_panel(panel, directions, baseline_panel=baseline_wide)

    logger.info("Stage 7: Trajectory computation (OLS slope  R)")
    trajectory = compute_trajectory(history, trend_window=config.trend_window)

    logger.info("Stage 8: Takeoff Detection Module (SIS + DGI + RVS  TOS)")

    gdp_pc_levels_var = "gdp_per_capita"
    gdp_pc_growth_var = "gdp_per_capita_growth"

    if gdp_pc_levels_var in history and not history[gdp_pc_levels_var].empty:
        gdp_pc_hist = history[gdp_pc_levels_var]
        logger.info("SIS: using GDP per capita levels for S-curve fitting")
    elif gdp_pc_growth_var in history:
        gdp_pc_hist = history[gdp_pc_growth_var]
        logger.info("SIS: falling back to GDP per capita growth (high-income guard inactive)")
    else:
        gdp_pc_hist = pd.DataFrame(index=countries)

    sis = compute_sis(gdp_pc_hist, target_year, scurve_years=config.scurve_years)
    sis = sis.reindex(countries).fillna(0.45)

    fdi_series = panel.get("net_fdi_pct_gdp", pd.Series(index=countries)).fillna(0)
    hhi_series = pd.Series(
        {c: hhi_export.get(c, 0.5) for c in countries} if hhi_export else
        {c: 0.5 for c in countries}
    )
    media_series = (
        pd.Series({c: gdelt_media_coverage.get(c, 0.5) for c in countries})
        if gdelt_media_coverage else None
    )
    pillar_d_full = trajectory.reindex(columns=list(TRAJECTORY_VARS.keys()))
    dgi = compute_dgi(
        normed[[c for c in normed.columns if c in PILLAR_A_PRIORS]],
        pillar_d_full,
        fdi_series, hhi_series,
        gdelt_media_coverage=media_series,
        gdelt_active=config.gdelt_active,
    )

    gov_traj = trajectory.get("governance_trend", pd.Series(0.5, index=countries))
    reform_sig = (
        pd.Series({c: gdelt_reform_signal.get(c, 0.5) for c in countries})
        if gdelt_reform_signal else None
    )
    rvs = compute_rvs(gov_traj, reform_sig, gdelt_active=config.gdelt_active)

    tos = compute_tos(
        sis, dgi, rvs,
        scurve_weight=config.scurve_weight,
        discovery_weight=config.discovery_weight,
        reform_weight=config.reform_weight,
        gdelt_active=config.gdelt_active,
    )

    logger.info("Stage 9: Pillar scoring")
    p_a = score_pillar_a(normed)
    p_b = score_pillar_b(normed)
    p_c = score_pillar_c(normed)
    p_d = score_pillar_d(trajectory)

    for s in [p_a, p_b, p_c, p_d]:
        s.reindex(countries).fillna(0.5)

    logger.info("Stage 10: Archetype clustering (K-Means)")
    archetypes = compute_archetypes(normed, n_clusters=config.n_clusters)

    logger.info(f"Stage 11: Bayesian Monte Carlo simulation ({config.n_simulations} runs)")
    cvar25, volatility, _ = run_bayesian_simulation(
        p_a.reindex(countries).fillna(0.5),
        p_b.reindex(countries).fillna(0.5),
        p_c.reindex(countries).fillna(0.5),
        p_d.reindex(countries).fillna(0.5),
        n_simulations=config.n_simulations,
        cvar_threshold=config.cvar_threshold,
        opp_weight=config.opp_weight,
        frag_weight=config.frag_weight,
    )

    logger.info("Stage 12: CVaR headline metric computed")

    logger.info("Stage 13: Anomaly detection (Isolation Forest + self-referential z-check)")
    normed_history_for_anomaly = {
        var: normalise_panel(
            history[var].T.rename_axis("variable").T
            if not history[var].empty else pd.DataFrame(index=countries),
            {var: directions.get(var, 1)}
        )
        for var in WB_INDICATORS.keys() if var in history
    }
    anomaly_flags = run_anomaly_detection(
        normed,
        normed_history_for_anomaly,
        contamination=config.contamination,
    )

    event_adj = compute_event_adjustment(
        gdelt_events, countries, date.today(),
        gdelt_active=config.gdelt_active,
    )
    fred_adj = compute_fred_adjustment(fred_conditions)
    total_event_adj = event_adj + fred_adj

    regime_codes, regime_labels = classify_regime(
        cvar25, volatility,
        tos_score=tos.reindex(countries).fillna(0.45)
    )

    logger.info("Stage 14: Final score assembly")
    base_rao, blended_01, rao_final = assemble_final_score(
        cvar25,
        tos.reindex(countries).fillna(0.45),
        total_event_adj.reindex(countries).fillna(0.0),
        anomaly_flags.reindex(countries).fillna(False),
        tos_weight=config.tos_weight,
        haircut_pts=config.anomaly_haircut,
    )

    results = []
    config_snap = config.snapshot()

    for country in countries:
        imputed = [
            var for var, imputed_list in imputation_log.items()
            if country in imputed_list
        ]
        tos_val = float(tos.get(country, 0.45))
        sis_val = float(sis.get(country, 0.45))
        dgi_val = float(dgi.get(country, 0.45))
        rvs_val = float(rvs.get(country, 0.45))

        result = RAOResult(
            country=country,
            country_name=COUNTRY_NAMES.get(country, country),
            target_year=target_year,
            rao_score=round(float(rao_final.get(country, 5.0)), 3),
            base_rao_score=round(float(base_rao.get(country, 5.0)), 3),
            tos_score=round(tos_val, 4),
            tos_signal_label=tos_label(tos_val, dgi_val, rvs_val, sis_val),
            sis_score=round(sis_val, 4),
            dgi_score=round(dgi_val, 4),
            rvs_score=round(rvs_val, 4),
            pillar_a=round(float(p_a.get(country, 0.5)), 4),
            pillar_b=round(float(p_b.get(country, 0.5)), 4),
            pillar_c=round(float(p_c.get(country, 0.5)), 4),
            pillar_d=round(float(p_d.get(country, 0.5)), 4),
            cvar25=round(float(cvar25.get(country, 0.5)), 4),
            volatility_band=round(float(volatility.get(country, 0.0)), 4),
            event_adjustment=round(float(total_event_adj.get(country, 0.0)), 3),
            anomaly_flag=bool(anomaly_flags.get(country, False)),
            anomaly_haircut_applied=bool(anomaly_flags.get(country, False)),
            regime=str(regime_codes.get(country, "SL")),
            regime_label=str(regime_labels.get(country, "Stable / Low Opportunity")),
            archetype=str(archetypes.get(country, "Unknown")),
            data_confidence=confidence_map.get(country, "MEDIUM"),
            imputed_vars=imputed,
            gdelt_active=config.gdelt_active,
            data_provenance=provenance.get(country, {}),
            config_snapshot=config_snap,
        )
        results.append(result)

    results.sort(key=lambda r: r.rao_score, reverse=True)
    summary_df = results_to_dataframe(results)

    logger.info(
        f"\n{'='*50}\n"
        f"  RAO v2.1 pipeline complete  {len(results)} countries scored\n"
        f"  Target year: {target_year}  |  Profile: {config.profile_label}\n"
        f"  GDELT: {'Active' if config.gdelt_active else 'Inactive (graceful degradation)'}\n"
        f"  Top score: {results[0].country_name} {results[0].rao_score:.2f}\n"
        f"{'='*50}"
    )

    return results, summary_df
