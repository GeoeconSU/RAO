import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit

logger = logging.getLogger(__name__)

ARCHETYPE_LABELS = {
    0: "Commodity Exporter",
    1: "Fragile / High-Risk State",
    2: "High-Growth Emerging Market",
    3: "Knowledge / Service Economy",
    4: "Stable / Moderate Growth",
}

REGIME_SCORE_HIGH = 0.575
REGIME_SCORE_LOW = 0.370
REGIME_VOLATILITY_HIGH = 0.12

def run_anomaly_detection(
    normed_panel: pd.DataFrame,
    normed_history: dict[str, pd.DataFrame],
    contamination: float = 0.05,
    zscore_threshold: float = 2.5,
) -> pd.Series:
    countries = normed_panel.index

    history_depth = 0
    for var, hist_df in normed_history.items():
        if not hist_df.empty:
            history_depth = max(history_depth, hist_df.shape[1])
        break

    if history_depth < 3:
        logger.info(
            "Anomaly detection: insufficient history depth "
            f"({history_depth} years < 3 required)  returning no flags. "
            "Will activate automatically when live data provides 3 years history."
        )
        return pd.Series(False, index=countries)

    from sklearn.ensemble import IsolationForest

    flags = pd.Series(False, index=countries)

    X = normed_panel.values
    if np.isnan(X).any():
        X = np.where(np.isnan(X), 0.5, X)

    n_estimators = min(200, max(50, len(countries) * 2))
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_labels = iso.fit_predict(X)
    isolation_flags = iso_labels == -1
    flags |= pd.Series(isolation_flags, index=countries)
    n_iso = isolation_flags.sum()
    if n_iso > 0:
        logger.info(f"Anomaly (Stage 1  Isolation Forest): {n_iso} flagged: "
                    f"{countries[isolation_flags].tolist()}")

    for country in countries:
        current = normed_panel.loc[country]
        for var in normed_panel.columns:
            if var not in normed_history:
                continue
            hist = normed_history.get(var)
            if hist is None or country not in hist.index:
                continue
            hist_vals = hist.loc[country].dropna().values
            if len(hist_vals) < 3:
                continue
            hist_mu = hist_vals.mean()
            hist_sd = hist_vals.std()
            if hist_sd < 1e-6:
                continue
            current_val = current.get(var, np.nan)
            if np.isnan(current_val):
                continue
            z = abs((current_val - hist_mu) / hist_sd)
            if z > zscore_threshold:
                flags[country] = True
                logger.info(
                    f"Anomaly (Stage 2  self-referential): {country} | {var} "
                    f"z={z:.2f} (threshold={zscore_threshold})"
                )
                break

    logger.info(f"Anomaly detection complete: {flags.sum()} countries flagged")
    return flags

def compute_archetypes(
    normed_panel: pd.DataFrame,
    n_clusters: int = 5,
) -> pd.Series:
    from sklearn.cluster import KMeans

    X = normed_panel.values
    if np.isnan(X).any():
        X = np.where(np.isnan(X), 0.5, X)

    k = min(n_clusters, len(normed_panel) - 1)
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    cluster_ids = km.fit_predict(X)

    pillar_b_cols = [c for c in normed_panel.columns if c in [
        "political_stability", "rule_of_law", "control_of_corruption",
        "government_debt_pct_gdp", "cpi_inflation",
        "current_account_pct_gdp", "external_debt_pct_gni"
    ]]
    pillar_c_cols = [c for c in normed_panel.columns if c in [
        "natural_resource_rents_pct_gdp", "gdp_per_capita_growth",
        "youth_unemployment", "vulnerable_employment", "regulatory_quality"
    ]]
    pillar_a_cols = [c for c in normed_panel.columns if c in [
        "gdp_growth", "export_growth", "net_fdi_pct_gdp",
        "gross_capital_formation", "secondary_school_enrolment"
    ]]

    cluster_labels = {}
    for cid in range(k):
        mask = cluster_ids == cid
        subset = normed_panel.iloc[mask]

        avg_b = subset[pillar_b_cols].mean().mean() if pillar_b_cols else 0.5
        avg_c = subset[pillar_c_cols].mean().mean() if pillar_c_cols else 0.5
        avg_a = subset[pillar_a_cols].mean().mean() if pillar_a_cols else 0.5

        if avg_b >= 0.63 and avg_c >= 0.65:
            label = "Advanced Economy"
        elif avg_b >= 0.52 and avg_c >= 0.52:
            label = "Knowledge / Service Economy"
        elif avg_a >= 0.52 and avg_c >= 0.38:
            label = "High-Growth Emerging Market"
        elif avg_c <= 0.28:
            label = "Commodity Exporter"
        else:
            label = "Diversified Emerging Market"

        cluster_labels[cid] = label
        logger.debug(
            f"Cluster {cid} ({mask.sum()} countries): "
            f"A={avg_a:.2f} B={avg_b:.2f} C={avg_c:.2f}  {label}"
        )

    labels = [cluster_labels[i] for i in cluster_ids]
    return pd.Series(labels, index=normed_panel.index)

def classify_regime(
    cvar25: pd.Series,
    volatility: pd.Series,
    tos_score: pd.Series = None,
) -> tuple[pd.Series, pd.Series]:
    codes = []
    labels = []

    for country in cvar25.index:
        score = cvar25[country]
        vol = volatility[country]
        tos = tos_score[country] if tos_score is not None else 0.5
        high_score = score > REGIME_SCORE_HIGH
        high_vol = vol >= REGIME_VOLATILITY_HIGH
        has_tos = tos >= 0.25

        if high_score and not high_vol and has_tos:
            codes.append("RO")
            labels.append("Robust Opportunity")
        elif high_score and (high_vol or not has_tos):
            codes.append("FO")
            labels.append("Fragile Opportunity")
        elif not high_score and not high_vol:
            codes.append("SL")
            labels.append("Stable / Low Opportunity")
        else:
            codes.append("SR")
            labels.append("Structural Risk")

    return pd.Series(codes, index=cvar25.index), pd.Series(labels, index=cvar25.index)

GDELT_EVENT_PARAMS = {
    "armed_conflict":     (-0.80, 0.04),
    "mass_violence":      (-1.00, 0.04),
    "sanctions_embargo":  (-0.45, 0.04),
    "protest_unrest":     (-0.225, 0.04),
    "policy_reform":      (+0.325, 0.01),
    "economic_cooperation": (+0.225, 0.01),
}

EVENT_ADJ_MIN = -1.5
EVENT_ADJ_MAX = +0.75

def compute_event_adjustment(
    events_by_country: Optional[dict[str, list[dict]]],
    countries: list[str],
    target_date,
    gdelt_active: bool = False,
) -> pd.Series:
    if not gdelt_active or events_by_country is None:
        if not gdelt_active:
            logger.info("Event adjustment: GDELT inactive  zero adjustment applied to all countries")
        return pd.Series(0.0, index=countries)

    adjustments = {}
    for country in countries:
        events = events_by_country.get(country, [])
        total_adj = 0.0
        for event in events:
            event_type = event.get("type", "")
            days_since = event.get("days_since", 0)
            if event_type not in GDELT_EVENT_PARAMS:
                continue
            base, lam = GDELT_EVENT_PARAMS[event_type]
            impact = base * np.exp(-lam * days_since)
            total_adj += impact

        adjustments[country] = np.clip(total_adj, EVENT_ADJ_MIN, EVENT_ADJ_MAX)

    return pd.Series(adjustments)

def compute_fred_adjustment(fred_conditions: Optional[dict] = None) -> float:
    if fred_conditions is None:
        return 0.0

    signals = []

    if "hy_spread" in fred_conditions and "hy_spread_mean" in fred_conditions:
        spread_z = (fred_conditions["hy_spread"] - fred_conditions["hy_spread_mean"]) \
                   / max(fred_conditions.get("hy_spread_std", 1.0), 0.1)
        signals.append(-0.10 * expit(spread_z))

    if "usd_index" in fred_conditions and "usd_index_mean" in fred_conditions:
        usd_z = (fred_conditions["usd_index"] - fred_conditions["usd_index_mean"]) \
                / max(fred_conditions.get("usd_index_std", 1.0), 0.1)
        signals.append(-0.08 * expit(usd_z))

    if not signals:
        return 0.0

    return float(np.clip(sum(signals), -0.30, 0.20))
