import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit

logger = logging.getLogger(__name__)

def _logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def compute_sis(
    gdp_pc_history: pd.DataFrame,
    target_year: int,
    scurve_years: int = 15,
) -> pd.Series:
    HIGH_INCOME_GDP_PC_THRESHOLD = 25000.0

    year_cols = sorted([c for c in gdp_pc_history.columns
                        if isinstance(c, (int, np.integer)) and c <= target_year])
    window = year_cols[-scurve_years:]

    sis_scores = {}

    for country in gdp_pc_history.index:
        row = gdp_pc_history.loc[country, window].values.astype(float)
        years = np.array(window)
        valid = ~np.isnan(row)

        if valid.sum() < 10:
            sis_scores[country] = _sis_fallback(row[valid], years[valid])
            continue

        y = row[valid]
        t = years[valid].astype(float)
        current_gdp_pc = y[-1] if not np.isnan(y[-1]) else np.nanmax(y)

        if current_gdp_pc > HIGH_INCOME_GDP_PC_THRESHOLD:
            sis_scores[country] = 0.25
            continue

        K_init = max(current_gdp_pc * 3.0, 1.0)
        r_init = 0.05
        t0_init = float(t.mean())

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                popt, _ = curve_fit(
                    _logistic,
                    t, y,
                    p0=[K_init, r_init, t0_init],
                    bounds=([y.max(), 0.01, t[0] - 20], [y.max() * 10, 1.0, t[-1] + 30]),
                    maxfev=5000
                )
            K_fit, r_fit, t0_fit = popt

            cap_utilisation = current_gdp_pc / K_fit if K_fit > 0 else 1.0
            if cap_utilisation > 0.80:
                sis_scores[country] = 0.35
                continue

            years_from_inflection = target_year - t0_fit
            sis_scores[country] = _sis_proximity_score(years_from_inflection)

        except (RuntimeError, ValueError, OverflowError):
            sis_scores[country] = _sis_fallback(y, t)

    return pd.Series(sis_scores)

def _sis_proximity_score(years_from_inflection: float) -> float:
    d = years_from_inflection
    if -2 <= d <= 3:
        return 0.875
    elif 3 < d <= 10:
        return 0.545
    elif -8 <= d < -2:
        return 0.450
    elif d < -8:
        return 0.200
    else:
        return 0.450

def _sis_fallback(y: np.ndarray, t: np.ndarray) -> float:
    if len(y) < 5:
        return 0.45

    mid = len(y) // 2
    recent_growth = np.nanmean(np.diff(y[mid:])) if len(y[mid:]) > 1 else 0.0
    early_growth = np.nanmean(np.diff(y[:mid])) if len(y[:mid]) > 1 else 0.0

    level = max(np.nanmean(y), 1.0)
    acceleration = (recent_growth - early_growth) / level

    return float(expit(acceleration * 10))

def compute_dgi(
    pillar_a_normed: pd.DataFrame,
    pillar_d_trajectory: pd.DataFrame,
    fdi_pct_gdp: pd.Series,
    hhi_export: pd.Series,
    gdelt_media_coverage: Optional[pd.Series] = None,
    gdelt_active: bool = False,
) -> pd.Series:
    countries = pillar_a_normed.index

    a_vals = pillar_a_normed.values
    d_vals = pillar_d_trajectory.reindex(pillar_a_normed.index).values
    all_fm = np.hstack([a_vals, d_vals])
    fm = pd.Series(np.nanmean(all_fm, axis=1), index=countries)

    fdi = fdi_pct_gdp.reindex(countries).fillna(fdi_pct_gdp.mean())
    fdi_mean = fdi.mean()
    fdi_std = fdi.std()
    fdi_recognition = pd.Series(
        expit((fdi.values - fdi_mean) / max(fdi_std, 1e-6)),
        index=countries
    )

    hhi = hhi_export.reindex(countries).fillna(0.5)
    hhi_recognition = pd.Series(expit((1 - hhi.values) * 4 - 2), index=countries)

    if gdelt_active and gdelt_media_coverage is not None:
        media = gdelt_media_coverage.reindex(countries).fillna(0.5)
        er = (fdi_recognition + media + hhi_recognition) / 3.0
        logger.debug("DGI: using full ER (FDI + GDELT media + HHI)")
    else:
        er = (fdi_recognition + hhi_recognition) / 2.0
        if not gdelt_active:
            logger.debug("DGI: GDELT inactive  using FDI + HHI only for ER (50/50)")

    gap = (fm - er) * 4.0
    dgi = pd.Series(expit(gap.values), index=countries)

    return dgi

def compute_rvs(
    governance_trajectory: pd.Series,
    gdelt_reform_signal: Optional[pd.Series] = None,
    gdelt_active: bool = False,
) -> pd.Series:
    countries = governance_trajectory.index

    wgi_norm = governance_trajectory.reindex(countries).fillna(0.5)

    if gdelt_active and gdelt_reform_signal is not None:
        gdelt_norm = gdelt_reform_signal.reindex(countries).fillna(0.5)
        rvs = 0.60 * wgi_norm + 0.40 * gdelt_norm
        logger.debug("RVS: using WGI (60%) + GDELT reform signal (40%)")
    else:
        rvs = wgi_norm.copy()
        if not gdelt_active:
            logger.debug("RVS: GDELT inactive  WGI-only mode (weight=1.0)")

    return pd.Series(rvs.values, index=countries)

def compute_tos(
    sis: pd.Series,
    dgi: pd.Series,
    rvs: pd.Series,
    scurve_weight: float = 0.35,
    discovery_weight: float = 0.35,
    reform_weight: float = 0.30,
    gdelt_active: bool = False,
) -> pd.Series:
    countries = sis.index
    eps = 1e-6

    s = sis.reindex(countries).clip(eps, 1 - eps).values
    d = dgi.reindex(countries).clip(eps, 1 - eps).values
    r = rvs.reindex(countries).clip(eps, 1 - eps).values

    log_tos = (
        scurve_weight * np.log(s)
        + discovery_weight * np.log(d)
        + reform_weight * np.log(r)
    )
    tos = np.exp(log_tos)
    tos = np.clip(tos, 0.0, 1.0)

    return pd.Series(tos, index=countries)

def tos_label(
    tos: float,
    dgi: float,
    rvs: float,
    sis: float,
) -> str:
    if tos >= 0.70:
        return "Strong Pre-Breakout Signal"
    elif tos >= 0.55:
        if dgi < 0.45 and sis >= 0.60:
            return "Emerging (Increasingly Discovered)"
        return "Pre-Breakout Signal"
    elif tos >= 0.40:
        if rvs >= 0.60 and dgi < 0.45:
            return "Reform Story"
        return "Moderate Potential"
    else:
        return "Low Takeoff Signal"
