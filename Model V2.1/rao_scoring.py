import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit

logger = logging.getLogger(__name__)

PILLAR_A_PRIORS = {
    "gdp_growth":                (0.25, 0.08),
    "export_growth":             (0.20, 0.07),
    "net_fdi_pct_gdp":           (0.20, 0.07),
    "gross_capital_formation":   (0.18, 0.06),
    "secondary_school_enrolment": (0.17, 0.06),
}

PILLAR_B_PRIORS = {
    "cpi_inflation":             (0.10, 0.04),
    "government_debt_pct_gdp":   (0.12, 0.04),
    "current_account_pct_gdp":   (0.10, 0.03),
    "external_debt_pct_gni":     (0.12, 0.04),
    "political_stability":       (0.18, 0.06),
    "rule_of_law":               (0.20, 0.06),
    "control_of_corruption":     (0.18, 0.05),
}

PILLAR_D_PRIORS = {
    "gdp_growth_trend":          (0.25, 0.08),
    "fdi_trend":                 (0.20, 0.07),
    "governance_trend":          (0.25, 0.08),
    "debt_trend":                (0.15, 0.05),
    "inflation_trend":           (0.15, 0.05),
}

PILLAR_LEVEL_PRIOR_SIGMA = 0.15

def _dirichlet_params(means: list[float], sigma: float = 0.07) -> np.ndarray:
    means = np.array(means, dtype=float)
    means = means / means.sum()
    mu_bar = means.mean()
    alpha_0 = max(mu_bar * (1 - mu_bar) / (sigma ** 2) - 1, 0.1)
    return np.maximum(means * alpha_0, 0.1)

def normalise_panel(
    panel: pd.DataFrame,
    directions: dict[str, int],
    baseline_panel: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if baseline_panel is None:
        ref = panel
    else:
        ref = baseline_panel

    normed = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)

    for var in panel.columns:
        if var not in panel.columns:
            continue
        direction = directions.get(var, +1)
        col = panel[var].values.astype(float)
        ref_col = ref[var].dropna().values.astype(float) if var in ref.columns else col

        mu = np.nanmean(ref_col)
        sigma = np.nanstd(ref_col)

        if sigma < 1e-10:
            normed[var] = 0.5
            continue

        z = (col - mu) / sigma
        if direction == -1:
            z = -z

        normed[var] = expit(z * 1.8)

    return normed

def compute_trajectory(
    history: dict[str, pd.DataFrame],
    trend_window: int = 5
) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    from rao_data import TRAJECTORY_VARS

    trajectory_raw = {}

    for traj_var, (source_var, direction, _) in TRAJECTORY_VARS.items():
        if source_var not in history:
            continue
        hist_df = history[source_var]

        year_cols = sorted([c for c in hist_df.columns if isinstance(c, (int, np.integer))])
        if len(year_cols) < 2:
            trajectory_raw[traj_var] = pd.Series(np.nan, index=hist_df.index)
            continue

        window_cols = year_cols[-trend_window:]
        scores = {}

        for country in hist_df.index:
            row = hist_df.loc[country, window_cols].values.astype(float)
            valid = ~np.isnan(row)
            if valid.sum() < 3:
                scores[country] = np.nan
                continue

            x = np.array(window_cols)[valid].reshape(-1, 1)
            y = row[valid]
            x_c = x - x.mean()
            reg = LinearRegression(fit_intercept=True).fit(x_c, y)
            slope = reg.coef_[0]
            ss_res = np.sum((y - reg.predict(x_c)) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = max(1 - ss_res / ss_tot, 0.0) if ss_tot > 1e-12 else 0.0

            raw = slope * r2
            if direction == -1:
                raw = -raw
            scores[country] = raw

        trajectory_raw[traj_var] = pd.Series(scores)

    raw_df = pd.DataFrame(trajectory_raw)

    normed = pd.DataFrame(index=raw_df.index, columns=raw_df.columns, dtype=float)
    for col in raw_df.columns:
        vals = raw_df[col].values.astype(float)
        p10 = np.nanpercentile(vals, 10)
        p90 = np.nanpercentile(vals, 90)
        vals = np.clip(vals, p10, p90)
        mu, sigma = np.nanmean(vals), np.nanstd(vals)
        if sigma < 1e-10:
            normed[col] = 0.5
        else:
            normed[col] = expit((vals - mu) / sigma * 1.8)

    return normed

def score_pillar_a(normed: pd.DataFrame) -> pd.Series:
    cols = [c for c in normed.columns if c in PILLAR_A_PRIORS]
    return normed[cols].mean(axis=1)

def score_pillar_b(normed: pd.DataFrame) -> pd.Series:
    cols = [c for c in normed.columns if c in PILLAR_B_PRIORS]
    return normed[cols].mean(axis=1)

def score_pillar_c(normed: pd.DataFrame) -> pd.Series:
    from rao_data import WB_INDICATORS
    cap_vars = [v for v, (_, _, p) in WB_INDICATORS.items() if p == "C"]
    cols = [c for c in normed.columns if c in cap_vars]
    if not cols:
        return pd.Series(0.5, index=normed.index)

    eps = 1e-6
    log_vals = np.log(normed[cols].clip(eps, 1 - eps).values)
    geo_mean = np.exp(log_vals.mean(axis=1))
    return pd.Series(geo_mean, index=normed.index)

def score_pillar_d(trajectory: pd.DataFrame) -> pd.Series:
    capped = trajectory.clip(upper=0.75)
    raw_mean = capped.mean(axis=1)

    floored = raw_mean.clip(lower=0.38)

    return floored

def run_bayesian_simulation(
    pillar_a: pd.Series,
    pillar_b: pd.Series,
    pillar_c: pd.Series,
    pillar_d: pd.Series,
    n_simulations: int = 1000,
    cvar_threshold: float = 0.25,
    opp_weight: float = 0.60,
    frag_weight: float = 0.40,
    rng: Optional[np.random.Generator] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if rng is None:
        rng = np.random.default_rng(42)

    countries = pillar_a.index
    n = len(countries)

    A = pillar_a.values
    B = pillar_b.values
    C = pillar_c.values
    D = pillar_d.values

    A_eff = A * C

    w_d_target = 0.20
    w_a_target = opp_weight - w_d_target
    w_b_target = frag_weight

    pillar_means = np.array([w_a_target, w_b_target, w_d_target])
    pillar_means = np.maximum(pillar_means, 0.01)
    pillar_means /= pillar_means.sum()

    alpha = np.maximum(pillar_means * 3.0, 0.1)

    pillar_weights = rng.dirichlet(alpha, size=n_simulations)
    w_A_sims = pillar_weights[:, 0]
    w_B_sims = pillar_weights[:, 1]
    w_D_sims = pillar_weights[:, 2]

    scores_matrix = (
        w_A_sims[:, None] * A_eff[None, :]
        + w_D_sims[:, None] * D[None, :]
        - w_B_sims[:, None] * (1 - B[None, :])
        + w_B_sims[:, None]
    )
    scores_matrix = np.clip(scores_matrix, 0.0, 1.0)

    quantile = np.quantile(scores_matrix, cvar_threshold, axis=0)
    cvar_vals = np.array([
        scores_matrix[:, i][scores_matrix[:, i] <= quantile[i]].mean()
        for i in range(n)
    ])

    q75 = np.quantile(scores_matrix, 0.75, axis=0)
    q25 = np.quantile(scores_matrix, 0.25, axis=0)
    volatility_vals = q75 - q25

    cvar_mu = cvar_vals.mean()
    cvar_sd = cvar_vals.std()
    if cvar_sd > 1e-6:
        cvar_z = (cvar_vals - cvar_mu) / cvar_sd
        cvar_scaled = expit(cvar_z * 1.2)
    else:
        cvar_scaled = np.full(n, 0.5)

    cvar25 = pd.Series(cvar_scaled, index=countries)
    volatility = pd.Series(volatility_vals, index=countries)

    sim_distributions = pd.Series(
        [scores_matrix[:, i] for i in range(n)],
        index=countries
    )

    return cvar25, volatility, sim_distributions

def assemble_final_score(
    cvar25: pd.Series,
    tos_score: pd.Series,
    event_adjustment: pd.Series,
    anomaly_haircut_applied: pd.Series,
    tos_weight: float = 0.25,
    haircut_pts: float = 0.50,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    base_rao = 1.0 + 9.0 * cvar25

    blended_01 = (1.0 - tos_weight) * cvar25 + tos_weight * tos_score
    rao_raw = 1.0 + 9.0 * blended_01

    haircut = anomaly_haircut_applied.astype(float) * haircut_pts
    rao_adj = rao_raw + event_adjustment - haircut
    rao_final = rao_adj.clip(1.0, 10.0)

    return base_rao, blended_01, rao_final
