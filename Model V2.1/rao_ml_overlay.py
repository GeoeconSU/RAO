import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TRAJECTORY_DISCOUNTS = {
    "structural":       1.00,
    "stable":           1.00,
    "cyclical":         0.60,
    "recovery_bounce":  0.50,
    "noise":            0.75,
}

COMMODITY_CYCLE_COUNTRIES = {
    "SAU", "OMN", "QAT", "BHR", "KWT", "ARE",,
    "NGA", "AGO", "GAB", "COG", "GNQ",,
    "AZE", "KAZ", "TKM",,
    "VEN", "ECU", "BOL",,
    "ZMB", "DRC", "COD",,
    "MNG",,
}

TOURISM_CYCLE_COUNTRIES = {
    "CRI", "PAN", "JAM", "DOM", "MUS", "MDV",
    "CPV", "SYC", "FJI", "VUT", "BHS",
}

CONFLICT_STATES = {
    "SYR", "YEM", "AFG", "SOM", "SSD", "CAF",
    "LBY", "IRQ", "MMR", "HTI", "MLI", "BFA",
    "NER", "TCD",
}

ECONOMIC_COLLAPSE_STATES = {
    "VEN",,
    "ZWE",,
    "SDN",,
    "LBN",,
    "SSD",,
    "ERI",,
}

AUTHORITARIAN_CLOSED_STATES = {
    "TKM",,
    "PRK",,
    "CUB",,
}

SANCTIONS_WATCH_LIST = {
    "RUS",,
    "IRN",,
    "BLR",,
    "MMR",,
}

NEAR_ZERO_CAPACITY_STATES = {
    "TLS",,
    "LBY",,
    "COG",,
    "SSD",,
    "COD",,
    "IRQ",,
    "GAB",,
}

SMALL_MARKET_DEPTH_WATCH = {
    "CRI",,
    "PAN",,
    "JAM",,
    "MNE",,
    "BTN",,
    "GMB",,
    "MDA",,
}

PILLAR_C_TOS_FLOOR = 0.10
PILLAR_C_MODERATE_FLOOR = 0.05

PILLAR_C_INVESTABILITY_FLOOR = 0.15

@dataclass
class TrajectoryAssessment:
    country: str
    classification: str
    confidence: float
    discount_factor: float
    pillar_d_original: float
    pillar_d_adjusted: float
    reason: str

def classify_trajectory(
    country: str,
    pillar_d: float,
    pillar_a: float,
    pillar_b: float,
    pillar_c: float,
    tos_score: float,
    rvs_score: float,
    dgi_score: float,
    archetype: str,
    data_confidence: str,
    raw_history: Optional[pd.DataFrame] = None,
) -> TrajectoryAssessment:

    if data_confidence == "LOW":
        return TrajectoryAssessment(
            country=country,
            classification="noise",
            confidence=0.90,
            discount_factor=TRAJECTORY_DISCOUNTS["noise"],
            pillar_d_original=pillar_d,
            pillar_d_adjusted=pillar_d * TRAJECTORY_DISCOUNTS["noise"],
            reason="Data confidence LOW  trajectory signal unreliable"
        )

    if archetype == "Advanced Economy" and pillar_b >= 0.55 and pillar_d < 0.45:
        return TrajectoryAssessment(
            country=country,
            classification="stable",
            confidence=0.85,
            discount_factor=TRAJECTORY_DISCOUNTS["stable"],
            pillar_d_original=pillar_d,
            pillar_d_adjusted=pillar_d,
            reason=(
                f"Advanced Economy with strong governance (Pillar B={pillar_b:.3f}) "
                f"and flat trajectory (Pillar D={pillar_d:.3f})  classified as stable "
                f"rather than noise. Flat trajectories on excellent fundamentals reflect "
                f"maintenance of quality, not decline. No discount applied."
            )
        )

    if pillar_d < 0.35:
        return TrajectoryAssessment(
            country=country,
            classification="noise",
            confidence=0.80,
            discount_factor=TRAJECTORY_DISCOUNTS["noise"],
            pillar_d_original=pillar_d,
            pillar_d_adjusted=pillar_d * TRAJECTORY_DISCOUNTS["noise"],
            reason=f"Pillar D ({pillar_d:.3f}) below signal threshold  weak trend"
        )

    is_commodity = country in COMMODITY_CYCLE_COUNTRIES
    is_tourism = country in TOURISM_CYCLE_COUNTRIES

    if (is_commodity or is_tourism) and pillar_d > 0.60:
        reform_supporting = rvs_score >= 0.50
        governance_threshold = 0.65 if is_tourism else 0.55
        governance_improving = pillar_b >= governance_threshold

        if not reform_supporting and not governance_improving:
            cycle_type = "commodity price cycle" if is_commodity else "tourism recovery cycle"
            return TrajectoryAssessment(
                country=country,
                classification="cyclical",
                confidence=0.85,
                discount_factor=TRAJECTORY_DISCOUNTS["cyclical"],
                pillar_d_original=pillar_d,
                pillar_d_adjusted=pillar_d * TRAJECTORY_DISCOUNTS["cyclical"],
                reason=(
                    f"High Pillar D ({pillar_d:.3f}) in {cycle_type}-exposed economy "
                    f"without governance reform support (RVS={rvs_score:.3f}, "
                    f"Pillar B={pillar_b:.3f})  classified as cyclical"
                )
            )

    if pillar_d > 0.65 and pillar_c < 0.40 and tos_score < 0.50:
        return TrajectoryAssessment(
            country=country,
            classification="recovery_bounce",
            confidence=0.80,
            discount_factor=TRAJECTORY_DISCOUNTS["recovery_bounce"],
            pillar_d_original=pillar_d,
            pillar_d_adjusted=pillar_d * TRAJECTORY_DISCOUNTS["recovery_bounce"],
            reason=(
                f"High Pillar D ({pillar_d:.3f}) but weak absorptive capacity "
                f"(Pillar C={pillar_c:.3f}) and no pre-breakout fundamentals "
                f"(TOS={tos_score:.3f})  trajectory unsustained, classified as "
                f"recovery bounce"
            )
        )

    confidence = min(0.70 + (rvs_score * 0.20) + (pillar_b * 0.10), 0.95)
    return TrajectoryAssessment(
        country=country,
        classification="structural",
        confidence=round(confidence, 3),
        discount_factor=TRAJECTORY_DISCOUNTS["structural"],
        pillar_d_original=pillar_d,
        pillar_d_adjusted=pillar_d,
        reason=(
            f"Trajectory supported by governance data (RVS={rvs_score:.3f}, "
            f"Pillar B={pillar_b:.3f})  classified as structural"
        )
    )

@dataclass
class UniverseFlag:
    country: str
    flag_type: str
    severity: str
    description: str
    score_adjustment: float
    requires_disclosure: bool

def assess_universe_integrity(
    country: str,
    rao_score: float,
    tos_score: float,
    tos_signal_label: str,
    pillar_b: float,
    pillar_c: float,
    data_confidence: str,
    archetype: str,
    imputed_vars: str,
) -> UniverseFlag:

    pre_breakout_signals = {
        "Strong Pre-Breakout Signal",
        "Pre-Breakout Signal",
        "Emerging (Increasingly Discovered)"
    }
    any_positive_signal = pre_breakout_signals | {"Moderate Potential", "Reform Story"}

    if country in CONFLICT_STATES and tos_signal_label in pre_breakout_signals:
        return UniverseFlag(
            country=country, flag_type="narrative_inconsistency", severity="HIGH",
            description=(
                f"Conflict/failed state ({country}) flagged with '{tos_signal_label}' "
                f" TOS signal driven by statistical undiscovery, not genuine investable "
                f"opportunity. Score requires analyst override before client distribution."
            ),
            score_adjustment=-1.0, requires_disclosure=True
        )

    if country in ECONOMIC_COLLAPSE_STATES and tos_signal_label in pre_breakout_signals:
        return UniverseFlag(
            country=country, flag_type="narrative_inconsistency", severity="HIGH",
            description=(
                f"Economic collapse state ({country}) flagged with '{tos_signal_label}' "
                f" TOS signal driven by statistical undiscovery on a collapsed economic base. "
                f"Pre-breakout label is not investable without structural recovery."
            ),
            score_adjustment=-1.0, requires_disclosure=True
        )

    if country in AUTHORITARIAN_CLOSED_STATES:
        return UniverseFlag(
            country=country, flag_type="market_access_restricted", severity="HIGH",
            description=(
                f"Authoritarian/closed economy ({country})  structurally non-investable "
                f"for institutional capital. Market access is legally or practically "
                f"restricted. Score should not be distributed to clients without "
                f"explicit non-investable designation."
            ),
            score_adjustment=-1.5, requires_disclosure=True
        )

    if country in SANCTIONS_WATCH_LIST:
        return UniverseFlag(
            country=country, flag_type="sanctions_compliance_risk", severity="HIGH",
            description=(
                f"({country}) is subject to major international sanctions or significant "
                f"AML/compliance concerns. Score reflects macro fundamentals only and does "
                f"not account for investability constraints. Mandatory compliance review required."
            ),
            score_adjustment=-1.0, requires_disclosure=True
        )

    if pillar_c < PILLAR_C_TOS_FLOOR and tos_signal_label in pre_breakout_signals:
        return UniverseFlag(
            country=country, flag_type="narrative_inconsistency", severity="HIGH",
            description=(
                f"Near-zero absorptive capacity (Pillar C={pillar_c:.3f}) with "
                f"'{tos_signal_label}'  economy cannot absorb investment flows at scale. "
                f"Pre-breakout label is structurally unsupported. Requires analyst review."
            ),
            score_adjustment=-0.75, requires_disclosure=True
        )

    if pillar_c < PILLAR_C_MODERATE_FLOOR and tos_signal_label in any_positive_signal:
        return UniverseFlag(
            country=country, flag_type="narrative_inconsistency", severity="MEDIUM",
            description=(
                f"Critically low absorptive capacity (Pillar C={pillar_c:.3f})  "
                f"investment absorption structurally impossible at scale. "
                f"'{tos_signal_label}' overstates investability. Disclosure required."
            ),
            score_adjustment=-0.50, requires_disclosure=True
        )

    if data_confidence == "LOW":
        return UniverseFlag(
            country=country, flag_type="data_poverty", severity="HIGH",
            description=(
                f"Data confidence LOW  score based on significant imputation. "
                f"Treat as indicative only; do not distribute without supplementary research."
            ),
            score_adjustment=-0.5, requires_disclosure=True
        )

    if archetype == "Commodity Exporter" and pillar_c < 0.30 and pillar_b > 0.55:
        return UniverseFlag(
            country=country, flag_type="commodity_trap", severity="MEDIUM",
            description=(
                f"Commodity monoculture: strong governance (Pillar B={pillar_b:.3f}) "
                f"masking near-zero absorptive capacity (Pillar C={pillar_c:.3f}). "
                f"Score reflects resource management quality, not diversified investment."
            ),
            score_adjustment=0.0, requires_disclosure=True
        )

    if archetype == "Advanced Economy" and pillar_c < 0.25:
        return UniverseFlag(
            country=country, flag_type="archetype_mismatch", severity="MEDIUM",
            description=(
                f"'Advanced Economy' archetype with very low absorptive capacity "
                f"(Pillar C={pillar_c:.3f}). Likely a commodity-dependent state. "
                f"Investment depth is significantly lower than archetype implies."
            ),
            score_adjustment=0.0, requires_disclosure=True
        )

    if tos_score > 0.65 and pillar_b < 0.25 and rao_score < 3.5:
        return UniverseFlag(
            country=country, flag_type="narrative_inconsistency", severity="MEDIUM",
            description=(
                f"High TOS ({tos_score:.3f}) on collapsed institutional base "
                f"(Pillar B={pillar_b:.3f}). Pre-breakout signal not investable "
                f"without institutional recovery. Requires analyst review."
            ),
            score_adjustment=-0.5, requires_disclosure=True
        )

    if country in SMALL_MARKET_DEPTH_WATCH and rao_score > 5.5:
        return UniverseFlag(
            country=country, flag_type="market_depth_limited", severity="MEDIUM",
            description=(
                f"({country}) scores highly on fundamentals but has limited market "
                f"depth for institutional capital deployment. Score reflects structural "
                f"quality; investors should apply position size constraints and verify "
                f"market access before allocation."
            ),
            score_adjustment=0.0, requires_disclosure=True
        )

    return UniverseFlag(
        country=country, flag_type="clean", severity="N/A",
        description="No universe integrity issues detected.",
        score_adjustment=0.0, requires_disclosure=False
    )

def apply_ml_overlay(
    summary_df: pd.DataFrame,
    verbose: bool = True,
    raw_data_path: Optional[str] = None,
) -> pd.DataFrame:
    enhanced = summary_df.copy()

    enhanced["rao_score_pre_overlay"] = enhanced["rao_score"].copy()
    enhanced["overlay_adjustment"] = 0.0

    trajectory_results = []
    universe_results = []

    for _, row in enhanced.iterrows():
        country = row["country"]

        traj = classify_trajectory(
            country=country,
            pillar_d=row["pillar_d"],
            pillar_a=row["pillar_a"],
            pillar_b=row["pillar_b"],
            pillar_c=row["pillar_c"],
            tos_score=row["tos_score"],
            rvs_score=row.get("rvs_score", 0.5),
            dgi_score=row.get("dgi_score", 0.5),
            archetype=row["archetype"],
            data_confidence=row["data_confidence"],
        )
        trajectory_results.append(traj)

        flag = assess_universe_integrity(
            country=country,
            rao_score=row["rao_score"],
            tos_score=row["tos_score"],
            tos_signal_label=row["tos_signal_label"],
            pillar_b=row["pillar_b"],
            pillar_c=row["pillar_c"],
            data_confidence=row["data_confidence"],
            archetype=row["archetype"],
            imputed_vars=str(row.get("imputed_vars", "")),
        )
        universe_results.append(flag)

    PILLAR_D_SCORE_WEIGHT = 0.20

    for i, (traj, flag) in enumerate(zip(trajectory_results, universe_results)):
        country = traj.country
        idx = enhanced[enhanced["country"] == country].index

        if len(idx) == 0:
            continue
        idx = idx[0]

        total_adj = 0.0

        if traj.discount_factor < 1.0:
            d_original = traj.pillar_d_original
            d_adjusted = traj.pillar_d_adjusted
            traj_adj = (d_adjusted - d_original) * PILLAR_D_SCORE_WEIGHT * 9.0
            total_adj += traj_adj

        total_adj += flag.score_adjustment

        if total_adj != 0.0:
            new_score = np.clip(
                enhanced.loc[idx, "rao_score"] + total_adj, 1.0, 10.0
            )
            enhanced.loc[idx, "rao_score"] = round(float(new_score), 3)
            enhanced.loc[idx, "overlay_adjustment"] = round(total_adj, 3)

    enhanced["trajectory_classification"] = [t.classification for t in trajectory_results]
    enhanced["trajectory_discount"] = [t.discount_factor for t in trajectory_results]
    enhanced["trajectory_reason"] = [t.reason for t in trajectory_results]
    enhanced["universe_flag"] = [f.flag_type for f in universe_results]
    enhanced["universe_flag_severity"] = [f.severity for f in universe_results]
    enhanced["universe_flag_description"] = [f.description for f in universe_results]
    enhanced["requires_disclosure"] = [f.requires_disclosure for f in universe_results]

    enhanced = enhanced.sort_values("rao_score", ascending=False).reset_index(drop=True)

    HARD_FLOOR_FLAGS = {
        "narrative_inconsistency", "market_access_restricted", "sanctions_compliance_risk"
    }
    for regime in enhanced["regime"].unique():
        regime_mask = enhanced["regime"] == regime
        clean_mask = ~enhanced["universe_flag"].isin(HARD_FLOOR_FLAGS)
        flagged_mask = enhanced["universe_flag"].isin(HARD_FLOOR_FLAGS)

        clean_in_regime = enhanced[regime_mask & clean_mask]["rao_score"]
        flagged_in_regime = enhanced[regime_mask & flagged_mask]

        if clean_in_regime.empty or flagged_in_regime.empty:
            continue

        regime_ceiling = clean_in_regime.min() - 0.10
        flagged_scores = flagged_in_regime["rao_score"]
        f_min, f_max = flagged_scores.min(), flagged_scores.max()

        for idx in flagged_in_regime.index:
            current = enhanced.loc[idx, "rao_score"]
            if current <= regime_ceiling:
                continue
            original = enhanced.loc[idx, "rao_score_pre_overlay"]
            if f_max > f_min:
                ratio = (current - f_min) / (f_max - f_min)
            else:
                ratio = 0.5
            new_score = round(1.0 + ratio * (regime_ceiling - 1.0), 3)
            enhanced.loc[idx, "rao_score"] = new_score
            enhanced.loc[idx, "overlay_adjustment"] = round(new_score - original, 3)

    enhanced = enhanced.sort_values("rao_score", ascending=False).reset_index(drop=True)

    MAX_SIZE_BONUS = 1.5

    POPULATION_M = {
        'IND': 1400, 'CHN': 1400, 'USA': 335, 'IDN': 275, 'PAK': 230,
        'BRA': 215, 'NGA': 220, 'BGD': 170, 'ETH': 125, 'RUS': 144,
        'MEX': 130, 'JPN': 124, 'PHL': 115, 'EGY': 105, 'VNM': 97,
        'IRN': 87, 'TUR': 85, 'DEU': 84, 'GBR': 67, 'FRA': 68,
        'TZA': 63, 'ZAF': 60, 'KOR': 52, 'ARG': 46, 'DZA': 45,
        'UKR': 44, 'SDN': 45, 'IRQ': 42, 'AFG': 40, 'UZB': 36,
        'SAU': 36, 'AGO': 35, 'CAN': 38, 'MOZ': 33, 'PER': 33,
        'GHA': 32, 'YEM': 33, 'NPL': 30, 'LKA': 22, 'CMR': 27,
        'CIV': 27, 'MYS': 33, 'SOM': 17, 'NER': 25, 'MLI': 22,
        'BFA': 22, 'KEN': 55, 'UGA': 47, 'RWA': 14, 'ZMB': 19,
        'ZWE': 15, 'TCD': 17, 'SEN': 17, 'SSD': 11, 'KHM': 17,
        'KAZ': 19, 'AZE': 10, 'ARE': 10, 'SYR': 21, 'NGA': 220,
        'THA': 71, 'POL': 38, 'MMR': 54, 'ARG': 46, 'ECU': 18,
        'BOL': 12, 'GTM': 17, 'HND': 10, 'SLV': 6, 'NIC': 7,
        'DOM': 11, 'JAM': 3, 'TTO': 1.4, 'CRI': 5, 'PAN': 4,
        'PRY': 7, 'URY': 3.5, 'CHL': 19, 'COL': 51, 'VEN': 29,
        'GUY': 0.8, 'SUR': 0.6, 'BLZ': 0.4, 'ARM': 3, 'GEO': 4,
        'MDA': 2.6, 'MNE': 0.6, 'ALB': 2.8, 'MKD': 2, 'SRB': 7,
        'BIH': 3.3, 'HRV': 4, 'SVN': 2, 'SVK': 5.5, 'CZE': 10.5,
        'HUN': 10, 'ROU': 19, 'BGR': 7, 'ISR': 9, 'CHE': 9,
        'AUT': 9, 'BEL': 11.5, 'NLD': 17.5, 'LUX': 0.6, 'DNK': 6,
        'NOR': 5, 'SWE': 10, 'FIN': 5.5, 'ISL': 0.37, 'IRL': 5,
        'GBR': 67, 'NZL': 5, 'AUS': 26, 'CAN': 38, 'SGP': 6,
        'MYS': 33, 'PHL': 115, 'IDN': 275, 'THA': 71, 'VNM': 97,
        'KOR': 52, 'JPN': 124, 'CHN': 1400, 'MNG': 3, 'KGZ': 7,
        'TJK': 10, 'TKM': 6, 'UZB': 36, 'BTN': 0.8, 'LAO': 7,
        'BWA': 2.5, 'NAM': 2.5, 'LSO': 2, 'SWZ': 1.2, 'MWI': 20,
        'MDG': 28, 'CAF': 5, 'GIN': 13, 'SLE': 8, 'LBR': 5,
        'MRT': 4.5, 'GMB': 2.6, 'GNB': 2, 'BEN': 13, 'TGO': 8,
        'DJI': 1, 'ERI': 3.5, 'BDI': 12, 'COG': 6, 'GAB': 2.3,
        'COD': 100, 'UVK': 1.8, 'LBN': 5, 'LBY': 7, 'TLS': 1.3,
        'SSD': 11, 'PNG': 10, 'MUS': 1.3, 'MLT': 0.5, 'CYP': 1.2,
        'BHR': 1.7, 'QAT': 2.9, 'KWT': 4.4, 'OMN': 4.5,
    }

    gdp_pc = {}
    if raw_data_path is not None:
        try:
            raw = pd.read_csv(raw_data_path)
            if 'gdp_per_capita' in raw.columns:
                latest = (raw.sort_values('year', ascending=False)
                          .groupby('country')['gdp_per_capita'].first())
                gdp_pc = latest.dropna().to_dict()
        except Exception:
            pass

    GDP_PC_FALLBACK = {
        'IND': 2695, 'CHN': 13303, 'USA': 84534, 'IDN': 4925, 'BRA': 10137,
        'JPN': 41580, 'DEU': 51645, 'GBR': 52590, 'FRA': 45340, 'KOR': 35765,
        'AUS': 63487, 'CAN': 57960, 'NZL': 47970, 'SGP': 90674, 'MYS': 13600,
        'THA': 7850, 'VNM': 4717, 'PHL': 3956, 'IDN': 4925, 'TUR': 15614,
        'SAU': 31050, 'ARE': 48952, 'ZAF': 7050, 'NGA': 2221, 'KEN': 2193,
        'DNK': 71026, 'SWE': 60180, 'NOR': 106149, 'CHE': 93720, 'NLD': 60461,
        'ISL': 76078, 'IRL': 103992, 'BEL': 50090, 'AUT': 52085, 'ISR': 55350,
        'CZE': 28280, 'POL': 20590, 'HUN': 20470, 'ROU': 15830, 'BGR': 14250,
        'HRV': 20700, 'SVN': 32370, 'SVK': 23920, 'LTU': 25410, 'LVA': 23060,
        'EST': 33490, 'MLT': 35280, 'CYP': 32150, 'LUX': 133750, 'GRC': 21490,
        'PRT': 25070, 'ESP': 33090, 'ITA': 36820, 'RUS': 15822, 'ARM': 8556,
        'GEO': 9241, 'KAZ': 12970, 'UZB': 2790, 'AZE': 7610, 'TKM': 9460,
    }

    countries = enhanced['country'].tolist()
    ln_gdp_vals = []
    for c in countries:
        pc = gdp_pc.get(c, GDP_PC_FALLBACK.get(c, 5000))
        pop = POPULATION_M.get(c, 8)
        gdp_total = pc * pop * 1e6
        ln_gdp_vals.append(np.log(max(gdp_total, 1.0)))

    ln_arr = np.array(ln_gdp_vals)
    ln_min_v, ln_max_v = ln_arr.min(), ln_arr.max()
    size_factors = (ln_arr - ln_min_v) / (ln_max_v - ln_min_v)
    size_bonus = size_factors * MAX_SIZE_BONUS

    enhanced["market_size_bonus"] = size_bonus.round(3)
    enhanced["rao_score"] = (enhanced["rao_score"] + enhanced["market_size_bonus"]).clip(upper=10.0).round(3)
    enhanced["overlay_adjustment"] = (enhanced["overlay_adjustment"] + enhanced["market_size_bonus"]).round(3)

    enhanced = enhanced.sort_values("rao_score", ascending=False).reset_index(drop=True)

    if verbose:
        n_cyclical = sum(1 for t in trajectory_results if t.classification == "cyclical")
        n_bounce = sum(1 for t in trajectory_results if t.classification == "recovery_bounce")
        n_noise = sum(1 for t in trajectory_results if t.classification == "noise")
        n_structural = sum(1 for t in trajectory_results if t.classification == "structural")
        n_stable = sum(1 for t in trajectory_results if t.classification == "stable")
        n_flags = sum(1 for f in universe_results if f.flag_type != "clean")
        n_disclosure = sum(1 for f in universe_results if f.requires_disclosure)

        adjusted = enhanced[enhanced["overlay_adjustment"] != 0]

        print("=" * 60)
        print("  ML OVERLAY  LAYER 1: TRAJECTORY CLASSIFICATION")
        print("=" * 60)
        print(f"  Structural:       {n_structural:3d} countries  (no adjustment)")
        print(f"  Stable:           {n_stable:3d} countries  (advanced economy  no adjustment)")
        print(f"  Cyclical:         {n_cyclical:3d} countries  (40% Pillar D discount)")
        print(f"  Recovery bounce:  {n_bounce:3d} countries  (50% Pillar D discount)")
        print(f"  Noise/weak:       {n_noise:3d} countries  (25% Pillar D discount)")

        if n_cyclical + n_bounce > 0:
            print()
            print("  Adjusted countries:")
            non_structural = enhanced[
                enhanced["trajectory_classification"].isin(
                    ["cyclical", "recovery_bounce", "noise"]
                )
            ][["country", "country_name", "rao_score_pre_overlay",
               "rao_score", "trajectory_classification",
               "overlay_adjustment"]].head(20)
            print(non_structural.to_string(index=False))

        print()
        print("=" * 60)
        print("  ML OVERLAY  LAYER 2: UNIVERSE INTEGRITY")
        print("=" * 60)
        print(f"  Clean:                    {len(enhanced) - n_flags:3d} countries")
        print(f"  Flagged:                  {n_flags:3d} countries")
        print(f"  Requiring disclosure:     {n_disclosure:3d} countries")

        flagged = enhanced[enhanced["universe_flag"] != "clean"][
            ["country", "country_name", "rao_score", "universe_flag",
             "universe_flag_severity", "universe_flag_description"]
        ]
        if len(flagged):
            print()
            for _, row in flagged.iterrows():
                print(f"  [{row['universe_flag_severity']:6s}] {row['country']} "
                      f"({row['universe_flag']}): {row['universe_flag_description'][:80]}...")

        print()
        print("=" * 60)
        print("  ML OVERLAY  LAYER 3: MARKET SIZE ADJUSTMENT")
        print("=" * 60)
        print(f"  Max bonus: +{MAX_SIZE_BONUS:.1f} pts  |  Method: ln(GDP_total) normalised")
        print(f"  Methodology: standard index practice (cf. MSCI, FTSE, JPMorgan weighting)")
        print()
        top_size = enhanced.nlargest(10, "market_size_bonus")[
            ["country", "country_name", "market_size_bonus", "rao_score"]
        ]
        print("  Largest bonuses:")
        for _, row in top_size.iterrows():
            rank = (enhanced.country == row["country"]).idxmax() + 1
            print(f"    {row['country']:4s} ({row['country_name']:20s}): "
                  f"+{row['market_size_bonus']:.3f} pts    #{rank} ({row['rao_score']:.2f})")
        print("=" * 60)

    return enhanced

def overlay_score_card(row: pd.Series) -> str:
    lines = [
        f"{'='*60}",
        f"  {row.get('country_name', row['country'])} ({row['country']})  {row.get('year', '')}",
        f"{'='*60}",
        f"  RAO Score:         {row['rao_score']:.2f} / 10",
    ]

    if row.get("overlay_adjustment", 0) != 0:
        lines.append(
            f"  Pre-overlay score: {row['rao_score_pre_overlay']:.2f} "
            f"(adjustment: {row['overlay_adjustment']:+.2f})"
        )
        lines.append(f"  Trajectory:        {row['trajectory_classification'].upper()}")
        lines.append(f"  Reason:            {row['trajectory_reason'][:70]}")

    lines += [
        f"",
        f"  Regime:            {row['regime']}  {row['regime_label']}",
        f"  Archetype:         {row['archetype']}",
        f"  TOS:               {row['tos_score']:.3f}  {row['tos_signal_label']}",
        f"",
        f"  Pillars: A={row['pillar_a']:.3f}  B={row['pillar_b']:.3f}  "
        f"C={row['pillar_c']:.3f}  D={row['pillar_d']:.3f}",
        f"  Data confidence:   {row['data_confidence']}",
    ]

    if row.get("requires_disclosure", False):
        lines += [
            f"",
            f"    DISCLOSURE REQUIRED",
            f"  {row.get('universe_flag_description', '')[:100]}",
        ]

    lines.append(f"{'='*60}")
    return "\n".join(lines)
