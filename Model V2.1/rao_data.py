import warnings
import logging
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WB_INDICATORS = {
    "gdp_growth":               ("NY.GDP.MKTP.KD.ZG",  +1, "A"),
    "export_growth":            ("NE.EXP.GNFS.KD.ZG",  +1, "A"),
    "net_fdi_pct_gdp":          ("BX.KLT.DINV.WD.GD.ZS", +1, "A"),
    "gross_capital_formation":  ("NE.GDI.TOTL.ZS",     +1, "A"),
    "secondary_school_enrolment": ("SE.SEC.ENRR",      +1, "A"),

    "cpi_inflation":            ("FP.CPI.TOTL.ZG",     -1, "B"),
    "government_debt_pct_gdp":  ("GC.DOD.TOTL.GD.ZS",  -1, "B"),
    "current_account_pct_gdp":  ("BN.CAB.XOKA.GD.ZS",  +1, "B"),
    "external_debt_pct_gni":    ("DT.DOD.DECT.GN.ZS",  -1, "B"),
    "political_stability":      ("PV.EST",             +1, "B"),
    "rule_of_law":              ("RL.EST",             +1, "B"),
    "control_of_corruption":    ("CC.EST",             +1, "B"),

    "natural_resource_rents_pct_gdp": ("NY.GDP.TOTL.RT.ZS", -1, "C"),
    "gdp_per_capita_growth":    ("NY.GDP.PCAP.KD.ZG",  +1, "C"),
    "youth_unemployment":       ("SL.UEM.1524.ZS",     -1, "C"),
    "vulnerable_employment":    ("SL.EMP.VULN.ZS",     -1, "C"),
    "regulatory_quality":       ("RQ.EST",             +1, "C"),

    "gdp_per_capita":           ("NY.GDP.PCAP.CD",     +1, "AUX"),
}

TRAJECTORY_VARS = {
    "gdp_growth_trend":      ("gdp_growth",             +1, "D"),
    "fdi_trend":             ("net_fdi_pct_gdp",        +1, "D"),
    "governance_trend":      ("rule_of_law",            +1, "D"),
    "debt_trend":            ("government_debt_pct_gdp", -1, "D"),
    "inflation_trend":       ("cpi_inflation",          -1, "D"),
}

IMF_INDICATORS = {
    "gdp_growth":            "NGDP_RPCH",,
    "cpi_inflation":         "PCPIPCH",,
    "government_debt_pct_gdp": "GGXWDG_NGDP",,
    "current_account_pct_gdp": "BCA_NGDPD",,
}

PARTIAL_COVERAGE_VARS = {
    "natural_resource_rents_pct_gdp": 0.773,
    "secondary_school_enrolment":     0.700,
}

HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.75

def fetch_worldbank_data(
    countries: list[str],
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    try:
        import wbgapi as wb
    except ImportError:
        raise ImportError("wbgapi not installed. Run: pip install wbgapi")

    wb_codes = {v: k for k, (v, _, _) in WB_INDICATORS.items()}
    indicator_list = [code for code, _, _ in WB_INDICATORS.values()]

    vintage = date.today().isoformat()
    records = []

    logger.info(f"Fetching {len(indicator_list)} WB indicators for {len(countries)} countries "
                f"({start_year}{end_year})...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df_raw = wb.data.DataFrame(
                indicator_list,
                economy=countries,
                time=range(start_year, end_year + 1),
                skipBlanks=True,
                numericTimeKeys=True
            ).reset_index()
        except Exception as e:
            logger.error(f"World Bank API error: {e}")
            return pd.DataFrame(columns=["country", "year", "variable", "value", "source", "vintage"])

    id_cols = [c for c in df_raw.columns if c in ("economy", "Country", "iso3c")]
    time_cols = [c for c in df_raw.columns if str(c).isdigit()]

    if not id_cols:
        logger.warning("Could not identify country column in WB response")
        return pd.DataFrame(columns=["country", "year", "variable", "value", "source", "vintage"])

    country_col = id_cols[0]

    for _, row in df_raw.iterrows():
        country = row[country_col]

        series_col = [c for c in df_raw.columns if c not in id_cols + time_cols]
        indicator_code = row[series_col[0]] if series_col else None

        for yr in time_cols:
            val = row[yr]
            if pd.isna(val):
                continue
            var_name = wb_codes.get(indicator_code, indicator_code)
            records.append({
                "country": country,
                "year": int(yr),
                "variable": var_name,
                "value": float(val),
                "source": "worldbank",
                "vintage": vintage,
            })

    return pd.DataFrame(records)

def fetch_worldbank_data_wbgapi(
    countries: list[str],
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    try:
        import wbgapi as wb
    except ImportError:
        raise ImportError("wbgapi not installed.")

    vintage = date.today().isoformat()
    all_records = []

    for var_name, (wb_code, direction, pillar) in WB_INDICATORS.items():
        try:
            df = wb.data.DataFrame(
                wb_code,
                economy=countries,
                time=range(start_year, end_year + 1),
                skipBlanks=True,
                numericTimeKeys=True
            )

            if df.index.name in ("economy", "Economy"):
                df = df.reset_index().melt(
                    id_vars=df.index.name,
                    var_name="year",
                    value_name="value"
                ).rename(columns={df.index.name: "country"})
            else:
                df = df.T.reset_index().melt(
                    id_vars="index",
                    var_name="country",
                    value_name="value"
                ).rename(columns={"index": "year"})

            df = df.dropna(subset=["value"])
            df["year"] = df["year"].astype(int)
            df["variable"] = var_name
            df["source"] = "worldbank"
            df["vintage"] = vintage
            all_records.append(df[["country", "year", "variable", "value", "source", "vintage"]])
            logger.debug(f"  WB {var_name}: {len(df)} rows")

        except Exception as e:
            logger.warning(f"  WB fetch failed for {var_name} ({wb_code}): {e}")
            continue

    if not all_records:
        return pd.DataFrame(columns=["country", "year", "variable", "value", "source", "vintage"])

    return pd.concat(all_records, ignore_index=True)

def fetch_imf_data(
    countries: list[str],
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    import requests

    BASE_URL = "https://www.imf.org/external/datamapper/api/v1"
    vintage = date.today().isoformat()
    records = []

    for var_name, imf_code in IMF_INDICATORS.items():
        url = f"{BASE_URL}/{imf_code}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"IMF fetch failed for {var_name} ({imf_code}): {e}")
            continue

        values_block = data.get("values", {}).get(imf_code, {})
        for iso3, year_dict in values_block.items():
            if iso3 not in countries:
                continue
            for yr_str, val in year_dict.items():
                try:
                    yr = int(yr_str)
                    if start_year <= yr <= end_year and val is not None:
                        records.append({
                            "country": iso3,
                            "year": yr,
                            "variable": var_name,
                            "value": float(val),
                            "source": "imf",
                            "vintage": vintage,
                        })
                except (ValueError, TypeError):
                    continue

    logger.info(f"IMF DataMapper: fetched {len(records)} data points across "
                f"{len(IMF_INDICATORS)} variables")
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["country", "year", "variable", "value", "source", "vintage"]
    )

SOURCE_PRIORITY = {"imf": 0, "worldbank": 1, "csv": 2, "external": 3}

def fuse_data_sources(*dataframes: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([df for df in dataframes if df is not None and len(df) > 0],
                         ignore_index=True)
    if combined.empty:
        return combined

    combined["vintage"] = pd.to_datetime(combined["vintage"], errors="coerce")
    combined["source_priority"] = combined["source"].map(SOURCE_PRIORITY).fillna(99)

    combined = combined.sort_values(
        ["country", "year", "variable", "vintage", "source_priority"],
        ascending=[True, True, True, False, True]
    )
    fused = combined.groupby(["country", "year", "variable"], as_index=False).first()
    fused["vintage"] = fused["vintage"].dt.date.astype(str)
    logger.info(f"Data fusion complete: {len(fused)} (country, year, variable) records "
                f"from {combined['source'].nunique()} sources")
    return fused

def build_panel(
    fused: pd.DataFrame,
    countries: list[str],
    target_year: int,
    lookback_years: int = 10,
    knn_neighbors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    from sklearn.impute import KNNImputer

    all_vars = [v for v, (_, _, p) in WB_INDICATORS.items() if p != "AUX"]
    aux_vars = [v for v, (_, _, p) in WB_INDICATORS.items() if p == "AUX"]
    year_range = list(range(target_year - lookback_years + 1, target_year + 1))

    latest = (
        fused[fused["year"].isin(year_range)]
        .sort_values("year", ascending=False)
        .groupby(["country", "variable"], as_index=False)
        .first()
    )
    panel_raw = latest.pivot(index="country", columns="variable", values="value")
    panel_raw = panel_raw.reindex(index=countries, columns=all_vars)

    RATIO_VARS_SUBJECT_TO_SIZE_BIAS = [
        "net_fdi_pct_gdp", "gross_capital_formation",
        "export_growth", "gdp_growth",
    ]
    gdp_pc_col = "gdp_per_capita"
    if gdp_pc_col in panel_raw.columns:
        gdp_pc = panel_raw[gdp_pc_col]
        median_gdp_pc = gdp_pc.median()
        small_economy_mask = gdp_pc < median_gdp_pc

        for var in RATIO_VARS_SUBJECT_TO_SIZE_BIAS:
            if var not in panel_raw.columns:
                continue

            p90 = panel_raw[var].quantile(0.90)

            panel_raw.loc[small_economy_mask, var] = panel_raw.loc[
                small_economy_mask, var
            ].clip(upper=p90)

        logger.info(
            f"GDP size guard: applied ratio variable cap (P90) to "
            f"{small_economy_mask.sum()} small economies"
        )

    history = {}
    for var in all_vars + aux_vars:
        h = fused[(fused["variable"] == var) & (fused["year"].isin(year_range))]
        h_pivot = h.pivot(index="country", columns="year", values="value")
        h_pivot = h_pivot.reindex(index=countries, columns=year_range)
        history[var] = h_pivot

    core_vars = [v for v in all_vars if v not in PARTIAL_COVERAGE_VARS]
    present_counts = panel_raw[core_vars].notna().sum(axis=1)
    total_core = len(core_vars)
    pct_present = present_counts / total_core

    confidence_map = {}
    for country in countries:
        p = pct_present.get(country, 0.0)
        if p >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_map[country] = "HIGH"
        elif p >= MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_map[country] = "MEDIUM"
        else:
            confidence_map[country] = "LOW"

    imputation_log = {var: [] for var in all_vars}

    missing_mask = panel_raw.isna()
    for var in all_vars:
        missing_countries = panel_raw.index[missing_mask[var]].tolist() if var in missing_mask else []
        imputation_log[var] = missing_countries

    imputer = KNNImputer(n_neighbors=min(knn_neighbors, len(countries) - 1))
    panel_imputed_values = imputer.fit_transform(panel_raw.values)
    panel_imputed = pd.DataFrame(
        panel_imputed_values,
        index=panel_raw.index,
        columns=panel_raw.columns
    )

    n_imputed = missing_mask.sum().sum()
    if n_imputed > 0:
        logger.info(f"KNN imputation: filled {n_imputed} missing cells across "
                    f"{len(all_vars)} variables for {len(countries)} countries")
        for var in PARTIAL_COVERAGE_VARS:
            n = len(imputation_log.get(var, []))
            if n > 0:
                logger.info(f"  {var}: {n} countries imputed "
                            f"(coverage was {PARTIAL_COVERAGE_VARS[var]*100:.1f}%)")

    return panel_imputed, history, imputation_log, confidence_map

def load_csv_override(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"country", "year"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    pipeline_var_names = set(WB_INDICATORS.keys())
    all_data_cols = [c for c in df.columns if c not in ("country", "year")]
    var_cols = [c for c in all_data_cols if c in pipeline_var_names]
    ignored_cols = [c for c in all_data_cols if c not in pipeline_var_names]

    if ignored_cols:
        logger.info(f"CSV override: ignoring {len(ignored_cols)} non-pipeline columns: "
                    f"{ignored_cols}")
    if not var_cols:
        raise ValueError(
            f"No recognised pipeline variables found in CSV. "
            f"Expected some of: {sorted(pipeline_var_names)}. "
            f"Found: {all_data_cols}"
        )

    melted = df.melt(id_vars=["country", "year"], value_vars=var_cols,
                     var_name="variable", value_name="value")
    melted = melted.dropna(subset=["value"])
    melted["source"] = "csv"
    melted["vintage"] = date.today().isoformat()
    logger.info(f"CSV override loaded: {len(melted)} rows, "
                f"{len(var_cols)}/{len(all_data_cols)} columns used as pipeline variables")
    return melted[["country", "year", "variable", "value", "source", "vintage"]]
