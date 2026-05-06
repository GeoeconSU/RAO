from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import pandas as pd

@dataclass
class RAOResult:

    country: str
    country_name: str = ""
    target_year: int = 0

    rao_score: float = 0.0
    base_rao_score: float = 0.0

    tos_score: float = 0.0
    tos_signal_label: str = ""
    sis_score: float = 0.0
    dgi_score: float = 0.0
    rvs_score: float = 0.0

    pillar_a: float = 0.0
    pillar_b: float = 0.0
    pillar_c: float = 0.0
    pillar_d: float = 0.0

    cvar25: float = 0.0
    volatility_band: float = 0.0

    event_adjustment: float = 0.0
    anomaly_flag: bool = False
    anomaly_haircut_applied: bool = False

    regime: str = ""
    regime_label: str = ""
    archetype: str = ""

    data_confidence: str = "MEDIUM"
    imputed_vars: list = field(default_factory=list)
    gdelt_active: bool = False

    data_provenance: dict = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def score_card(self) -> str:
        gdelt_note = "" if self.gdelt_active else " [GDELT inactive: structural data only]"
        conf_note = f"[Data confidence: {self.data_confidence}]"
        if self.imputed_vars:
            conf_note += f" [Imputed: {', '.join(self.imputed_vars)}]"

        lines = [
            f"{'='*60}",
            f"  RAO FRAMEWORK v2.1  {self.country_name or self.country} ({self.target_year})",
            f"{'='*60}",
            f"  RAO Score:        {self.rao_score:.2f} / 10",
            f"  Base RAO:         {self.base_rao_score:.2f} / 10",
            f"  Regime:           {self.regime}  {self.regime_label}",
            f"  Archetype:        {self.archetype}",
            f"",
            f"  Takeoff Signal:   {self.tos_score:.3f}    {self.tos_signal_label}{gdelt_note}",
            f"    S-Curve (SIS):   {self.sis_score:.3f}",
            f"    Discovery (DGI): {self.dgi_score:.3f}",
            f"    Reform (RVS):    {self.rvs_score:.3f}",
            f"",
            f"  Pillar Scores:",
            f"    A (Opportunity):  {self.pillar_a:.3f}",
            f"    B (Fragility):    {self.pillar_b:.3f}  (higher = more resilient)",
            f"    C (Capacity):     {self.pillar_c:.3f}",
            f"    D (Trajectory):   {self.pillar_d:.3f}",
            f"",
            f"  CVaR:           {self.cvar25:.3f}",
            f"  Volatility (IQR):  {self.volatility_band:.3f}",
            f"  Event Adjustment:  {self.event_adjustment:+.2f} pts",
            f"  Anomaly Flag:      {'YES  haircut applied' if self.anomaly_haircut_applied else 'No'}",
            f"",
            f"  {conf_note}",
            f"{'='*60}",
        ]
        return "\n".join(lines)

def results_to_dataframe(results: list[RAOResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "country": r.country,
            "country_name": r.country_name,
            "year": r.target_year,
            "rao_score": round(r.rao_score, 3),
            "base_rao_score": round(r.base_rao_score, 3),
            "regime": r.regime,
            "regime_label": r.regime_label,
            "tos_score": round(r.tos_score, 3),
            "tos_signal_label": r.tos_signal_label,
            "sis_score": round(r.sis_score, 3),
            "dgi_score": round(r.dgi_score, 3),
            "rvs_score": round(r.rvs_score, 3),
            "pillar_a": round(r.pillar_a, 3),
            "pillar_b": round(r.pillar_b, 3),
            "pillar_c": round(r.pillar_c, 3),
            "pillar_d": round(r.pillar_d, 3),
            "cvar25": round(r.cvar25, 3),
            "volatility_band": round(r.volatility_band, 3),
            "event_adjustment": round(r.event_adjustment, 3),
            "anomaly_flag": r.anomaly_flag,
            "anomaly_haircut_applied": r.anomaly_haircut_applied,
            "archetype": r.archetype,
            "data_confidence": r.data_confidence,
            "imputed_vars": "; ".join(r.imputed_vars) if r.imputed_vars else "",
            "gdelt_active": r.gdelt_active,
        })
    return pd.DataFrame(rows).sort_values("rao_score", ascending=False).reset_index(drop=True)
