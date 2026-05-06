
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

@dataclass
class RAOConfig:

    opp_weight: float = 0.60
    frag_weight: float = 0.40

    tos_weight: float = 0.25

    scurve_weight: float = 0.35
    discovery_weight: float = 0.35
    reform_weight: float = 0.30

    n_simulations: int = 1000
    cvar_threshold: float = 0.25

    n_clusters: int = 5
    contamination: float = 0.05

    anomaly_haircut: float = 0.50

    gdelt_lookback_days: int = 90
    lookback_years: int = 10
    trend_window: int = 5
    scurve_years: int = 15

    gdelt_active: bool = False

    knn_neighbors: int = 5

    profile_label: str = "Commercial Default"

    def validate(self) -> None:
        if abs(self.opp_weight + self.frag_weight - 1.0) > 1e-9:
            raise ValueError(
                f"opp_weight ({self.opp_weight}) + frag_weight ({self.frag_weight}) "
                f"must equal 1.0"
            )
        if not (0.0 <= self.tos_weight <= 0.5):
            raise ValueError(f"tos_weight must be in [0.0, 0.5], got {self.tos_weight}")
        tdm_sum = self.scurve_weight + self.discovery_weight + self.reform_weight
        if abs(tdm_sum - 1.0) > 1e-9:
            raise ValueError(
                f"TDM weights must sum to 1.0, got {tdm_sum:.4f}"
            )
        if self.n_simulations < 100:
            raise ValueError("n_simulations must be >= 100 for reliable CVaR estimates")
        if not (0.05 <= self.cvar_threshold <= 0.5):
            raise ValueError(f"cvar_threshold must be in [0.05, 0.5], got {self.cvar_threshold}")

    def snapshot(self) -> dict:
        return asdict(self)

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2)

COMMERCIAL_DEFAULT = RAOConfig(
    opp_weight=0.60, frag_weight=0.40, tos_weight=0.40,
    scurve_weight=0.35, discovery_weight=0.35, reform_weight=0.30,
    profile_label="Commercial Default"
)

FRONTIER_MANDATE = RAOConfig(
    opp_weight=0.55, frag_weight=0.45, tos_weight=0.35,
    scurve_weight=0.40, discovery_weight=0.35, reform_weight=0.25,
    profile_label="Frontier Mandate"
)

DEFENSIVE = RAOConfig(
    opp_weight=0.65, frag_weight=0.35, tos_weight=0.15,
    scurve_weight=0.30, discovery_weight=0.35, reform_weight=0.35,
    cvar_threshold=0.20,
    profile_label="Defensive"
)

HIGH_CONVICTION = RAOConfig(
    opp_weight=0.52, frag_weight=0.48, tos_weight=0.40,
    scurve_weight=0.40, discovery_weight=0.30, reform_weight=0.30,
    cvar_threshold=0.30,
    anomaly_haircut=0.25,
    profile_label="High-Conviction Pre-Breakout"
)
