"""All parameters are experimental assumptions, not empirical estimates."""
from dataclasses import asdict, dataclass
import math


MECHANISMS = ("information", "liquidity", "herding", "deleveraging")
CONTROLS = ("baseline", "no_social", "homogeneous", "random_emotion", "simple_rule", "no_shock")


@dataclass(frozen=True)
class Config:
    agents: int = 32
    assets: int = 8
    steps: int = 90
    shock_step: int = 30
    shock_size: float = 0.12
    stress_duration: int = 10
    recovery_steps: int = 10
    lookback: int = 5
    horizons: tuple = (1, 5, 20)
    cost_bps_per_side: tuple = (0.0, 10.0, 30.0)
    agent_fee_bps: float = 2.0
    impact: float = 0.045
    depth_fraction: float = 0.035
    value_anchor: float = 0.6
    emotion_decay: float = 0.82
    panic_threshold: float = 0.35
    disagreement_cuts: tuple = (0.02, 0.06)
    seed: int = 42

    def __post_init__(self):
        for name in ("agents", "assets", "steps", "shock_step", "stress_duration", "recovery_steps", "lookback", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.agents < 4 or self.assets < 4 or self.seed < 0:
            raise ValueError("Require >=4 agents, >=4 assets and a nonnegative seed")
        if self.lookback < 1 or self.shock_step < max(20, self.lookback) or self.steps <= self.shock_step + 3:
            raise ValueError("Need >=20 pre-shock steps and a post-shock evaluation window")
        if self.stress_duration < 1 or self.recovery_steps < 1:
            raise ValueError("Durations must be positive")
        if not self.horizons or any(type(h) is not int or h < 1 for h in self.horizons):
            raise ValueError("Horizons must be positive integer trading steps")
        if max(self.horizons) + self.shock_step + 3 >= self.steps:
            raise ValueError("Increase steps to cover the fixed event signal plus longest horizon")
        for name in ("shock_size", "agent_fee_bps", "impact", "depth_fraction", "value_anchor", "emotion_decay", "panic_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.shock_size > 0.5 or self.value_anchor > 1 or self.impact > 0.2 or self.depth_fraction <= 0 or self.agent_fee_bps >= 1000:
            raise ValueError("Parameter outside the supported experimental range")
        if not 0 <= self.emotion_decay < 1 or not 0 < self.panic_threshold < 1:
            raise ValueError("Invalid emotion persistence or threshold")
        if not self.cost_bps_per_side or any(not math.isfinite(c) or not 0 <= c < 1000 for c in self.cost_bps_per_side):
            raise ValueError("Costs must be finite, in [0,1000) bps per side")
        if len(self.disagreement_cuts) != 2 or not 0 < self.disagreement_cuts[0] < self.disagreement_cuts[1] < 1:
            raise ValueError("Need two increasing, predeclared disagreement thresholds")

    def to_dict(self):
        return asdict(self)
