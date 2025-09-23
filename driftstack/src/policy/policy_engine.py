
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class PolicyConfig:
    cooldown: int = 200
    hysteresis: float = 0.05
    actions: dict = None

class PolicyEngine:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.last_action_t = -10**9

    def decide(self, t: int, detector_name: str, score: float, drift_kind: str = 'abrupt') -> Optional[str]:
        if t - self.last_action_t < self.cfg.cooldown:
            return None
        action = (self.cfg.actions or {}).get(drift_kind, 'reset')
        # hysteresis hook: require score beyond margin to fire again
        self.last_action_t = t
        return action
