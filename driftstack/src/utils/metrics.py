
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

@dataclass
class DriftEvent:
    t: int
    detector: str
    score: float
    kind: str = "unknown"   # abrupt/gradual/incremental/recurring/unknown

@dataclass
class Metrics:
    preq_correct: int = 0
    seen: int = 0
    runtime_ms: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)  # placeholder, fill later
    detections: List[DriftEvent] = field(default_factory=list)
    true_drifts: List[int] = field(default_factory=list)

    def update_preq(self, y_true, y_pred):
        self.seen += 1
        if y_true == y_pred:
            self.preq_correct += 1

    @property
    def preq_accuracy(self):
        return self.preq_correct / max(1, self.seen)

    def mttd(self, tolerance=0):
        # mean time-to-detect against true drifts list (greedy match forward)
        if not self.true_drifts or not self.detections:
            return None
        det_ts = [d.t for d in self.detections]
        delays = []
        j = 0
        for td in self.true_drifts:
            while j < len(det_ts) and det_ts[j] < td - tolerance:
                j += 1
            if j < len(det_ts):
                delays.append(max(0, det_ts[j] - td))
                j += 1
        return np.mean(delays) if delays else None

    def confusion_counts(self, window=200, tolerance=0):
        # TP if a detection within [td, td+window]
        if not self.true_drifts:
            return 0, 0, len(self.detections)
        det_ts = [d.t for d in self.detections]
        TP = 0; used = set()
        for td in self.true_drifts:
            for i, t in enumerate(det_ts):
                if i in used: 
                    continue
                if td - tolerance <= t <= td + window:
                    TP += 1; used.add(i); break
        FP = len(det_ts) - len(used)
        FN = max(0, len(self.true_drifts) - TP)
        return TP, FP, FN
