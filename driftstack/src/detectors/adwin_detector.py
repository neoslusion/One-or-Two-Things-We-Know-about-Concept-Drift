
from river.drift import ADWIN

class ADWINDetector:
    def __init__(self, delta=0.002, name='adwin'):
        self.name = name
        self.detector = ADWIN(delta=delta)

    def update(self, value) -> float:
        # value: error stream (0/1) or numeric metric
        in_drift, _ = self.detector.update(value)
        return 1.0 if in_drift else 0.0
