import json, time, os, sys
import numpy as np
from confluent_kafka import Producer
from pathlib import Path

# Import shared configuration
from config import BROKERS, TOPIC

# Make experiments/backup importable for gen_random
REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DATA_DIR = REPO_ROOT / "experiments" / "backup"
if str(GEN_DATA_DIR) not in sys.path:
    sys.path.append(str(GEN_DATA_DIR))

from gen_data import gen_random

# Allow environment variable overrides
BROKERS = os.getenv("BROKERS", BROKERS)
TOPIC = os.getenv("TOPIC", TOPIC)


p = Producer({"bootstrap.servers": BROKERS})

def emit(record):
    p.produce(TOPIC, json.dumps(record).encode("utf-8"))
    p.poll(0)

def main():
    # Stream forever in randomized segments with drift using gen_random
    global_idx = 0
    rng = np.random.default_rng(123)
    while True:
        # Randomize segment parameters similar to experiments
        length = int(rng.integers(3000, 12000))
        number = int(rng.integers(1, 6))
        intens = float(rng.uniform(0.1, 0.6))
        dims = 2
        dist = "unif"
        alt = True
        X, y = gen_random(number=number, dims=dims, intens=intens, dist=dist, alt=alt, length=length)

        for i in range(length):
            x = X[i]
            drift_indicator = int(y[i])  # y is the drift indicator from gen_random
            rec = {"ts": time.time(), "idx": global_idx, "x": x.tolist(), "drift": drift_indicator}
            emit(rec)
            global_idx += 1
            time.sleep(0.002)

    p.flush()

if __name__ == "__main__":
    main()
