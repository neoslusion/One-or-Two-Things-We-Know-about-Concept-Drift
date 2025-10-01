import json, time, os, sys
import numpy as np
from confluent_kafka import Producer
from pathlib import Path

# Make experiments/backup importable for gen_random
REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DATA_DIR = REPO_ROOT / "experiments" / "backup"
if str(GEN_DATA_DIR) not in sys.path:
    sys.path.append(str(GEN_DATA_DIR))

from gen_data import gen_random

BROKERS = os.getenv("BROKERS", "localhost:9092")
TOPIC = os.getenv("TOPIC", "sensor.stream")


p = Producer({"bootstrap.servers": BROKERS})

def emit(record):
    p.produce(TOPIC, json.dumps(record).encode("utf-8"))
    p.poll(0)

def main():
    # Generate a full sequence with change points using shared utility
    n = 10000
    X, y = gen_random(number=10,  # number of change points
                      dims=2,     # number of dimensions
                      intens=0.5, # intensity of drift
                      dist="unif",# distribution type
                      alt=True,   # alternating drift
                      length=n)

    for i in range(n):
        x = X[i]
        rec = {"ts": time.time(), "idx": i, "x": x.tolist()}
        emit(rec)
        time.sleep(0.002)

    p.flush()

if __name__ == "__main__":
    main()
