"""
Producer matching DriftMonitoring.ipynb notebook.
Generates SEA dataset with sudden drift for real-time streaming.
"""
import json, time, os
import numpy as np
from confluent_kafka import Producer
from river.datasets import synth

# Import shared configuration
from config import BROKERS, TOPIC

# Allow environment variable overrides
BROKERS = os.getenv("BROKERS", BROKERS)
TOPIC = os.getenv("TOPIC", TOPIC)

# Data generation parameters (matching DriftMonitoring.ipynb)
STREAM_SIZE = int(os.getenv("STREAM_SIZE", "10000"))
DRIFT_POSITION = int(os.getenv("DRIFT_POSITION", "1500"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


def generate_stream_with_sudden_drift(total_size, drift_position, seed=42):
    """
    Generate stream with sudden drift matching DriftMonitoring.ipynb.
    
    - Pre-drift: SEA variant 0 (baseline concept)
    - Post-drift: SEA variant 3 + moderate transformation
    """
    np.random.seed(seed)
    X_list, y_list = [], []
    
    print(f"Generating pre-drift data (SEA variant 0, {drift_position} samples)...")
    stream_pre = synth.SEA(seed=seed, variant=0)
    for i, (x, y) in enumerate(stream_pre.take(drift_position)):
        X_list.append(list(x.values()))
        y_list.append(y)
    
    print(f"Generating post-drift data (SEA variant 3 + transformation, {total_size - drift_position} samples)...")
    stream_post = synth.SEA(seed=seed + 100, variant=3)
    for i, (x, y) in enumerate(stream_post.take(total_size - drift_position)):
        x_vals = list(x.values())
        # Moderate transformation: detectable but learnable
        x_transformed = [
            x_vals[0] * 1.8 + 5.0,
            x_vals[1] * 1.5 - 3.0,
            x_vals[2] * 2.0 + 8.0
        ]
        X_list.append(x_transformed)
        y_list.append(y)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Verify drift
    pre_mean = np.mean(X[:drift_position], axis=0)
    post_mean = np.mean(X[drift_position:], axis=0)
    print(f"\n✓ SUDDEN DRIFT created at sample {drift_position}")
    print(f"  Pre-drift means:  [{pre_mean[0]:.2f}, {pre_mean[1]:.2f}, {pre_mean[2]:.2f}]")
    print(f"  Post-drift means: [{post_mean[0]:.2f}, {post_mean[1]:.2f}, {post_mean[2]:.2f}]")
    print(f"  Feature shift magnitude: {np.linalg.norm(post_mean - pre_mean):.2f}\n")
    
    return X, y


def main():
    p = Producer({"bootstrap.servers": BROKERS})
    
    # Generate complete stream once (matching notebook)
    X_stream, y_stream = generate_stream_with_sudden_drift(
        total_size=STREAM_SIZE,
        drift_position=DRIFT_POSITION,
        seed=RANDOM_SEED
    )
    
    print(f"Starting streaming {STREAM_SIZE} samples to topic '{TOPIC}'...")
    print(f"Drift position: {DRIFT_POSITION}")
    print(f"Streaming rate: 1 sample every 0.002s\n")
    
    # Stream samples one by one
    for idx in range(len(X_stream)):
        x = X_stream[idx]
        y_label = int(y_stream[idx])
        
        # Create record with features and label
        rec = {
            "ts": time.time(),
            "idx": idx,
            "x": x.tolist(),
            "y": y_label,  # Include ground truth label for training
            "drift_indicator": 1 if idx >= DRIFT_POSITION else 0  # True drift marker
        }
        
        p.produce(TOPIC, json.dumps(rec).encode("utf-8"))
        p.poll(0)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Sent {idx + 1}/{STREAM_SIZE} samples...")
        
        time.sleep(0.002)  # 2ms per sample
    
    p.flush()
    print(f"\n✓ Streaming complete - {STREAM_SIZE} samples sent")
    print("Stream will restart in 5 seconds...")
    time.sleep(5)

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nProducer stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Restarting in 5 seconds...")
            time.sleep(5)
