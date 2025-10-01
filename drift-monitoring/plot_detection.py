import os, csv
import numpy as np
import matplotlib.pyplot as plt

def load_points(log_path):
    idx, score, drift = [], [], []
    with open(log_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            idx.append(int(row["batch_end_idx"]))
            score.append(float(row["score"]))
            drift.append(int(row["drift"]))
    return np.array(idx), np.array(score), np.array(drift)

def main():
    log_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    idx, score, drift = load_points(log_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(idx, score, label="batch score (max peak)")
    ax.scatter(idx[drift==1], score[drift==1], color="red", s=30, label="DRIFT")
    ax.set_xlabel("sample index (batch end)")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


