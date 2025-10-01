import os, csv
import numpy as np
import matplotlib.pyplot as plt

def load_points(log_path):
    idx, score, p_min, drift, est_cp = [], [], [], [], []
    with open(log_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            idx.append(int(row["batch_end_idx"]))
            score.append(float(row["score"]))
            p_min.append(float(row.get("p_min", "nan")))
            drift.append(int(row["drift"]))
            est_cp.append(int(row.get("est_change_idx", -1)))
    return np.array(idx), np.array(score), np.array(p_min), np.array(drift), np.array(est_cp)

def main():
    log_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    idx, score, p_min, drift, est_cp = load_points(log_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: shape-adaptive statistics (score) and detected drift points
    axes[0].plot(idx, score, label="max peak score", color="#1f77b4")
    axes[0].scatter(idx[drift==1], score[drift==1], color="red", s=24, label="DRIFT (batch)")
    # Mark estimated change points
    valid_cp = est_cp[est_cp >= 0]
    if valid_cp.size > 0:
        axes[0].vlines(valid_cp, ymin=min(score)*0.95, ymax=max(score)*1.05, colors="red", linestyles=":", alpha=0.4, label="est. change point")
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Bottom: p-values
    if np.isfinite(p_min).any():
        axes[1].plot(idx, p_min, label="min p-value", color="#2ca02c")
        axes[1].axhline(0.05, color="#d62728", linestyle="--", alpha=0.7, label="alpha=0.05")
        axes[1].set_ylabel("p-value")
        axes[1].set_xlabel("sample index (batch end)")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


