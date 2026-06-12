import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure output directory exists
out_dir = 'report/latex/image'
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. MMD Intuition Visualization
# ---------------------------------------------------------
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: P and Q are similar (Small MMD)
P_x = np.random.normal(0, 1, 200)
P_y = np.random.normal(0, 1, 200)
Q_x = np.random.normal(0.2, 1, 200)
Q_y = np.random.normal(-0.2, 1, 200)

axes[0].scatter(P_x, P_y, alpha=0.6, label='Reference Window (P)', c='#003366', edgecolors='w', s=50)
axes[0].scatter(Q_x, Q_y, alpha=0.6, label='Current Window (Q)', c='#cc0000', edgecolors='w', s=50)
axes[0].set_title("No Drift: P \u2248 Q \n(MMD is close to 0)", fontsize=14, pad=15)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].legend(loc='upper right')

# Right: P and Q are different (Large MMD)
P_x = np.random.normal(0, 1, 200)
P_y = np.random.normal(0, 1, 200)
Q_x = np.random.normal(3, 1, 200)
Q_y = np.random.normal(2, 1, 200)

axes[1].scatter(P_x, P_y, alpha=0.6, label='Reference Window (P)', c='#003366', edgecolors='w', s=50)
axes[1].scatter(Q_x, Q_y, alpha=0.6, label='Current Window (Q)', c='#cc0000', edgecolors='w', s=50)

# Draw an arrow to represent MMD distance
axes[1].annotate('', xy=(3, 2), xytext=(0, 0), arrowprops=dict(arrowstyle='<->', color='black', lw=2))
axes[1].text(1.5, 1.2, 'MMD Distance', fontsize=12, ha='center', va='center', rotation=30, backgroundcolor='white')

axes[1].set_title("Concept Drift: P \u2260 Q \n(MMD is large)", fontsize=14, pad=15)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mmd_intuition.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 2. ShapeDD Triangle Visualization
# ---------------------------------------------------------
# We'll simulate a 1D sequence with a sudden shift and compute a rolling MMD trace
# (or just a theoretical mock-up for cleaner visualization)
t = np.arange(0, 2000)
# Data distribution means
data_mean = np.zeros_like(t, dtype=float)
data_mean[1000:] = 2.0

window_size = 200 # l_1 = 100
l = window_size // 2

# Theoretical MMD trace (triangle)
mmd_trace = np.zeros_like(t, dtype=float)
drift_point = 1000
for i in range(len(t)):
    # distance to drift point
    dist = abs(i - drift_point)
    if dist < l:
        mmd_trace[i] = 1.0 * (1 - dist / l)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

# Top: Raw Data Stream
axes[0].plot(t, data_mean, color='#003366', lw=3, label='Concept Mean')
axes[0].axvline(1000, color='#cc0000', linestyle='--', lw=2, label='Drift Occurs')
axes[0].set_ylabel("Data Value")
axes[0].set_title("Streaming Data (Sudden Drift at t=1000)", fontsize=12)
axes[0].legend(loc='upper left')

# Bottom: ShapeDD MMD Trace
axes[1].plot(t, mmd_trace, color='#008000', lw=3, label='MMD Trace $\sigma(t)$')
axes[1].axvline(1000, color='#cc0000', linestyle='--', lw=2)

# Highlight the triangle
axes[1].fill_between(t, mmd_trace, color='#008000', alpha=0.2)
axes[1].annotate('Triangle Shape\nTheorem', xy=(1000, 1.0), xytext=(1150, 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=11, ha='left')

axes[1].set_xlabel("Time step (t)")
axes[1].set_ylabel("MMD Distance")
axes[1].set_title("ShapeDD Output: The Triangle Signature", fontsize=12)
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'shapedd_intuition.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Images generated successfully in report/latex/image/")
