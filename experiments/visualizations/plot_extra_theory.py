import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from core.config import PLOTS_DIR

def setup_matplotlib_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def create_sliding_window_illustration():
    setup_matplotlib_style()
    fig = plt.figure(figsize=(12, 7))
    
    # Create grid for layout
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
    ax_data = fig.add_subplot(gs[0])
    ax_mmd = fig.add_subplot(gs[1])
    
    # 1. Top Plot: Data Stream
    t = np.linspace(0, 400, 400)
    drift_point = 200
    
    # Generate data stream with a drift (mean shift)
    data_pre = np.random.normal(0, 0.5, size=200)
    data_post = np.random.normal(2, 0.5, size=200)
    data = np.concatenate([data_pre, data_post])
    
    ax_data.scatter(t[:200], data[:200], c='gray', alpha=0.5, s=10, label='Concept A')
    ax_data.scatter(t[200:], data[200:], c='teal', alpha=0.5, s=10, label='Concept B')
    ax_data.axvline(drift_point, color='red', linestyle='--', alpha=0.5)
    ax_data.text(drift_point, 3.5, 'True Drift Point', color='red', ha='center')
    
    # Draw Reference and Test Windows
    # Position them so the boundary is exactly at the drift point to show maximum MMD
    window_center = drift_point
    w_ref_len = 50
    w_test_len = 150
    
    ref_start = window_center - w_ref_len
    test_end = window_center + w_test_len
    
    rect_ref = patches.Rectangle((ref_start, -2), w_ref_len, 6, linewidth=2, 
                                 edgecolor='blue', facecolor='blue', alpha=0.15, label='Reference Window ($W_{ref}$)')
    rect_test = patches.Rectangle((window_center, -2), w_test_len, 6, linewidth=2, 
                                  edgecolor='orange', facecolor='orange', alpha=0.15, label='Test Window ($W_{test}$)')
    
    ax_data.add_patch(rect_ref)
    ax_data.add_patch(rect_test)
    ax_data.text(ref_start + w_ref_len/2, 2.5, '$W_{ref}$', color='blue', ha='center', fontweight='bold')
    ax_data.text(window_center + w_test_len/2, 2.5, '$W_{test}$', color='orange', ha='center', fontweight='bold')
    
    ax_data.set_ylim(-2, 4)
    ax_data.set_xlim(0, 400)
    ax_data.set_title('A) Data Stream & Sliding Windows')
    ax_data.set_ylabel('Feature Value (X)')
    ax_data.set_xticks([]) # Hide x ticks for top plot
    ax_data.spines['top'].set_visible(False)
    ax_data.spines['right'].set_visible(False)
    ax_data.spines['bottom'].set_visible(False)
    
    # 2. Bottom Plot: MMD Signal
    # Simulate MMD signal
    mmd_signal = np.zeros(400)
    for i in range(50, 350):
        # MMD is high when W_ref and W_test cover different concepts
        dist = abs(i - drift_point)
        if dist < 60:
            mmd_signal[i] = 0.8 * (1 - dist/60)
        else:
            mmd_signal[i] = np.random.normal(0.05, 0.02)
            
    mmd_signal[:50] = np.nan
    mmd_signal[350:] = np.nan
            
    ax_mmd.plot(t, mmd_signal, color='purple', lw=2)
    ax_mmd.axvline(drift_point, color='red', linestyle='--', alpha=0.5)
    
    # Highlight the specific MMD point corresponding to the windows above
    ax_mmd.plot(window_center, mmd_signal[window_center], 'ro', markersize=8)
    
    # Draw arrow from top plot to bottom plot
    ax_mmd.annotate('MMD Calculation\n$\sigma(t) = \mathrm{MMD}(W_{ref}, W_{test})$', 
                    xy=(window_center, mmd_signal[window_center] + 0.05), 
                    xytext=(window_center, 1.0),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    ha='center', va='bottom')
    
    ax_mmd.set_ylim(0, 1.2)
    ax_mmd.set_xlim(0, 400)
    ax_mmd.set_title('B) Resulting MMD Distance Signal $\sigma(t)$')
    ax_mmd.set_xlabel('Time Step ($t$)')
    ax_mmd.set_ylabel('MMD Distance')
    ax_mmd.spines['top'].set_visible(False)
    ax_mmd.spines['right'].set_visible(False)
    
    out_path = PLOTS_DIR / "theory_sliding_window.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def create_idw_weighting_illustration():
    setup_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate 2D Gaussian Blob
    np.random.seed(42)
    x = np.random.normal(0, 1, 500)
    y = np.random.normal(0, 1, 500)
    data = np.vstack([x, y])
    
    # Calculate density using KDE
    kde = gaussian_kde(data)
    density = kde(data)
    
    # Calculate Inverse Density Weights (IDW)
    # Adding a small epsilon to prevent division by zero
    weights = 1.0 / (np.sqrt(density) + 0.1)
    
    # Normalize weights for visualization purposes
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Plot 1: Standard MMD (Uniform Weighting)
    ax1 = axes[0]
    # Uniform size and opacity
    ax1.scatter(x, y, s=40, c='royalblue', alpha=0.6, edgecolors='none')
    ax1.set_title('A) Standard MMD (Uniform Weights)\nAll points contribute equally')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Annotate dense region
    ax1.annotate('High Density Core\n(Dominates distance metric)', xy=(0, 0), xytext=(2, 2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
    
    # Plot 2: IDW-MMD (Inverse Density Weighting)
    ax2 = axes[1]
    
    # Map weights to size and alpha
    sizes = 10 + 150 * weights_norm
    alphas = 0.2 + 0.8 * weights_norm
    
    # Scatter plot with varying size and color
    scatter = ax2.scatter(x, y, s=sizes, c=weights_norm, cmap='YlOrRd', alpha=0.7, edgecolors='gray', linewidth=0.5)
    
    ax2.set_title('B) IDW-MMD (Inverse Density Weights)\nBoundary points are amplified, core is suppressed')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Annotations
    ax2.annotate('Low Weight Core\n(Suppressed noise)', xy=(0, 0), xytext=(2, 2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
    
    # Find a boundary point
    boundary_idx = np.argmax(weights)
    ax2.annotate('High Weight Boundary\n(Sensitive to drift)', xy=(x[boundary_idx], y[boundary_idx]), 
                 xytext=(x[boundary_idx]-1.5, y[boundary_idx]-1.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Weight Magnitude (w_i)')
    
    out_path = PLOTS_DIR / "theory_idw_weighting.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    print("Generating Extra SE-CDT theory visualizations...")
    create_sliding_window_illustration()
    create_idw_weighting_illustration()
    print("Done.")
