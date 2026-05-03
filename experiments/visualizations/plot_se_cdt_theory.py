import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from core.config import PLOTS_DIR

def setup_matplotlib_style():
    """Setup consistent matplotlib style."""
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

def create_peak_anatomy_plot():
    """Create a plot illustrating SE-CDT signal peak anatomy."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate synthetic peak
    t = np.linspace(0, 200, 200)
    center = 100
    baseline = 0.2
    noise = np.random.normal(0, 0.02, size=200)
    
    # Triangular/Gaussian-like peak
    peak_height = 0.8
    sigma = 15
    signal = baseline + peak_height * np.exp(-0.5 * ((t - center) / sigma)**2) + noise
    
    # Plot signal
    ax.plot(t, signal, color='blue', alpha=0.8, label=r'MMD Signal $\sigma(t)$')
    
    # Annotate Peak
    t_peak = center
    h_peak = baseline + peak_height
    ax.plot(t_peak, h_peak, 'ro', markersize=8)
    ax.annotate(r'$t_{peak}$', xy=(t_peak, h_peak), xytext=(t_peak - 10, h_peak + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    
    # Annotate Baseline
    ax.axhline(baseline, color='gray', linestyle='--', alpha=0.7)
    ax.text(10, baseline + 0.05, 'Baseline Threshold', color='gray')
    
    # Annotate FWHM (Half Width)
    half_height = baseline + peak_height / 2
    idx_left = np.where(signal[:100] >= half_height)[0][0]
    idx_right = np.where(signal[100:] >= half_height)[0][-1] + 100
    
    ax.hlines(y=half_height, xmin=t[idx_left], xmax=t[idx_right], color='red', linestyle='-', lw=2)
    ax.text(center, half_height - 0.05, r'FWHM (Full Width at Half Maximum)' + '\n' + r'$\rightarrow$ Width Ratio (WR)', 
            color='red', ha='center', va='top')
    
    # Annotate Prominence / SNR
    ax.annotate('', xy=(150, h_peak), xytext=(150, baseline),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(152, baseline + peak_height/2, 'Prominence (SNR > 2.0)', color='green', va='center')
    
    # Annotate Left and Right windows for LTS calculation
    rect_left = patches.Rectangle((t_peak - 40, 0.1), 30, 0.9, linewidth=1, edgecolor='purple', facecolor='purple', alpha=0.1)
    rect_right = patches.Rectangle((t_peak + 10, 0.1), 30, 0.9, linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.1)
    ax.add_patch(rect_left)
    ax.add_patch(rect_right)
    ax.text(t_peak - 25, 0.95, 'Left Window\n(Pre-peak)', color='purple', ha='center', fontsize=10)
    ax.text(t_peak + 25, 0.95, 'Right Window\n(Post-peak)', color='orange', ha='center', fontsize=10)
    
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('MMD Distance')
    
    # Hide top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = PLOTS_DIR / "theory_peak_anatomy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def create_drift_shapes_comparison():
    """Create a 2x2 grid comparing drift signal shapes."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    t = np.linspace(0, 300, 300)
    center = 150
    noise = np.random.normal(0, 0.02, size=300)
    baseline = 0.2
    
    # 1. Sudden Drift (Triangle / Sharp Peak)
    sig_sudden = baseline + noise.copy()
    peak_width = 30
    for i in range(300):
        dist = abs(i - center)
        if dist < peak_width:
            sig_sudden[i] += 0.8 * (1 - dist/peak_width)
    axes[0].plot(t, sig_sudden, 'b-')
    axes[0].set_title('A) Sudden Drift\n(Sharp, symmetric peak, low WR)')
    axes[0].annotate('Peak (TCD)', xy=(center, 1.0), xytext=(center+30, 0.9),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    # 2. Gradual Drift (Plateau / Broad / Low LTS)
    sig_gradual = baseline + noise.copy()
    plateau_width = 60
    for i in range(300):
        dist = abs(i - center)
        if dist < plateau_width:
            sig_gradual[i] += 0.5 * (1 - (dist/plateau_width)**2)
    axes[1].plot(t, sig_gradual, 'g-')
    axes[1].set_title('B) Gradual Drift\n(Broad, plateau, low LTS)')
    axes[1].text(center, 0.6, 'Oscillation Zone (PCD)', color='green', ha='center')
    
    # 3. Incremental Drift (Ramp / Monotonic)
    sig_inc = baseline + noise.copy()
    for i in range(300):
        if i > 100 and i < 250:
            sig_inc[i] += 0.6 * ((i - 100) / 150)
        elif i >= 250:
            sig_inc[i] += 0.6 - 0.6 * ((i - 250) / 50) # drops back eventually or stays high
            if sig_inc[i] < baseline: sig_inc[i] = baseline + noise[i]
    axes[2].plot(t, sig_inc, 'r-')
    axes[2].set_title('C) Incremental Drift\n(Ramp-like, high LTS, high MS)')
    
    # Draw trend arrow
    axes[2].annotate('', xy=(200, 0.6), xytext=(120, 0.3),
                     arrowprops=dict(arrowstyle='->', color='red', lw=3, ls='--'))
    
    # 4. Blip Drift (Extremely sharp spike)
    sig_blip = baseline + noise.copy()
    blip_width = 5
    for i in range(300):
        dist = abs(i - center)
        if dist < blip_width:
            sig_blip[i] += 0.9 * (1 - dist/blip_width)
    axes[3].plot(t, sig_blip, 'm-')
    axes[3].set_title('D) Blip Drift\n(Extremely sharp spike, WR < 0.05)')
    
    for ax in axes:
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('MMD Signal')
        ax.axhline(baseline, color='gray', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    out_path = PLOTS_DIR / "theory_drift_shapes.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    print("Generating SE-CDT theory visualizations...")
    create_peak_anatomy_plot()
    create_drift_shapes_comparison()
    print("Done.")
