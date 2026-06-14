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
    """Create a two-panel plot illustrating all 9 SE-CDT signal features."""
    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # -------------------------------------------------------------
    # PANEL 1: Single Peak Anatomy (WR, SNR, LTS, SDS, MS)
    # -------------------------------------------------------------
    np.random.seed(42)  # For reproducible noise
    t1 = np.linspace(0, 150, 150)
    center1 = 75
    baseline1 = 0.2
    noise1 = np.random.normal(0, 0.015, size=150)
    
    peak_height1 = 0.7
    sigma1 = 12
    signal1 = baseline1 + peak_height1 * np.exp(-0.5 * ((t1 - center1) / sigma1)**2) + noise1
    
    # Plot signal 1
    ax1.plot(t1, signal1, color='#1f77b4', lw=2.5, label=r'MMD Signal $\sigma(t)$')
    
    # Peak point & label
    max_idx = np.argmax(signal1)
    t_peak1 = t1[max_idx]
    h_peak1 = signal1[max_idx]
    ax1.plot(t_peak1, h_peak1, 'ro', markersize=8, label='Detected Peak')
    ax1.text(t_peak1, h_peak1 + 0.03, r'$t_{peak}$', ha='center', fontweight='bold')
    
    # Baseline
    ax1.axhline(baseline1, color='gray', linestyle='--', alpha=0.7)
    ax1.text(10, baseline1 - 0.05, 'Baseline (Median)', color='gray', fontsize=10)
    
    # SNR Annotation
    ax1.annotate('', xy=(130, h_peak1), xytext=(130, baseline1),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(128, (baseline1 + h_peak1)/2, 'SNR = max / median\n(Signal-to-Noise)', color='green', ha='right', va='center', fontsize=9)
    
    # FWHM / WR Annotation
    half_height = baseline1 + (h_peak1 - baseline1) / 2
    idx_left = np.where(signal1[:center1] >= half_height)[0][0]
    idx_right = np.where(signal1[center1:] >= half_height)[0][-1] + center1
    t_l, t_r = t1[idx_left], t1[idx_right]
    
    ax1.hlines(y=half_height, xmin=t_l, xmax=t_r, color='red', linestyle='-', lw=2.5)
    ax1.plot([t_l, t_r], [half_height, half_height], 'rx', markersize=8)
    ax1.text((t_l + t_r)/2, half_height + 0.03, 'FWHM (Full Width at Half Max)', color='red', ha='center', fontsize=9)
    ax1.text((t_l + t_r)/2, half_height - 0.07, r'$\rightarrow$ WR = FWHM / $2l_1$', color='red', ha='center', fontsize=9, fontweight='bold')
    
    # Pre-peak & Post-peak slopes (LTS, SDS, MS)
    # Pre-peak slope annotation
    ax1.annotate('Pre-peak Slope\n(LTS, SDS, MS)', xy=(center1 - 15, baseline1 + 0.3), xytext=(center1 - 65, baseline1 + 0.5),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, connectionstyle="arc3,rad=-0.1"))
    # Post-peak slope annotation
    ax1.annotate('Post-peak Slope\n(Trend Decay)', xy=(center1 + 20, baseline1 + 0.25), xytext=(center1 + 40, baseline1 + 0.45),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, connectionstyle="arc3,rad=0.1"))
    
    ax1.set_ylim(0.0, 1.1)
    ax1.set_xlabel('Time Steps (t)', fontsize=11)
    ax1.set_ylabel('MMD Distance', fontsize=11)
    ax1.set_title('1. Peak-Level Geometry & Temporal Trends', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # -------------------------------------------------------------
    # PANEL 2: Multi-Peak Anatomy (n_p, CV, PPR, DPAR)
    # -------------------------------------------------------------
    t2 = np.linspace(0, 200, 200)
    baseline2 = 0.2
    noise2 = np.random.normal(0, 0.015, size=200)
    
    # Generate 3 peaks (recurrent/multi-peak pattern)
    p1_center, p1_height, p1_sigma = 50, 0.6, 8
    p2_center, p2_height, p2_sigma = 110, 0.55, 8
    p3_center, p3_height, p3_sigma = 165, 0.5, 8
    
    signal2 = (baseline2 + 
               p1_height * np.exp(-0.5 * ((t2 - p1_center) / p1_sigma)**2) +
               p2_height * np.exp(-0.5 * ((t2 - p2_center) / p2_sigma)**2) +
               p3_height * np.exp(-0.5 * ((t2 - p3_center) / p3_sigma)**2) +
               noise2)
    
    # Plot signal 2
    ax2.plot(t2, signal2, color='#2ca02c', lw=2.5, label=r'MMD Signal $\sigma(t)$')
    
    # Mark peaks
    peaks_x = [p1_center, p2_center, p3_center]
    peaks_y = [signal2[p1_center], signal2[p2_center], signal2[p3_center]]
    ax2.plot(peaks_x, peaks_y, 'ro', markersize=8, label='Detected Peaks')
    
    # Label np
    ax2.text(20, 0.95, r'Peak Count $n_p = 3$', fontsize=11, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Peak spacing for CV
    ax2.annotate('', xy=(p1_center, 0.35), xytext=(p2_center, 0.35),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax2.text((p1_center + p2_center)/2, 0.38, r'$\Delta P_1$', color='purple', ha='center', fontweight='bold')
    
    ax2.annotate('', xy=(p2_center, 0.35), xytext=(p3_center, 0.35),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax2.text((p2_center + p3_center)/2, 0.38, r'$\Delta P_2$', color='purple', ha='center', fontweight='bold')
    
    ax2.text(110, 0.23, r'Periodicity $CV = \frac{\mathrm{std}(\Delta P)}{\mathrm{mean}(\Delta P)}$', color='purple', ha='center', fontsize=9)
    
    # PPR (Peak Proximity Ratio between top 2 peaks: peak 1 and peak 2)
    ax2.annotate('', xy=(p1_center, 0.82), xytext=(p2_center, 0.82),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax2.text((p1_center + p2_center)/2, 0.85, 'PPR = $|p_2 - p_1| / T$\n(Proximity of Top 2)', color='orange', ha='center', fontsize=9)
    
    # DPAR (Dual-Peak Amplitude Ratio between top 2 peaks)
    h1_val = peaks_y[0]
    h2_val = peaks_y[1]
    ax2.axhline(h1_val, color='red', linestyle=':', alpha=0.5, xmin=p1_center/200, xmax=p2_center/200)
    ax2.axhline(h2_val, color='red', linestyle=':', alpha=0.5, xmin=p1_center/200, xmax=p2_center/200)
    ax2.annotate('', xy=(p2_center + 5, h2_val), xytext=(p2_center + 5, h1_val),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax2.text(p2_center + 8, (h1_val + h2_val)/2, r'DPAR = $\frac{\min(h_1, h_2)}{\max(h_1, h_2)}$', color='red', va='center', fontsize=9)
    
    ax2.set_ylim(0.0, 1.1)
    ax2.set_xlabel('Time Steps (t)', fontsize=11)
    ax2.set_ylabel('MMD Distance', fontsize=11)
    ax2.set_title('2. Multi-Peak Geometry & Periodicity', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = PLOTS_DIR / "theory_peak_anatomy.png"
    plt.savefig(out_path, dpi=200)
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
