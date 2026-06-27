"""
Variance Ratio (VR) theory visualization.

Generates `theory_variance_ratio.png`: a figure illustrating WHY the Variance
Ratio feature separates Gradual drift (probabilistic mixture -> variance
inflates) from Incremental drift (continuous mean shift -> variance stays
flat), grounded in the Law of Total Variance.

The figure uses the SAME data generators as the benchmark
(ConceptDriftStreamGenerator) and the SAME VR computation as the SE-CDT
classifier (core/detectors/se_cdt.py), so the numbers shown are faithful to
what the system actually computes.

Style matches experiments/visualizations/plot_se_cdt_theory.py.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from core.config import PLOTS_DIR
from data.generators.drift_generators import ConceptDriftStreamGenerator


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
        'savefig.pad_inches': 0.1,
    })


def sliding_window_total_variance(X, l2, step=5):
    """Total (sum-over-dims) variance in each L2 sliding window.

    Mirrors the VR numerator in core/detectors/se_cdt.py:
        w_var = sum(np.var(data_window[i:i+l2], axis=0))
    """
    positions, variances = [], []
    for i in range(0, len(X) - l2 + 1, step):
        v = np.sum(np.var(X[i:i + l2], axis=0))
        positions.append(i + l2 // 2)   # center the window for plotting
        variances.append(v)
    return np.array(positions), np.array(variances)


def compute_vr(X, l1, l2, step=5):
    """Variance Ratio exactly as in core/detectors/se_cdt.py extract_features."""
    baseline_var = np.sum(np.var(X[:l1], axis=0))
    if baseline_var <= 1e-10:
        return 1.0, baseline_var
    max_var = baseline_var
    for i in range(0, len(X) - l2 + 1, step):
        w_var = np.sum(np.var(X[i:i + l2], axis=0))
        if w_var > max_var:
            max_var = w_var
    return max_var / baseline_var, baseline_var


def create_variance_ratio_plot():
    setup_matplotlib_style()

    # ---- Parameters (match SE-CDT defaults) ----
    l1, l2 = 50, 150
    length = 800
    drift_pos = 300
    transition_width = 250
    magnitude = 2.0
    seed = 42

    gen = ConceptDriftStreamGenerator(n_features=5, base_std=1.0, seed=seed)
    X_grad, _, _ = gen.generate_gradual_drift(length, drift_pos, transition_width, magnitude)

    gen2 = ConceptDriftStreamGenerator(n_features=5, base_std=1.0, seed=seed)
    X_inc, _, _ = gen2.generate_incremental_drift(length, drift_pos, transition_width, magnitude)

    vr_grad, base_grad = compute_vr(X_grad, l1, l2)
    vr_inc, base_inc = compute_vr(X_inc, l1, l2)

    pos_g, var_g = sliding_window_total_variance(X_grad, l2)
    pos_i, var_i = sliding_window_total_variance(X_inc, l2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # Use the first shifted feature for the 1-D raw-data illustration (top row).
    feat = 0
    t = np.arange(length)
    trans_end = drift_pos + transition_width

    # ---------- TOP-LEFT: Gradual raw data ----------
    ax = axes[0, 0]
    ax.scatter(t, X_grad[:, feat], s=6, alpha=0.5, color='green')
    ax.axvspan(drift_pos, trans_end, color='gray', alpha=0.12, label='Transition window')
    ax.set_title('A) Gradual drift — raw data (feature 0)\n'
                 'Mixture of two concepts $\\Rightarrow$ points scatter across BOTH levels')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Feature value')
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---------- TOP-RIGHT: Incremental raw data ----------
    ax = axes[0, 1]
    ax.scatter(t, X_inc[:, feat], s=6, alpha=0.5, color='red')
    ax.axvspan(drift_pos, trans_end, color='gray', alpha=0.12, label='Transition window')
    ax.set_title('B) Incremental drift — raw data (feature 0)\n'
                 'Single distribution, mean slides $\\Rightarrow$ tight band follows the ramp')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Feature value')
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---------- BOTTOM-LEFT: sliding-window variance ----------
    ax = axes[1, 0]
    ax.plot(pos_g, var_g, color='green', label='Gradual: total variance')
    ax.plot(pos_i, var_i, color='red', label='Incremental: total variance')
    ax.axhline(base_grad, color='green', ls='--', alpha=0.6,
               label=f'Gradual baseline ($l_1$={l1})')
    ax.axhline(base_inc, color='red', ls=':', alpha=0.6,
               label=f'Incremental baseline ($l_1$={l1})')
    ax.set_title('C) Sliding-window total variance (window $l_2$=%d)\n'
                 'Gradual variance SPIKES; Incremental stays near baseline' % l2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(r'$\sum_d \mathrm{Var}(X^{(d)})$ in window')
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---------- BOTTOM-RIGHT: resulting VR + decision bands ----------
    ax = axes[1, 1]
    tau_low, tau_high = 1.1, 1.3
    ax.axhspan(0.8, tau_low, color='red', alpha=0.10)
    ax.axhspan(tau_low, tau_high, color='gray', alpha=0.12)
    ax.axhspan(tau_high, max(vr_grad, vr_inc) * 1.15, color='green', alpha=0.10)

    bars = ax.bar(['Gradual', 'Incremental'], [vr_grad, vr_inc],
                  color=['green', 'red'], alpha=0.75, width=0.5)
    for b, v in zip(bars, [vr_grad, vr_inc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'VR = {v:.2f}',
                ha='center', va='bottom', fontweight='bold')

    ax.axhline(tau_high, color='green', ls='--', lw=1.5)
    ax.axhline(tau_low, color='red', ls='--', lw=1.5)
    ax.text(1.52, tau_high, r'$\tau_{VR}^{+}=1.3$', color='green', va='center', fontsize=9)
    ax.text(1.52, tau_low, r'$\tau_{VR}^{-}=1.1$', color='red', va='center', fontsize=9)
    ax.text(-0.45, (tau_high + max(vr_grad, vr_inc) * 1.15) / 2, 'Gradual\nzone',
            color='green', va='center', ha='center', fontsize=9)
    ax.text(-0.45, (tau_low + tau_high) / 2, 'Fallback\n(geometric)',
            color='gray', va='center', ha='center', fontsize=9)
    ax.text(-0.45, (0.8 + tau_low) / 2, 'Incremental\nzone',
            color='red', va='center', ha='center', fontsize=9)

    ax.set_ylim(0.8, max(vr_grad, vr_inc) * 1.15)
    ax.set_title('D) Resulting Variance Ratio & decision rule\n'
                 r'VR $> \tau_{VR}^{+}\Rightarrow$ Gradual;  VR $< \tau_{VR}^{-}\Rightarrow$ Incremental')
    ax.set_ylabel('Variance Ratio (VR)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = PLOTS_DIR / "theory_variance_ratio.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")
    print(f"  Gradual VR={vr_grad:.3f} (baseline={base_grad:.3f})")
    print(f"  Incremental VR={vr_inc:.3f} (baseline={base_inc:.3f})")


if __name__ == '__main__':
    print("Generating Variance Ratio theory visualization...")
    create_variance_ratio_plot()
    print("Done.")
