"""
Generate `theory_detection_flow.png` for the presentation.

Visualises the ShapeDD-IDW detection pipeline end-to-end on a synthetic stream
that contains a single known sudden drift:

    (a) Raw data stream            -> what arrives
    (b) MMD trace sigma(t)         -> compute MMD + isolate candidate peaks
    (c) Bonferroni validation      -> Gamma-null p-value per candidate vs the
                                      adjusted alpha = alpha / n_peaks; spurious
                                      peaks are rejected and the true drift
                                      survives -> a single confirmed event.

The trace, candidate peaks and p-values are produced by the *real* detector
(`shapedd_idw_mmd_proper` / `wmmd_gamma`), not hand-drawn, so the figure is
faithful to the implementation.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from core.config import PLOTS_DIR
from core.detectors.mmd_variants import shapedd_idw_mmd_proper, wmmd_gamma

# ---- house style (matches plot_se_cdt_theory.py) ---------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

UNIVBLUE = "#003366"
ALERTRED = "#cc0000"
DARKGREEN = "#0a7a0a"
GREY = "#888888"

# ---------------------------------------------------------------------------
# 1. Synthetic stream: stable -> sudden mean shift -> stable
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
DIM = 5
N_PRE, N_POST = 600, 600
TRUE_DRIFT = N_PRE

pre = rng.normal(0.0, 1.0, size=(N_PRE, DIM))
shift = np.zeros(DIM)
shift[:4] = 1.8                       # decisive sustained shift on a feature subset
post = rng.normal(0.0, 1.0, size=(N_POST, DIM)) + shift
X = np.vstack([pre, post])
# A short transient blip in the stable region: spikes the MMD trace but is too
# brief to pass the sustained two-sample test -> rejected by the Gamma p-value.
BLIP_T, BLIP_LEN = 290, 22
X[BLIP_T:BLIP_T + BLIP_LEN, :2] += 2.4
n = len(X)

L1, L2, ALPHA = 50, 150, 0.05

# ---------------------------------------------------------------------------
# 2. Run the real detector to obtain trace, confirmed positions, p-values
# ---------------------------------------------------------------------------
is_drift, positions, mmd_trace, conf_pvals = shapedd_idw_mmd_proper(
    X, l1=L1, l2=L2, alpha=ALPHA, weight_method="variance_reduction"
)

# Re-derive the trace x-axis exactly as the detector does
step = max(1, L1 // 2)
mmd_positions = np.array([i + L1 for i in range(0, n - L1 - L2, step)])
mmd_trace = np.asarray(mmd_trace)
assert len(mmd_positions) == len(mmd_trace), (len(mmd_positions), len(mmd_trace))

# Replicate the detector's peak-isolation step for display
smooth = gaussian_filter1d(mmd_trace, sigma=4)
threshold = smooth.mean() + smooth.std()
peak_idx, _ = find_peaks(smooth, height=threshold)
if len(peak_idx) == 0:
    peak_idx = np.array([int(np.argmax(mmd_trace))])
cand_pos = mmd_positions[peak_idx]

# Bonferroni-adjusted alpha and per-candidate Gamma-null p-values
adj_alpha = ALPHA / max(1, len(peak_idx))
cand_pvals = []
for pk in peak_idx:
    pos = mmd_positions[pk]
    a, b = max(0, pos - L2 // 2), min(n, pos + L2 // 2)
    s = pos - a
    if s < 10 or (b - a - s) < 10:
        cand_pvals.append(1.0)
        continue
    _, p = wmmd_gamma(X[a:b], s, "variance_reduction")
    cand_pvals.append(float(p))
cand_pvals = np.array(cand_pvals)
accepted = cand_pvals < adj_alpha

# ---------------------------------------------------------------------------
# 3. Plot: 3 stacked panels sharing the time axis
# ---------------------------------------------------------------------------
fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(9.2, 8.4), sharex=True,
    gridspec_kw={"height_ratios": [1.0, 1.35, 1.1], "hspace": 0.18},
)

# (a) raw stream ------------------------------------------------------------
ax0.plot(X[:, 0], color=GREY, lw=0.7, alpha=0.9)
ax0.axvline(TRUE_DRIFT, color=ALERTRED, ls="--", lw=2)
ax0.set_ylabel("Feature $x_1$")
ax0.set_title("(a) Raw data stream", loc="left", color=UNIVBLUE, fontweight="bold")
ax0.text(TRUE_DRIFT + 12, ax0.get_ylim()[1] * 0.62, "true drift",
         color=ALERTRED, fontsize=10, fontweight="bold")

# (b) MMD trace + peaks -----------------------------------------------------
ax1.plot(mmd_positions, mmd_trace, color=UNIVBLUE, lw=1.0, alpha=0.45,
         label=r"MMD trace $\sigma(t)$")
ax1.plot(mmd_positions, smooth, color=UNIVBLUE, lw=2.4, label="smoothed")
ax1.axhline(threshold, color=DARKGREEN, ls=":", lw=2,
            label=r"threshold ($\bar\sigma+\mathrm{std}$)")
# shade every contiguous run where the smoothed trace exceeds the threshold:
# each run is a burst of consecutive raw alarms that collapses to one event.
above = smooth > threshold
edges = np.diff(above.astype(int))
starts = mmd_positions[np.where(edges == 1)[0] + 1] if np.any(edges == 1) else []
ends = mmd_positions[np.where(edges == -1)[0] + 1] if np.any(edges == -1) else []
if above[0]:
    starts = np.r_[mmd_positions[0], starts]
if above[-1]:
    ends = np.r_[ends, mmd_positions[-1]]
for si, (s0, e0) in enumerate(zip(starts, ends)):
    ax1.axvspan(s0, e0, color=ALERTRED, alpha=0.10,
                label="super-threshold burst" if si == 0 else None)
ax1.plot(cand_pos, smooth[peak_idx], "o", ms=11, mfc="none",
         mec=ALERTRED, mew=2.2, label="candidate peaks")
ax1.axvline(TRUE_DRIFT, color=ALERTRED, ls="--", lw=1.2, alpha=0.6)
# annotate the transient blip bump that stays below threshold
blip_mask = (mmd_positions > BLIP_T - 60) & (mmd_positions < BLIP_T + 90)
if np.any(blip_mask):
    bx = mmd_positions[blip_mask][int(np.argmax(smooth[blip_mask]))]
    ax1.annotate("transient blip\n(sub-threshold $\\rightarrow$ ignored)",
                 xy=(bx, smooth[mmd_positions == bx][0]),
                 xytext=(bx - 30, threshold * 1.7), fontsize=8.5, color=GREY,
                 ha="center", arrowprops=dict(arrowstyle="->", color=GREY))
ax1.set_ylabel(r"$\sigma(t)$")
ax1.set_title("(b) MMD trace $\\rightarrow$ isolate candidate peaks",
              loc="left", color=UNIVBLUE, fontweight="bold")
ax1.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=9)

# (c) decision lane: collapse the super-threshold burst into ONE event -------
ax2.set_ylim(0, 1)
ax2.set_yticks([])
for si, (s0, e0) in enumerate(zip(starts, ends)):
    burst_pts = mmd_positions[(mmd_positions >= s0) & (mmd_positions <= e0)]
    ax2.vlines(burst_pts, 0.46, 0.64, color=ALERTRED, lw=1.6, alpha=0.55)
    ax2.text((s0 + e0) / 2, 0.74, "raw alarms\n(whole window)", color=ALERTRED,
             ha="center", va="bottom", fontsize=9)
ev = cand_pos[int(np.argmax(smooth[peak_idx]))]            # dominant peak
ax2.axvspan(TRUE_DRIFT - 75, TRUE_DRIFT + 75, color=DARKGREEN, alpha=0.10,
            label=r"tolerance $\delta=75$")
ax2.vlines(ev, 0.0, 0.46, color=DARKGREEN, lw=3)
ax2.plot([ev], [0.46], marker="v", ms=15, color=DARKGREEN)
ax2.axvline(TRUE_DRIFT, color=ALERTRED, ls="--", lw=1.2, alpha=0.6)
ax2.annotate("1 confirmed event\nGamma-null $p<\\alpha$ (Bonferroni)\n"
             r"$|\hat t-t^\ast|=%d<\delta$" % abs(ev - TRUE_DRIFT),
             xy=(ev, 0.30), xytext=(ev + 95, 0.46), color=DARKGREEN,
             fontsize=9.5, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=DARKGREEN))
ax2.set_xlabel("Time $t$")
ax2.set_title("(c) Gamma-null validation $\\rightarrow$ collapse the burst into a single drift event",
              loc="left", color=UNIVBLUE, fontweight="bold")
ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

fig.align_ylabels()
out = Path(PLOTS_DIR) / "theory_detection_flow.png"
fig.savefig(out)
print(f"saved -> {out}")
print(f"is_drift={is_drift} positions={list(positions)} "
      f"candidates={list(cand_pos)} pvals={[round(p,4) for p in cand_pvals]} "
      f"adj_alpha={adj_alpha:.4f}")
