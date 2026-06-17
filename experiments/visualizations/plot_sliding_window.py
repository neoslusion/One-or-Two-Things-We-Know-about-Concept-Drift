"""
Generate `theory_window_to_trace.png`: a talkable visual of how the ShapeDD
sliding window turns a data stream into the MMD trace.

Top panel  : the raw stream with a drift, and the [W_ref | W_test] window drawn
             at three positions (before / straddling / after the drift).
Bottom panel: the real MMD trace, with the three corresponding points marked and
             linked to the windows above --- so you can literally point and say
             "window here -> this MMD value".
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from core.config import PLOTS_DIR
from core.detectors.mmd_variants import shapedd_idw_mmd_proper

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "lines.linewidth": 2, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})

UNIVBLUE = "#003366"; GREEN = "#0a7a0a"; RED = "#cc0000"; GREY = "#888888"
L1, L2 = 50, 150
TRUE_DRIFT = 600

# ---- synthetic stream: stable -> sudden shift -> stable ----
rng = np.random.default_rng(3)
DIM = 5
pre = rng.normal(0, 1, (TRUE_DRIFT, DIM))
post = rng.normal(0, 1, (600, DIM)); post[:, :4] += 1.9
X = np.vstack([pre, post]); n = len(X)

# ---- real MMD trace ----
_, _, trace, _ = shapedd_idw_mmd_proper(X, l1=L1, l2=L2, alpha=0.05,
                                        weight_method="variance_reduction")
step = max(1, L1 // 2)
pos = np.array([i + L1 for i in range(0, n - L1 - L2, step)])
trace = np.asarray(trace)

# three window split-points (trace position = split between ref and test)
p_peak = int(pos[int(np.argmax(trace))])           # straddling the drift
p_pre  = 250                                        # before the drift
p_post = 980                                        # after, restabilised
def tval(pp):  # trace value nearest a split position
    return trace[int(np.argmin(np.abs(pos - pp)))]

# ---------------------------------------------------------------- plot
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12})

# (top) stream + windows
ax0.plot(X[:, 0], color=GREY, lw=0.7, alpha=0.9)
ax0.axvline(TRUE_DRIFT, color=RED, ls="--", lw=2)
ax0.text(TRUE_DRIFT + 10, ax0.get_ylim()[1]*0.7, "drift", color=RED,
         fontsize=11, fontweight="bold")
ax0.set_ylabel("Feature $x_1$")
ax0.set_title("A sliding window runs over the stream …", loc="left",
              color=UNIVBLUE, fontweight="bold")

ytop, ybot = ax0.get_ylim()
hi = ybot + 0.92*(ytop-ybot); lo = ybot + 0.70*(ytop-ybot)
labels = {p_pre: "①", p_peak: "②", p_post: "③"}
for pp in (p_pre, p_peak, p_post):
    # ref half (green) and test half (salmon)
    ax0.add_patch(Rectangle((pp-L1, lo), L1, hi-lo, fc=GREEN, ec=GREEN, alpha=0.25, lw=1.2))
    ax0.add_patch(Rectangle((pp, lo), L2, hi-lo, fc=RED, ec=RED, alpha=0.16, lw=1.2))
    ax0.text(pp, hi+0.04*(ytop-ybot), labels[pp], ha="center", color=UNIVBLUE,
             fontsize=13, fontweight="bold")
# small legend for the two halves (placed on window 1, in clear space)
ax0.text(p_pre-L1/2, lo-0.12*(ytop-ybot), "$W_{ref}$ ($l_1{=}50$)", ha="center",
         va="top", color=GREEN, fontsize=8.5)
ax0.text(p_pre+L2/2, lo-0.12*(ytop-ybot), "$W_{test}$ ($l_2{=}150$)", ha="center",
         va="top", color=RED, fontsize=8.5)
ax0.annotate("", xy=(p_post+40, hi+0.02*(ytop-ybot)), xytext=(p_pre-40, hi+0.02*(ytop-ybot)),
             arrowprops=dict(arrowstyle="->", color=UNIVBLUE, lw=1.4))
ax0.text((p_pre+p_post)/2, hi+0.10*(ytop-ybot), "slides by $l_1/2 = 25$", ha="center",
         color=UNIVBLUE, fontsize=9)

# (bottom) trace + the 3 points
ax1.plot(pos, trace, color=UNIVBLUE, lw=2.3)
ax1.axvline(TRUE_DRIFT, color=RED, ls="--", lw=1.2, alpha=0.6)
for pp in (p_pre, p_peak, p_post):
    v = tval(pp)
    ax1.plot(pp, v, "o", ms=11, color=UNIVBLUE)
    ax1.text(pp, v+0.03, labels[pp], ha="center", va="bottom", color=UNIVBLUE,
             fontsize=12, fontweight="bold")
    # dashed link from window (top) to its trace value (bottom)
    ax1.axvline(pp, color=GREY, ls=":", lw=1, alpha=0.7)
ax1.set_ylabel(r"$\sigma(t)$")
ax1.set_xlabel("Time $t$  (sample index)")
ax1.set_title("… each window position gives one MMD value $\\to$ the trace $\\sigma(t)$",
              loc="left", color=UNIVBLUE, fontweight="bold")
ax1.set_ylim(0, max(trace)*1.25)

out = Path(PLOTS_DIR) / "theory_window_to_trace.png"
fig.savefig(out)
print("saved ->", out, "| peak split pos =", p_peak)
