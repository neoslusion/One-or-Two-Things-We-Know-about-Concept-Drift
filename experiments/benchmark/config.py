
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ============================================================================
# DATA STREAM CONFIGURATION
# ============================================================================
# Standard benchmark sizes (per MOA/River conventions):
# - 10,000-40,000 samples for synthetic datasets
# - With 10 drift events: ~1,000-4,000 samples per segment
# - Minimum: STREAM_SIZE > (n_drift_events + 1) * max(CHUNK_SIZE, SHAPE_L2) * 2
STREAM_SIZE = 10000  # 10 drifts = ~909 samples per segment (sufficient for L2=150)
RANDOM_SEED = 42  # Fixed seed for reproducibility (deprecated - use RANDOM_SEEDS instead)

import os

# ============================================================================
# RELIABILITY CONFIGURATION (Multiple Independent Runs)
# ============================================================================
# Statistical validation requires multiple runs with different random seeds.
# The historical default is 30 runs for 80% statistical power; we expose
# two environment-variable overrides for the master-thesis benchmark loop:
#
#   QUICK_MODE=True              -> N_RUNS = 2   (developer smoke test)
#   BENCHMARK_N_RUNS=<int>       -> N_RUNS = int (overrides everything)
#
# If both are unset we default to 15 seeds, which gives a good
# Friedman/Nemenyi sample without spending several hours on a laptop.
_env_n_runs = os.environ.get("BENCHMARK_N_RUNS")
if _env_n_runs is not None:
    N_RUNS = int(_env_n_runs)
elif os.environ.get("QUICK_MODE") == "True":
    N_RUNS = 2
else:
    N_RUNS = 15

RANDOM_SEEDS = [42 + i * 137 for i in range(N_RUNS)]  # Prime spacing avoids correlation
N_JOBS = -1 # Number of parallel jobs (-1 = all cores)

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================
CHUNK_SIZE = 150        # Detection window size
OVERLAP = 100           # Overlap between windows
SHAPE_L1 = 50          # ShapeDD reference window
SHAPE_L2 = 150         # ShapeDD test window
SHAPE_N_PERM = 2500    # ShapeDD permutation count
COOLDOWN = 75          # Minimum samples between detections

# ============================================================================
# SPECTRA-DRIFT PARAMETERS
# ============================================================================
SPECTRA_WINDOW = 500         # Window size for spectral analysis
SPECTRA_K = None             # Number of neighbors (None = auto sqrt(window_size))
SPECTRA_EIGENVALUES = 10     # Number of eigenvalues to extract
SPECTRA_ALPHA = 0.01         # False positive rate (auto-calibrated threshold)

# ============================================================================
# STREAMING DETECTOR CONFIGURATION
# ============================================================================
INITIAL_TRAINING_SIZE = 500    # Initial batch for model training
PREQUENTIAL_WINDOW = 100       # Window for prequential accuracy

# ============================================================================
# METHOD LISTS
# ============================================================================

# Window-based methods (RECOMMENDED for thesis benchmark)
WINDOW_METHODS = [
    # === BASELINES ===
    'MMD',             # Standard MMD with permutation test (Gretton et al., 2012)
    'KS',              # Kolmogorov-Smirnov test (classical)
    # === LEGACY METHODS ===
    'D3',              # Deep learning-based (different paradigm)
    'DAWIDD',          # Distance-aware windowed
    'IDW_MMD',         # Just IDW-MMD without ShapeDD (ablation)
    
    # === SHAPEDD VARIANTS ===
    'ShapeDD',                # Original ShapeDD with permutation test (baseline)
    'ShapeDD_IDW',            # RECOMMENDED: ShapeDD + IDW-MMD + asymptotic p-value (FAST)
    
    # === UNIFIED SYSTEM ===
    'SE_CDT',                 # Detection + Classification (most complete)
]

# Streaming methods (require model for accuracy signal)
STREAMING_METHODS = [
    # 'ADWIN',        # Adaptive Windowing
    # 'DDM',          # Drift Detection Method
    # 'EDDM',         # Early Drift Detection Method
    # 'HDDM_A',       # Hoeffding's Drift Detection Method (Average)
    # 'HDDM_W',       # Hoeffding's Drift Detection Method (Weighted)
]

