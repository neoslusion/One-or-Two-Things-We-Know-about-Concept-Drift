"""
Configuration module for drift detection benchmark.

Contains all hyperparameters, detection settings, and method configurations.
"""

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
STREAM_SIZE = 2000
RANDOM_SEED = 42  # Fixed seed for reproducibility (deprecated - use RANDOM_SEEDS instead)

# ============================================================================
# RELIABILITY CONFIGURATION (Multiple Independent Runs)
# ============================================================================
# Statistical validation requires multiple runs with different random seeds
# Benchmark standard: 30-500 runs (we use 30 for 80% statistical power)
N_RUNS = 10  # Number of independent runs (minimum for statistical validity)
RANDOM_SEEDS = [42 + i * 137 for i in range(N_RUNS)]  # Prime spacing avoids correlation

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================
CHUNK_SIZE = 150        # Detection window size
OVERLAP = 100           # Overlap between windows
SHAPE_L1 = 50          # ShapeDD reference window
SHAPE_L2 = 150         # ShapeDD test window (matches CHUNK_SIZE)
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

# Window-based methods
WINDOW_METHODS = [
    'D3',           # Margin density drift detector
    'DAWIDD',       # Distance-aware windowed drift detector
    'MMD',          # Maximum Mean Discrepancy
    'KS',           # Kolmogorov-Smirnov test
    'ShapeDD',      # Original method
    'ShapeDD_SNR_Adaptive',  # SNR-Aware Hybrid
    'MMD_OW',       # Optimally-Weighted MMD estimator
    'ShapeDD_OW_MMD',  # ShapeDD + OW-MMD Hybrid
]

# Streaming methods (require model for accuracy signal)
STREAMING_METHODS = [
    # 'ADWIN',        # Adaptive Windowing
    # 'DDM',          # Drift Detection Method
    # 'EDDM',         # Early Drift Detection Method
    # 'HDDM_A',       # Hoeffding's Drift Detection Method (Average)
    # 'HDDM_W',       # Hoeffding's Drift Detection Method (Weighted)
]

