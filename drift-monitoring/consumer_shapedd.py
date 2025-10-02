from collections import deque
import numpy as np
import sys
from pathlib import Path

# Import shared configuration
from config import BUFFER_SIZE, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM, DRIFT_PVALUE

# Ensure we can import shape_adaptive from experiments/backup/shape_dd.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.append(str(SHAPE_DD_DIR))

from shape_dd import shape_adaptive

W = BUFFER_SIZE  # Use consistent buffer size (10000)

buf = deque(maxlen=W)

def update_and_check(x_vec):
    """Return True if drift detected at the end of the window."""
    buf.append(np.array(x_vec, float))
    if len(buf) < W:
        return False
    X = np.vstack(buf)              # (W, d)
    res = shape_adaptive(X, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)  # Use consistent parameters
    
    # Use p-values from column 2 for proper statistical testing
    p_values = res[:, 2]
    valid_p_values = p_values[p_values < 1.0]  # Filter out invalid p-values
    
    if len(valid_p_values) == 0:
        return False
    
    # Find minimum p-value for drift detection
    min_p_value = float(np.min(valid_p_values))
    return min_p_value < DRIFT_PVALUE  # Correct statistical test


