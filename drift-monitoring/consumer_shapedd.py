from collections import deque
import numpy as np
import sys
from pathlib import Path

# Ensure we can import shape_adaptive from experiments/backup/shape_dd.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.append(str(SHAPE_DD_DIR))

from shape_dd import shape_adaptive

W = 200  # window length for detection

buf = deque(maxlen=W)

def update_and_check(x_vec):
    """Return True if drift detected at the end of the window."""
    buf.append(np.array(x_vec, float))
    if len(buf) < W:
        return False
    X = np.vstack(buf)              # (W, d)
    res = shape_adaptive(X, 50, 100, 100)  # (n, 3), col 0 = peak statistic
    # Convert detector output to a scalar: use max peak value
    score = float(np.max(res[:, 0]))
    return score > 0.05


