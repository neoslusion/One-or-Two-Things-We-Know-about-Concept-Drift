import numpy as np
from mmd import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from scipy.ndimage import uniform_filter1d

def shape(X, l1, l2, n_perm):
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    K = apply_kernel(X, metric="rbf")
    W = np.zeros( (n-2*l1,n) )
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    shape = np.convolve(stat,w)
    shape_prime = shape[1:]*shape[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1
    for pos in np.where(shape_prime < 0)[0]:
        if shape[pos] > 0:
            res[pos,0] = shape[pos]
            a,b = max(0,pos-int(l2/2)),min(n,pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
    return res

def shape_adaptive(X, l1, l2, n_perm, 
                  sensitivity_mode='balanced',
                  auto_tune_parameters=True):
    """
    Adaptive ShapeDD that preserves the original algorithm's strengths while fixing key issues.
    
    Key improvements:
    1. Auto-parameter tuning based on data characteristics
    2. Adaptive thresholds without overly restrictive filtering
    3. Smarter candidate selection
    4. Preserves the original MMD-based drift detection logic
    """
    
    # Auto-tune parameters based on data characteristics if requested
    if auto_tune_parameters:
        n_samples = X.shape[0]
        # Scale l1 based on data size (but keep reasonable bounds)
        l1 = max(20, min(100, n_samples // 60))  # ~1.5% of data, bounded
        # l2 should be larger than l1 for meaningful test windows
        l2 = max(l1 * 2, min(300, n_samples // 20))  # ~5% of data, bounded
        # Adjust permutations based on required precision vs speed
        n_perm = max(500, min(n_perm, 2000))
    
    w = np.array(l1*[1.] + l1*[-1.]) / float(l1)
    n = X.shape[0]
    
    # Ensure we have enough data
    if n < 4 * l1:
        res = np.zeros((n, 3))
        res[:, 2] = 1.0
        return res
    
    # Compute kernel matrix with bandwidth selection
    try:
        K = apply_kernel(X, metric="rbf", gamma='scale')
    except:
        K = apply_kernel(X, metric="rbf")
    
    # Build convolution matrix
    W = np.zeros((n - 2*l1, n))
    for i in range(n - 2*l1):
        W[i, i:i+2*l1] = w
    
    # Compute shape statistics
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)
    shape = np.convolve(stat, w)
    
    # Apply light temporal smoothing (less aggressive than improved version)
    if len(shape) > 7:
        shape_smoothed = uniform_filter1d(shape, size=3, mode='nearest')
    else:
        shape_smoothed = shape
    
    shape_prime = shape_smoothed[1:] * shape_smoothed[:-1]
    
    # Initialize results
    res = np.zeros((n, 3))
    res[:, 2] = 1.0  # Default: no drift detected
    
    # Find zero-crossings (potential drift points)
    zero_crossings = np.where(shape_prime < 0)[0]
    
    if len(zero_crossings) == 0:
        return res
    
    # Adaptive thresholding based on data characteristics
    shape_values = shape_smoothed[shape_smoothed > 0]
    if len(shape_values) > 0:
        if sensitivity_mode == 'high':
            # More sensitive - lower threshold
            shape_threshold = max(0.0, np.percentile(shape_values, 25))
        elif sensitivity_mode == 'conservative':
            # Less sensitive - higher threshold  
            shape_threshold = max(0.05, np.percentile(shape_values, 75))
        else:  # balanced
            # Moderate sensitivity
            shape_threshold = max(0.01, np.percentile(shape_values, 50))
    else:
        shape_threshold = 0.01
    
    # Process candidates with smart filtering
    processed_positions = set()
    
    for pos in zero_crossings:
        # Skip if already processed nearby
        if any(abs(pos - p) < l1//2 for p in processed_positions):
            continue
            
        shape_val = shape_smoothed[pos]
        
        # Only test candidates with sufficient shape statistic
        if shape_val <= shape_threshold:
            continue
        
        # Define test window around potential drift
        window_start = max(0, pos - l2//2)
        window_end = min(n, pos + l2//2)
        
        # Ensure minimum window size for meaningful test
        if window_end - window_start < 2 * l1:
            # Expand window if possible
            needed = 2 * l1 - (window_end - window_start)
            expand_left = min(needed//2, window_start)
            expand_right = min(needed - expand_left, n - window_end)
            window_start = max(0, window_start - expand_left)
            window_end = min(n, window_end + expand_right)
            
            # Skip if still insufficient
            if window_end - window_start < 2 * l1:
                continue
        
        # Perform MMD test
        try:
            X_window = X[window_start:window_end]
            split_point = pos - window_start
            
            # Ensure split point is valid
            if split_point <= l1//2 or split_point >= len(X_window) - l1//2:
                split_point = len(X_window) // 2
            
            mmd_stat, p_value = mmd(X_window, split_point, n_perm)
            
            # Store results
            res[pos, 0] = shape_val
            res[pos, 1] = mmd_stat
            res[pos, 2] = p_value
            
            processed_positions.add(pos)
            
        except Exception as e:
            # Skip this candidate on error
            continue
    
    return res

def shape_smart(X, l1=None, l2=None, n_perm=1000):
    """
    Smart wrapper that automatically selects good parameters and sensitivity.
    """
    return shape_adaptive(X, l1, l2, n_perm, 
                         sensitivity_mode='balanced',
                         auto_tune_parameters=True)

def shape_sensitive(X, l1=None, l2=None, n_perm=1500):
    """
    High-sensitivity version for detecting subtle drifts.
    """
    return shape_adaptive(X, l1, l2, n_perm,
                         sensitivity_mode='high', 
                         auto_tune_parameters=True)

def shape_robust(X, l1=None, l2=None, n_perm=800):
    """
    Robust version for noisy data with lower false positive rate.
    """
    return shape_adaptive(X, l1, l2, n_perm,
                         sensitivity_mode='conservative',
                         auto_tune_parameters=True)
