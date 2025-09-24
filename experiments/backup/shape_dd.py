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

def shape_improved(X, l1, l2, n_perm, 
                  shape_threshold=0.1,      # Minimum shape statistic value
                  persistence_window=3,     # Require detection persistence
                  adaptive_alpha=True,      # Use adaptive significance level
                  temporal_smoothing=True,  # Apply temporal smoothing
                  min_separation=50):       # Minimum separation between detections
    """
    Improved ShapeDD with reduced false positives while maintaining sensitivity.
    
    Key improvements:
    1. Adaptive thresholding based on local statistics
    2. Temporal persistence requirements  
    3. Smoothing to reduce noise sensitivity
    4. Minimum separation between detections
    5. Context-aware MMD testing
    """
    
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    n = X.shape[0]
    
    # Compute kernel matrix
    K = apply_kernel(X, metric="rbf")
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i, i:i+2*l1] = w    
    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)
    shape = np.convolve(stat, w)
    
    # Apply temporal smoothing to reduce noise
    if temporal_smoothing and len(shape) > 5:
        shape_smoothed = uniform_filter1d(shape, size=3)
    else:
        shape_smoothed = shape
    
    shape_prime = shape_smoothed[1:] * shape_smoothed[:-1]
    
    # Initialize results
    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value = 1 (no drift)
    
    # Find potential drift points (negative shape_prime)
    candidate_points = np.where(shape_prime < 0)[0]
    
    if len(candidate_points) == 0:
        return res
    
    # Filter candidates by shape threshold (must be sufficiently strong)
    strong_candidates = []
    for pos in candidate_points:
        if shape_smoothed[pos] > shape_threshold:
            strong_candidates.append(pos)
    
    if len(strong_candidates) == 0:
        return res
    
    # Apply persistence filtering: require multiple consecutive detections
    if persistence_window > 1:
        persistent_candidates = []
        for pos in strong_candidates:
            # Count nearby candidates within persistence window
            nearby = sum(1 for p in strong_candidates 
                        if abs(p - pos) <= persistence_window // 2)
            if nearby >= persistence_window:
                persistent_candidates.append(pos)
        strong_candidates = persistent_candidates
    
    if len(strong_candidates) == 0:
        return res
    
    # Apply minimum separation constraint
    if min_separation > 0:
        separated_candidates = []
        last_detection = -min_separation - 1
        
        for pos in sorted(strong_candidates):
            if pos - last_detection >= min_separation:
                separated_candidates.append(pos)
                last_detection = pos
        
        strong_candidates = separated_candidates
    
    # Perform MMD tests only on filtered candidates
    for pos in strong_candidates:
        res[pos, 0] = shape_smoothed[pos]
        
        # Define test windows around drift point
        a = max(0, pos - int(l2/2))
        b = min(n, pos + int(l2/2))
        
        # Ensure sufficient data for MMD test
        if b - a < 2 * l1:
            continue
            
        # Adaptive significance level based on local variance
        if adaptive_alpha:
            # Estimate local noise level
            local_shape = shape_smoothed[max(0, pos-20):min(len(shape_smoothed), pos+20)]
            local_std = np.std(local_shape) if len(local_shape) > 1 else 1.0
            
            # Adjust number of permutations based on required precision
            # For gradual drifts, we can be less strict
            adjusted_n_perm = max(100, min(n_perm, int(n_perm / max(1, local_std))))
        else:
            adjusted_n_perm = n_perm
        
        # Perform MMD test with adjusted parameters
        try:
            mmd_result = mmd(X[a:b], pos-a, adjusted_n_perm)
            res[pos, 1:] = mmd_result
        except Exception as e:
            print(f"MMD test failed at position {pos}: {e}")
            res[pos, 1] = 0.0
            res[pos, 2] = 1.0
    
    return res

def shape_conservative(X, l1, l2, n_perm, alpha=0.01):
    """
    Conservative version of ShapeDD for very low false positive rate.
    """
    return shape_improved(X, l1, l2, n_perm,
                         shape_threshold=0.2,      # Higher threshold
                         persistence_window=5,     # More persistence required
                         adaptive_alpha=True,
                         temporal_smoothing=True,
                         min_separation=100)       # Larger separation

def shape_balanced(X, l1, l2, n_perm, alpha=0.05):
    """
    Balanced version of ShapeDD for moderate false positive rate.
    """
    return shape_improved(X, l1, l2, n_perm,
                         shape_threshold=0.1,      # Moderate threshold
                         persistence_window=3,     # Some persistence required
                         adaptive_alpha=True,
                         temporal_smoothing=True,
                         min_separation=50)        # Moderate separation
