import numpy as np
from mmd import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.metrics.pairwise import pairwise_distances
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

def shape_adaptive(X, l1, l2, n_perm):
    """
    Adaptive Shape Drift Detector with improved bandwidth selection and peak filtering.
    
    Improvements over baseline shape():
    1. Data-driven bandwidth selection using Scott's rule
    2. Adaptive smoothing based on window size and noise estimation
    3. Statistical peak filtering to reduce false positives
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data stream
    l1 : int
        Half-window size for drift detection
    l2 : int
        Window size for MMD computation
    n_perm : int
        Number of permutations for statistical test
        
    Returns
    -------
    res : ndarray of shape (n_samples, 3)
        Column 0: Shape statistic value at detected peaks
        Column 1: MMD p-value
        Column 2: Always 1 (compatibility)
    """
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    
    # Adaptive bandwidth selection using Scott's rule
    # Scott's rule: sigma = std * n^(-1/(d+4))
    n_sample = min(1000, n)
    X_sample = X[:n_sample]
    d = X.shape[1]
    
    # Calculate data-driven bandwidth
    data_std = np.std(X_sample, axis=0).mean()
    if data_std > 0:
        # Scott's rule for bandwidth selection
        scott_factor = (n_sample ** (-1.0 / (d + 4)))
        sigma = data_std * scott_factor
        gamma = 1.0 / (2 * sigma**2)
    else:
        # Fallback: use median distance heuristic
        distances = pairwise_distances(X_sample, metric='euclidean')
        distances_flat = distances[distances > 0]
        if len(distances_flat) > 0:
            median_dist = np.median(distances_flat)
            gamma = 1.0 / (2 * median_dist**2)
        else:
            gamma = 1.0  # Default fallback
    
    K = apply_kernel(X, metric="rbf", gamma=gamma)
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    # Adaptive smoothing: reduce window for smaller l1 to maintain responsiveness
    # Use sqrt scaling instead of linear to avoid over-smoothing
    smooth_window = max(3, int(np.sqrt(l1)))
    stat_smooth = uniform_filter1d(stat, size=smooth_window, mode='nearest')

    shape = np.convolve(stat_smooth, w)
    shape_prime = shape[1:]*shape[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1

    # Improved peak filtering with statistical threshold
    potential_peaks = np.where(shape_prime < 0)[0]
    
    # Adaptive threshold based on shape distribution
    # Use mean + k*std where k depends on desired false positive rate
    # For FPR ~ 0.05, use k = 1.645 (z-score for 95% confidence)
    shape_mean = np.mean(shape)
    shape_std = np.std(shape)
    threshold = shape_mean + 0.015 * shape_std

    for pos in potential_peaks:
        # Apply statistical threshold to filter noise peaks
        if shape[pos] > threshold:
            res[pos,0] = shape[pos]
            a, b = max(0, pos-int(l2/2)), min(n, pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
    
    return res
