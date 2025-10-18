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
def shape_adaptive(X, l1, l2, n_perm, sensitivity='medium'):
    """
    sensitivity: 'low' (conservative), 'medium' (balanced), 'high' (aggressive), 'none' (no filtering)
    """
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    n_sample = min(1000, n)
    X_sample = X[:n_sample]
    d = X.shape[1]
    
    data_std = np.std(X_sample, axis=0).mean()
    if data_std > 0:
        scott_factor = (n_sample ** (-1.0 / (d + 4)))
        sigma = data_std * scott_factor
        gamma = 1.0 / (2 * sigma**2)
    else:
        distances = pairwise_distances(X_sample, metric='euclidean')
        distances_flat = distances[distances > 0]
        if len(distances_flat) > 0:
            median_dist = np.median(distances_flat)
            gamma = 1.0 / (2 * median_dist**2)
        else:
            gamma = 1.0
    
    K = apply_kernel(X, metric="rbf", gamma=gamma)
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    smooth_window = max(3, int(np.sqrt(l1)))
    stat_smooth = uniform_filter1d(stat, size=smooth_window, mode='nearest')

    shape = np.convolve(stat_smooth, w)
    shape_prime = shape[1:]*shape[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1

    potential_peaks = np.where(shape_prime < 0)[0]
    
    # Adjustable threshold based on sensitivity
    if sensitivity == 'none':
        threshold = 0
    else:
        shape_mean = np.mean(shape)
        shape_std = np.std(shape)
        threshold_factors = {'low': 0.005, 'medium': 0.01, 'high': 0.015}
        k = threshold_factors.get(sensitivity, 0.01)
        threshold = shape_mean + k * shape_std
    
    p_values = []
    positions = []

    for pos in potential_peaks:
        if shape[pos] > threshold:
            res[pos,0] = shape[pos]
            a, b = max(0, pos-int(l2/2)), min(n, pos+int(l2/2))
            mmd_result = mmd(X[a:b], pos-a, n_perm)
            res[pos,1:] = mmd_result
            p_values.append(mmd_result[1])
            positions.append(pos)
    
    # Optional FDR: only apply if sensitivity is not 'none'
    if len(p_values) > 1 and sensitivity != 'none':
        p_values_array = np.array(p_values)
        
        # Adjust alpha based on sensitivity
        alpha_values = {'low': 0.01, 'medium': 0.05, 'high': 0.10}
        alpha = alpha_values.get(sensitivity, 0.05)
        
        significant_indices = benjamini_hochberg_correction(p_values_array, alpha=alpha)
        significant_set = set(significant_indices)
        
        for i, pos in enumerate(positions):
            if i not in significant_set:
                res[pos,0] = 0
                res[pos,1] = 0
                res[pos,2] = 1.0
    
    return res

def benjamini_hochberg_correction(p_values, alpha=0.05):
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    m = len(p_values)
    
    for k in range(m-1, -1, -1):
        if sorted_p[k] <= (k+1) / m * alpha:
            significant_indices = sorted_indices[:k+1]
            return significant_indices
    
    return np.array([])
