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
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    distances = pairwise_distances(X[:min(1000, n)], metric='euclidean')

    # CHANGE 1: Make kernel bandwidth more conservative (less restrictive)
    # median_dist = np.median(distances[distances > 0])
    # gamma = 1.0 / (2 * median_dist**2) if median_dist > 0 else 1.0

    # CHANGE 2: Use 75th percentile distance instead of median (larger bandwidth = more sensitive)
    # percentile_75_dist = np.percentile(distances[distances > 0], 50)
    # gamma = 1.0 / (4 * percentile_75_dist**2) if percentile_75_dist > 0 else 'scale'

    # CHANGE 3:
    distances_flat = distances[distances > 0]
    percentile_dist = np.percentile(distances_flat, 75)
    gamma = 1.0 / (15 * percentile_dist**2) if percentile_dist > 0 else 0.05
    
    K = apply_kernel(X, metric="rbf", gamma=gamma)
    W = np.zeros( (n-2*l1,n) )
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    # Smooth statistic to reduce noise
    smooth_window = max(5, l1//2)
    stat_smooth = uniform_filter1d(stat, size=smooth_window, mode='nearest')

    shape = np.convolve(stat_smooth,w)
    shape_prime = shape[1:]*shape[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1

    # Improvement: Peak filtering with prominence
    potential_peaks = np.where(shape_prime < 0)[0]

    for pos in potential_peaks:
        if shape[pos] > 0:
        # if shape[pos] > (np.std(shape) * 1.5):
            res[pos,0] = shape[pos]
            a,b = max(0,pos-int(l2/2)),min(n,pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
    return res
