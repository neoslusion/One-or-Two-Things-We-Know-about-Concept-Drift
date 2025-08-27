import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel

def gen_window_matrix(l1,l2, n_perm, cache=dict()):
    if (l1,l2, n_perm) not in cache.keys():
        w = np.array(l1*[1./l1]+(l2)*[-1./(l2)])
        W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])
        cache[(l1,l2,n_perm)] = W
    return cache[(l1,l2,n_perm)]
def mmd(X, s=None, n_perm=2500):
    K = apply_kernel(X, metric="rbf")
    if s is None:
        s = int(X.shape[0]/2)
    
    W = gen_window_matrix(s,K.shape[0]-s, n_perm)
    s = np.einsum('ij,ij->i', np.dot(W, K), W)
    p = (s[0] < s).sum() / n_perm
    
    return s[0], p

