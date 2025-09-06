import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel

def get_time_kernel(n_size, n_perm, kernel, cache=dict()):
    if (n_size,n_perm,kernel) not in cache.keys():
        T = np.linspace(-1,1,n_size).reshape(-1,1)
        H = np.eye(n_size) - (1/n_size) * np.ones((n_size, n_size))
        if kernel == "min":
            K = [H @ np.minimum(T[None,:],T[:,None])[:,:,0] @ H] 
            for _ in range(n_perm):
                T = np.random.permutation(T)
                K.append(H @ np.minimum(T[None,:],T[:,None])[:,:,0] @ H)
        else:
            K = [H @ apply_kernel(T, metric=kernel) @ H] 
            for _ in range(n_perm):
                T = np.random.permutation(T)
                K.append(H @ apply_kernel(T, metric=kernel) @ H)
        K = np.array(K)
        K = K.reshape(n_perm+1,n_size*n_size)
        cache[(n_size,n_perm,kernel)] = K
    return cache[(n_size,n_perm,kernel)]
def dawidd(X, T_kernel="rbf", n_perm=2500):
    n_size = X.shape[0]
    
    K_X = apply_kernel(X, metric="rbf")
    s = get_time_kernel(n_size, n_perm, T_kernel) @ K_X.ravel()
    p = (s[0] < s).sum() / n_perm
    
    return (1/n_size)**2 * s[0], p

