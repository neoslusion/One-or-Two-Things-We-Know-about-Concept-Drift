import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
import scipy
import scipy.special
from chapydette import cp_estimation

## Kernel Drift Detector, Code based on https://github.com/cjones6/chapydette

def kernel_dd(X, min_ncp=0, max_ncp=100):
    n = X.shape[0]
    cp,objs = cp_estimation.mkcpe(apply_kernel(X, metric="rbf"),n_cp=(min_ncp,max_ncp),kernel_type='precomputed',est_ncp=False, return_obj=True)
    
    # Regress the objective values on log (n-1 choose ncp) and ncp+1 and store the resultant slopes
    ncp_range_subset = np.array(range(max(int(0.6*(max_ncp)), min_ncp), max_ncp))
    x1_subset = scipy.special.loggamma(n*np.ones_like(ncp_range_subset)) - scipy.special.loggamma(n - ncp_range_subset)\
                - scipy.special.loggamma(ncp_range_subset + 1)
    x2_subset = ncp_range_subset+1
    x = np.column_stack((np.ones_like(x1_subset), x1_subset, x2_subset))
    objs_subset = np.array([objs[ncp] for ncp in ncp_range_subset])
    beta = np.linalg.solve(x.T.dot(x), x.T.dot(objs_subset))
    s1, s2 = beta[1], beta[2]

    # Obtain the penalized objectives
    ncp_range = np.array(range(min_ncp, max_ncp+1))
    x1 = scipy.special.loggamma(n*np.ones_like(ncp_range)) - scipy.special.loggamma(n - ncp_range) \
                - scipy.special.loggamma(ncp_range + 1)
    x2 = ncp_range+1
    objs_all = np.array([objs[key] for key in ncp_range]).flatten()
    
    
    ## TODO: This is real bad, fix me
    penalty0 = (s1*np.array(x1) + s2*np.array(x2))
    hi = 1
    while (objs_all-hi*penalty0).min() < objs_all[0]-hi*penalty0[0]:
        hi *= 2
    lo = 0
    while hi-lo > 1e-3:
        mid = (hi+lo)/2
        if (objs_all-mid*penalty0).min() < objs_all[0]-mid*penalty0[0]:
            lo = mid
        else:
            hi = mid
    lo,lohi = 0,hi
    while lohi-lo > 1e-3:
        mid = (lohi+lo)/2
        if (objs_all-mid*penalty0).min() >= objs_all[-1]-mid*penalty0[-1]:
            lo = mid
        else:
            lohi = mid
    alphas =  np.linspace(lo-0.5,hi+0.5, 10000)
    nums = np.array([np.argmin(objs_all-alpha*penalty0) for alpha in alphas])
    pos = np.where(np.diff(nums) != 0)[0]
    
    out = np.zeros(n)
    for p in pos:
        out[cp[nums[p]]] = -alphas[p]
    
    return out
