import numpy as np
from scipy.stats import ks_2samp

def ks(X, s=None):
    if s is None:
        s = int(X.shape[0]/2)
    return min([ks_2samp(X[:,i][:s], X[:,i][s:],mode="exact")[1] for i in range(X.shape[1])])

