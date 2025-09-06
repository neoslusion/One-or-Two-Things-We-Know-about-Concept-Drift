from copy import deepcopy
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from river.base import DriftDetector
from mmd import mmd

class ShapeDD(DriftDetector):
    r"""Shape Drift Detector

    Shape Drift Detector is a meta-statistic-based concept drift in data streams. The algorithm employs MMD as its core statistical measure and follows a systematic approach consisting of four main stages.
    First stage: Data Collection, it collects data using sliding window techniques.
    Second stage: Feature Construction, it constructs a similarity matrix using a kernel function to capture the relationships between data points.
    Third stage: Difference Computation, it computes the statistical difference between consecutive data segments using MMD.
    Fourth stage: Statistical Validation, it normalizes the MMD statistics and identifies potential change points through zero-crossing detection.

    Parameters
    ----------
    data_stream
        The data stream that will be used to detect drift.
    window_size_l1
        The size of the data window l1.
    window_size_l2
        The size of the data window l2.
    n_perm
        The number of permutations to be used in the permutation test, default is 2500.
    """


    def __init__(
        self,
        window_size_l1=50,
        window_size_l2=100,
        n_perm=2500,
        number_of_elements=0,
    ):
        super().__init__()
        self.window_size_l1 = window_size_l1
        self.window_size_l2 = window_size_l2 
        self.n_perm = n_perm
        self.number_of_elements = number_of_elements
        self.drift_detected = False

    def calculate_mmd(self, X):
        w = np.array(self.window_size_l1*[1.]+self.window_size_l1*[-1.]) / float(self.window_size_l1)
    
        self.number_of_elements = X.shape[0]
        # n = X.shape[0]
        n = self.number_of_elements
        K = apply_kernel(X, metric="rbf")
        W = np.zeros( (n-2*self.window_size_l1,n) )
        
        for i in range(n-2*self.window_size_l1):
            W[i,i:i+2*self.window_size_l1] = w    
        stat = np.einsum('ij,ij->i', np.dot(W, K), W)

        shape_values = np.convolve(stat,w)
        shape_prime = shape_values[1:]*shape_values[:-1] 
        return shape_values, shape_prime


    def permutation_test(self, shape_values, shape_prime, X, n_perm):
        res = np.zeros((self.number_of_elements,3))
        res[:,2] = 1
        for pos in np.where(shape_prime < 0)[0]:
            if shape_values[pos] > 0:
                res[pos,0] = shape_values[pos]
                a,b = max(0,pos-int(self.window_size_l2/2)),min(self.number_of_elements,pos+int(self.window_size_l2/2))
                res[pos,1:] = mmd(X[a:b], pos-a, n_perm)

        if np.any(res[:,2] < 0.05):
            self.drift_detected = True

        return res

