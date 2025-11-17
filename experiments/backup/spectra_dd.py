import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel

class SpectraDrift:
    def __init__(self, k=10, n_eigenvalues=10, threshold=0.15):
        self.k = k
        self.n_eigenvalues = n_eigenvalues
        self.threshold = threshold
        self.reference_features = None
        
    def _build_graph(self, X):
        n = len(X)
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        sigma = np.median(distances[:, -1])
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        row_ind = np.repeat(np.arange(n), self.k)
        col_ind = indices.flatten()
        data = weights.flatten()
        W = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
        W = (W + W.T) / 2
        
        return W
    
    def _compute_laplacian(self, W):
        D = np.array(W.sum(axis=1)).flatten()
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        D_mat = csr_matrix((D_inv_sqrt, (range(len(D)), range(len(D)))))
        L_sym = D_mat @ (csr_matrix(np.diag(D)) - W) @ D_mat
        return L_sym
    
    def _extract_features(self, eigenvalues):
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
        mid_scale = np.mean(eigenvalues[2:5]) if len(eigenvalues) > 4 else 0
        local_scale = np.mean(eigenvalues[-5:]) if len(eigenvalues) >= 5 else 0
        
        probs = eigenvalues / eigenvalues.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        
        features = np.array([
            lambda_2,
            mid_scale,
            local_scale,
            entropy,
            eigenvalues.std(),
            eigenvalues.max(),
            eigenvalues.min(),
            np.median(eigenvalues),
            np.percentile(eigenvalues, 25),
            np.percentile(eigenvalues, 75),
            eigenvalues.sum()
        ])
        return features
    
    def fit(self, X):
        W = self._build_graph(X)
        L = self._compute_laplacian(W)
        
        k = min(self.n_eigenvalues, L.shape[0] - 2)
        eigenvalues, _ = eigsh(L, k=k, which='SM')
        eigenvalues = np.sort(eigenvalues)
        
        self.reference_features = self._extract_features(eigenvalues)
        return self
    
    def detect(self, X):
        W = self._build_graph(X)
        L = self._compute_laplacian(W)
        
        k = min(self.n_eigenvalues, L.shape[0] - 2)
        eigenvalues, _ = eigsh(L, k=k, which='SM')
        eigenvalues = np.sort(eigenvalues)
        
        current_features = self._extract_features(eigenvalues)
        score = np.linalg.norm(current_features - self.reference_features)
        drift_type = self._classify_drift(current_features, eigenvalues)
        
        return {
            'drift_detected': score > self.threshold,
            'score': score,
            'drift_type': drift_type,
            'eigenvalues': eigenvalues,
            'features': current_features
        }
    
    def _classify_drift(self, features, eigenvalues):
        lambda_2_change = abs(features[0] - self.reference_features[0])
        
        if lambda_2_change > 0.3:
            return 'sudden'
        elif features[1] > self.reference_features[1] * 1.2:
            return 'gradual'
        elif features[4] > self.reference_features[4] * 1.5:
            return 'incremental'
        else:
            return 'stable'

def spectra(X_ref, X_current, k=10, n_eigenvalues=10, threshold=0.15):
    """
    SPECTRA-DRIFT: Spectral Graph Theory cho Drift Detection
    
    Parameters:
    -----------
    X_ref : array (n_ref, d)
        Reference window (stable data)
    X_current : array (n_curr, d)
        Current window (test for drift)
    k : int
        Số neighbors trong k-NN graph
    n_eigenvalues : int
        Số eigenvalues để trích xuất
    threshold : float
        Ngưỡng phát hiện drift
        
    Returns:
    --------
    drift_detected : bool
    score : float
    drift_type : str
    """
    detector = SpectraDrift(k=k, n_eigenvalues=n_eigenvalues, threshold=threshold)
    detector.fit(X_ref)
    result = detector.detect(X_current)
    
    return result['drift_detected'], result['score'], result['drift_type']

def spectra_streaming(X, window_size=500, k=10, n_eigenvalues=10, threshold=0.15):
    """
    Streaming version - tương tự shape() trong shape_dd.py
    
    Returns:
    --------
    res : array (n_samples, 3)
        [:, 0] = spectral score
        [:, 1] = drift_type_encoded (0=stable, 1=sudden, 2=gradual, 3=incremental)
        [:, 2] = binary detection (1=drift, 0=no_drift)
    """
    n = X.shape[0]
    res = np.zeros((n, 3))
    
    if n < window_size * 2:
        return res
    
    detector = SpectraDrift(k=k, n_eigenvalues=n_eigenvalues, threshold=threshold)
    X_ref = X[:window_size]
    detector.fit(X_ref)
    
    drift_type_map = {'stable': 0, 'sudden': 1, 'gradual': 2, 'incremental': 3}
    
    for i in range(window_size, n - window_size):
        X_curr = X[i:i+window_size]
        
        result = detector.detect(X_curr)
        
        res[i, 0] = result['score']
        res[i, 1] = drift_type_map[result['drift_type']]
        res[i, 2] = 1 if result['drift_detected'] else 0
        
        if result['drift_detected']:
            detector.fit(X_curr)
    
    return res
