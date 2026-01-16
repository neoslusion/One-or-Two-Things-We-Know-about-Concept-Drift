import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class CDT_MSW:
    """Concept Drift Type identification based on Multi-Sliding Windows"""
    
    def __init__(self, window_size=50, tau=0.85, delta=0.005, n_adjoint=4, k_tracking=10):
        self.s = window_size
        self.tau = tau
        self.delta = delta
        self.n = n_adjoint
        self.k = k_tracking
    
    def _train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train on X_train, test on X_test"""
        if len(np.unique(y_train)) < 2 or len(X_train) < 2:
            return 0.5
        model = SVC(kernel='rbf', C=1.0)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))
    
    def detection_process(self, stream_X, stream_y):
        """Algorithm 1: Detect drift position
        Train on W_A (reference), Test on W_B (current)
        p_det = accuracy(train W_A, test W_B) / accuracy(train W_A, test W_A)
        """
        W_A_X = stream_X[:self.s]
        W_A_y = stream_y[:self.s]
        
        # Baseline accuracy on W_A
        alpha_A = self._train_and_evaluate(W_A_X, W_A_y, W_A_X, W_A_y)
        
        t = 1
        max_pos = (len(stream_X) - self.s) // self.s
        
        while t < max_pos:
            start_idx = t * self.s
            end_idx = start_idx + self.s
            
            W_B_X = stream_X[start_idx:end_idx]
            W_B_y = stream_y[start_idx:end_idx]
            
            # Train on W_A, Test on W_B
            alpha_cross = self._train_and_evaluate(W_A_X, W_A_y, W_B_X, W_B_y)
            p_det = alpha_cross / (alpha_A + 1e-10)
            
            if p_det < self.tau:
                return t
            t += 1
        
        return -1
    
    def growth_process(self, stream_X, stream_y, drift_pos):
        """Algorithm 2: Detect drift length and category
        Use cross-window accuracy to measure stability
        """
        # Reference window (before drift)
        W_ref_X = stream_X[:self.s]
        W_ref_y = stream_y[:self.s]
        
        start_R = (drift_pos + 1) * self.s
        
        def compute_variance_in_WR(start_pos):
            accuracies = []
            for j in range(self.n):
                idx_start = start_pos + j * self.s
                idx_end = idx_start + self.s
                if idx_end > len(stream_X):
                    break
                W_j_X = stream_X[idx_start:idx_end]
                W_j_y = stream_y[idx_start:idx_end]
                
                # Cross-window accuracy
                acc = self._train_and_evaluate(W_ref_X, W_ref_y, W_j_X, W_j_y)
                accuracies.append(acc)
            
            if len(accuracies) < 2:
                return None, accuracies
            return np.var(accuracies), accuracies
        
        variance, _ = compute_variance_in_WR(start_R)
        if variance is None:
            return 1, "TCD"
        
        if variance <= self.delta:
            return 1, "TCD"
        
        drift_length = 1
        for i in range(1, 10):
            new_start = start_R + i * self.s
            variance, _ = compute_variance_in_WR(new_start)
            
            if variance is None:
                break
            
            drift_length = i + 1
            
            if variance <= self.delta:
                return drift_length, "PCD"
        
        return drift_length, "PCD"
    
    def tracking_process(self, stream_X, stream_y, drift_pos, drift_length):
        """Algorithm 3: Compute TFR curve
        W'_B: static at drift position
        W'_A: sliding from position 0
        TFR = accuracy(train W'_A, test W'_B) / accuracy(train W'_B, test W'_B)
        """
        m = max(drift_length, 1)
        
        W_B_start = drift_pos * self.s
        W_B_end = W_B_start + m * self.s
        
        if W_B_end > len(stream_X):
            return []
        
        W_B_X = stream_X[W_B_start:W_B_end]
        W_B_y = stream_y[W_B_start:W_B_end]
        
        alpha_B_star = self._train_and_evaluate(W_B_X, W_B_y, W_B_X, W_B_y)
        
        tfr_values = []
        
        for i in range(self.k):
            W_A_start = i * self.s
            W_A_end = W_A_start + m * self.s
            
            if W_A_end > len(stream_X):
                break
            
            W_A_X = stream_X[W_A_start:W_A_end]
            W_A_y = stream_y[W_A_start:W_A_end]
            
            # Train on W'_A, Test on W'_B
            alpha_A = self._train_and_evaluate(W_A_X, W_A_y, W_B_X, W_B_y)
            tfr = alpha_A / (alpha_B_star + 1e-10)
            tfr_values.append(tfr)
        
        return tfr_values
    
    def identify_subcategory(self, tfr_values, drift_category):
        """Identify subcategory from TFR curve"""
        if len(tfr_values) < 3:
            return "Unknown"
        
        if drift_category == "TCD":
            initial = np.mean(tfr_values[:2])
            final = np.mean(tfr_values[-2:])
            
            if initial < 1 and abs(final - 1) < 0.15:
                return "Sudden"
            
            mid_idx = len(tfr_values) // 2
            middle = np.mean(tfr_values[mid_idx-1:mid_idx+1])
            if initial < 1 and middle > initial and final < 1:
                return "Blip"
            
            if np.std(tfr_values) > 0.2:
                return "Recurrent"
            
            return "Sudden"
        
        else:  # PCD
            if np.var(tfr_values) > 0.1:
                return "Gradual"
            return "Incremental"
    
    def detect(self, stream_X, stream_y):
        result = {
            'drift_detected': False,
            'drift_position': None,
            'drift_length': None,
            'drift_category': None,
            'drift_subcategory': None,
            'tfr_curve': None
        }
        
        drift_pos = self.detection_process(stream_X, stream_y)
        
        if drift_pos == -1:
            return result
        
        result['drift_detected'] = True
        result['drift_position'] = drift_pos
        
        drift_length, drift_category = self.growth_process(stream_X, stream_y, drift_pos)
        result['drift_length'] = drift_length
        result['drift_category'] = drift_category
        
        tfr_values = self.tracking_process(stream_X, stream_y, drift_pos, drift_length)
        result['tfr_curve'] = tfr_values
        
        subcategory = self.identify_subcategory(tfr_values, drift_category)
        result['drift_subcategory'] = subcategory
        
        return result


# === DEMO ===
if __name__ == "__main__":
    np.random.seed(42)
    
    window_size = 50
    
    # Old distribution
    X_old = np.random.randn(500, 5)
    y_old = (X_old[:, 0] + X_old[:, 1] > 0).astype(int)
    
    # New distribution
    X_new = np.random.randn(500, 5) + 2
    y_new = (X_new[:, 0] - X_new[:, 1] > 0).astype(int)
    
    stream_X = np.vstack([X_old, X_new])
    stream_y = np.hstack([y_old, y_new])
    
    detector = CDT_MSW(window_size=50, tau=0.8, delta=0.01)
    result = detector.detect(stream_X, stream_y)
    
    print("=== CDT-MSW Results ===")
    print(f"Drift detected: {result['drift_detected']}")
    print(f"Drift position: window {result['drift_position']}")
    print(f"Sample index: {result['drift_position'] * window_size if result['drift_position'] else 'N/A'}")
    print(f"Expected: window 10 (sample 500)")
    print(f"Drift category: {result['drift_category']}")
    print(f"Drift subcategory: {result['drift_subcategory']}")

    # Debug growth process
    print("\n=== Debug Growth ===")
    drift_pos = result['drift_position']
    start_R = (drift_pos + 1) * window_size
    
    W_ref_X = stream_X[:window_size]
    W_ref_y = stream_y[:window_size]
    
    accuracies = []
    for j in range(4):  # n_adjoint = 4
        idx_start = start_R + j * window_size
        idx_end = idx_start + window_size
        W_j_X = stream_X[idx_start:idx_end]
        W_j_y = stream_y[idx_start:idx_end]
        
        acc = detector._train_and_evaluate(W_ref_X, W_ref_y, W_j_X, W_j_y)
        accuracies.append(acc)
        print(f"Window {j}: accuracy = {acc:.4f}")
    
    variance = np.var(accuracies)
    print(f"Variance: {variance:.6f}")
    print(f"Delta threshold: {detector.delta}")
    print(f"Variance <= delta? {variance <= detector.delta}")
