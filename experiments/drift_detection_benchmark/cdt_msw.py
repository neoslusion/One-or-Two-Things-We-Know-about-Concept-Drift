"""
CDT_MSW: Concept Drift Type identification based on Multi-Sliding Windows
Paper: Guo et al., Information Sciences 585 (2022) 1-23

Implementation by: [Your Name]
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class CDT_MSW:
    """
    Concept Drift Type identification based on Multi-Sliding Windows.
    
    This method identifies:
    - Drift position (where drift occurs)
    - Drift category (TCD: Transient, PCD: Progressive)
    - Drift length (number of blocks during transition)
    - TFR curve (for subcategory identification)
    
    Parameters

    s : int, default=40
        Basic window size (number of samples per block)
    sigma : float, default=0.75
        Drift position detection threshold (lower = more sensitive)
    d : float, default=0.01
        Drift length detection threshold (variance threshold)
    k : int, default=10
        Tracking process stop parameter
    n : int, default=4
        Number of basic windows in adjoint window
    
    References

    [1] Guo, H., Li, H., Ren, Q., & Wang, W. (2022). Concept drift type 
        identification based on multi-sliding windows. Information Sciences, 
        585, 1-23.
    """
    
    def __init__(self, s=40, sigma=0.85, d=0.005, k=10, n=6):
        self.s = s
        self.sigma = sigma
        self.d = d
        self.k = k
        self.n = n
        
    def _self_accuracy(self, X, y):
        """
        Compute real-time accuracy using cross-validation.
        
        As per paper Section 4.1: "the accuracy of the learner trained 
        by the data in window W is called the real-time accuracy of W"
        """
        if len(X) < 5 or len(np.unique(y)) < 2:
            return 1.0
        try:
            clf = SVC(kernel='rbf', C=1.0, gamma='scale')
            cv = min(5, len(X) // 2)
            if cv < 2:
                return 1.0
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            return np.mean(scores)
        except:
            return 1.0
    
    def _cross_accuracy(self, X_train, y_train, X_test, y_test):
        """Train on reference window, test on current window."""
        if len(np.unique(y_train)) < 2:
            return 0.5
        try:
            clf = SVC(kernel='rbf', C=1.0, gamma='scale')
            clf.fit(X_train, y_train)
            return clf.score(X_test, y_test)
        except:
            return 0.5

    def detect(self, stream_X, stream_y):
        """
        Main detection pipeline.
        
        Parameters

        stream_X : array-like of shape (n_samples, n_features)
            Feature matrix of streaming data
        stream_y : array-like of shape (n_samples,)
            Labels of streaming data
            
        Returns

        results : dict
            Dictionary containing:
            - 'drift_positions': list of detected drift positions
            - 'drift_categories': list of categories ('TCD' or 'PCD')
            - 'drift_lengths': list of drift lengths
            - 'tfr_curves': list of TFR curves for subcategory identification
        """
        # Split stream into blocks of size s
        n_blocks = len(stream_X) // self.s
        blocks_X = [stream_X[i*self.s:(i+1)*self.s] for i in range(n_blocks)]
        blocks_y = [stream_y[i*self.s:(i+1)*self.s] for i in range(n_blocks)]
        
        if n_blocks < self.n + 3:
            return {'drift_positions': [], 'drift_categories': [], 
                    'drift_lengths': [], 'tfr_curves': []}
        
        results = {
            'drift_positions': [],
            'drift_categories': [],
            'drift_lengths': [],
            'tfr_curves': [],
            'drift_subcategories': []
        }
        
        # === DETECTION PROCESS (Algorithm 1) ===
        # Use multiple initial blocks for stable baseline
        init_blocks = min(3, n_blocks // 4)
        ref_X = np.vstack(blocks_X[:init_blocks])
        ref_y = np.concatenate(blocks_y[:init_blocks])
        
        t = init_blocks
        
        while t < n_blocks:
            curr_X = blocks_X[t]
            curr_y = blocks_y[t]
            
            # Cross-accuracy: train on reference, test on current
            a_t_B = self._cross_accuracy(ref_X, ref_y, curr_X, curr_y)
            baseline = self._cross_accuracy(ref_X, ref_y, ref_X, ref_y)
            baseline = max(baseline, 0.5)
            
            # Detection flow ratio (Formula 2 in paper)
            P_det = a_t_B / baseline
            
            if P_det < self.sigma:
                drift_pos = t
                
                # Verification: check if next block also shows drift
                if drift_pos + 2 < n_blocks:
                    next_acc = self._cross_accuracy(ref_X, ref_y, 
                                                     blocks_X[drift_pos+1], 
                                                     blocks_y[drift_pos+1])
                    if next_acc / baseline >= self.sigma:
                        t += 1
                        continue  # False alarm
                
                # === GROWTH PROCESS (Algorithm 2) ===
                if drift_pos + self.n >= n_blocks:
                    break
                
                # Adjoint window W_R accuracies
                acc_R = []
                for j in range(self.n):
                    idx = drift_pos + 1 + j
                    if idx >= n_blocks:
                        break
                    acc_R.append(self._self_accuracy(blocks_X[idx], blocks_y[idx]))
                
                if len(acc_R) < self.n:
                    break
                
                # Variance of accuracies (Formula 3 in paper)
                var_R = np.var(acc_R)
                m = 1
                
                if var_R <= self.d:
                    category = "TCD"
                else:
                    category = "PCD"
                    max_m = min(20, n_blocks - drift_pos - self.n)
                    while var_R > self.d and m < max_m:
                        m += 1
                        new_idx = drift_pos + m + self.n - 1
                        if new_idx >= n_blocks:
                            break
                        acc_R.pop(0)
                        acc_R.append(self._self_accuracy(blocks_X[new_idx], 
                                                          blocks_y[new_idx]))
                        var_R = np.var(acc_R)
                
                # === TRACKING PROCESS (Algorithm 3) ===
                tfr_vector = []
                end_idx = min(drift_pos + m, n_blocks)
                if end_idx > drift_pos:
                    X_B = np.vstack(blocks_X[drift_pos:end_idx])
                    y_B = np.concatenate(blocks_y[drift_pos:end_idx])
                    
                    # TFR curve (Formula 5 in paper): Cross-Accuracy Ratio
                    # Measures similarity of drift block B to historical block A
                    for i in range(min(self.k + 1, n_blocks - m + 1)):
                        X_A = np.vstack(blocks_X[i:i+m])
                        y_A = np.concatenate(blocks_y[i:i+m])
                        
                        a_A = self._self_accuracy(X_A, y_A)
                        a_B_given_A = self._cross_accuracy(X_A, y_A, X_B, y_B)
                        
                        tfr = a_B_given_A / a_A if a_A > 0 else 0.0
                        tfr_vector.append(round(tfr, 3))
                
                # Identify subcategory based on TFR curve
                subcategory = self._identify_subcategory(category, m, tfr_vector)

                # Store results
                results['drift_positions'].append(drift_pos * self.s)
                results['drift_categories'].append(category)
                results['drift_lengths'].append(m)
                results['tfr_curves'].append(tfr_vector)
                results['drift_subcategories'].append(subcategory)
                
                # Update reference to new distribution
                new_ref_start = drift_pos + m
                new_ref_end = min(new_ref_start + init_blocks, n_blocks)
                if new_ref_end > new_ref_start:
                    ref_X = np.vstack(blocks_X[new_ref_start:new_ref_end])
                    ref_y = np.concatenate(blocks_y[new_ref_start:new_ref_end])
                
                t = new_ref_end
            else:
                t += 1
        
        return results

    def _identify_subcategory(self, category, length, tfr_vector):
        """
        Identify drift subcategory based on TFR curve properties and drift length.
        """
        if not tfr_vector:
            return "sudden" if category == "TCD" else "gradual"
        
        tfr = np.array(tfr_vector)
        max_tfr = np.max(tfr)
        mean_tfr = np.mean(tfr)
        
        if category == "TCD":
            # Recurrent: High similarity to some past block
            if max_tfr > 0.8:
                return "recurrent"
            # Blip: Extremely short duration
            if length <= 1:
                return "blip"
            # Sudden: Distinct new concept (low similarity to past)
            return "sudden"
            
        else: # PCD
            # Incremental: Higher average similarity (slow shift retains some past knowledge)
            if mean_tfr > 0.6:
                return "incremental"
            # Gradual: Lower similarity (mixed distribution causes confusion)
            return "gradual"
