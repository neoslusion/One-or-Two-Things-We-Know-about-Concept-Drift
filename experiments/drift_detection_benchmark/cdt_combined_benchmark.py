"""
Combined CDT Benchmark: CDT_MSW vs SE-CDT
==========================================

Fair comparison of:
1. CDT_MSW (Guo et al. 2022) - Accuracy-based, supervised
2. SE-CDT - MMD-based, unsupervised

Both tested on same synthetic datasets.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_drift_data(drift_type, n_samples=10000, n_features=10, seed=42):
    """Generate drift data with labels and true drift info."""
    np.random.seed(seed)
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    drift_points = []
    
    if drift_type == "sudden":
        segment = n_samples // 6
        for i in range(6):
            start, end = i * segment, min((i+1) * segment, n_samples)
            X[start:end] = np.random.randn(end-start, n_features)
            y[start:end] = (X[start:end, 0] > 0).astype(int) if i % 2 == 0 else (X[start:end, 0] < 0).astype(int)
            if i > 0:
                drift_points.append(start)
        return X, y, drift_points, "TCD", "sudden"
    
    elif drift_type == "gradual":
        segment = n_samples // 4
        tw = segment // 2
        for t in range(n_samples):
            X[t] = np.random.randn(n_features)
            seg_idx = t // segment
            pos = t % segment
            if pos < tw and seg_idx > 0:
                alpha = pos / tw
                concept = seg_idx % 2 if np.random.random() < alpha else (seg_idx-1) % 2
            else:
                concept = seg_idx % 2
            y[t] = int(X[t, 0] > 0) if concept == 0 else int(X[t, 0] < 0)
            if t > 0 and t == seg_idx * segment:
                drift_points.append(t)
        return X, y, drift_points, "PCD", "gradual"
    
    elif drift_type == "incremental":
        X = np.random.randn(n_samples, n_features)
        boundary = np.linspace(-1, 1, n_samples)
        y = (X[:, 0] > boundary).astype(int)
        return X, y, [], "PCD", "incremental"
    
    elif drift_type == "recurrent":
        period = 2000
        for t in range(n_samples):
            X[t] = np.random.randn(n_features)
            concept = (t // period) % 2
            y[t] = int(X[t, 0] > 0) if concept == 0 else int(X[t, 0] < 0)
            if t > 0 and t % period == 0:
                drift_points.append(t)
        return X, y, drift_points, "PCD", "recurrent"
    
    elif drift_type == "blip":
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)
        for i in range(5):
            blip_start = (i+1) * n_samples // 6
            blip_end = min(blip_start + 100, n_samples)
            y[blip_start:blip_end] = (X[blip_start:blip_end, 0] < 0).astype(int)
            drift_points.append(blip_start)
        return X, y, drift_points, "TCD", "blip"
    
    raise ValueError(f"Unknown type: {drift_type}")


# =============================================================================
# CDT_MSW CLASSIFIER (Accuracy-based, Supervised)
# =============================================================================

class CDT_MSW_Classifier:
    def __init__(self, window_size=200, threshold=0.85, step=50):
        self.window_size = window_size
        self.threshold = threshold
        self.step = step
    
    def classify(self, X, y):
        """Classify drift type using accuracy ratio."""
        n = len(X)
        s = self.window_size
        
        if n < 2 * s:
            return None, None
        
        # Detection phase
        X_A, y_A = X[:s], y[:s]
        model_A = LogisticRegression(max_iter=100).fit(X_A, y_A)
        a_star_A = model_A.score(X_A, y_A)
        
        drift_pos = None
        for t in range(s, n - s, self.step):
            X_B, y_B = X[t:t+s], y[t:t+s]
            if len(np.unique(y_B)) < 2:
                continue
            model_B = LogisticRegression(max_iter=100).fit(X_B, y_B)
            a_t_B = model_B.score(X_A, y_A)
            if a_t_B / (a_star_A + 1e-10) < self.threshold:
                drift_pos = t
                break
        
        if drift_pos is None:
            return "incremental", "PCD"  # No discrete drift
        
        # Growth phase - simplified
        m = 1
        t_end = drift_pos + s
        while t_end < n:
            X_cur = X[t_end-s:t_end]
            y_cur = y[t_end-s:t_end]
            if len(np.unique(y_cur)) < 2:
                break
            model_cur = LogisticRegression(max_iter=100).fit(X[:s], y[:s])
            acc = model_cur.score(X_cur, y_cur)
            if acc > 0.7:  # Stabilized
                break
            m += 1
            t_end += self.step
            if m > 10:
                break
        
        category = "TCD" if m <= 2 else "PCD"
        
        # Tracking phase - simplified
        tfr = []
        for i in range(5):
            t = drift_pos + i * self.step
            if t + s > n:
                break
            X_B = X[t:t+s]
            try:
                acc_B = model_A.score(X_B, y[t:t+s])
                tfr.append(acc_B)
            except:
                tfr.append(0.5)
        
        # Classify subcategory
        if len(tfr) >= 2:
            if tfr[-1] > 0.7 and min(tfr) < 0.5:
                return "blip", "TCD"
            if np.std(tfr) < 0.1:
                return "incremental", "PCD"
            if category == "TCD":
                return "sudden", "TCD"
            return "gradual", "PCD"
        
        return "sudden" if category == "TCD" else "gradual", category


# =============================================================================
# SE-CDT CLASSIFIER (MMD-based, Unsupervised)
# =============================================================================

class SE_CDT_Classifier:
    def __init__(self, window_size=200, stride=50):
        self.window_size = window_size
        self.stride = stride
    
    def compute_mmd(self, X, Y):
        if len(X) < 2 or len(Y) < 2:
            return 0.0
        combined = np.vstack([X, Y])
        dists = pdist(combined)
        if len(dists) == 0:
            return 0.0
        gamma = 1.0 / (2 * np.median(dists)**2 + 1e-10)
        m, n = len(X), len(Y)
        K_XX = np.exp(-gamma * squareform(pdist(X, 'sqeuclidean')))
        K_YY = np.exp(-gamma * squareform(pdist(Y, 'sqeuclidean')))
        K_XY = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2))
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        mmd_sq = np.sum(K_XX)/(m*(m-1)) + np.sum(K_YY)/(n*(n-1)) - 2*np.mean(K_XY)
        return max(0, mmd_sq)**0.5
    
    def classify(self, X, y=None):
        """Classify drift type using MMD signal (ignores y - unsupervised)."""
        n = len(X)
        s = self.window_size
        
        # Compute MMD signal
        times, sigma = [], []
        for t in range(s, n - s, self.stride):
            sigma.append(self.compute_mmd(X[t-s:t], X[t:t+s]))
            times.append(t)
        
        if len(sigma) < 3:
            return None, None
        
        times = np.array(times)
        sigma = gaussian_filter1d(np.array(sigma), sigma=2)
        
        # Statistics
        max_s, mean_s, std_s, median_s = np.max(sigma), np.mean(sigma), np.std(sigma), np.median(sigma)
        snr = max_s / (median_s + 1e-10)
        
        # Peak detection
        thresh = mean_s + 0.5 * std_s
        peaks, _ = find_peaks(sigma, height=thresh, distance=5, prominence=0.05)
        n_peaks = len(peaks)
        
        # Width analysis
        widths = []
        for p in peaks:
            hm = sigma[p] / 2
            l, r = p, p
            while l > 0 and sigma[l] > hm: l -= 1
            while r < len(sigma) - 1 and sigma[r] > hm: r += 1
            widths.append(r - l)
        avg_width = np.mean(widths) if widths else 0
        
        # Periodicity
        period_cv = 1.0
        if n_peaks >= 3:
            intervals = np.diff(times[peaks])
            if len(intervals) > 0 and np.mean(intervals) > 0:
                period_cv = np.std(intervals) / np.mean(intervals)
        
        # Classification
        below_baseline = np.sum(sigma < mean_s + 0.2 * std_s) / len(sigma)
        
        if n_peaks == 0:
            return "incremental", "PCD"
        if n_peaks <= 3 and below_baseline > 0.8 and snr > 4:
            return "blip", "TCD"
        if n_peaks >= 2 and n_peaks <= 5 and snr > 3:
            if period_cv > 0.3:
                return "sudden", "TCD"
        if n_peaks >= 4 and period_cv < 0.3:
            return "recurrent", "PCD"
        if avg_width * self.stride / (2 * s) > 2:
            return "gradual", "PCD"
        if n_peaks <= 5 and snr > 2.5:
            return "sudden", "TCD"
        return "gradual", "PCD"


# =============================================================================
# BENCHMARK
# =============================================================================

def run_combined_benchmark():
    """Run fair comparison between CDT_MSW and SE-CDT."""
    print("=" * 80)
    print("COMBINED CDT BENCHMARK: CDT_MSW vs SE-CDT")
    print("=" * 80)
    
    drift_types = ["sudden", "gradual", "incremental", "recurrent", "blip"]
    n_runs = 5
    
    cdt_msw = CDT_MSW_Classifier()
    se_cdt = SE_CDT_Classifier()
    
    results = defaultdict(lambda: defaultdict(list))
    
    print(f"\nRunning {len(drift_types)} types × {n_runs} runs × 2 methods...")
    print("-" * 80)
    
    for dtype in drift_types:
        print(f"\n{dtype.upper()}:", end=" ", flush=True)
        
        for run in range(n_runs):
            X, y, drift_points, true_cat, true_type = generate_drift_data(dtype, seed=run*100)
            
            # CDT_MSW (supervised)
            start = time.time()
            pred_type, pred_cat = cdt_msw.classify(X, y)
            time_msw = time.time() - start
            
            if pred_type is not None:
                results[("CDT_MSW", dtype)]["type"].append(pred_type == true_type)
                results[("CDT_MSW", dtype)]["cat"].append(pred_cat == true_cat)
                results[("CDT_MSW", dtype)]["time"].append(time_msw)
            
            # SE-CDT (unsupervised)
            start = time.time()
            pred_type, pred_cat = se_cdt.classify(X)
            time_se = time.time() - start
            
            if pred_type is not None:
                results[("SE-CDT", dtype)]["type"].append(pred_type == true_type)
                results[("SE-CDT", dtype)]["cat"].append(pred_cat == true_cat)
                results[("SE-CDT", dtype)]["time"].append(time_se)
            
            print(".", end="", flush=True)
        
        print(" Done")
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY DRIFT TYPE")
    print("=" * 80)
    
    for dtype in drift_types:
        print(f"\n{dtype.upper()}:")
        print("-" * 60)
        print(f"{'Method':<12} {'Type Acc':<12} {'Cat Acc':<12} {'Time (ms)':<12}")
        print("-" * 60)
        
        for method in ["CDT_MSW", "SE-CDT"]:
            key = (method, dtype)
            if results[key]["type"]:
                type_acc = np.mean(results[key]["type"]) * 100
                cat_acc = np.mean(results[key]["cat"]) * 100
                avg_time = np.mean(results[key]["time"]) * 1000
                print(f"{method:<12} {type_acc:>8.1f}%    {cat_acc:>8.1f}%    {avg_time:>8.1f}")
            else:
                print(f"{method:<12} {'N/A':>8}     {'N/A':>8}     {'N/A':>8}")
    
    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"\n{'Method':<12} {'Type Acc':<12} {'Cat Acc':<12} {'Supervised':<12}")
    print("-" * 50)
    
    for method in ["CDT_MSW", "SE-CDT"]:
        all_type, all_cat = [], []
        for dtype in drift_types:
            all_type.extend(results[(method, dtype)]["type"])
            all_cat.extend(results[(method, dtype)]["cat"])
        
        if all_type:
            type_acc = np.mean(all_type) * 100
            cat_acc = np.mean(all_cat) * 100
            supervised = "Yes" if method == "CDT_MSW" else "No"
            print(f"{method:<12} {type_acc:>8.1f}%    {cat_acc:>8.1f}%    {supervised:<12}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_combined_benchmark()
