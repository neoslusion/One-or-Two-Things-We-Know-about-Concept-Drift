import numpy as np
from collections import deque
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from .mmd_variants import (
    compute_gamma_median_heuristic,
    rbf_kernel,
    shapedd_idw_mmd_proper,
)


# ---------------------------------------------------------------------------
# Rolling-quantile feature baseline (Phase 3-mini (b))
# ---------------------------------------------------------------------------
class _RollingFeatureBaseline:
    """Per-feature rolling-quantile baseline.

    Used by the SE-CDT classifier to *self-calibrate* the SNR / WR / CV
    rule thresholds: instead of hard-coding "WR < 0.15" we ask whether
    the WR observed at a candidate drift is **lower than the 25th
    percentile** of WR observed on recent *no-drift* windows.

    The baseline is fed with features extracted from the MMD trace of
    windows where no drift was detected.  When fewer than ``min_history``
    samples have been collected we report ``None`` and the caller is
    expected to fall back on the original hard-coded threshold.
    """

    def __init__(self, max_size: int = 200, min_history: int = 20) -> None:
        self.max_size = int(max_size)
        self.min_history = int(min_history)
        self._history: dict[str, deque[float]] = {}

    def update(self, features: dict) -> None:
        for k, v in features.items():
            if not isinstance(v, (int, float, np.floating, np.integer)):
                continue
            v = float(v)
            if not np.isfinite(v):
                continue
            if k not in self._history:
                self._history[k] = deque(maxlen=self.max_size)
            self._history[k].append(v)

    def quantile(self, key: str, q: float) -> Optional[float]:
        h = self._history.get(key)
        if h is None or len(h) < self.min_history:
            return None
        return float(np.quantile(h, q))

    def n_samples(self, key: str) -> int:
        h = self._history.get(key)
        return 0 if h is None else len(h)

    def reset(self) -> None:
        self._history = {}

@dataclass
class SECDTResult:
    """Result from SE-CDT unified detection and classification."""
    is_drift: bool = False
    drift_type: str = "Unknown"  # TCD (Temporary), PCD (Permanent)
    subcategory: str = "Unknown" # Sudden, Blip, Gradual, Incremental, Recurrent
    features: Dict[str, float] = None
    score: float = 0.0
    p_value: float = 1.0  # NEW: Statistical p-value for drift detection
    mmd_trace: np.ndarray = None
    drift_positions: List[int] = field(default_factory=list)  # NEW: Detected positions
    classification_time: float = 0.0
    # NEW (Phase 3-mini a):
    # `recurrent_match_idx` is the index inside the concept memory of the
    # snapshot that most resembled the post-drift concept; -1 if no match.
    # `recurrent_distance` is the corresponding Standard-MMD distance, or
    # NaN when no match (or when concept memory is disabled).
    recurrent_match_idx: int = -1
    recurrent_distance: float = float("nan")


def _standard_mmd_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased squared MMD between two iid samples X, Y under an RBF kernel.

    Used by the concept-memory comparator below.  We use the *standard*
    (unweighted) MMD rather than IDW-MMD so that two snapshots from the
    same concept are scored similarly across density regimes.
    """
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    m, n = X.shape[0], Y.shape[0]
    # Unbiased U-statistic estimator
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    mmd_sq = (
        K_XX.sum() / max(m * (m - 1), 1)
        + K_YY.sum() / max(n * (n - 1), 1)
        - 2.0 * K_XY.mean()
    )
    return float(np.sqrt(max(0.0, mmd_sq)))


class SE_CDT:
    """
    SE-CDT (ShapeDD-Enhanced Concept Drift Type identification).
    Unified Detector-Classifier System.
    
    This implementation PROPERLY combines:
    1. ShapeDD shape statistic for drift DETECTION (candidate identification)
    2. IDW-MMD with asymptotic p-value for VALIDATION
    3. Signal shape analysis for drift TYPE CLASSIFICATION
    4. Concept Memory + Standard-MMD comparison for RECURRENT relabelling
       (Phase 3-mini (a); enabled by ``use_concept_memory=True``)

    Attributes
    ----------
    l1 : int
        Reference window size (half-window for shape statistic).
    l2 : int
        Test window size for MMD validation.
    alpha : float
        Significance level for drift detection (default 0.05).
    use_concept_memory : bool
        If True (default), keep a bounded ring buffer of past stable
        concepts and relabel a newly detected drift as "Recurrent" when
        its post-drift snapshot is closer (in Standard MMD) to a stored
        concept than ``concept_match_threshold``.
    concept_memory_size : int
        Maximum number of snapshots kept in the ring buffer.
    concept_snapshot_size : int
        Number of post-drift samples used as the snapshot per concept.
    concept_match_threshold : float
        Standard-MMD distance below which two snapshots are considered to
        come from the *same* concept. Default 0.15. This is calibrated
        empirically: with snapshot_size=150 and 5-D Gaussian noise, the
        Standard-MMD null distribution has a mean of ~0.08, while two
        snapshots that come from a +2.0 shifted concept produce
        distances of 0.6+. A threshold of 0.15 (~2× noise floor) cleanly
        separates recurrent matches from genuine new concepts. The
        previous 0.05 was below the noise floor and never triggered.
    """

    def __init__(
        self,
        window_size: int = 50,
        l2: int = 150,
        threshold: float = 0.15,
        alpha: float = 0.05,
        use_proper: bool = True,
        *,
        use_concept_memory: bool = True,
        concept_memory_size: int = 8,
        concept_snapshot_size: int = 150,
        concept_match_threshold: float = 0.15,
        use_self_calibration: bool = True,
        baseline_size: int = 200,
        baseline_min_history: int = 20,
    ):
        """
        Initialize SE-CDT detector.

        Parameters
        ----------
        window_size : int
            Reference window size (l1). Default 50.
        l2 : int
            Test window size for MMD. Default 150.
        threshold : float
            Legacy threshold parameter (kept for API compatibility).
        alpha : float
            Significance level for proper mode. Default 0.05.
        use_proper : bool
            Legacy parameter (ignored, always uses proper mode).
        use_concept_memory, concept_memory_size, concept_snapshot_size,
        concept_match_threshold :
            See class docstring (Phase 3-mini (a)).
        use_self_calibration : bool
            If True, learn a per-feature rolling-quantile baseline from
            no-drift windows and use it to **self-calibrate** the
            SNR/WR/CV thresholds in the classifier.  Falls back on the
            hard-coded thresholds while the baseline is still building
            up (Phase 3-mini (b)).
        baseline_size, baseline_min_history :
            Parameters of the rolling baseline.  Default 200/20.
        """
        self.l1 = window_size
        self.l2 = l2
        self.threshold = threshold
        self.alpha = alpha

        # Concept memory (Phase 3-mini (a)).
        self.use_concept_memory = use_concept_memory
        self.concept_memory_size = int(concept_memory_size)
        self.concept_snapshot_size = int(concept_snapshot_size)
        self.concept_match_threshold = float(concept_match_threshold)
        self._concept_memory: list[tuple[np.ndarray, float]] = []

        # Self-calibration (Phase 3-mini (b)).
        self.use_self_calibration = bool(use_self_calibration)
        self._feature_baseline = _RollingFeatureBaseline(
            max_size=baseline_size, min_history=baseline_min_history
        )

    # ------------------------------------------------------------------ #
    #  Concept-memory helpers                                              #
    # ------------------------------------------------------------------ #
    def reset_concept_memory(self) -> None:
        """Clear the concept memory (use between independent runs)."""
        self._concept_memory = []

    def _extract_post_drift_snapshot(
        self, window: np.ndarray, position: int
    ) -> Optional[np.ndarray]:
        """Pick a clean ``self.concept_snapshot_size`` slice *after* the drift.

        The slice starts ``self.l1`` samples after ``position`` so we skip
        the transition region, and it is centered inside ``window``.
        Returns ``None`` if the window is too short.
        """
        n = len(window)
        start = min(n, max(0, position + self.l1))
        end = min(n, start + self.concept_snapshot_size)
        if end - start < max(20, self.concept_snapshot_size // 2):
            return None
        return np.asarray(window[start:end], dtype=float)

    def _match_or_store_concept(
        self, snapshot: np.ndarray
    ) -> tuple[int, float]:
        """Match ``snapshot`` against memory; if new, push it.

        Returns (idx, distance). ``idx >= 0`` means "matched memory[idx]".
        ``idx == -1`` means "no match -> snapshot pushed to memory".
        """
        gamma_new = float(compute_gamma_median_heuristic(snapshot))
        best_idx = -1
        best_dist = float("inf")

        for i, (mem_snap, gamma_mem) in enumerate(self._concept_memory):
            # Use a single shared bandwidth: average of the two median
            # heuristics. This is invariant to ordering and avoids the
            # asymmetry that a per-side bandwidth would introduce.
            gamma = 0.5 * (gamma_new + gamma_mem)
            dist = _standard_mmd_unbiased(snapshot, mem_snap, gamma)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_dist <= self.concept_match_threshold:
            return best_idx, best_dist

        # No match -> store snapshot in the ring buffer.
        if len(self._concept_memory) >= self.concept_memory_size:
            # Drop oldest.
            self._concept_memory.pop(0)
        self._concept_memory.append((snapshot, gamma_new))
        return -1, best_dist if np.isfinite(best_dist) else float("nan")

    def monitor(self, window: np.ndarray) -> SECDTResult:
        """
        Monitor the stream window for drift.
        If drift is detected, automatically classify it.
        
        Uses PROPER ShapeDD + IDW-MMD design:
        1. ShapeDD shape statistic detects candidate drift points
        2. IDW-MMD with asymptotic p-value validates candidates
        3. MMD signal shape classifies drift type
        
        Parameters:
        -----------
        window : np.ndarray
            Current data window to check (buffer).
            
        Returns:
        --------
        result : SECDTResult
            Unified result containing:
            - is_drift: Whether drift was detected
            - p_value: Statistical significance (only in proper mode)
            - drift_type: TCD or PCD
            - subcategory: Sudden, Blip, Gradual, Incremental, Recurrent
            - mmd_trace: MMD signal for analysis
            - features: Extracted geometric features
        """
        return self._monitor_proper(window)
    
    def _monitor_proper(self, window: np.ndarray) -> SECDTResult:
        """
        PROPER implementation using ShapeDD + IDW-MMD with p-value.

        Pipeline:
        1. Detection:    ShapeDD + IDW-MMD finds drift candidates.
        2. Growth:       Width Ratio analysis measures drift length -> TCD vs PCD.
        3. Classification: Shape + temporal features -> subcategory.
        4. Concept memory (optional, Phase 3-mini (a)): if the post-drift
           segment closely matches a stored concept (by Standard-MMD), the
           classification is overridden to "Recurrent".
        """
        is_drift, drift_positions, mmd_trace, p_values = shapedd_idw_mmd_proper(
            window, l1=self.l1, l2=self.l2, alpha=self.alpha
        )

        if p_values:
            min_p = min(p_values)
            score = 1.0 - min_p
        else:
            min_p = 1.0
            score = 0.0

        result = SECDTResult(
            is_drift=is_drift,
            score=score,
            p_value=min_p,
            mmd_trace=mmd_trace,
            drift_positions=drift_positions,
        )

        # Self-calibration baseline (Phase 3-mini (b)):
        # update only on NO-drift windows so the baseline tracks the
        # null distribution of trace-shape features.  We accept trace
        # lengths as small as 3 (the bare minimum for shape statistics)
        # because the benchmark sliding window typically yields only
        # a handful of trace points per call.
        if (
            self.use_self_calibration
            and not is_drift
            and mmd_trace is not None
            and len(mmd_trace) >= 3
        ):
            baseline_features = self.extract_features(mmd_trace)
            if baseline_features:
                self._feature_baseline.update(baseline_features)

        if is_drift and len(mmd_trace) > 0:
            t0 = time.time()
            drift_length = self._growth_process(window, mmd_trace=mmd_trace)
            classification_res = self.classify(mmd_trace, drift_length=drift_length)

            # Concept-memory pass (recurrent relabelling).
            recurrent_idx = -1
            recurrent_dist = float("nan")
            if self.use_concept_memory and drift_positions:
                snapshot = self._extract_post_drift_snapshot(
                    window, drift_positions[-1]
                )
                if snapshot is not None:
                    recurrent_idx, recurrent_dist = self._match_or_store_concept(
                        snapshot
                    )
                    if recurrent_idx >= 0:
                        # Override the rule-based label: this is recurrent
                        # because the concept itself is in our memory.
                        classification_res.drift_type = "TCD"
                        classification_res.subcategory = "Recurrent"

            t1 = time.time()
            result.drift_type = classification_res.drift_type
            result.subcategory = classification_res.subcategory
            result.features = classification_res.features
            result.classification_time = t1 - t0
            result.recurrent_match_idx = int(recurrent_idx)
            result.recurrent_distance = float(recurrent_dist)

        return result
    
    def _growth_process(self, data_window: np.ndarray, mmd_trace: np.ndarray = None) -> int:
        """
        Growth process (CDT-MSW Algorithm 2, adapted for unsupervised MMD).

        Measures the Width Ratio (WR) of the dominant MMD peak to distinguish:
        - TCD (WR < WR_THRESHOLD): Sharp, narrow peak  → sudden / transient
        - PCD (WR >= WR_THRESHOLD): Wide peak         → gradual / incremental

        For PCD we report a *bucketed* drift length in trace-window units so
        that downstream code only uses it as a binary "1 vs >1" discriminator.
        Concretely we map the FWHM (in trace samples) to the number of
        non-overlapping reference windows (each of length self.l1) it spans,
        rounded up. This gives a length-independent, dimensionally-meaningful
        integer instead of the previous unexplained `fwhm / 3` magic factor
        (D4 fix).

        Parameters
        ----------
        data_window : np.ndarray
            Raw data buffer (kept for API parity; unused in this branch).
        mmd_trace : np.ndarray, optional
            Pre-computed MMD trace from the detection step.

        Returns
        -------
        drift_length : int
            1     -> TCD (sharp change, single window).
            >= 2  -> PCD, ceil(FWHM / l1) reference windows wide.
        """
        from scipy.signal import find_peaks, peak_widths

        if mmd_trace is None or len(mmd_trace) < 10:
            return 1

        sigma_s = gaussian_filter1d(mmd_trace, sigma=4)
        threshold = np.mean(sigma_s) + 0.3 * np.std(sigma_s)
        peaks, properties = find_peaks(sigma_s, height=threshold)

        if len(peaks) == 0:
            return 1

        best_peak_idx = np.argmax(properties['peak_heights'])
        widths, _, _, _ = peak_widths(sigma_s, [peaks[best_peak_idx]], rel_height=0.5)
        fwhm = float(widths[0])

        # Width Ratio: peak FWHM relative to the full reference-test window
        # 2*l1 is used as the normaliser to keep parity with extract_features().
        wr = fwhm / (2 * self.l1)

        WR_THRESHOLD = 0.12  # CDT-MSW-derived; sharp vs wide cutoff
        if wr < WR_THRESHOLD:
            return 1  # TCD

        # PCD: report drift length in *reference-window units*. ceil(fwhm/l1)
        # so a peak narrower than one window still counts as 2 (i.e. PCD),
        # and longer drifts are reported in proportionally more units.
        drift_length = int(np.ceil(fwhm / float(self.l1)))
        return max(2, drift_length)

    def extract_features(self, sigma_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric features from the MMD signal.
        Expected input: A window of MMD values centered around the detected drift.
        """
        if len(sigma_signal) == 0:
            return {}
            
        # 1. Smoothing (Moderate)
        sigma_s = gaussian_filter1d(sigma_signal, sigma=4) # Reverted to 4
        
        # 2. Peak Detection
        threshold = np.mean(sigma_s) + 0.3 * np.std(sigma_s)
        peaks, properties = find_peaks(sigma_s, height=threshold)
        
        n_p = len(peaks)
        
        # 3. Calculate Features
        features = {
            'n_p': n_p,
            'WR': 0.0,    # Width Ratio
            'SNR': 0.0,   # Signal-to-Noise Ratio
            'CV': 0.0,    # Coefficient of Variation (Periodicity)
            'Mean': np.mean(sigma_s),
            'peak_positions': peaks.tolist() if n_p > 0 else [],  # For Blip detection
            'PPR': 0.0,   # Peak Proximity Ratio (for Blip)
            'DPAR': 0.0,  # Dual-Peak Amplitude Ratio (for Blip)
            # Temporal features (for Incremental vs Gradual)
            'LTS': 0.0,   # Linear Trend Strength
            'SDS': 0.0,   # Step Detection Score
            'MS': 0.0     # Monotonicity Score
        }
        
        # SNR
        median_val = np.median(sigma_s)
        max_val = np.max(sigma_s) if len(sigma_s) > 0 else 0
        features['SNR'] = max_val / (median_val + 1e-10)
        
        if n_p > 0:
            best_peak_idx = np.argmax(properties['peak_heights'])
            
            # calculate widths at half height
            widths, width_heights, left_ips, right_ips = peak_widths(
                sigma_s, [peaks[best_peak_idx]], rel_height=0.5
            )
            fwhm = widths[0]
            features['WR'] = fwhm / (2 * self.l1)
            
            # Periodicity (CV) if multiple peaks
            if n_p >= 2:
                peak_distances = np.diff(peaks)
                if len(peak_distances) > 0:
                    features['CV'] = np.std(peak_distances) / (np.mean(peak_distances) + 1e-10)
                
                # PPR (Peak Proximity Ratio) - for Blip detection
                # Ratio of closest peak distance to signal length
                min_peak_distance = np.min(peak_distances)
                features['PPR'] = min_peak_distance / len(sigma_s)
                
                # DPAR (Dual-Peak Amplitude Ratio) - for Blip detection
                # If 2 peaks close together with similar heights = Blip
                if n_p == 2:
                    peak_heights = properties['peak_heights']
                    h1, h2 = peak_heights[0], peak_heights[1]
                    features['DPAR'] = min(h1, h2) / (max(h1, h2) + 1e-10)
        
        # 4. Extract Temporal Features (for Incremental vs Gradual)
        temporal_features = self.extract_temporal_features(sigma_s)
        features.update(temporal_features)
        
        return features

    def extract_temporal_features(self, sigma_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features to distinguish Incremental from Gradual drift.
        
        Incremental: Stepwise changes with monotonic trend
        Gradual: Smooth curve with oscillations
        
        Returns:
        --------
        temporal_features : dict
            - LTS (Linear Trend Strength): R² of linear fit
            - SDS (Step Detection Score): Number of significant steps
            - MS (Monotonicity Score): Ratio of monotonic segments
        """
        if len(sigma_signal) < 10:
            return {'LTS': 0.0, 'SDS': 0.0, 'MS': 0.0}
        
        signal = sigma_signal.copy()
        n = len(signal)
        
        # 1. LTS (Linear Trend Strength) - R² coefficient
        # High LTS = strong linear trend (Incremental)
        # Low LTS = curved/oscillating (Gradual)
        x = np.arange(n)
        if np.std(signal) > 1e-10:
            # Linear regression
            A = np.vstack([x, np.ones(n)]).T
            m, c = np.linalg.lstsq(A, signal, rcond=None)[0]
            y_pred = m * x + c
            
            # R² calculation
            ss_tot = np.sum((signal - np.mean(signal))**2)
            ss_res = np.sum((signal - y_pred)**2)
            lts = 1 - (ss_res / (ss_tot + 1e-10))
            lts = max(0.0, min(1.0, lts))  # Clamp to [0, 1]
        else:
            lts = 0.0
        
        # 2. SDS (Step Detection Score) - Count significant jumps
        # High SDS = many steps (Incremental)
        # Low SDS = smooth changes (Gradual)
        diffs = np.diff(signal)
        if len(diffs) > 0:
            diff_threshold = np.std(diffs) * 1.5  # 1.5 std = significant step
            significant_steps = np.sum(np.abs(diffs) > diff_threshold)
            sds = significant_steps / len(diffs)  # Normalized
        else:
            sds = 0.0
        
        # 3. MS (Monotonicity Score) - Ratio of monotonic direction
        # High MS = mostly increasing or decreasing (Incremental)
        # Low MS = back-and-forth oscillations (Gradual)
        if len(diffs) > 0:
            positive_changes = np.sum(diffs > 0)
            negative_changes = np.sum(diffs < 0)
            total_changes = positive_changes + negative_changes
            
            if total_changes > 0:
                # Monotonicity = how much one direction dominates
                ms = abs(positive_changes - negative_changes) / total_changes
            else:
                ms = 0.0
        else:
            ms = 0.0
        
        return {
            'LTS': float(lts),
            'SDS': float(sds),
            'MS': float(ms)
        }

    def classify(self, sigma_signal: np.ndarray, drift_length: int = None) -> SECDTResult:
        """
        Classify drift type based on Growth process + signal shape.

        Enhanced algorithm (CDT-MSW inspired):
        1. If drift_length > 1 (Growth process says PCD): use temporal features.
        2. Otherwise: shape-based decision tree using *adaptive* SNR/WR/CV
           thresholds (Phase 3-mini (b) self-calibration).  When the
           rolling baseline has not yet collected ``min_history`` samples
           we silently fall back on the original hard-coded thresholds.
        """
        features = self.extract_features(sigma_signal)
        if not features:
            return SECDTResult()

        n_p = features['n_p']
        wr = features['WR']
        snr = features['SNR']
        cv = features['CV']
        mean_val = features['Mean']
        peak_positions = features.get('peak_positions', [])
        ppr = features.get('PPR', 0.0)
        dpar = features.get('DPAR', 0.0)
        lts = features.get('LTS', 0.0)
        sds = features.get('SDS', 0.0)
        ms = features.get('MS', 0.0)

        result = SECDTResult(features=features)

        # ---- Adaptive thresholds (Phase 3-mini (b)) ---------------------
        # Strategy: keep the legacy hard-coded thresholds as a *safety
        # floor* and only *loosen* them based on the baseline.  This gives
        # us:
        #   - On clean datasets (baseline degenerate, mostly zeros) the
        #     thresholds stay at their original values -- behaviour is
        #     identical to the non-self-calibrated detector.
        #   - On noisy datasets (baseline shows non-trivial peaks even
        #     without drift) the thresholds shift to remain *significant*
        #     relative to that noise, e.g. snr_thresh becomes the 90th
        #     percentile of no-drift SNR if that exceeds 2.0.
        # The asymmetry is intentional: the legacy thresholds were
        # tuned to be conservative; we accept losing some recall on
        # already-noisy streams, in exchange for not introducing false
        # positives on clean streams.
        wr_thresh = 0.15
        snr_thresh = 2.0
        cv_thresh = 0.30
        if self.use_self_calibration:
            # SNR: real signal must be at least the 90th percentile of
            # no-drift SNR (or 2.0, whichever is larger).
            q_snr_high = self._feature_baseline.quantile("SNR", 0.90)
            if q_snr_high is not None and q_snr_high > snr_thresh:
                snr_thresh = float(q_snr_high)
            # WR: narrow peak must be narrower than the 25th percentile
            # of no-drift peak widths *if* the baseline actually has
            # peaks (i.e. the 25th percentile is non-trivial).  Otherwise
            # keep the hard-coded 0.15.
            q_wr_low = self._feature_baseline.quantile("WR", 0.25)
            if q_wr_low is not None and q_wr_low > wr_thresh:
                wr_thresh = float(q_wr_low)
            # CV: regular peaks must be more regular than the 25th
            # percentile of no-drift CV (when informative).
            q_cv_low = self._feature_baseline.quantile("CV", 0.25)
            if q_cv_low is not None and q_cv_low > cv_thresh:
                cv_thresh = float(q_cv_low)

        # =====================================================================
        # PRIMARY: Use drift_length from Growth process (CDT-MSW Algorithm 2)
        # drift_length > 1 -> PCD (distribution changed gradually)
        # =====================================================================
        if drift_length is not None and drift_length > 1:
            result.drift_type = "PCD"
            is_incremental = (
                (lts > 0.5) or
                (ms > 0.6 and lts > 0.3) or
                (sds > 0.12 and lts > 0.3) or
                (n_p >= 7) or
                (n_p == 0 and mean_val > 0.0001 and lts > 0.5)
            )
            result.subcategory = "Incremental" if is_incremental else "Gradual"
            return result

        # =====================================================================
        # FALLBACK: shape-based decision tree (drift_length == 1)
        # =====================================================================

        # 1. Blip Drift (TCD)
        # A Blip is a short, transient up-then-down pattern.  Detected
        # using only *relative* (length-independent) features (D3 fix):
        #   n_p == 2, PPR < 0.20, DPAR > 0.60, WR < 0.30
        if n_p == 2 and len(peak_positions) >= 2:
            is_blip = (
                ppr > 0 and ppr < 0.20
                and dpar > 0.60
                and wr < 0.30
            )
            if is_blip:
                result.drift_type = "TCD"
                result.subcategory = "Blip"
                return result

        # 2. Sudden Drift (TCD) -- adaptive WR & SNR thresholds.
        if n_p <= 3 and wr < wr_thresh and snr > snr_thresh:
            result.drift_type = "TCD"
            result.subcategory = "Sudden"
            return result

        # 3. Recurrent Drift (TCD) -- adaptive CV threshold.
        if n_p >= 4 and cv < cv_thresh and lts < 0.5:
            result.drift_type = "TCD"
            result.subcategory = "Recurrent"
            return result

        # 4. Gradual vs Incremental (PCD) -- fallback.
        result.drift_type = "PCD"
        is_incremental = (
            (lts > 0.5) or
            (ms > 0.6 and lts > 0.3) or
            (sds > 0.12 and lts > 0.3) or
            (n_p >= 7) or
            (n_p == 0 and mean_val > 0.0001 and lts > 0.5)
        )
        result.subcategory = "Incremental" if is_incremental else "Gradual"
        return result

