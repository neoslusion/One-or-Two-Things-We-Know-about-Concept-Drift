"""
Metric calculation functions for drift detection evaluation.

Implements standard drift detection metrics including:
- Beta score (F-beta score from original ShapeDD paper)
- Precision, Recall, F1-Score
- Mean Time To Detection (MTTD)
- Mean Time Between False Alarms (MTFA)
- Mean Time Ratio (MTR)
"""

import numpy as np


def calculate_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Calculate β-score (F-beta score) - Original ShapeDD paper metric.

    β-score = (1 + β²) * (precision * recall) / (β² * precision + recall)

    Args:
        precision: Detection precision
        recall: Detection recall
        beta: Beta parameter
            β=0.5: Emphasizes precision (minimizes false alarms) - Original paper uses this
            β=1.0: Standard F1-score (equal weight)
            β=2.0: Emphasizes recall (catches all drifts)

    Returns:
        Beta score value
    """
    if precision + recall == 0:
        return 0.0

    beta_squared = beta ** 2
    numerator = (1 + beta_squared) * precision * recall
    denominator = beta_squared * precision + recall

    return numerator / denominator if denominator > 0 else 0.0


def calculate_detection_metrics_enhanced(detections, true_drifts, stream_length,
                                         acceptable_delta=150):
    """
    Calculate detection performance metrics following standard drift detection practice.

    This implementation follows the event-based approach used in most drift detection
    papers, which do NOT compute True Negatives (TN) due to definitional ambiguity.

    Implements metrics from:
    - Basseville & Nikiforov (1993): MTFA, MTR
    - Bifet et al. (2020): F1 as primary metric
    - Standard drift detection practice: Precision, Recall, MTTD

    Args:
        detections: List of detected drift positions
        true_drifts: List of true drift positions
        stream_length: Total length of data stream
        acceptable_delta: Acceptable delay window (default: 150)

    Returns:
        dict: Comprehensive metrics including:
            - Basic: TP, FP, FN
            - Primary: Precision, Recall, F1
            - Temporal: MTTD, Median TTD, MTFA, MTR, MDR
            - Other: Detection rate, n_detections
    """
    detections = sorted([int(d) for d in detections])

    # Handle no-drift case
    if not true_drifts or len(true_drifts) == 0:
        fp = len(detections)

        return {
            # Basic counts
            'tp': 0, 'fp': fp, 'fn': 0,

            # Primary metrics
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'beta_score': 0.0,

            # Temporal metrics
            'mttd': float('inf'),  # Mean Time To Detection
            'median_ttd': float('inf'),  # Median TTD (more robust)
            'mtfa': stream_length / fp if fp > 0 else float('inf'),  # Mean Time Between False Alarms
            'mtr': 0.0,  # Mean Time Ratio
            'mdr': 0.0,  # Missed Detection Rate

            # Additional
            'detection_rate': 0.0,
            'false_alarm_rate': fp / stream_length if stream_length > 0 else 0.0,
            'n_detections': fp,
            'n_true_drifts': 0
        }

    # Convert true_drifts to list
    if isinstance(true_drifts, (int, float)):
        true_drifts = [int(true_drifts)]
    else:
        true_drifts = [int(d) for d in true_drifts]

    # ==================================================================
    # STEP 1: Match detections to true drifts
    # ==================================================================
    matched_detections = set()
    per_drift_delays = []

    for true_drift in true_drifts:
        # Find all detections within acceptable window
        valid_detections = [(d, abs(d - true_drift)) for d in detections
                           if abs(d - true_drift) <= acceptable_delta
                           and d not in matched_detections]

        if valid_detections:
            # Match to closest detection
            closest_det, delay = min(valid_detections, key=lambda x: x[1])
            matched_detections.add(closest_det)
            per_drift_delays.append(delay)

    # ==================================================================
    # STEP 2: Calculate basic confusion matrix (NO TN)
    # ==================================================================
    tp = len(matched_detections)  # True Positives
    fn = len(true_drifts) - tp     # False Negatives
    fp = len(detections) - len(matched_detections)  # False Positives

    # ==================================================================
    # STEP 3: Primary Detection Metrics
    # ==================================================================
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    beta_score = calculate_beta_score(precision, recall, beta=0.5)  # Original paper metric

    # ==================================================================
    # STEP 4: Temporal Metrics
    # ==================================================================

    # Mean Time To Detection (MTTD)
    mttd = np.mean(per_drift_delays) if per_drift_delays else float('inf')

    # Median Time To Detection (more robust to outliers)
    median_ttd = np.median(per_drift_delays) if per_drift_delays else float('inf')

    # Mean Time Between False Alarms (MTFA)
    # Source: Basseville & Nikiforov (1993)
    # This is the standard way to measure false alarm rate in drift detection
    # without needing True Negatives
    mtfa = stream_length / fp if fp > 0 else float('inf')

    # False Alarm Rate (alarms per sample)
    # Alternative representation of MTFA
    false_alarm_rate = fp / stream_length if stream_length > 0 else 0.0

    # Mean Time Ratio (MTR)
    # Combines detection rate, MTTD, and MTFA into single score
    # Source: Basseville & Nikiforov (1993)
    if mttd > 0 and mttd != float('inf'):
        mtr = (recall * mtfa) / mttd
    else:
        mtr = 0.0

    # Missed Detection Rate (MDR)
    # Emphasizes cost of missed drifts
    mdr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as 1 - recall

    # ==================================================================
    # STEP 5: Additional Metrics
    # ==================================================================
    detection_rate = tp / len(true_drifts)

    # ==================================================================
    # RETURN: Standard drift detection metrics (NO TN)
    # ==================================================================
    return {
        # Basic confusion matrix (TP, FP, FN only - NO TN)
        'tp': tp,
        'fp': fp,
        'fn': fn,

        # Primary detection metrics (STANDARD)
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'beta_score': beta_score,

        # Temporal metrics (STANDARD)
        'mttd': mttd,              # Mean Time To Detection
        'median_ttd': median_ttd,  # Median TTD (robust)
        'mtfa': mtfa,              # Mean Time Between False Alarms ⭐ Replaces FPR
        'mtr': mtr,                # Mean Time Ratio
        'mdr': mdr,                # Missed Detection Rate

        # Additional metrics
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,  # FP per sample (alternative to MTFA)
        'n_detections': len(detections),
        'n_true_drifts': len(true_drifts)
    }


def calculate_detection_metrics(detections, true_drifts, acceptable_delta=150):
    """
    DEPRECATED: Use calculate_detection_metrics_enhanced instead.
    This wrapper maintains compatibility with old code.

    Note: Returns subset of metrics (without MTFA, MTR, etc.)
    """
    detections = sorted([int(d) for d in detections])

    if not true_drifts or len(true_drifts) == 0:
        return {
            'tp': 0, 'fp': len(detections), 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'beta_score': 0.0,
            'mttd': float('inf'),
            'detection_rate': 0.0,
            'n_detections': len(detections)
        }

    # Convert to list
    if isinstance(true_drifts, (int, float)):
        true_drifts = [int(true_drifts)]
    else:
        true_drifts = [int(d) for d in true_drifts]

    # Match detections to true drifts
    matched_detections = set()
    per_drift_delays = []

    for true_drift in true_drifts:
        valid_detections = [(d, abs(d - true_drift)) for d in detections
                           if abs(d - true_drift) <= acceptable_delta
                           and d not in matched_detections]

        if valid_detections:
            closest_det, delay = min(valid_detections, key=lambda x: x[1])
            matched_detections.add(closest_det)
            per_drift_delays.append(delay)

    # Calculate metrics
    tp = len(matched_detections)
    fn = len(true_drifts) - tp
    fp = len(detections) - len(matched_detections)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    beta_score = calculate_beta_score(precision, recall, beta=0.5)  # Original paper metric

    mttd = np.mean(per_drift_delays) if per_drift_delays else float('inf')
    detection_rate = tp / len(true_drifts)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'beta_score': beta_score,
        'mttd': mttd,
        'detection_rate': detection_rate,
        'n_detections': len(detections)
    }

