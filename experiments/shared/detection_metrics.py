"""
Shared Detection Metrics Calculation.

This module provides unified metrics calculation for drift detection evaluation,
ensuring consistency between benchmark_proper.py and evaluate_prequential.py.

Metrics:
    - TP (True Positives): Detections matching a true drift within tolerance
    - FP (False Positives): Detections not matching any true drift
    - FN (False Negatives): True drifts not detected
    - EDR (Detection Rate / Recall): TP / (TP + FN)
    - MDR (Missed Detection Rate): FN / (TP + FN) = 1 - EDR
    - Precision: TP / (TP + FP)
    - Mean Delay: Average detection delay for TPs
"""

import numpy as np
from typing import List, Dict, Union

# Default tolerance for matching detections to ground truth
DEFAULT_TOLERANCE = 250


def calculate_detection_metrics(
    detections: List[Union[int, Dict]],
    ground_truth_events: List[Union[int, Dict]],
    tolerance: int = DEFAULT_TOLERANCE,
    early_tolerance: int = 50
) -> Dict:
    """
    Calculate detection metrics: TP, FP, FN, EDR, MDR, Precision, Mean Delay.
    
    Args:
        detections: List of detection positions. Can be:
            - List of integers (sample indices)
            - List of dicts with 'pos' key (e.g., [{'pos': 1000}, {'pos': 2000}])
        ground_truth_events: List of true drift positions. Same format as detections.
        tolerance: Maximum allowed delay for a detection to be considered TP (samples after drift)
        early_tolerance: Maximum allowed early detection before drift (samples before drift)
    
    Returns:
        dict with keys:
            - TP: True Positives count
            - FP: False Positives count
            - FN: False Negatives count
            - EDR: Detection Rate (Recall) = TP / total_events
            - MDR: Missed Detection Rate = FN / total_events
            - Precision: TP / (TP + FP)
            - Mean_Delay: Average detection delay for TPs (can be negative for early detections)
            - delays: List of individual delays for each TP
    
    Example:
        >>> detections = [1050, 2100, 3500]
        >>> ground_truth = [1000, 2000, 4000]
        >>> metrics = calculate_detection_metrics(detections, ground_truth, tolerance=250)
        >>> print(f"EDR: {metrics['EDR']:.2f}, FP: {metrics['FP']}")
        EDR: 0.67, FP: 1
    """
    # Normalize inputs to position lists
    def get_positions(items):
        if not items:
            return []
        if isinstance(items[0], dict):
            return sorted([item['pos'] for item in items])
        return sorted(items)
    
    det_positions = get_positions(detections)
    gt_positions = get_positions(ground_truth_events)
    
    # Handle edge case: no ground truth events
    if not gt_positions:
        return {
            "TP": 0,
            "FP": len(det_positions),
            "FN": 0,
            "EDR": 0.0,
            "MDR": 0.0,
            "Precision": 0.0,
            "Mean_Delay": 0.0,
            "delays": []
        }
    
    tp = 0
    fp = 0
    delays = []
    
    # Track which events are detected to avoid double counting
    detected_events = set()
    
    # Match each detection to ground truth
    for det_pos in det_positions:
        matched = False
        
        for i, gt_pos in enumerate(gt_positions):
            if i in detected_events:
                continue
            
            # Acceptance window: [gt_pos - early_tolerance, gt_pos + tolerance]
            if gt_pos - early_tolerance <= det_pos <= gt_pos + tolerance:
                tp += 1
                delay = det_pos - gt_pos
                delays.append(delay)
                detected_events.add(i)
                matched = True
                break
        
        if not matched:
            fp += 1
    
    fn = len(gt_positions) - tp
    total_events = len(gt_positions)
    
    # Calculate rates
    edr = tp / total_events if total_events > 0 else 0.0
    mdr = fn / total_events if total_events > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mean_delay = float(np.mean(delays)) if delays else 0.0
    
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "EDR": edr,
        "MDR": mdr,
        "Precision": precision,
        "Mean_Delay": mean_delay,
        "delays": delays
    }


def format_detection_metrics(metrics: Dict, compact: bool = False) -> str:
    """
    Format detection metrics as a string for printing.
    
    Args:
        metrics: Dict from calculate_detection_metrics()
        compact: If True, return single-line format
    
    Returns:
        Formatted string
    """
    if compact:
        return (f"EDR={metrics['EDR']:.3f}, MDR={metrics['MDR']:.3f}, "
                f"Prec={metrics['Precision']:.3f}, FP={metrics['FP']}, "
                f"Delay={metrics['Mean_Delay']:.1f}")
    
    lines = [
        f"Detection Metrics:",
        f"  TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']}",
        f"  EDR (Recall):    {metrics['EDR']:.3f}",
        f"  MDR (Miss Rate): {metrics['MDR']:.3f}",
        f"  Precision:       {metrics['Precision']:.3f}",
        f"  Mean Delay:      {metrics['Mean_Delay']:.1f} samples"
    ]
    return "\n".join(lines)
