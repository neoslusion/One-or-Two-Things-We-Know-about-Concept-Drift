# Benchmark Consistency: Unified Threshold and Comprehensive Metrics

## Problem Statement

Two key evaluation scripts have inconsistencies and gaps:

### 1. `experiments/benchmark_proper.py`
- **Purpose**: Detection + Classification benchmark (TP, FP, EDR, MDR, CAT/SUB accuracy)
- **Issue**: Line 400 uses hardcoded `height=0.001, prom=0.0005` which is too low → **1513 FP**
- **Result**: Table `table_comparison_aggregate.tex` shows unrealistic EDR=0.980 with 1513 FP

### 2. `experiments/drift_monitoring_system/evaluate_prequential.py`
- **Purpose**: Full system demo (Detect → Classify → Adapt → Recover)
- **Issue**: Only measures adaptation effectiveness (Prequential Accuracy), NOT detection quality
- **Missing**: TP, FP, FN, EDR, MDR, Precision metrics
- **Threshold**: Uses `--sudden_thresh` CLI arg but disconnected from benchmark

## Goals

1. **Unified threshold tuning** in `benchmark_proper.py` as authoritative source
2. **Comprehensive metrics** in `evaluate_prequential.py` showing BOTH detection AND adaptation
3. **Shared utility** for detection metrics to avoid code duplication
4. **Consistent configuration** between both scripts

## Architecture

```
experiments/
├── benchmark_proper.py          # Detection/Classification benchmark
│   └── Uses: SHAPE_HEIGHT, SHAPE_PROMINENCE (configurable)
│
├── shared/
│   └── detection_metrics.py     # NEW: Shared TP/FP/EDR/MDR calculation
│
└── drift_monitoring_system/
    ├── evaluate_prequential.py  # Full system demo (UPDATED)
    │   └── Now reports: Detection + Adaptation metrics
    └── config.py                # SE-CDT threshold config
```

## Tasks

### Task 1: Create Shared Detection Metrics Utility
**File**: `experiments/shared/detection_metrics.py`

Extract `calculate_metrics()` from `benchmark_proper.py` into shared module:
```python
def calculate_detection_metrics(detections, ground_truth_events, tolerance=300):
    """
    Calculate detection metrics: TP, FP, FN, EDR, MDR, Precision, Mean Delay.
    
    Args:
        detections: List of detection positions (int or dict with 'pos' key)
        ground_truth_events: List of true drift positions (int or dict with 'pos' key)
        tolerance: Maximum allowed delay for a detection to be considered TP
    
    Returns:
        dict with keys: TP, FP, FN, EDR, MDR, Precision, Mean_Delay
    """
```

### Task 2: Fix Threshold in benchmark_proper.py
**File**: `experiments/benchmark_proper.py`

**Current (line 400)**:
```python
se_det_std = detect_peaks_from_signal(mmd_sig_std, height=0.001, prom=0.0005)  # Very low!
```

**Fixed**:
```python
# Use configurable thresholds from top of file
se_det_std = detect_peaks_from_signal(mmd_sig_std, height=SHAPE_HEIGHT, prom=SHAPE_PROMINENCE)
```

**Also update config section (lines 26-29)**:
```python
# Detection Thresholds - Tuned for balanced precision/recall
SHAPE_HEIGHT = 0.015      # Raised from 0.01 to reduce FP
SHAPE_PROMINENCE = 0.008  # Raised from 0.005
DETECTION_TOLERANCE = 250 # Standard tolerance
```

### Task 3: Add Detection Metrics to evaluate_prequential.py
**File**: `experiments/drift_monitoring_system/evaluate_prequential.py`

**Add import**:
```python
from experiments.shared.detection_metrics import calculate_detection_metrics
```

**Update `calculate_metrics()` function (around line 311)**:
```python
def calculate_metrics(results: Dict[str, Dict], drift_points: List[int]) -> Dict:
    """Calculate comparison metrics for all modes."""
    metrics = {}
    
    for mode, data in results.items():
        # ... existing accuracy calculations ...
        
        # NEW: Add detection metrics
        detections = data.get('detections', [])
        det_metrics = calculate_detection_metrics(detections, drift_points, tolerance=250)
        
        metrics[mode] = {
            'overall_accuracy': mean_acc,
            'post_drift_accuracy': post_drift_mean,
            'n_detections': len(detections),
            'n_adaptations': len(data['adaptations']),
            # NEW detection metrics
            'TP': det_metrics['TP'],
            'FP': det_metrics['FP'],
            'FN': det_metrics['FN'],
            'EDR': det_metrics['EDR'],
            'MDR': det_metrics['MDR'],
            'Precision': det_metrics['Precision'],
            'Mean_Delay': det_metrics['Mean_Delay'],
        }
```

**Update output (around line 446)**:
```python
print(f"{mode:20s}: Accuracy = {m.get('overall_accuracy', 0):.4f}, "
      f"EDR = {m.get('EDR', 0):.3f}, MDR = {m.get('MDR', 0):.3f}, "
      f"FP = {m.get('FP', 0)}, "
      f"Restoration = {s.get('avg_restoration_time', 0):.1f} samples")
```

### Task 4: Sync Threshold Configuration
**File**: `experiments/drift_monitoring_system/config.py`

Add threshold constants matching benchmark:
```python
# SE-CDT Detection Thresholds (synced with benchmark_proper.py)
SE_CDT_THRESHOLD = 0.15       # For SE_CDT.monitor() internal threshold
SHAPE_HEIGHT = 0.015          # For peak detection
SHAPE_PROMINENCE = 0.008      # For peak detection
DETECTION_TOLERANCE = 250     # Samples for TP matching
```

### Task 5: Re-run Benchmarks and Update Tables
1. Run `python experiments/benchmark_proper.py`
2. Verify `report/latex/tables/table_comparison_aggregate.tex` has reasonable FP count
3. Run `python experiments/drift_monitoring_system/evaluate_prequential.py --drift_type sudden`
4. Verify output shows both detection metrics AND adaptation metrics

## Expected Output After Fix

### benchmark_proper.py Output
```
Method          | MDR   | EDR (Recall) | Precision | FP
-----------------------------------------------------------------
CDT             | 0.942 | 0.058        | 0.060     | 91
SE_STD          | 0.150 | 0.850        | 0.750     | ~50   # Better balance
SE_ADW          | 0.446 | 0.554        | 0.650     | ~40
```

### evaluate_prequential.py Output
```
================================================================================
RESULTS - COMPREHENSIVE EVALUATION
================================================================================

Drift Type: SUDDEN
----------------------------------------
                     | Detection         | Adaptation
Mode                 | EDR   MDR   FP    | Accuracy  Restoration
-----------------------------------------------------------------
type_specific        | 0.800 0.200 3     | 85.46%    193 samples
simple_retrain       | 0.800 0.200 3     | 84.20%    210 samples
no_adaptation        | 0.800 0.200 3     | 81.26%    N/A

Improvement (Type-Specific vs No-Adaptation): +5.2%
================================================================================
```

## Validation Criteria

1. **FP count reasonable**: SE-CDT (Std) should have FP < 200 (not 1513)
2. **EDR balanced with Precision**: EDR ~0.80-0.90 with Precision > 0.70
3. **Both metrics reported**: `evaluate_prequential.py` shows detection AND adaptation
4. **Threshold consistency**: Same threshold values used in both scripts
5. **Tables updated**: `table_comparison_aggregate.tex` reflects tuned thresholds

## Files Modified

| File | Changes |
|------|---------|
| `experiments/shared/detection_metrics.py` | NEW - shared utility |
| `experiments/benchmark_proper.py` | Fix line 400, update thresholds |
| `experiments/drift_monitoring_system/evaluate_prequential.py` | Add detection metrics |
| `experiments/drift_monitoring_system/config.py` | Add threshold constants |
| `report/latex/tables/table_comparison_aggregate.tex` | Regenerated |
