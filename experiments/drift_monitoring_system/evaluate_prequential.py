"""
Prequential Accuracy Evaluation for Drift Adaptation (Enhanced Version).

System Architecture:
--------------------
This script implements the evaluation of the Closed-Loop Adaptive Learning System defined in the thesis.
It integrates the following modules:

1.  **Stream Interface:** Generates synthetic data streams (Sudden, Gradual, Recurrent, etc.).
2.  **Inference Module:** Uses an incremental Learner (SGD or Retrainable Batch Classifier) to make predictions.
3.  **SE-CDT System (Monitor):** A Unified Detector-Classifier that:
    *   Detects drift using ShapeDD-ADW (Adaptive Density-Weighted MMD).
    *   Classifies drift type (Sudden/Gradual/etc.) using Signal Shape Analysis (Algorithm 3.4).
4.  **Adaptation Manager:** Selects the optimal adaptation strategy based on the classified drift type:
    *   Sudden -> Reset/Retrain
    *   Recurrent -> Retrieve from Cache
    *   Gradual -> Partial Update (or Retrain in this implementation)
    *   Blip -> Ignore

Usage:
------
    python evaluate_prequential.py --n_samples 5000 --drift_type sudden --w_ref 50 --sudden_thresh 0.5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Callable, Optional, Dict, List

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
SHARED_DIR = REPO_ROOT / "experiments" / "shared"

if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPE_DD_DIR))
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import unified output configuration
from output_config import PREQUENTIAL_OUTPUTS, escape_latex

# Import SE_CDT detector (Unified Detector-Classifier)
try:
    from se_cdt import SE_CDT
except ImportError:
    print("[WARNING] SE_CDT not found, using dummy detector")
    SE_CDT = None

# Import adaptation strategies
from adaptation_strategies import (
    adapt_sudden_drift,
    adapt_incremental_drift,
    adapt_gradual_drift,
    adapt_recurrent_drift,
    adapt_blip_drift,
)

# Configuration
from config import (
    BUFFER_SIZE,
    CHUNK_SIZE,
    SHAPE_L1,
    SHAPE_L2,
    DRIFT_ALPHA,
    SHAPE_N_PERM,
    INITIAL_TRAINING_SIZE,
    PREQUENTIAL_WINDOW,
    ADAPTATION_WINDOW,
)

# Import shared detection metrics
try:
    from experiments.drift_detection_benchmark.evaluation.metrics import (
        calculate_detection_metrics_enhanced,
    )

    def calculate_detection_metrics(
        detections, ground_truth, tolerance=250, early_tolerance=50
    ):
        """Adapter for the enhanced metrics from benchmark."""
        stream_len = 10000

        metrics = calculate_detection_metrics_enhanced(
            detections,
            ground_truth,
            stream_length=stream_len,
            acceptable_delta=tolerance,
        )

        return {
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "EDR": metrics["recall"],
            "MDR": metrics["mdr"],
            "Precision": metrics["precision"],
            "Mean_Delay": metrics["mttd"],
            "delays": [],
        }

except ImportError:
    print("[WARNING] Shared metrics not found, using fallback")

    def calculate_detection_metrics(
        detections, ground_truth, tolerance=250, early_tolerance=50
    ):
        # Handle both old format (int) and new format (dict with 'idx')
        det_pos = []
        for d in detections if detections else []:
            if isinstance(d, int):
                det_pos.append(d)
            elif isinstance(d, dict):
                det_pos.append(d.get("idx", d.get("pos", 0)))
            else:
                det_pos.append(d)

        gt_pos = ground_truth if ground_truth else []
        tp, fp, delays = 0, 0, []
        detected = set()
        for d in det_pos:
            matched = False
            for i, g in enumerate(gt_pos):
                if i not in detected and g - early_tolerance <= d <= g + tolerance:
                    tp += 1
                    delays.append(d - g)
                    detected.add(i)
                    matched = True
                    break
            if not matched:
                fp += 1
        fn = len(gt_pos) - tp
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "EDR": tp / len(gt_pos) if gt_pos else 0,
            "MDR": fn / len(gt_pos) if gt_pos else 0,
            "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "Mean_Delay": float(np.mean(delays)) if delays else 0,
            "delays": delays,
        }


# Strategy mapping
STRATEGY_MAP = {
    "sudden": adapt_sudden_drift,
    "incremental": adapt_incremental_drift,
    "gradual": adapt_gradual_drift,
    "recurrent": adapt_recurrent_drift,
    "blip": adapt_blip_drift,
}


def create_model():
    """Create sklearn Pipeline for classification."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def model_factory():
    """Factory function for creating new models (used by adapt_sudden_drift)."""
    return create_model()


# Import SOTA Generators
from data_generators import (
    generate_sea_concepts,
    generate_mixed_drift_dataset,
    generate_rotating_hyperplane,
)


def generate_synthetic_stream(
    n_samples: int = 5000,
    n_drifts: int = 5,
    n_features: int = 5,
    drift_type: str = "sudden",
    random_seed: int = 42,
):
    """
    Wrapper for SOTA generators.
    """
    if drift_type == "mixed":
        return generate_mixed_drift_dataset(n_samples, random_seed)
    elif drift_type == "sudden":
        return generate_sea_concepts(n_samples, n_drifts, random_seed)
    elif drift_type == "gradual":
        # Use rotating hyperplane for gradual
        return generate_rotating_hyperplane(
            n_samples,
            n_drifts,
            n_features,
            drift_type="gradual",
            random_seed=random_seed,
        )
    elif drift_type == "incremental":
        return generate_rotating_hyperplane(
            n_samples,
            n_drifts,
            n_features,
            drift_type="incremental",
            random_seed=random_seed,
        )
    elif drift_type == "recurrent":
        # Use mixed for now as it has recurrent parts, or fallback
        # Ideally we'd have a specific recurrent generator
        return generate_mixed_drift_dataset(n_samples, random_seed)
    else:
        # Default fallback
        return generate_sea_concepts(n_samples, n_drifts, random_seed)


# ... (Configuration and imports preserved) ...


class AdaptationMode:
    """Enum-like class for adaptation modes."""

    NONE = "no_adaptation"
    SIMPLE = "simple_retrain"
    TYPE_SPECIFIC = "type_specific"


def evaluate_with_adaptation(
    X,
    y,
    drift_points,
    mode: str = AdaptationMode.TYPE_SPECIFIC,
    cache_dir: Optional[Path] = None,
    w_ref: int = 50,
    sudden_thresh: float = 0.5,
):
    """
    Evaluate adaptation strategies using SE-CDT Unified System.

    Modules:
    1. Stream Interface: Sliding window processing.
    2. Inference Module: Model prediction & accuracy tracking.
    3. SE-CDT System: Unified Detector-Classifier (Monitor).
    4. Adaptation Manager: Strategy selection and model update.
    """
    n_samples = len(X)

    # Initialize
    model = create_model()
    buffer = deque(maxlen=BUFFER_SIZE)
    recent_correct = deque(maxlen=PREQUENTIAL_WINDOW)

    accuracy_history = []
    drift_detections = []
    adaptations = []
    classification_times = []

    # Initialize SE-CDT Unified System
    # Note: window_size corresponds to l1 (reference window)
    se_cdt_system = (
        SE_CDT(window_size=w_ref, l2=SHAPE_L2, threshold=sudden_thresh)
        if SE_CDT
        else None
    )

    # Phase 1: Initial training
    train_end = min(INITIAL_TRAINING_SIZE, n_samples)
    model.fit(X[:train_end], y[:train_end])
    print(f"  [{mode}] Initial training on {train_end} samples")

    # Init buffer
    for i in range(train_end):
        buffer.append({"idx": i, "x": X[i], "y": y[i]})

    drift_detected = False
    samples_since_drift = 0
    adaptation_delay = 50
    current_drift_type = "sudden"  # Default

    # CRITICAL: For No Adaptation, we must NOT update this model object
    # We keep 'model' as is.

    for idx in range(train_end, n_samples):
        x = X[idx : idx + 1]
        y_true = y[idx]

        # Predict
        y_pred = model.predict(x)[0]
        is_correct = y_pred == y_true
        recent_correct.append(is_correct)

        accuracy = np.mean(recent_correct)
        accuracy_history.append({"idx": idx, "accuracy": accuracy})

        buffer.append({"idx": idx, "x": X[idx], "y": y_true})

        # Drift monitoring via SE-CDT System
        if len(buffer) >= BUFFER_SIZE and idx % CHUNK_SIZE == 0:
            if se_cdt_system is not None and not drift_detected:
                buffer_X = np.array([item["x"] for item in buffer])
                try:
                    # Call Unified Monitor
                    result = se_cdt_system.monitor(buffer_X)

                    if result.is_drift:
                        drift_detected = True
                        classification_times.append(result.classification_time)

                        # Use classified drift type immediately
                        detected_drift_type = result.subcategory
                        print(
                            f"    [{mode}] Drift detected at sample {idx} (Score={result.score:.4f}, Type={detected_drift_type})"
                        )

                        # Store BOTH detection position AND classified type
                        drift_detections.append(
                            {
                                "idx": idx,
                                "score": result.score,
                                "classified_type": detected_drift_type,
                            }
                        )

                        # Store result for adaptation step logic
                        current_drift_type = detected_drift_type

                except Exception as e:
                    print(f"Error in SE-CDT monitor: {e}")
                    pass

        # Adaptation Logic
        if drift_detected:
            samples_since_drift += 1
            if samples_since_drift >= adaptation_delay:
                # 1. NO ADAPTATION
                if mode == AdaptationMode.NONE:
                    # Do nothing to the model!
                    drift_detected = False
                    continue

                # Prepare data
                adapt_start = max(0, len(buffer) - ADAPTATION_WINDOW)
                adapt_data = list(buffer)[adapt_start:]
                adapt_X = np.array([item["x"] for item in adapt_data])
                adapt_y = np.array([item["y"] for item in adapt_data])

                # 2. SIMPLE RETRAIN
                if mode == AdaptationMode.SIMPLE:
                    # Naively retrain on recent window
                    # Force a new model instance to be sure
                    model = create_model()
                    model.fit(adapt_X, adapt_y)
                    strategy_used = "simple_retrain"

                # 3. TYPE SPECIFIC
                elif mode == AdaptationMode.TYPE_SPECIFIC:
                    # Use the type detected by SE-CDT
                    drift_type = current_drift_type
                    print(f"    [{mode}] Adapting for: {drift_type}")

                    if (
                        drift_type == "Sudden" or drift_type == "TCD"
                    ):  # Handle capitalization variations
                        model = adapt_sudden_drift(model_factory, adapt_X, adapt_y)
                        strategy_used = "sudden"
                    elif drift_type == "Recurrent":
                        # Try to load from cache
                        model = adapt_recurrent_drift(
                            model_factory, model, adapt_X, adapt_y, cache_dir=cache_dir
                        )
                        strategy_used = "recurrent"
                    elif drift_type == "Gradual":
                        # Use weighted approach or wait?
                        # For now, standard adapt
                        model = adapt_gradual_drift(model, adapt_X, adapt_y)
                        strategy_used = "gradual"
                    elif drift_type == "Blip":
                        # Blip: Do nothing or filter
                        print(f"    [{mode}] Blip detected - Ignoring adaptation")
                        strategy_used = "blip_ignored"
                        # We don't update model for Blip
                    else:
                        model = adapt_incremental_drift(model, adapt_X, adapt_y)
                        strategy_used = "incremental"

                adaptations.append(
                    {
                        "idx": idx,
                        "strategy": strategy_used
                        if mode == AdaptationMode.TYPE_SPECIFIC
                        else "simple",
                    }
                )
                drift_detected = False
                samples_since_drift = 0

    return accuracy_history, drift_detections, adaptations, classification_times


def plot_prequential_comparison(
    results: Dict[str, Dict],
    drift_points: List[int],
    output_path: Path,
    drift_types: List[str] = None,
    metrics: Dict = None,
    X: np.ndarray = None,
    y: np.ndarray = None,
):
    """
    Plot comprehensive prequential accuracy comparison with clear annotations.

    Features:
    - Data stream panel: Shows feature distribution and class labels over time
    - Main plot: Accuracy curves for all adaptation modes
    - Ground truth drift points with vertical bands and labels
    - Detection markers showing when SE-CDT detected drift
    - Adaptation markers showing strategy used
    - Summary statistics panel
    """
    # Determine if we have data stream to show
    has_data_stream = X is not None and y is not None

    # Create figure with 4 subplots if data stream available, else 3
    if has_data_stream:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 2.5, 0.8, 0.6], hspace=0.25)
        ax_stream = fig.add_subplot(gs[0])  # Data stream visualization
        ax_main = fig.add_subplot(gs[1], sharex=ax_stream)  # Main accuracy plot
        ax_det = fig.add_subplot(gs[2], sharex=ax_stream)  # Detection timeline
        ax_summary = fig.add_subplot(gs[3])  # Summary panel
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.8], hspace=0.25)
        ax_stream = None
        ax_main = fig.add_subplot(gs[0])  # Main accuracy plot
        ax_det = fig.add_subplot(gs[1], sharex=ax_main)  # Detection timeline
        ax_summary = fig.add_subplot(gs[2])  # Summary panel

    ax_summary.axis("off")

    # Color schemes
    mode_colors = {
        AdaptationMode.TYPE_SPECIFIC: "#2ecc71",  # Green
        AdaptationMode.SIMPLE: "#3498db",  # Blue
        AdaptationMode.NONE: "#e74c3c",  # Red
    }
    mode_labels = {
        AdaptationMode.TYPE_SPECIFIC: "Type-Specific Adaptation",
        AdaptationMode.SIMPLE: "Simple Retrain",
        AdaptationMode.NONE: "No Adaptation (Baseline)",
    }

    strategy_colors = {
        "sudden": "#e74c3c",  # Red
        "gradual": "#f39c12",  # Orange
        "incremental": "#9b59b6",  # Purple
        "recurrent": "#1abc9c",  # Teal
        "blip_ignored": "#95a5a6",  # Gray
        "simple_retrain": "#3498db",  # Blue
        "simple": "#3498db",  # Blue
    }
    strategy_markers = {
        "sudden": "s",  # Square
        "gradual": "D",  # Diamond
        "incremental": "^",  # Triangle up
        "recurrent": "o",  # Circle
        "blip_ignored": "x",  # X
        "simple_retrain": "v",  # Triangle down
        "simple": "v",
    }

    # =========================================================================
    # DATA STREAM PANEL: Shows feature distribution and labels over time
    # =========================================================================

    if ax_stream is not None and X is not None and y is not None:
        n_samples = len(X)
        sample_indices = np.arange(n_samples)

        # Calculate rolling statistics for visualization
        window = 100
        n_features = X.shape[1]

        # Use first principal component or feature mean for visualization
        if n_features > 1:
            # Simple: use mean of first 2 features as "signal"
            feature_signal = (X[:, 0] + X[:, 1]) / 2
        else:
            feature_signal = X[:, 0]

        # Rolling mean and std to show distribution changes
        from scipy.ndimage import uniform_filter1d

        rolling_mean = uniform_filter1d(feature_signal, size=window, mode="nearest")
        rolling_std = np.sqrt(
            uniform_filter1d(
                (feature_signal - rolling_mean) ** 2, size=window, mode="nearest"
            )
        )

        # Plot feature distribution band
        ax_stream.fill_between(
            sample_indices,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.3,
            color="#3498db",
            label="Feature Distribution (±1σ)",
        )
        ax_stream.plot(
            sample_indices,
            rolling_mean,
            color="#2980b9",
            linewidth=1.5,
            label="Feature Mean",
        )

        # Scatter plot for class labels (subsample for performance)
        subsample = max(1, n_samples // 500)  # Show at most 500 points
        class_0_mask = y[::subsample] == 0
        class_1_mask = y[::subsample] == 1
        scatter_indices = sample_indices[::subsample]

        # Create a y-position based on feature value for scatter
        scatter_y = feature_signal[::subsample]

        ax_stream.scatter(
            scatter_indices[class_0_mask],
            scatter_y[class_0_mask],
            c="#e74c3c",
            s=15,
            alpha=0.5,
            label="Class 0",
            marker="o",
        )
        ax_stream.scatter(
            scatter_indices[class_1_mask],
            scatter_y[class_1_mask],
            c="#27ae60",
            s=15,
            alpha=0.5,
            label="Class 1",
            marker="s",
        )

        # Mark drift points with vertical lines
        for i, dp in enumerate(drift_points):
            ax_stream.axvline(
                x=dp, color="#c0392b", linestyle="-", linewidth=2, alpha=0.8
            )
            # Add drift type label at top
            if drift_types and i < len(drift_types):
                dtype = drift_types[i]
                dtype_colors = {
                    "sudden": "#e74c3c",
                    "gradual": "#f39c12",
                    "incremental": "#9b59b6",
                    "recurrent": "#1abc9c",
                }
                ax_stream.annotate(
                    dtype.upper()[:3],
                    xy=(dp, 1.05),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color=dtype_colors.get(dtype, "#c0392b"),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                )

        ax_stream.set_ylabel("Feature\nValues", fontsize=10, fontweight="bold")
        ax_stream.set_title(
            "Data Stream: Feature Distribution and Class Labels Over Time",
            fontsize=12,
            fontweight="bold",
        )
        ax_stream.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)
        ax_stream.grid(True, alpha=0.3)
        plt.setp(ax_stream.get_xticklabels(), visible=False)

    # =========================================================================
    # MAIN PLOT: Accuracy Curves
    # =========================================================================

    # First, draw ground truth drift regions (shaded bands)
    for i, dp in enumerate(drift_points):
        # Shaded region around drift point (drift impact zone: ±200 samples)
        ax_main.axvspan(
            dp - 50,
            dp + 300,
            alpha=0.15,
            color="red",
            label="Drift Impact Zone" if i == 0 else "_nolegend_",
        )
        # Vertical line at exact drift point
        ax_main.axvline(
            x=dp,
            color="#c0392b",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label="True Drift Point" if i == 0 else "_nolegend_",
        )
        # Label the drift point
        drift_label = f"D{i + 1}"
        if drift_types and i < len(drift_types):
            drift_label = f"D{i + 1}\n({drift_types[i][:3]})"
        ax_main.annotate(
            drift_label,
            xy=(dp, 1.02),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#c0392b",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#c0392b",
                alpha=0.9,
            ),
        )

    # Plot accuracy curves
    for mode, data in results.items():
        idx_list = [r["idx"] for r in data["accuracy"]]
        acc_list = [r["accuracy"] for r in data["accuracy"]]

        ax_main.plot(
            idx_list,
            acc_list,
            color=mode_colors.get(mode, "gray"),
            linewidth=2 if mode == AdaptationMode.TYPE_SPECIFIC else 1.5,
            label=mode_labels.get(mode, mode),
            alpha=0.9 if mode == AdaptationMode.TYPE_SPECIFIC else 0.7,
            zorder=3 if mode == AdaptationMode.TYPE_SPECIFIC else 2,
        )

    # Mark adaptation events on the accuracy curve (Type-Specific only)
    ts_data = results.get(AdaptationMode.TYPE_SPECIFIC, {})
    if ts_data.get("adaptations"):
        acc_map = {r["idx"]: r["accuracy"] for r in ts_data.get("accuracy", [])}

        for adapt in ts_data["adaptations"]:
            adapt_idx = adapt["idx"]
            strategy = adapt.get("strategy", "unknown")
            acc_at_adapt = acc_map.get(adapt_idx, 0.85)

            # Plot marker at adaptation point
            marker = strategy_markers.get(strategy, "o")
            color = strategy_colors.get(strategy, "purple")
            ax_main.scatter(
                adapt_idx,
                acc_at_adapt,
                marker=marker,
                color=color,
                s=150,
                zorder=5,
                edgecolors="black",
                linewidth=1.5,
            )

    ax_main.set_ylabel("Prequential Accuracy", fontsize=12, fontweight="bold")
    ax_main.set_ylim([0.35, 1.08])
    ax_main.set_xlim([0, max(idx_list) if idx_list else 5000])
    ax_main.grid(True, alpha=0.3, linestyle="-")
    ax_main.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax_main.set_title(
        "Prequential Accuracy: Adaptive Learning with Drift Detection & Classification",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Hide x-axis labels for main plot (shared with detection timeline)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # =========================================================================
    # DETECTION TIMELINE: Shows when detections occurred
    # =========================================================================

    # Draw ground truth markers on timeline
    for i, dp in enumerate(drift_points):
        ax_det.axvline(x=dp, color="#c0392b", linestyle="-", linewidth=2, alpha=0.8)
        ax_det.scatter(
            dp,
            0,
            marker="v",
            color="#c0392b",
            s=200,
            zorder=5,
            label="Ground Truth" if i == 0 else "_nolegend_",
        )

    # Plot detections for each mode
    y_positions = {
        AdaptationMode.TYPE_SPECIFIC: 1.5,
        AdaptationMode.SIMPLE: 0,
        AdaptationMode.NONE: -1,
    }

    # Classification type colors and markers for detected drifts
    classify_colors = {
        "Sudden": "#e74c3c",  # Red
        "Gradual": "#f39c12",  # Orange
        "Incremental": "#9b59b6",  # Purple
        "Recurrent": "#1abc9c",  # Teal
        "TCD": "#3498db",  # Blue (generic)
        "PCD": "#27ae60",  # Green (generic)
    }
    classify_markers = {
        "Sudden": "s",  # Square
        "Gradual": "D",  # Diamond
        "Incremental": "^",  # Triangle up
        "Recurrent": "o",  # Circle
        "TCD": "P",  # Plus (filled)
        "PCD": "X",  # X (filled)
    }

    for mode, data in results.items():
        detections = data.get("detections", [])
        y_pos = y_positions.get(mode, 0)
        color = mode_colors.get(mode, "gray")

        if detections:
            # Handle both old format (int) and new format (dict)
            for i, det in enumerate(detections):
                if isinstance(det, dict):
                    det_idx = det.get("idx", 0)
                    classified_type = det.get("classified_type", "Unknown")
                else:
                    det_idx = det
                    classified_type = "Unknown"

                # For Type-Specific mode, show classification with colored markers
                if mode == AdaptationMode.TYPE_SPECIFIC:
                    marker = classify_markers.get(classified_type, "o")
                    m_color = classify_colors.get(classified_type, color)
                    ax_det.scatter(
                        det_idx,
                        y_pos,
                        marker=marker,
                        color=m_color,
                        s=200,
                        zorder=5,
                        edgecolors="black",
                        linewidth=1,
                        label=f"{classified_type}" if i == 0 else "_nolegend_",
                    )

                    # Add text label above marker
                    label_text = classified_type[:3].upper()
                    ax_det.annotate(
                        label_text,
                        xy=(det_idx, y_pos + 0.35),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        color=m_color,
                        bbox=dict(
                            boxstyle="round,pad=0.1",
                            facecolor="white",
                            edgecolor=m_color,
                            alpha=0.8,
                        ),
                    )

                    # Draw arrow to closest ground truth
                    if drift_points:
                        closest_gt = min(drift_points, key=lambda x: abs(x - det_idx))
                        delay = det_idx - closest_gt
                        if -100 < delay < 500:
                            arrow_color = (
                                "#27ae60"
                                if delay < 100
                                else "#f39c12"
                                if delay < 250
                                else "#e74c3c"
                            )
                            ax_det.annotate(
                                "",
                                xy=(closest_gt, 0),
                                xytext=(det_idx, y_pos),
                                arrowprops=dict(
                                    arrowstyle="->", color=arrow_color, alpha=0.4, lw=1
                                ),
                            )
                else:
                    # Simple marker for other modes
                    ax_det.scatter(
                        det_idx,
                        y_pos,
                        marker="|",
                        color=color,
                        s=200,
                        linewidth=2,
                        zorder=4,
                    )

    ax_det.set_ylim([-1.5, 2.5])
    ax_det.set_yticks([1.5, 0, -1])
    ax_det.set_yticklabels(
        ["Detected + Classified", "Ground Truth", "Simple/None"], fontsize=9
    )
    ax_det.set_xlabel("Sample Index", fontsize=12, fontweight="bold")
    ax_det.set_ylabel("Detection &\nClassification", fontsize=10, fontweight="bold")
    ax_det.grid(True, alpha=0.3, axis="x")
    ax_det.axhline(y=0, color="#c0392b", linestyle="--", alpha=0.5)

    # =========================================================================
    # SUMMARY PANEL: Key metrics and legend
    # =========================================================================

    # Create summary text
    summary_lines = []
    summary_lines.append("SUMMARY")
    summary_lines.append("-" * 60)
    summary_lines.append(f"Ground Truth: {len(drift_points)} drift points")

    # Add metrics if available
    if metrics:
        for mode in [
            AdaptationMode.TYPE_SPECIFIC,
            AdaptationMode.SIMPLE,
            AdaptationMode.NONE,
        ]:
            m = metrics.get(mode, {})
            mode_name = mode_labels.get(mode, mode)[:20]
            edr = m.get("EDR", 0)
            fp = m.get("FP", 0)
            acc = m.get("overall_accuracy", 0)
            summary_lines.append(
                f"{mode_name:<20}: Acc={acc:.1%}  EDR={edr:.0%}  FP={fp}"
            )

    summary_lines.append("-" * 60)

    # Classification legend
    classify_legend = "Classification Markers:  "
    for ctype, marker in [
        ("Sudden", "s"),
        ("Gradual", "D"),
        ("Incremental", "^"),
        ("Recurrent", "o"),
    ]:
        classify_legend += f" {marker}={ctype[:3]}"
    summary_lines.append(classify_legend)

    summary_text = "\n".join(summary_lines)

    ax_summary.text(
        0.02,
        0.5,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8f9fa",
            edgecolor="#dee2e6",
            alpha=0.95,
        ),
    )

    # Add color legend for classification and arrows
    color_legend = (
        "Classification Colors:\n"
        "  SUD=Red  GRA=Orange\n"
        "  INC=Purple  REC=Teal\n\n"
        "Arrow Delay:\n"
        "  Green=<100 samples\n"
        "  Orange=<250 samples\n"
        "  Red=>250 samples"
    )
    ax_summary.text(
        0.55,
        0.5,
        color_legend,
        transform=ax_summary.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#fff3cd",
            edgecolor="#ffc107",
            alpha=0.9,
        ),
    )

    # =========================================================================
    # Save figure
    # =========================================================================

    # Use subplots_adjust instead of tight_layout for better control with GridSpec
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def calculate_metrics(results: Dict[str, Dict], drift_points: List[int]) -> Dict:
    """
    Calculate comparison metrics for all modes.
    Now includes BOTH adaptation metrics AND detection metrics.
    """
    metrics = {}

    for mode, data in results.items():
        acc_list = [r["accuracy"] for r in data["accuracy"]]
        idx_list = [r["idx"] for r in data["accuracy"]]

        # Overall accuracy
        mean_acc = np.mean(acc_list)

        # Post-drift accuracy (500 samples after each drift)
        post_drift_acc = []
        for dp in drift_points:
            for i, idx in enumerate(idx_list):
                if dp <= idx < dp + 500:
                    post_drift_acc.append(acc_list[i])

        post_drift_mean = np.mean(post_drift_acc) if post_drift_acc else 0

        # Classification time (only for Type Specific)
        avg_class_time = 0
        if "classification_times" in data and len(data["classification_times"]) > 0:
            avg_class_time = (
                np.mean(data["classification_times"]) * 1000
            )  # Convert to ms

        # NEW: Calculate detection metrics (TP, FP, EDR, MDR, etc.)
        detections = data.get("detections", [])
        det_metrics = calculate_detection_metrics(
            detections, drift_points, tolerance=250
        )

        metrics[mode] = {
            # Adaptation metrics
            "overall_accuracy": mean_acc,
            "post_drift_accuracy": post_drift_mean,
            "n_detections": len(detections),
            "n_adaptations": len(data["adaptations"]),
            "avg_classification_time_ms": avg_class_time,
            # Detection metrics
            "TP": det_metrics["TP"],
            "FP": det_metrics["FP"],
            "FN": det_metrics["FN"],
            "EDR": det_metrics["EDR"],
            "MDR": det_metrics["MDR"],
            "Precision": det_metrics["Precision"],
            "Mean_Delay": det_metrics["Mean_Delay"],
        }

    # Calculate improvements
    if AdaptationMode.NONE in metrics and AdaptationMode.TYPE_SPECIFIC in metrics:
        baseline = metrics[AdaptationMode.NONE]["overall_accuracy"]
        type_spec = metrics[AdaptationMode.TYPE_SPECIFIC]["overall_accuracy"]
        simple = metrics.get(AdaptationMode.SIMPLE, {}).get("overall_accuracy", 0)

        metrics["improvement_vs_none"] = type_spec - baseline
        metrics["improvement_pct_vs_none"] = (
            (type_spec - baseline) / baseline * 100 if baseline > 0 else 0
        )
        metrics["improvement_vs_simple"] = type_spec - simple

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Prequential Accuracy Evaluation (Enhanced)"
    )
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument(
        "--n_drifts", type=int, default=5, help="Number of drift points"
    )
    parser.add_argument(
        "--drift_type",
        type=str,
        default="mixed",
        choices=["sudden", "gradual", "incremental", "recurrent", "mixed"],
        help='Type of drift to simulate (use "mixed" for realistic scenario with multiple drift types)',
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: use unified config path)",
    )
    # Added Arguments
    parser.add_argument(
        "--w_ref", type=int, default=50, help="Reference window size (l1)"
    )
    parser.add_argument(
        "--sudden_thresh",
        type=float,
        default=0.5,
        help="Threshold for sudden drift detection",
    )

    args = parser.parse_args()

    # Create output directory - use unified config if not specified
    if args.output_dir is None:
        output_dir = PREQUENTIAL_OUTPUTS["results_pkl"].parent
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cache directory for recurrent drift
    cache_dir = output_dir / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREQUENTIAL ACCURACY EVALUATION (Enhanced with 5 Strategies)")
    print("=" * 80)
    print(
        f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Type: {args.drift_type}, Seed: {args.seed}"
    )
    print(f"SE-CDT Config: w_ref={args.w_ref}, thresh={args.sudden_thresh}")
    print("=" * 80)

    # Generate data
    print("\n[1] Generating synthetic data stream...")
    X, y, drift_points, drift_types = generate_synthetic_stream(
        n_samples=args.n_samples,
        n_drifts=args.n_drifts,
        drift_type=args.drift_type,
        random_seed=args.seed,
    )

    results = {}

    # Evaluate WITH type-specific adaptation
    print("\n[2] Evaluating WITH type-specific adaptation (5 strategies)...")
    acc_ts, det_ts, adapt_ts, times_ts = evaluate_with_adaptation(
        X,
        y,
        drift_points,
        mode=AdaptationMode.TYPE_SPECIFIC,
        cache_dir=cache_dir,
        w_ref=args.w_ref,
        sudden_thresh=args.sudden_thresh,
    )
    results[AdaptationMode.TYPE_SPECIFIC] = {
        "accuracy": acc_ts,
        "detections": det_ts,
        "adaptations": adapt_ts,
        "classification_times": times_ts,
    }

    # Evaluate WITH simple adaptation
    print("\n[3] Evaluating WITH simple retrain adaptation...")
    acc_simple, det_simple, adapt_simple, _ = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.SIMPLE
    )
    results[AdaptationMode.SIMPLE] = {
        "accuracy": acc_simple,
        "detections": det_simple,
        "adaptations": adapt_simple,
    }

    # Evaluate WITHOUT adaptation
    print("\n[4] Evaluating WITHOUT adaptation (baseline)...")
    acc_none, det_none, adapt_none, _ = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.NONE
    )
    results[AdaptationMode.NONE] = {
        "accuracy": acc_none,
        "detections": det_none,
        "adaptations": adapt_none,
    }

    # Calculate metrics
    print("\n[5] Calculating metrics...")
    metrics = calculate_metrics(results, drift_points)

    # Calculate SOTA Adaptation Metrics (Recovery Speed, etc.)
    print(
        "\n[5.1] Calculating SOTA Adaptation Metrics (Recovery Speed, Performance Loss)..."
    )
    sota_metrics = {}
    for mode, data in results.items():
        sota_metrics[mode] = calculate_sota_adaptation_metrics(
            data["accuracy"],
            drift_points,
            n_samples=args.n_samples,
            adaptation_window=500,
        )

    print("\n" + "=" * 90)
    print("RESULTS - COMPREHENSIVE EVALUATION")
    print("=" * 90)
    print(f"\nDrift Type: {args.drift_type.upper()}")
    print(f"Ground Truth: {len(drift_points)} drift points")
    print("-" * 90)

    # Header
    print(f"{'Mode':<20} | {'Detection':<30} | {'Adaptation':<25}")
    print(
        f"{'':20} | {'EDR':>6} {'MDR':>6} {'Prec':>6} {'FP':>4} | {'Acc':>8} {'Restore':>10}"
    )
    print("-" * 90)

    for mode in [
        AdaptationMode.TYPE_SPECIFIC,
        AdaptationMode.SIMPLE,
        AdaptationMode.NONE,
    ]:
        m = metrics.get(mode, {})
        s = sota_metrics.get(mode, {})

        # Detection metrics
        edr = m.get("EDR", 0)
        mdr = m.get("MDR", 0)
        prec = m.get("Precision", 0)
        fp = m.get("FP", 0)

        # Adaptation metrics
        acc = m.get("overall_accuracy", 0)
        restore = s.get("avg_restoration_time", float("inf"))
        restore_str = f"{restore:.0f}" if restore < 1000 else "N/A"

        print(
            f"{mode:<20} | {edr:>6.3f} {mdr:>6.3f} {prec:>6.3f} {fp:>4} | {acc:>7.2%} {restore_str:>10}"
        )

    print("-" * 90)
    print(
        f"\nImprovement (Type-Specific vs No-Adaptation): "
        f"+{metrics.get('improvement_vs_none', 0):.4f} "
        f"(+{metrics.get('improvement_pct_vs_none', 0):.1f}%)"
    )
    print(
        f"Improvement (Type-Specific vs Simple):        "
        f"+{metrics.get('improvement_vs_simple', 0):.4f}"
    )
    print("=" * 90)

    # Generate plot
    print("\n[6] Generating plot...")
    # Use drift type to determine output filename from unified config
    drift_type_map = {
        "sudden": "sudden_accuracy",
        "gradual": "gradual_accuracy",
        "incremental": "incremental_accuracy",
        "recurrent": "recurrent_accuracy",
        "mixed": "mixed_accuracy",
    }
    plot_key = drift_type_map.get(args.drift_type, "mixed_accuracy")
    plot_path = PREQUENTIAL_OUTPUTS.get(plot_key, output_dir / f"prequential_accuracy_{args.drift_type}.pdf")
    
    plot_prequential_comparison(
        results,
        drift_points,
        plot_path,
        drift_types=drift_types,
        metrics=metrics,
        X=X,
        y=y,
    )

    # Save metrics to file
    metrics_path = output_dir / f"metrics_{args.drift_type}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Prequential Accuracy Evaluation Results (Comprehensive)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(
            f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Type: {args.drift_type}, Seed: {args.seed}\n\n"
        )
        f.write(f"Config: w_ref={args.w_ref}, thresh={args.sudden_thresh}\n")
        f.write(f"Ground Truth: {len(drift_points)} drift points at {drift_points}\n")

        for mode in [
            AdaptationMode.TYPE_SPECIFIC,
            AdaptationMode.SIMPLE,
            AdaptationMode.NONE,
        ]:
            m = metrics.get(mode, {})
            s = sota_metrics.get(mode, {})
            f.write(f"\n{mode}:\n")
            f.write(f"  --- Detection Metrics ---\n")
            f.write(f"  EDR (Recall):     {m.get('EDR', 0):.4f}\n")
            f.write(f"  MDR (Miss Rate):  {m.get('MDR', 0):.4f}\n")
            f.write(f"  Precision:        {m.get('Precision', 0):.4f}\n")
            f.write(
                f"  TP: {m.get('TP', 0)}, FP: {m.get('FP', 0)}, FN: {m.get('FN', 0)}\n"
            )
            f.write(f"  Mean Delay:       {m.get('Mean_Delay', 0):.1f} samples\n")
            f.write(f"  --- Adaptation Metrics ---\n")
            f.write(f"  Overall Accuracy: {m.get('overall_accuracy', 0):.4f}\n")
            f.write(f"  Post-Drift Acc:   {m.get('post_drift_accuracy', 0):.4f}\n")
            f.write(f"  Adaptations:      {m.get('n_adaptations', 0)}\n")
            if mode == AdaptationMode.TYPE_SPECIFIC:
                f.write(
                    f"  Avg Class. Time:  {m.get('avg_classification_time_ms', 0):.4f} ms\n"
                )
            f.write(
                f"  Restoration Time: {s.get('avg_restoration_time', float('inf')):.1f} samples\n"
            )
            f.write(f"  Performance Loss: {s.get('avg_performance_loss', 0):.4f}\n")
            f.write(f"  Convergence Rate: {s.get('convergence_rate', 0):.5f}\n")

    print(f"Metrics saved to: {metrics_path}")

    print("\n✓ Evaluation complete!")
    return metrics, sota_metrics


def calculate_sota_adaptation_metrics(
    accuracy_history, drift_points, n_samples, adaptation_window=500
):
    """
    Calculate SOTA adaptation metrics:
    1. Restoration Time: Samples to reach 95% of pre-drift accuracy (or 90% absolute).
    2. Performance Loss: Average accuracy drop during drift period.
    3. Convergence Rate: Slope of recovery.
    """
    metrics = {
        "avg_restoration_time": 0.0,
        "avg_performance_loss": 0.0,
        "convergence_rate": 0.0,
    }

    if not drift_points:
        return metrics

    restoration_times = []
    perf_losses = []
    slopes = []

    # Convert history to easier format
    acc_map = {item["idx"]: item["accuracy"] for item in accuracy_history}

    for drift_idx in drift_points:
        # Pre-drift accuracy (baseline) - avg of 100 samples before
        pre_drift = [acc_map.get(i, 0) for i in range(drift_idx - 100, drift_idx)]
        baseline = np.mean(pre_drift) if pre_drift else 0.8  # Default fallback
        target_acc = max(0.85, baseline * 0.95)  # Target to be considered "restored"

        # Analyze post-drift window
        window_end = min(n_samples, drift_idx + adaptation_window)
        post_drift_accs = [acc_map.get(i, 0) for i in range(drift_idx, window_end)]

        # 1. Restoration Time
        restored_at = float("inf")
        for i, acc in enumerate(post_drift_accs):
            # Check if stabilized above target for at least 30 samples
            if acc >= target_acc:
                # Look ahead 30 samples to ensure stability
                future_accs = post_drift_accs[i : i + 30]
                if len(future_accs) > 0 and np.mean(future_accs) >= target_acc:
                    restored_at = i
                    break

        restoration_times.append(
            restored_at if restored_at != float("inf") else adaptation_window
        )

        # 2. Performance Loss (Integral of error)
        # Loss = Sum(Baseline - Current) for samples where Current < Baseline
        loss = (
            np.sum([max(0, baseline - acc) for acc in post_drift_accs])
            / len(post_drift_accs)
            if post_drift_accs
            else 0
        )
        perf_losses.append(loss)

        # 3. Convergence Rate (Slope of linear fit to first 100 samples)
        if len(post_drift_accs) >= 10:
            fit_window = min(100, len(post_drift_accs))
            y_vals = post_drift_accs[:fit_window]
            x_vals = np.arange(len(y_vals))
            slope, _ = np.polyfit(x_vals, y_vals, 1)
            slopes.append(slope)

    metrics["avg_restoration_time"] = np.mean(restoration_times)
    metrics["avg_performance_loss"] = np.mean(perf_losses)
    metrics["convergence_rate"] = np.mean(slopes) if slopes else 0

    return metrics


if __name__ == "__main__":
    main()
