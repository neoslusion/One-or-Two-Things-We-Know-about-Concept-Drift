#!/usr/bin/env python3
"""
Test script to demonstrate the simplified plotting functionality without Kafka.
This shows how the data and detection points would be visualized.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the experiments/backup directory to import gen_random
REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DATA_DIR = REPO_ROOT / "experiments" / "backup"
if str(GEN_DATA_DIR) not in sys.path:
    sys.path.append(str(GEN_DATA_DIR))

from gen_data import gen_random

def simulate_detections(true_drift_positions, detection_accuracy=0.7, false_positive_rate=0.1):
    """Simulate drift detections with some accuracy and false positives."""
    detections = []
    
    # Add true positives (with some noise in position)
    for pos in true_drift_positions:
        if np.random.random() < detection_accuracy:
            # Add some noise to detection position
            noise = np.random.randint(-50, 51)
            det_pos = max(0, pos + noise)
            detections.append(det_pos)
    
    # Add some false positives
    length = 10000
    n_false_positives = int(length * false_positive_rate / 1000)
    for _ in range(n_false_positives):
        false_pos = np.random.randint(0, length)
        detections.append(false_pos)
    
    return sorted(detections)

def main():
    print("Generating synthetic data with concept drift...")
    
    # Generate data similar to what producer.py would create
    X, y = gen_random(number=3, dims=2, intens=0.3, dist="unif", alt=True, length=10000)
    
    # Find true drift positions
    true_drift_positions = np.where(np.diff(y) != 0)[0] + 1
    print(f"True drift positions: {true_drift_positions}")
    
    # Simulate some detections
    detected_positions = simulate_detections(true_drift_positions)
    print(f"Simulated detections: {detected_positions}")
    
    # Create the simplified 2-plot visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Data with true drift points (from gen_random)
    scatter1 = ax1.scatter(range(len(X)), X[:, 0], c=y, 
                          cmap='coolwarm', alpha=0.7, s=15)
    ax1.set_title('Data Stream with True Drift Points (from gen_random)')
    ax1.set_ylabel('Feature Value (First Dimension)')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for true drift points
    for pos in true_drift_positions:
        ax1.axvline(x=pos, color='blue', linestyle='-', alpha=0.5, linewidth=1)
        ax1.text(pos, ax1.get_ylim()[1]*0.9, f'True\n{pos}', 
                rotation=90, ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))
    
    # Add colorbar for drift indicator
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('True Drift Level', rotation=270, labelpad=15)
    
    # Plot 2: Detection points (aligned with data)
    detection_signal = np.zeros(len(X))
    if len(detected_positions) > 0:
        for d_pos in detected_positions:
            if d_pos < len(detection_signal):
                detection_signal[d_pos] = 1
        
        # Plot detection signal
        scatter2 = ax2.scatter(range(len(X)), detection_signal, c=detection_signal, 
                              cmap='Reds', alpha=0.8, s=50)
        ax2.set_title(f'ShapeDD Detection Points ({len(detected_positions)} detections)')
        
        # Add vertical lines for detections
        for d_pos in detected_positions:
            if d_pos < len(X):
                ax2.axvline(x=d_pos, color='red', linestyle='--', alpha=0.7)
                ax2.text(d_pos, 0.8, f'Det\n{d_pos}', 
                        rotation=90, ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
    else:
        ax2.scatter(range(len(X)), detection_signal, c='gray', alpha=0.3, s=15)
        ax2.set_title('ShapeDD Detection Points (No detections)')
    
    ax2.set_ylabel('Detection (0=No, 1=Yes)')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Print summary
    true_drift_points = len(true_drift_positions)
    print(f"\nSummary:")
    print(f"  Total samples: {len(X):,}")
    print(f"  True drift points: {true_drift_points}")
    print(f"  True drift positions: {list(true_drift_positions)}")
    print(f"  Detected drift points: {len(detected_positions)}")
    print(f"  Detection positions: {detected_positions}")
    
    # Calculate basic performance metrics
    if len(detected_positions) > 0 and len(true_drift_positions) > 0:
        # Simple performance check - how many detections are near true drifts
        tolerance = 100  # samples
        true_positives = 0
        for true_pos in true_drift_positions:
            for det_pos in detected_positions:
                if abs(det_pos - true_pos) <= tolerance:
                    true_positives += 1
                    break
        
        precision = true_positives / len(detected_positions) if len(detected_positions) > 0 else 0
        recall = true_positives / len(true_drift_positions) if len(true_drift_positions) > 0 else 0
        
        print(f"  Performance (tolerance={tolerance}):")
        print(f"    True positives: {true_positives}")
        print(f"    Precision: {precision:.2f}")
        print(f"    Recall: {recall:.2f}")
    
    plt.tight_layout()
    
    # Save plot instead of showing (for testing)
    plt.savefig('drift_detection_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'drift_detection_demo.png'")
    print("This demonstrates the simplified visualization system.")

if __name__ == "__main__":
    main()