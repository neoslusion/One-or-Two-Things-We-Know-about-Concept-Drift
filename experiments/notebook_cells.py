#!/usr/bin/env python3
"""
Notebook cells for integrating the comprehensive validation pipeline 
into the existing ConceptDrift_Pipeline.ipynb
"""

# Cell 1: Updated imports and setup
cell_1_imports = '''
# Enhanced imports for comprehensive validation
import sys
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core scientific computing libraries
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import drift detection methods from methods directory
sys.path.insert(0, os.path.abspath('../methods'))

# Import River drift detectors
from river.drift import ADWIN
from river.drift.binary import DDM, EDDM, FHDDM, HDDM_A, HDDM_W
from river.datasets import synth

# Import custom methods (River-formatted)
from dawidd import DAWIDD
from shape_dd import ShapeDD
from new_d3 import D3

# Import validation pipeline
from validation_pipeline import ValidationPipeline, DriftDetectorEvaluator, DataStreamGenerator, ValidationMetrics
from visualization_utils import DriftVisualization

# Set style and seeds
plt.style.use('seaborn-v0_8')
np.random.seed(42)
random.seed(42)
warnings.filterwarnings('ignore')

print("✅ All modules imported successfully!")
print("✅ Comprehensive validation pipeline ready!")
'''

# Cell 2: Quick method comparison
cell_2_quick_test = '''
# Quick Test: Compare All Methods on Simple Data Stream
print("Quick Comparison of All Drift Detection Methods")
print("=" * 50)

# Create a simple data generator
generator = DataStreamGenerator()

# Generate test streams
univariate_stream, uni_drift_info = generator.generate_univariate_stream(length=1000, drift_points=[500])
multivariate_stream, mv_drift_info = generator.generate_multivariate_stream(length=800, n_features=3, drift_points=[300, 600])
binary_stream, bin_drift_info = generator.generate_binary_error_stream(length=1000, drift_points=[500])

print(f"Generated streams:")
print(f"- Univariate: {len(univariate_stream)} points, drift at {[d.position for d in uni_drift_info]}")
print(f"- Multivariate: {len(multivariate_stream)} points, drift at {[d.position for d in mv_drift_info]}")
print(f"- Binary: {len(binary_stream)} points, drift at {[d.position for d in bin_drift_info]}")
'''

# Cell 3: Individual method testing
cell_3_individual_tests = '''
# Test Each Method Category Separately

# 1. Streaming Methods (Univariate)
print("\\n1. STREAMING METHODS (Univariate Data)")
print("-" * 40)

adwin = ADWIN(delta=0.002)
detections_adwin = []
for i, val in enumerate(univariate_stream):
    adwin.update(val)
    if adwin.drift_detected:
        detections_adwin.append(i)
        print(f"ADWIN detected drift at position {i}")

print(f"ADWIN: {len(detections_adwin)} detections at {detections_adwin}")

# 2. Binary Error Methods
print("\\n2. BINARY ERROR METHODS")
print("-" * 30)

ddm = DDM()
detections_ddm = []
for i, error in enumerate(binary_stream):
    ddm.update(error)
    if ddm.drift_detected:
        detections_ddm.append(i)
        print(f"DDM detected drift at position {i}")

print(f"DDM: {len(detections_ddm)} detections at {detections_ddm}")

# 3. Multivariate Methods
print("\\n3. MULTIVARIATE METHODS")
print("-" * 30)

# D3 (Incremental)
d3 = D3(window_size=150, auc_threshold=0.7)
detections_d3 = []
for i, sample in enumerate(multivariate_stream):
    d3.update(sample)
    if d3.drift_detected:
        detections_d3.append(i)
        print(f"D3 detected drift at position {i}")

print(f"D3: {len(detections_d3)} detections at {detections_d3}")

# ShapeDD (Window-based)
shapedd = ShapeDD(window_size=120, l1=12, l2=18, n_perm=300, alpha=0.05)
detections_shapedd = []
for i, sample in enumerate(multivariate_stream):
    shapedd.update(sample)
    if i % 15 == 0 and shapedd.drift_detected:  # Check every 15 points
        detections_shapedd.append(i)
        print(f"ShapeDD detected drift at position {i}")

print(f"ShapeDD: {len(detections_shapedd)} detections at {detections_shapedd}")

# DAWIDD (Window-based)
dawidd = DAWIDD(window_size=120, n_perm=300, alpha=0.05)
detections_dawidd = []
for i, sample in enumerate(multivariate_stream):
    dawidd.update(sample)
    if i % 15 == 0 and dawidd.drift_detected:  # Check every 15 points
        detections_dawidd.append(i)
        print(f"DAWIDD detected drift at position {i}")

print(f"DAWIDD: {len(detections_dawidd)} detections at {detections_dawidd}")
'''

# Cell 4: Comprehensive validation
cell_4_comprehensive = '''
# Run Comprehensive Validation Pipeline
print("\\n" + "=" * 60)
print("COMPREHENSIVE VALIDATION PIPELINE")
print("=" * 60)

# Initialize and run the validation pipeline
pipeline = ValidationPipeline()
pipeline.run_comprehensive_validation()
'''

# Cell 5: Method recommendations
cell_5_recommendations = '''
# Method Selection Guide
print("\\n" + "=" * 60)
print("DRIFT DETECTION METHOD SELECTION GUIDE")
print("=" * 60)

guide_data = {
    'Method': ['ADWIN', 'DDM/EDDM', 'D3', 'ShapeDD', 'DAWIDD'],
    'Data Type': ['Univariate', 'Binary/Error', 'Multivariate', 'Multivariate', 'Multivariate'],
    'Processing': ['Streaming', 'Streaming', 'Incremental', 'Batch', 'Batch'],
    'Best For': [
        'Real-time, guaranteed performance',
        'Classification error monitoring',
        'Moderate computational resources',
        'High accuracy requirements',
        'Statistical rigor, offline analysis'
    ],
    'Computational Cost': ['Low', 'Very Low', 'Medium', 'High', 'Very High'],
    'Latency': ['Very Low', 'Very Low', 'Low', 'Medium', 'High']
}

df_guide = pd.DataFrame(guide_data)
print(df_guide.to_string(index=False))

print("\\n\\nRecommendations by Use Case:")
print("-" * 30)
print("🚀 Real-time applications: ADWIN, DDM")
print("📊 Multivariate data: D3, ShapeDD")
print("🎯 High accuracy needs: ShapeDD, DAWIDD")
print("⚡ Low latency requirements: ADWIN, DDM")
print("🔍 Research/offline analysis: DAWIDD, ShapeDD")
print("💰 Limited computational budget: ADWIN, DDM")

print("\\n\\nStreaming Characteristics:")
print("-" * 25)
print("✅ True Streaming (1-point-at-a-time): ADWIN, DDM, EDDM")
print("🔄 Incremental (optimized for streaming): D3")
print("⏰ Batch (check every N points): ShapeDD, DAWIDD")
'''

# Cell 6: Custom validation function
cell_6_custom_validation = '''
# Custom Validation Function for Your Data
def validate_drift_detectors_on_custom_data(data_stream, true_drift_points, 
                                          detector_configs=None, 
                                          tolerance=50, 
                                          verbose=True):
    """
    Validate drift detectors on custom data.
    
    Parameters:
    -----------
    data_stream : list
        Your data stream (univariate, multivariate, or binary)
    true_drift_points : list
        Known drift point positions
    detector_configs : dict
        Configuration for each detector type
    tolerance : int
        Tolerance for detection delay
    verbose : bool
        Print detailed results
    
    Returns:
    --------
    dict : Validation results
    """
    if detector_configs is None:
        detector_configs = {
            'adwin': {'delta': 0.002},
            'd3': {'window_size': min(200, len(data_stream)//4), 'auc_threshold': 0.7},
            'shapedd': {'window_size': min(150, len(data_stream)//5), 'l1': 15, 'l2': 20, 'n_perm': 500},
            'dawidd': {'window_size': min(150, len(data_stream)//5), 'n_perm': 500}
        }
    
    results = {}
    
    # Determine data type
    sample = data_stream[0]
    is_multivariate = isinstance(sample, dict) or (hasattr(sample, '__len__') and len(sample) > 1)
    is_binary = all(isinstance(x, bool) or x in [0, 1] for x in data_stream[:10])
    
    if verbose:
        print(f"Data characteristics:")
        print(f"  Length: {len(data_stream)}")
        print(f"  Type: {'Multivariate' if is_multivariate else 'Binary' if is_binary else 'Univariate'}")
        print(f"  True drift points: {true_drift_points}")
        print()
    
    # Test appropriate detectors based on data type
    if not is_multivariate:
        # Test ADWIN
        adwin = ADWIN(**detector_configs['adwin'])
        detections = []
        for i, val in enumerate(data_stream):
            adwin.update(val)
            if adwin.drift_detected:
                detections.append(i)
        results['ADWIN'] = {
            'detections': detections,
            'precision': _calculate_precision(detections, true_drift_points, tolerance),
            'recall': _calculate_recall(detections, true_drift_points, tolerance)
        }
    
    if is_multivariate:
        # Test D3
        d3 = D3(**detector_configs['d3'])
        detections = []
        for i, sample in enumerate(data_stream):
            d3.update(sample)
            if d3.drift_detected:
                detections.append(i)
        results['D3'] = {
            'detections': detections,
            'precision': _calculate_precision(detections, true_drift_points, tolerance),
            'recall': _calculate_recall(detections, true_drift_points, tolerance)
        }
        
        # Test ShapeDD (with reduced frequency)
        shapedd = ShapeDD(**detector_configs['shapedd'])
        detections = []
        check_freq = max(10, len(data_stream) // 100)
        for i, sample in enumerate(data_stream):
            shapedd.update(sample)
            if i % check_freq == 0 and shapedd.drift_detected:
                detections.append(i)
        results['ShapeDD'] = {
            'detections': detections,
            'precision': _calculate_precision(detections, true_drift_points, tolerance),
            'recall': _calculate_recall(detections, true_drift_points, tolerance)
        }
        
        # Test DAWIDD (with reduced frequency)
        dawidd = DAWIDD(**detector_configs['dawidd'])
        detections = []
        for i, sample in enumerate(data_stream):
            dawidd.update(sample)
            if i % check_freq == 0 and dawidd.drift_detected:
                detections.append(i)
        results['DAWIDD'] = {
            'detections': detections,
            'precision': _calculate_precision(detections, true_drift_points, tolerance),
            'recall': _calculate_recall(detections, true_drift_points, tolerance)
        }
    
    if verbose:
        print("Validation Results:")
        print("-" * 50)
        for method, metrics in results.items():
            f1 = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
            print(f"{method:10} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {f1:.3f} | Detections: {metrics['detections']}")
    
    return results

def _calculate_precision(detections, true_drifts, tolerance):
    if not detections:
        return 0.0
    true_positives = 0
    for detection in detections:
        if any(abs(detection - true_drift) <= tolerance for true_drift in true_drifts):
            true_positives += 1
    return true_positives / len(detections)

def _calculate_recall(detections, true_drifts, tolerance):
    if not true_drifts:
        return 1.0
    detected_drifts = set()
    for detection in detections:
        for true_drift in true_drifts:
            if abs(detection - true_drift) <= tolerance:
                detected_drifts.add(true_drift)
                break
    return len(detected_drifts) / len(true_drifts)

print("✅ Custom validation function ready!")
print("Usage: validate_drift_detectors_on_custom_data(your_data, your_drift_points)")
'''

# Cell 7: Example usage
cell_7_example = '''
# Example: Test on Your Own Data
# Uncomment and modify the following example:

# # Example 1: Custom univariate data
# my_univariate_data = [your_data_here]  # List of numbers
# my_drift_points = [300, 600]  # Known drift positions
# results = validate_drift_detectors_on_custom_data(my_univariate_data, my_drift_points)

# # Example 2: Custom multivariate data
# my_multivariate_data = [
#     {"feature1": 1.0, "feature2": 2.0, "feature3": 1.5},  # Format: list of dicts
#     # ... more samples
# ]
# my_mv_drift_points = [150, 300]
# results = validate_drift_detectors_on_custom_data(my_multivariate_data, my_mv_drift_points)

print("Replace the example data above with your actual data stream!")
print("The validation function will automatically detect data type and test appropriate methods.")
'''

def create_notebook_content():
    """Create the complete notebook content as a string."""
    content = f"""
# Cell 1: Enhanced Imports and Setup
{cell_1_imports}

# Cell 2: Quick Method Comparison
{cell_2_quick_test}

# Cell 3: Individual Method Testing
{cell_3_individual_tests}

# Cell 4: Comprehensive Validation
{cell_4_comprehensive}

# Cell 5: Method Selection Guide
{cell_5_recommendations}

# Cell 6: Custom Validation Function
{cell_6_custom_validation}

# Cell 7: Example Usage
{cell_7_example}
"""
    return content

if __name__ == "__main__":
    content = create_notebook_content()
    print("Notebook cells created successfully!")
    print("Copy and paste the content above into your Jupyter notebook.")
