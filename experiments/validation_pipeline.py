#!/usr/bin/env python3
"""
Comprehensive Validation Pipeline for Concept Drift Detection Methods

This pipeline validates different types of drift detectors based on their algorithmic approaches:
1. Streaming-optimized methods (ADWIN, DDM, EDDM, etc.)
2. Window-based methods with incremental processing (D3)
3. Window-based methods requiring batch processing (ShapeDD, DAWIDD)
"""

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
from visualization_utils import DriftVisualization

# Add methods to path
sys.path.insert(0, os.path.abspath('../methods'))

# Import drift detection methods
from river.drift import ADWIN
from river.drift.binary import DDM, EDDM, FHDDM, HDDM_A, HDDM_W
from river.datasets import synth

# Import custom methods
from methods.dawidd import DAWIDD
from methods.shape_dd import ShapeDD
from methods.new_d3 import D3

warnings.filterwarnings('ignore')


@dataclass
class DriftPoint:
    """Information about a known drift point in the data stream."""
    position: int
    description: str
    severity: str = "moderate"  # "mild", "moderate", "severe"


@dataclass
class DetectionResult:
    """Result of drift detection on a data stream."""
    detector_name: str
    detections: List[int]
    processing_times: List[float]
    total_time: float
    stream_length: int
    true_drift_points: List[int]


class DataStreamGenerator:
    """Generator for various types of synthetic data streams with known drift points."""
    
    @staticmethod
    def generate_univariate_stream(length: int = 2000, drift_points: List[int] = None) -> Tuple[List[float], List[DriftPoint]]:
        """Generate univariate data stream with concept drift."""
        if drift_points is None:
            drift_points = [length // 2]
        
        stream = []
        drift_info = []
        current_mean = 0.0
        current_std = 1.0
        
        for i in range(length):
            # Check for drift points
            if i in drift_points:
                # Shift distribution parameters
                current_mean += random.uniform(2.0, 4.0)
                current_std *= random.uniform(0.5, 1.5)
                drift_info.append(DriftPoint(i, f"Mean shift to {current_mean:.2f}, std to {current_std:.2f}"))
            
            # Generate data point
            value = np.random.normal(current_mean, current_std)
            stream.append(value)
        
        return stream, drift_info
    
    @staticmethod
    def generate_multivariate_stream(length: int = 2000, n_features: int = 5, 
                                   drift_points: List[int] = None) -> Tuple[List[Dict], List[DriftPoint]]:
        """Generate multivariate data stream with concept drift."""
        if drift_points is None:
            drift_points = [length // 3, 2 * length // 3]
        
        stream = []
        drift_info = []
        current_means = np.zeros(n_features)
        current_cov = np.eye(n_features)
        
        for i in range(length):
            # Check for drift points
            if i in drift_points:
                # Shift distribution parameters
                current_means += np.random.uniform(-3, 3, n_features)
                # Add some correlation changes
                rotation = np.random.uniform(-0.5, 0.5, (n_features, n_features))
                current_cov = current_cov @ (np.eye(n_features) + rotation)
                drift_info.append(DriftPoint(i, f"Multivariate distribution shift"))
            
            # Generate data point
            values = np.random.multivariate_normal(current_means, current_cov)
            sample = {f"feature_{j}": values[j] for j in range(n_features)}
            stream.append(sample)
        
        return stream, drift_info
    
    @staticmethod
    def generate_binary_error_stream(length: int = 2000, drift_points: List[int] = None) -> Tuple[List[bool], List[DriftPoint]]:
        """Generate binary error stream for binary drift detectors."""
        if drift_points is None:
            drift_points = [length // 2]
        
        stream = []
        drift_info = []
        current_error_rate = 0.1
        
        for i in range(length):
            # Check for drift points
            if i in drift_points:
                current_error_rate = random.uniform(0.3, 0.7)
                drift_info.append(DriftPoint(i, f"Error rate change to {current_error_rate:.2f}"))
            
            # Generate error indicator
            error = np.random.random() < current_error_rate
            stream.append(error)
        
        return stream, drift_info


class DriftDetectorEvaluator:
    """Evaluator for different types of drift detection methods."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_streaming_detector(self, detector, stream: List[Union[float, bool]], 
                                  detector_name: str, true_drifts: List[int]) -> DetectionResult:
        """Evaluate streaming detectors (ADWIN, DDM, etc.) - process one point at a time."""
        detections = []
        processing_times = []
        
        start_time = time.time()
        
        for i, value in enumerate(stream):
            point_start = time.time()
            
            # Update detector
            detector.update(value)
            
            # Check for detection
            if detector.drift_detected:
                detections.append(i)
            
            point_time = time.time() - point_start
            processing_times.append(point_time)
        
        total_time = time.time() - start_time
        
        return DetectionResult(
            detector_name=detector_name,
            detections=detections,
            processing_times=processing_times,
            total_time=total_time,
            stream_length=len(stream),
            true_drift_points=true_drifts
        )
    
    def evaluate_window_detector(self, detector, stream: List[Dict], 
                                detector_name: str, true_drifts: List[int],
                                check_frequency: int = 10) -> DetectionResult:
        """Evaluate window-based detectors (ShapeDD, DAWIDD) - with controlled checking frequency."""
        detections = []
        processing_times = []
        
        start_time = time.time()
        
        for i, sample in enumerate(stream):
            point_start = time.time()
            
            # Update detector
            detector.update(sample)
            
            # Check for detection (only at specified intervals to reduce computational load)
            if i % check_frequency == 0 or detector.drift_detected:
                if detector.drift_detected:
                    detections.append(i)
            
            point_time = time.time() - point_start
            processing_times.append(point_time)
        
        total_time = time.time() - start_time
        
        return DetectionResult(
            detector_name=detector_name,
            detections=detections,
            processing_times=processing_times,
            total_time=total_time,
            stream_length=len(stream),
            true_drift_points=true_drifts
        )
    
    def evaluate_incremental_detector(self, detector, stream: List[Dict], 
                                    detector_name: str, true_drifts: List[int]) -> DetectionResult:
        """Evaluate incremental window detectors (D3) - designed for streaming."""
        detections = []
        processing_times = []
        
        start_time = time.time()
        
        for i, sample in enumerate(stream):
            point_start = time.time()
            
            # Update detector
            detector.update(sample)
            
            # Check for detection
            if detector.drift_detected:
                detections.append(i)
            
            point_time = time.time() - point_start
            processing_times.append(point_time)
        
        total_time = time.time() - start_time
        
        return DetectionResult(
            detector_name=detector_name,
            detections=detections,
            processing_times=processing_times,
            total_time=total_time,
            stream_length=len(stream),
            true_drift_points=true_drifts
        )


class ValidationMetrics:
    """Compute validation metrics for drift detection performance."""
    
    @staticmethod
    def compute_detection_metrics(result: DetectionResult, tolerance: int = 50) -> Dict[str, float]:
        """Compute detection performance metrics."""
        true_positives = 0
        false_positives = 0
        detected_drifts = set()
        
        # Count true positives and false positives
        for detection in result.detections:
            is_true_positive = False
            for true_drift in result.true_drift_points:
                if abs(detection - true_drift) <= tolerance:
                    if true_drift not in detected_drifts:
                        true_positives += 1
                        detected_drifts.add(true_drift)
                        is_true_positive = True
                        break
            
            if not is_true_positive:
                false_positives += 1
        
        # Count false negatives
        false_negatives = len(result.true_drift_points) - len(detected_drifts)
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute detection delay
        delays = []
        for true_drift in result.true_drift_points:
            min_delay = float('inf')
            for detection in result.detections:
                if detection >= true_drift:
                    delay = detection - true_drift
                    if delay <= tolerance:
                        min_delay = min(min_delay, delay)
            if min_delay != float('inf'):
                delays.append(min_delay)
        
        avg_delay = np.mean(delays) if delays else float('inf')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'avg_detection_delay': avg_delay,
            'avg_processing_time': np.mean(result.processing_times),
            'total_time': result.total_time
        }


class ValidationPipeline:
    """Main validation pipeline for drift detection methods."""
    
    def __init__(self):
        self.evaluator = DriftDetectorEvaluator()
        self.generator = DataStreamGenerator()
        self.metrics = ValidationMetrics()
        self.visualizer = DriftVisualization()
        self.results = []
        self.stream_data = None
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation across different detector types and data streams."""
        print("Starting Comprehensive Drift Detection Validation Pipeline")
        print("=" * 60)
        
        # Test 1: Univariate streaming detectors
        print("\n1. Testing Univariate Streaming Detectors")
        print("-" * 40)
        
        stream, drift_info = self.generator.generate_univariate_stream(length=2000, drift_points=[500, 1000, 1500])
        true_drifts = [d.position for d in drift_info]
        
        streaming_detectors = [
            (ADWIN(delta=0.002), "ADWIN"),
        ]
        
        for detector, name in streaming_detectors:
            print(f"Testing {name}...")
            result = self.evaluator.evaluate_streaming_detector(detector, stream, name, true_drifts)
            metrics = self.metrics.compute_detection_metrics(result)
            self.results.append((result, metrics))
            self._print_metrics(name, metrics, result)
        
        # Test 2: Binary error detectors
        print("\n2. Testing Binary Error Detectors")
        print("-" * 40)
        
        error_stream, error_drift_info = self.generator.generate_binary_error_stream(length=2000, drift_points=[600, 1200])
        error_true_drifts = [d.position for d in error_drift_info]
        
        binary_detectors = [
            (DDM(), "DDM"),
            (EDDM(), "EDDM"),
            (HDDM_A(), "HDDM_A"),
        ]
        
        for detector, name in binary_detectors:
            print(f"Testing {name}...")
            result = self.evaluator.evaluate_streaming_detector(detector, error_stream, name, error_true_drifts)
            metrics = self.metrics.compute_detection_metrics(result)
            self.results.append((result, metrics))
            self._print_metrics(name, metrics, result)
        
        # Test 3: Multivariate incremental detectors
        print("\n3. Testing Multivariate Incremental Detectors")
        print("-" * 40)
        
        mv_stream, mv_drift_info = self.generator.generate_multivariate_stream(length=1500, n_features=5, drift_points=[400, 800, 1200])
        mv_true_drifts = [d.position for d in mv_drift_info]
        self.stream_data = mv_stream  # Store for visualization
        
        incremental_detectors = [
            (D3(window_size=200, auc_threshold=0.7), "D3"),
        ]
        
        for detector, name in incremental_detectors:
            print(f"Testing {name}...")
            result = self.evaluator.evaluate_incremental_detector(detector, mv_stream, name, mv_true_drifts)
            metrics = self.metrics.compute_detection_metrics(result)
            self.results.append((result, metrics))
            self._print_metrics(name, metrics, result)
        
        # Test 4: Window-based batch detectors (with reduced frequency)
        print("\n4. Testing Window-based Batch Detectors")
        print("-" * 40)
        
        window_detectors = [
            (ShapeDD(window_size=150, l1=15, l2=20, n_perm=500, alpha=0.05), "ShapeDD"),
            (DAWIDD(window_size=150, n_perm=500, alpha=0.05), "DAWIDD"),
        ]
        
        for detector, name in window_detectors:
            print(f"Testing {name} (with check_frequency=20)...")
            result = self.evaluator.evaluate_window_detector(detector, mv_stream, name, mv_true_drifts, check_frequency=20)
            metrics = self.metrics.compute_detection_metrics(result, tolerance=100)  # More tolerance for batch methods
            self.results.append((result, metrics))
            self._print_metrics(name, metrics, result)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Generate visualizations
        if self.results:
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATIONS")
            print("=" * 60)
            self.generate_visualizations()
    
    def _print_metrics(self, detector_name: str, metrics: Dict[str, float], result: DetectionResult):
        """Print metrics for a detector."""
        print(f"  {detector_name} Results:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1-Score: {metrics['f1_score']:.3f}")
        print(f"    Avg Detection Delay: {metrics['avg_detection_delay']:.1f} points")
        print(f"    Avg Processing Time: {metrics['avg_processing_time']*1000:.3f} ms/point")
        print(f"    Total Detections: {len(result.detections)}")
        print(f"    True Drift Points: {result.true_drift_points}")
        print(f"    Detected Points: {result.detections}")
        print()
    
    def generate_summary_report(self):
        """Generate a summary report of all validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY REPORT")
        print("=" * 60)
        
        # Create comparison table
        comparison_data = []
        for result, metrics in self.results:
            comparison_data.append({
                'Detector': result.detector_name,
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'Avg Delay': f"{metrics['avg_detection_delay']:.1f}",
                'Avg Time (ms)': f"{metrics['avg_processing_time']*1000:.3f}",
                'Total Detections': len(result.detections)
            })
        
        df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(df.to_string(index=False))
        
        # Method categorization summary
        print("\n\nMethod Categories and Characteristics:")
        print("-" * 40)
        print("1. Streaming Optimized (One-point-at-a-time):")
        print("   - ADWIN: Adaptive windowing with statistical guarantees")
        print("   - DDM/EDDM/HDDM: Binary error rate monitoring")
        print("   - Best for: Real-time applications, low latency requirements")
        
        print("\n2. Incremental Window-based:")
        print("   - D3: Discriminative drift detection with incremental learning")
        print("   - Best for: Multivariate data, moderate computational resources")
        
        print("\n3. Batch Window-based (Reduced frequency checking):")
        print("   - ShapeDD: Shape analysis with MMD testing")
        print("   - DAWIDD: Kernel-based independence testing")
        print("   - Best for: High accuracy requirements, offline analysis")
        
        print("\n\nRecommendations:")
        print("-" * 20)
        print("- For real-time streaming: Use ADWIN or DDM family")
        print("- For multivariate data: Use D3 or ShapeDD")
        print("- For high accuracy: Use ShapeDD or DAWIDD with tuned parameters")
        print("- For computational efficiency: Use ADWIN or DDM")
    
    def generate_visualizations(self):
        """Generate and save visualization plots."""
        try:
            # Create performance comparison plot
            print("Creating performance comparison plot...")
            fig1 = self.visualizer.plot_performance_comparison(self.results)
            fig1.savefig('drift_detection_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Create method category comparison
            print("Creating method category comparison...")
            fig2 = self.visualizer.plot_method_category_comparison(self.results)
            fig2.savefig('drift_detection_category_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # Create detection timeline if we have multivariate data
            if self.stream_data:
                print("Creating detection timeline...")
                fig3 = self.visualizer.plot_detection_timeline(self.results, self.stream_data)
                fig3.savefig('drift_detection_timeline.png', dpi=300, bbox_inches='tight')
                plt.close(fig3)
            
            # Create comprehensive dashboard
            print("Creating comprehensive dashboard...")
            fig4 = self.visualizer.create_summary_dashboard(self.results, self.stream_data)
            fig4.savefig('drift_detection_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            print("Visualizations saved as PNG files in the current directory.")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            print("Continuing without visualizations...")


def main():
    """Run the validation pipeline."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run validation
    pipeline = ValidationPipeline()
    pipeline.run_comprehensive_validation()


if __name__ == "__main__":
    main()
