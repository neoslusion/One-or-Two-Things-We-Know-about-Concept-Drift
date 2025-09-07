#!/usr/bin/env python3
"""
Visualization utilities for drift detection validation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


class DriftVisualization:
    """Visualization utilities for drift detection results."""
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_detection_timeline(self, results: List[Tuple[Any, Dict]], 
                              stream_data: List[Any] = None, figsize: Tuple[int, int] = (15, 10)):
        """Plot detection timeline for multiple detectors."""
        fig, axes = plt.subplots(len(results), 1, figsize=figsize, sharex=True)
        if len(results) == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(results))
        
        for i, (result, metrics) in enumerate(results):
            ax = axes[i]
            
            # Plot stream data if available
            if stream_data and hasattr(stream_data[0], '__iter__') and not isinstance(stream_data[0], str):
                # For multivariate data, plot first feature
                if isinstance(stream_data[0], dict):
                    first_feature = list(stream_data[0].keys())[0]
                    y_data = [sample[first_feature] for sample in stream_data]
                else:
                    y_data = [sample[0] if hasattr(sample, '__iter__') else sample for sample in stream_data]
                ax.plot(range(len(y_data)), y_data, alpha=0.3, color='gray', linewidth=0.5)
            
            # Plot true drift points
            for true_drift in result.true_drift_points:
                ax.axvline(x=true_drift, color='red', linestyle='--', alpha=0.7, 
                          label='True Drift' if true_drift == result.true_drift_points[0] else "")
            
            # Plot detected drift points
            for detection in result.detections:
                ax.axvline(x=detection, color=colors[i], linestyle='-', alpha=0.8,
                          label='Detection' if detection == result.detections[0] else "")
            
            # Customize subplot
            ax.set_title(f'{result.detector_name} - F1: {metrics["f1_score"]:.3f}, '
                        f'Delay: {metrics["avg_detection_delay"]:.1f}')
            ax.set_ylabel('Signal')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.xlabel('Time (data points)')
        plt.title('Drift Detection Timeline Comparison', fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    
    def plot_performance_comparison(self, results: List[Tuple[Any, Dict]], 
                                  figsize: Tuple[int, int] = (12, 8)):
        """Plot performance comparison across detectors."""
        # Prepare data
        detector_names = [result[0].detector_name for result in results]
        metrics_data = {
            'Precision': [result[1]['precision'] for result in results],
            'Recall': [result[1]['recall'] for result in results],
            'F1-Score': [result[1]['f1_score'] for result in results]
        }
        
        # Create subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Precision, Recall, F1-Score comparison
        x = np.arange(len(detector_names))
        width = 0.25
        
        ax1.bar(x - width, metrics_data['Precision'], width, label='Precision', alpha=0.8)
        ax1.bar(x, metrics_data['Recall'], width, label='Recall', alpha=0.8)
        ax1.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Detectors')
        ax1.set_ylabel('Score')
        ax1.set_title('Detection Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(detector_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Detection delay comparison
        delays = [result[1]['avg_detection_delay'] for result in results]
        finite_delays = [d if d != float('inf') else 0 for d in delays]
        
        bars = ax2.bar(detector_names, finite_delays, alpha=0.8)
        ax2.set_xlabel('Detectors')
        ax2.set_ylabel('Average Delay (points)')
        ax2.set_title('Average Detection Delay')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, delay in zip(bars, delays):
            height = bar.get_height()
            label = 'inf' if delay == float('inf') else f'{delay:.1f}'
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom')
        
        # 3. Processing time comparison
        processing_times = [result[1]['avg_processing_time'] * 1000 for result in results]  # Convert to ms
        
        bars = ax3.bar(detector_names, processing_times, alpha=0.8, color='orange')
        ax3.set_xlabel('Detectors')
        ax3.set_ylabel('Processing Time (ms/point)')
        ax3.set_title('Average Processing Time per Data Point')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')  # Log scale for better visualization
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, processing_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Detection count comparison
        total_detections = [len(result[0].detections) for result in results]
        true_drift_count = len(results[0][0].true_drift_points) if results else 0
        
        bars = ax4.bar(detector_names, total_detections, alpha=0.8, color='green')
        ax4.axhline(y=true_drift_count, color='red', linestyle='--', alpha=0.7, 
                   label=f'True Drifts ({true_drift_count})')
        ax4.set_xlabel('Detectors')
        ax4.set_ylabel('Number of Detections')
        ax4.set_title('Total Number of Detections')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, total_detections):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_method_category_comparison(self, results: List[Tuple[Any, Dict]], 
                                      figsize: Tuple[int, int] = (12, 6)):
        """Plot comparison by method categories."""
        # Categorize methods
        streaming_methods = ['ADWIN', 'DDM', 'EDDM', 'HDDM_A', 'HDDM_W']
        incremental_methods = ['D3']
        batch_methods = ['ShapeDD', 'DAWIDD']
        
        categories = {
            'Streaming': [],
            'Incremental': [],
            'Batch': []
        }
        
        for result, metrics in results:
            name = result.detector_name
            if name in streaming_methods:
                categories['Streaming'].append((result, metrics))
            elif name in incremental_methods:
                categories['Incremental'].append((result, metrics))
            elif name in batch_methods:
                categories['Batch'].append((result, metrics))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. F1-Score by category
        category_f1_scores = {}
        for category, methods in categories.items():
            if methods:
                f1_scores = [metrics['f1_score'] for _, metrics in methods]
                category_f1_scores[category] = f1_scores
        
        # Box plot for F1-scores by category
        if category_f1_scores:
            data_for_box = []
            labels_for_box = []
            for category, scores in category_f1_scores.items():
                data_for_box.extend(scores)
                labels_for_box.extend([category] * len(scores))
            
            df_box = pd.DataFrame({'Category': labels_for_box, 'F1-Score': data_for_box})
            sns.boxplot(data=df_box, x='Category', y='F1-Score', ax=ax1)
            ax1.set_title('F1-Score Distribution by Method Category')
            ax1.grid(True, alpha=0.3)
        
        # 2. Processing time by category
        category_times = {}
        for category, methods in categories.items():
            if methods:
                times = [metrics['avg_processing_time'] * 1000 for _, metrics in methods]  # Convert to ms
                category_times[category] = times
        
        if category_times:
            data_for_box_time = []
            labels_for_box_time = []
            for category, times in category_times.items():
                data_for_box_time.extend(times)
                labels_for_box_time.extend([category] * len(times))
            
            df_box_time = pd.DataFrame({'Category': labels_for_box_time, 'Processing Time (ms)': data_for_box_time})
            sns.boxplot(data=df_box_time, x='Category', y='Processing Time (ms)', ax=ax2)
            ax2.set_title('Processing Time Distribution by Method Category')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, results: List[Tuple[Any, Dict]], 
                               stream_data: List[Any] = None, figsize: Tuple[int, int] = (20, 15)):
        """Create a comprehensive dashboard with all visualizations."""
        fig = plt.figure(figsize=figsize)
        
        # Create a complex grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)
        
        # 1. Detection timeline (top, full width)
        ax_timeline = fig.add_subplot(gs[0, :])
        
        # Plot timeline for top 4 performers based on F1-score
        sorted_results = sorted(results, key=lambda x: x[1]['f1_score'], reverse=True)[:4]
        colors = sns.color_palette("husl", len(sorted_results))
        
        if stream_data and hasattr(stream_data[0], '__iter__') and not isinstance(stream_data[0], str):
            if isinstance(stream_data[0], dict):
                first_feature = list(stream_data[0].keys())[0]
                y_data = [sample[first_feature] for sample in stream_data]
            else:
                y_data = [sample[0] if hasattr(sample, '__iter__') else sample for sample in stream_data]
            ax_timeline.plot(range(len(y_data)), y_data, alpha=0.2, color='gray', linewidth=0.5, label='Data Stream')
        
        # Plot true drift points
        true_drifts = sorted_results[0][0].true_drift_points if sorted_results else []
        for true_drift in true_drifts:
            ax_timeline.axvline(x=true_drift, color='red', linestyle='--', alpha=0.7, linewidth=2,
                              label='True Drift' if true_drift == true_drifts[0] else "")
        
        # Plot detections for each method
        for i, (result, metrics) in enumerate(sorted_results):
            for detection in result.detections:
                ax_timeline.axvline(x=detection, color=colors[i], alpha=0.6, linewidth=1,
                                  label=f'{result.detector_name}' if detection == result.detections[0] else "")
        
        ax_timeline.set_title('Top Performing Drift Detectors - Detection Timeline', fontsize=14)
        ax_timeline.set_xlabel('Time (data points)')
        ax_timeline.set_ylabel('Signal Value')
        ax_timeline.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_timeline.grid(True, alpha=0.3)
        
        # 2. Performance metrics (bottom left)
        ax_metrics = fig.add_subplot(gs[1, 0])
        
        detector_names = [result[0].detector_name for result in results]
        f1_scores = [result[1]['f1_score'] for result in results]
        
        bars = ax_metrics.bar(detector_names, f1_scores, alpha=0.7)
        ax_metrics.set_title('F1-Score Comparison')
        ax_metrics.set_ylabel('F1-Score')
        ax_metrics.tick_params(axis='x', rotation=45)
        ax_metrics.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Processing time (bottom right)
        ax_time = fig.add_subplot(gs[1, 1])
        
        processing_times = [result[1]['avg_processing_time'] * 1000 for result in results]
        bars = ax_time.bar(detector_names, processing_times, alpha=0.7, color='orange')
        ax_time.set_title('Processing Time Comparison')
        ax_time.set_ylabel('Time (ms/point)')
        ax_time.set_yscale('log')
        ax_time.tick_params(axis='x', rotation=45)
        ax_time.grid(True, alpha=0.3)
        
        # 4. Summary statistics table (bottom)
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        # Create summary table
        table_data = []
        for result, metrics in results:
            table_data.append([
                result.detector_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['avg_detection_delay']:.1f}" if metrics['avg_detection_delay'] != float('inf') else "∞",
                f"{metrics['avg_processing_time']*1000:.3f}",
                str(len(result.detections))
            ])
        
        table = ax_table.table(cellText=table_data,
                              colLabels=['Detector', 'Precision', 'Recall', 'F1-Score', 
                                       'Avg Delay', 'Time (ms)', 'Detections'],
                              cellLoc='center',
                              loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')
        
        plt.suptitle('Drift Detection Validation Dashboard', fontsize=16, y=0.95)
        return fig


def main():
    """Example usage of visualization utilities."""
    # This would typically be called from the validation pipeline
    print("Visualization utilities loaded successfully!")


if __name__ == "__main__":
    main()
