"""
Benchmark logging utilities.

Provides structured, analyzable logging for the drift detection benchmark.
Logs are formatted consistently with timestamps and context for easy parsing.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from core.config import LOGS_DIR


class BenchmarkLogger:
    """
    Structured logger for drift detection benchmark.
    
    Features:
    - Consistent timestamp formatting
    - Progress tracking with ETA
    - Structured result logging (parseable)
    - Summary statistics at each level
    - Optional file output for later analysis
    """
    
    def __init__(self, output_dir=LOGS_DIR, log_to_file=True, verbose=True):
        """
        Initialize benchmark logger.
        
        Args:
            output_dir: Directory to save log file
            log_to_file: Whether to save logs to file
            verbose: Whether to print detailed per-method output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.log_to_file = log_to_file
        
        # Tracking
        self.start_time = None
        self.run_times = []
        self.method_results = {}  # {method: [f1_scores]}
        self.dataset_results = {}  # {dataset: [f1_scores]}
        
        # Setup file logging
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.output_dir / f"benchmark_{timestamp}.log"
            self.file_handler = open(self.log_file, 'w', encoding='utf-8')
        else:
            self.log_file = None
            self.file_handler = None
    
    def _log(self, message, level="INFO"):
        """Internal logging function."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level:5}] {message}"
        print(formatted, flush=True)
        if self.file_handler:
            self.file_handler.write(formatted + "\n")
            self.file_handler.flush()
    
    def header(self, title, width=80):
        """Print a section header."""
        line = "=" * width
        self._log(line)
        self._log(title.center(width))
        self._log(line)
    
    def subheader(self, title, width=80):
        """Print a subsection header."""
        self._log("-" * width)
        self._log(f"  {title}")
        self._log("-" * width)
    
    def config(self, n_runs, n_datasets, n_methods, datasets, methods):
        """Log benchmark configuration."""
        self.header("BENCHMARK CONFIGURATION")
        self._log(f"  Runs:     {n_runs}")
        self._log(f"  Datasets: {n_datasets}")
        self._log(f"  Methods:  {n_methods}")
        self._log(f"  Total experiments: {n_runs * n_datasets * n_methods}")
        self._log("")
        self._log(f"  Datasets: {', '.join(datasets)}")
        self._log(f"  Methods:  {', '.join(methods)}")
        self._log("=" * 80)
    
    def start_benchmark(self):
        """Mark benchmark start."""
        self.start_time = time.time()
        self._log("Benchmark started")
    
    def start_run(self, run_idx, total_runs, seed, experiments_so_far):
        """Log run start with progress."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Calculate ETA
        if run_idx > 1 and self.run_times:
            avg_run_time = sum(self.run_times) / len(self.run_times)
            remaining_runs = total_runs - run_idx + 1
            eta_seconds = avg_run_time * remaining_runs
            eta_str = f" | ETA: {eta_seconds/60:.1f}min"
        else:
            eta_str = ""
        
        self._log("")
        self.header(f"RUN {run_idx}/{total_runs} (seed={seed})")
        self._log(f"  Progress: {(run_idx-1)/total_runs*100:.1f}% | Experiments: {experiments_so_far} | Elapsed: {elapsed/60:.1f}min{eta_str}")
    
    def start_dataset(self, run_idx, total_runs, dataset_idx, total_datasets, dataset_name, n_features, n_drifts):
        """Log dataset start."""
        self.subheader(f"Dataset {dataset_idx}/{total_datasets}: {dataset_name}")
        self._log(f"  Features: {n_features} | Drift events: {n_drifts}")
    
    def log_method_start(self, method_name):
        """Log method evaluation start (minimal output)."""
        if self.verbose:
            self._log(f"    → {method_name}...", level="DEBUG")
    
    def log_method_result(self, method_name, dataset_name, result):
        """Log method result in structured format."""
        f1 = result.get('f1_score', 0)
        precision = result.get('precision', 0)
        recall = result.get('recall', 0)
        tp = result.get('tp', 0)
        fp = result.get('fp', 0)
        fn = result.get('fn', 0)
        runtime = result.get('runtime_s', 0)
        n_detect = len(result.get('detections', []))
        
        # Track for summary
        if method_name not in self.method_results:
            self.method_results[method_name] = []
        self.method_results[method_name].append(f1)
        
        if dataset_name not in self.dataset_results:
            self.dataset_results[dataset_name] = []
        self.dataset_results[dataset_name].append(f1)
        
        # Compact structured output
        status = "✓" if f1 > 0.5 else "○" if f1 > 0 else "✗"
        self._log(f"    {status} {method_name:20} | F1={f1:.3f} P={precision:.3f} R={recall:.3f} | TP={tp:2} FP={fp:2} FN={fn:2} | {runtime:.2f}s")
    
    def log_drift_detection(self, method_name, position, p_value=None):
        """Log individual drift detection (only in verbose mode)."""
        if self.verbose and p_value is not None:
            self._log(f"      ↳ Drift at {position} (p={p_value:.4f})", level="DEBUG")
    
    def end_dataset(self, dataset_name, n_methods, avg_f1):
        """Log dataset summary."""
        self._log(f"  Summary: {n_methods} methods evaluated | Avg F1: {avg_f1:.3f}")
    
    def end_run(self, run_idx, run_time, n_experiments):
        """Log run completion."""
        self.run_times.append(run_time)
        self._log(f"  Run {run_idx} completed in {run_time:.1f}s | Total experiments: {n_experiments}")
    
    def summary(self, total_experiments, expected_experiments, total_time):
        """Print final summary with statistics."""
        self._log("")
        self.header("BENCHMARK SUMMARY")
        
        # Validation
        if total_experiments == expected_experiments:
            self._log(f"  ✓ All {expected_experiments} experiments completed successfully")
        else:
            self._log(f"  ✗ WARNING: Expected {expected_experiments}, got {total_experiments}", level="WARN")
        
        self._log(f"  Total runtime: {total_time/60:.1f} minutes")
        
        # Method rankings
        self._log("")
        self._log("  METHOD RANKINGS (by mean F1):")
        method_means = {m: sum(scores)/len(scores) for m, scores in self.method_results.items() if scores}
        for rank, (method, mean_f1) in enumerate(sorted(method_means.items(), key=lambda x: -x[1]), 1):
            n = len(self.method_results[method])
            std_f1 = (sum((x - mean_f1)**2 for x in self.method_results[method]) / n) ** 0.5
            bar = "█" * int(mean_f1 * 20)
            self._log(f"    {rank}. {method:20} | F1={mean_f1:.3f}±{std_f1:.3f} | {bar}")
        
        # Dataset performance
        self._log("")
        self._log("  DATASET DIFFICULTY (by mean F1):")
        dataset_means = {d: sum(scores)/len(scores) for d, scores in self.dataset_results.items() if scores}
        for dataset, mean_f1 in sorted(dataset_means.items(), key=lambda x: -x[1]):
            difficulty = "Easy" if mean_f1 > 0.6 else "Medium" if mean_f1 > 0.3 else "Hard"
            self._log(f"    {dataset:25} | F1={mean_f1:.3f} | {difficulty}")
        
        self._log("=" * 80)
        
        # Log file location
        if self.log_file:
            self._log(f"  Log saved to: {self.log_file}")
    
    def close(self):
        """Close file handler."""
        if self.file_handler:
            self.file_handler.close()


# Global logger instance
_logger = None


def get_logger(output_dir=LOGS_DIR, log_to_file=True, verbose=True):
    """Get or create the global benchmark logger."""
    global _logger
    if _logger is None:
        _logger = BenchmarkLogger(output_dir, log_to_file, verbose)
    return _logger


def reset_logger():
    """Reset the global logger."""
    global _logger
    if _logger:
        _logger.close()
    _logger = None
