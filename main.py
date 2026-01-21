#!/usr/bin/env python3
"""
Concept Drift Benchmark Dispatcher

Usage:
    python main.py benchmark [--quick]
    python main.py compare
    python main.py monitoring
    python main.py plot
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_script(path, args=None):
    """Runs a python script as a module."""
    cmd = [sys.executable, "-m", path.replace("/", ".").replace(".py", "")]
    if args:
        cmd.extend(args)
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"Error executing {path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Concept Drift Research Dispatcher")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run comprehensive drift detection benchmark")
    benchmark_parser.add_argument("--quick", action="store_true", help="Run in quick mode (fewer iterations)")

    # Compare command (SE-CDT vs CDT_MSW)
    compare_parser = subparsers.add_parser("compare", help="Run SE-CDT vs CDT_MSW comparison benchmark")

    # Monitoring command
    monitoring_parser = subparsers.add_parser("monitoring", help="Run Prequential Accuracy Evaluation (Kafka Monitoring simulation)")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate all publication figures")

    args = parser.parse_args()

    if args.command == "benchmark":
        # Pass quick mode via environment variable if needed, or handle in script
        if args.quick:
            os.environ["QUICK_MODE"] = "True"
        run_script("experiments/benchmark/main.py")
    
    elif args.command == "compare":
        run_script("experiments/benchmark/benchmark_proper.py")
        
    elif args.command == "monitoring":
        run_script("experiments/monitoring/evaluate_prequential.py")
        
    elif args.command == "plot":
        run_script("experiments/visualizations/plot_all_figures.py")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
