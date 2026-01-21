#!/usr/bin/env python3
"""  
Master Script: Generate ALL Thesis Figures

Consolidated and refactored to use standardized results/plots directory.
"""
import os
import sys
from pathlib import Path
import subprocess

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PLOTS_DIR

print("="*80)
print("GENERATING ALL THESIS FIGURES")
print("="*80)
print()

def run_viz(script_name):
    print(f"Executing: {script_name}")
    try:
        subprocess.check_call([sys.executable, script_name], cwd=str(Path(__file__).parent))
        print(f"✓ {script_name} completed")
    except Exception as e:
        print(f"✗ Error generating {script_name}: {e}")

# Generate all figures using the renamed scripts
viz_scripts = [
    'plot_architecture.py',
    'plot_strategy_selection.py',
    'plot_benchmark.py'
]

for script in viz_scripts:
    if os.path.exists(os.path.join(os.path.dirname(__file__), script)):
        run_viz(script)
    else:
        print(f"Skipping {script} (not found)")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Check output in: {PLOTS_DIR}")
print("="*80)
print()
