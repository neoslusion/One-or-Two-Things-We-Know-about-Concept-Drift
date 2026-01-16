"""  
Master Script: Generate ALL Thesis Figures

This script generates all visualizations for the thesis:
1. Architecture diagram
2. Related visualization figures

Usage:
    cd experiments/drift_detection_benchmark
    python3 visualizations/generate_all_figures.py

Output:
    - report/latex/image/ - Architecture and related diagrams

Note: Main benchmark figures (heatmaps, timelines, CD diagram) are generated
      automatically when running main.py via analysis/visualization.py
"""
import os
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)

print("="*80)
print("GENERATING ALL THESIS FIGURES")
print("="*80)
print()

# Check if we're in the right directory
if not os.path.exists(os.path.join(BENCHMARK_DIR, 'main.py')):
    print("ERROR: Please run this script from experiments/drift_detection_benchmark/ directory")
    sys.exit(1)

os.chdir(SCRIPT_DIR)

print("Step 1/1: Generating architecture diagram...")
print("-" * 80)
try:
    exec(open('create_architecture_diagram.py').read())
    print("✓ Architecture diagram completed")
except Exception as e:
    print(f"✗ Error generating architecture diagram: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("SUMMARY")
print("="*80)

# Check generated files
output_dir = os.path.join(SCRIPT_DIR, '../../../report/latex/image/')
expected_files = [
    'drift_detection_architecture.png',
]

print()
print(f"Checking files in: {output_dir}")
print()

for i, filename in enumerate(expected_files, 1):
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {i}. ✓ {filename:40s} ({size_kb:6.1f} KB)")
    else:
        print(f"  {i}. ✗ {filename:40s} (MISSING)")

print()
print("="*80)
print("NOTE: Main benchmark figures (heatmaps, timelines, CD diagram)")
print("      are generated automatically when running main.py")
print("      via analysis/visualization.py")
print("      Output location: publication_figures/")
print("="*80)
print()
