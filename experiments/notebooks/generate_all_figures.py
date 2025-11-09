"""
Master Script: Generate ALL Thesis Figures (English)

This is the COMPLETE script that generates all 7 visualizations for the thesis.
Run this from within the notebook (Cell 11) OR standalone.

Usage:
    From notebook: %run generate_all_figures.py
    Standalone:    python3 generate_all_figures.py

Output: 7 PNG files in ../../report/latex/image/
"""

import os
import sys

print("="*80)
print("GENERATING ALL THESIS FIGURES (ENGLISH LABELS)")
print("="*80)
print()

# Check if we're in the right directory
if not os.path.exists('create_all_visualizations_v2.py'):
    print("ERROR: Please run this script from experiments/notebooks/ directory")
    sys.exit(1)

# Check if output directory exists
output_dir = '../../report/latex/image/'
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

print("Step 1/2: Generating performance visualizations (6 figures)...")
print("-" * 80)
try:
    exec(open('create_all_visualizations_v2.py').read())
    print("✓ Performance visualizations completed (6 figures)")
except Exception as e:
    print(f"✗ Error generating performance visualizations: {e}")
    import traceback
    traceback.print_exc()

print()
print("Step 2/2: Generating architecture diagram (English)...")
print("-" * 80)
try:
    exec(open('create_architecture_diagram_english.py').read())
    print("✓ Architecture diagram completed (English version)")
except Exception as e:
    print(f"✗ Error generating architecture diagram: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("SUMMARY: Generated Figures (ALL IN ENGLISH)")
print("="*80)

# List all generated files
expected_files = [
    'strategy_selection.png',
    'f1_comparison.png',
    'optimization_comparison.png',
    'method_ranking.png',
    'threshold_sensitivity.png',
    'buffer_dilution.png',
    'snr_adaptive_architecture.png'
]

print()
print(f"Checking generated files in: {output_dir}")
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
print("NEXT STEPS:")
print("="*80)
print()
print("1. Review generated figures in: report/latex/image/")
print("2. All labels are in ENGLISH (international readability)")
print("3. Insert figures into thesis chapters (see LATEX_FIGURE_INSERTIONS.md)")
print("4. Compile thesis:")
print("   cd ../../report/latex")
print("   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")
print("5. Verify all figures appear correctly in the PDF")
print()
