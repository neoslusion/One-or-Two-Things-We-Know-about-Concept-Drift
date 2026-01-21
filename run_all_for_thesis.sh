#!/bin/bash
# ============================================================================
# COMPLETE THESIS BENCHMARK RUNNER
# ============================================================================
# This script runs all necessary benchmarks and generates outputs for thesis.
#
# Usage:
#   chmod +x run_all_for_thesis.sh
#   ./run_all_for_thesis.sh
#
# Estimated time: ~30-60 minutes (depending on hardware)
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "           CONCEPT DRIFT THESIS - COMPLETE BENCHMARK RUNNER"
echo "============================================================================"
echo ""
echo "This will run:"
echo "  1. Detection Benchmark (ShapeDD vs MMD vs KS vs SE_CDT)"
echo "  2. SE-CDT vs CDT_MSW Comparison"
echo "  3. Prequential Accuracy Evaluation (Monitoring)"
echo "  4. Generate all publication figures"
echo ""
echo "Output directories:"
echo "  - results/tables/    : LaTeX tables"
echo "  - results/plots/     : Figures (PNG)"
echo "  - results/raw/       : Raw metrics (JSON/PKL)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[OK] Virtual environment activated"
else
    echo "[ERROR] .venv not found. Please run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# ============================================================================
# STEP 1: Detection Benchmark
# ============================================================================
echo ""
echo "============================================================================"
echo "[STEP 1/4] Running Detection Benchmark..."
echo "  Methods: MMD, KS, ShapeDD, ShapeDD_WMMD_PROPER, SE_CDT"
echo "  Datasets: 10 synthetic datasets"
echo "  Runs: 30 (for statistical validity)"
echo "============================================================================"
echo ""

python main.py benchmark

echo ""
echo "[STEP 1/4] ✓ Detection Benchmark Complete"
echo "  Output: results/tables/table_*.tex"
echo "          results/plots/critical_difference_*.png"
echo ""

# ============================================================================
# STEP 2: SE-CDT vs CDT_MSW Comparison
# ============================================================================
echo "============================================================================"
echo "[STEP 2/4] Running SE-CDT vs CDT_MSW Comparison..."
echo "  Comparing supervised (CDT_MSW) vs unsupervised (SE-CDT)"
echo "============================================================================"
echo ""

python main.py compare

echo ""
echo "[STEP 2/4] ✓ Comparison Benchmark Complete"
echo "  Output: results/tables/table_comparison_*.tex"
echo ""

# ============================================================================
# STEP 3: Prequential Evaluation (Monitoring)
# ============================================================================
echo "============================================================================"
echo "[STEP 3/4] Running Prequential Accuracy Evaluation..."
echo "  Testing adaptation strategies: Type-Specific, Simple Retrain, No Adaptation"
echo "============================================================================"
echo ""

python main.py monitoring

echo ""
echo "[STEP 3/4] ✓ Prequential Evaluation Complete"
echo "  Output: results/plots/fig_prequential_*.png"
echo "          results/raw/metrics_*.txt"
echo ""

# ============================================================================
# STEP 4: Generate All Figures
# ============================================================================
echo "============================================================================"
echo "[STEP 4/4] Generating Publication Figures..."
echo "============================================================================"
echo ""

python main.py plot

echo ""
echo "[STEP 4/4] ✓ Figure Generation Complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================================================"
echo "                        ALL BENCHMARKS COMPLETE!"
echo "============================================================================"
echo ""
echo "Generated outputs:"
echo ""
echo "TABLES (for LaTeX inclusion):"
ls -la results/tables/*.tex 2>/dev/null || echo "  (no tables found)"
echo ""
echo "PLOTS (for figures):"
ls -la results/plots/*.png 2>/dev/null || echo "  (no plots found)"
echo ""
echo "Next step: Build LaTeX document"
echo "  cd report/latex"
echo "  pdflatex main.tex"
echo "  bibtex main"
echo "  pdflatex main.tex"
echo "  pdflatex main.tex"
echo ""
echo "============================================================================"
