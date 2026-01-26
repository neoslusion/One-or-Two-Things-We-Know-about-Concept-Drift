#!/bin/bash
# =============================================================================
# UNIFIED BENCHMARK & THESIS PIPELINE
# =============================================================================
# This script unifies the workflow for running benchmarks, generating results,
# and building the final thesis report.
#
# Usage:
#   ./run_all.sh [options]
#
# Options:
#   --quick             Run benchmarks in quick mode (fewer iterations)
#   --skip-benchmark    Skip benchmark execution (only plot/build)
#   --help              Show this help message
#
# Workflow:
#   1. Setup Environment
#   2. Run Detection Benchmark (main.py benchmark)
#   3. Run Comparison Benchmark (main.py compare)
#   4. Run Monitoring Benchmark (main.py monitoring)
#   5. Generate Visualizations (main.py plot)
#   6. Build LaTeX Report
# =============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
QUICK_MODE=false
SKIP_BENCHMARK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            export QUICK_MODE=True
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --help)
            echo "Usage: ./run_all.sh [options]"
            echo "  --quick             Run in quick mode"
            echo "  --skip-benchmark    Skip benchmark runs"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            shift
            ;;
    esac
done

# Logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/run_all_${TIMESTAMP}.log"
mkdir -p results/logs results/plots results/tables results/raw

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# =============================================================================
# 1. Environment Setup
# =============================================================================
print_header "1. Environment Setup"

if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
print_success "Environment ready"

# =============================================================================
# 2. Run Benchmarks
# =============================================================================
if [ "$SKIP_BENCHMARK" = false ]; then
    
    # 2a. Detection Benchmark
    print_header "2a. Detection Benchmark (Window-based)"
    if [ "$QUICK_MODE" = true ]; then
        echo "Running in QUICK mode..."
        python3 main.py benchmark --quick 2>&1 | tee -a "$LOG_FILE"
    else
        echo "Running in FULL mode (this may take time)..."
        python3 main.py benchmark 2>&1 | tee -a "$LOG_FILE"
    fi
    print_success "Detection Benchmark Complete"

    # 2b. Comparison Benchmark
    print_header "2b. Comparison Benchmark (SE-CDT vs CDT_MSW)"
    # Note: benchmark_proper.py now respects QUICK_MODE env var
    python3 main.py compare 2>&1 | tee -a "$LOG_FILE"
    print_success "Comparison Benchmark Complete"

    # 2c. Monitoring Benchmark (Prequential)
    print_header "2c. Monitoring Benchmark (Prequential)"
    echo "Running adaptation evaluation for all drift types..."
    for dtype in sudden gradual incremental recurrent mixed; do
        echo "  → Scenario: $dtype"
        python3 main.py monitoring -- --drift_type $dtype 2>&1 | tee -a "$LOG_FILE"
    done
    print_success "Monitoring Benchmark Complete"

else
    echo "Skipping benchmarks..."
fi

# =============================================================================
# 3. Generate Visualizations
# =============================================================================
print_header "3. Generating Visualizations"
python3 main.py plot 2>&1 | tee -a "$LOG_FILE"
print_success "Visualizations generated in results/plots/"

# =============================================================================
# 4. Build LaTeX Report
# =============================================================================
print_header "4. Building LaTeX Report"

cd report/latex
# Clean build
rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf

echo "Compiling LaTeX..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

if [ -f "main.pdf" ]; then
    print_success "Report built: report/latex/main.pdf"
else
    print_error "LaTeX build failed. Check logs in report/latex/"
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Summary
# =============================================================================
print_header "Execution Summary"
echo "Log file: $LOG_FILE"
echo "Tables:   results/tables/"
echo "Plots:    results/plots/"
echo "Report:   report/latex/main.pdf"
echo ""
