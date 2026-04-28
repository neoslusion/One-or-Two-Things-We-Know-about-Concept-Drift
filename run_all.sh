#!/bin/bash
# =============================================================================
# FINAL REPRODUCIBILITY PIPELINE (N=30)
# =============================================================================
# Runs the thesis experiments and rebuilds all generated thesis artifacts using
# one consistent final configuration:
#
#   - H0 calibration: 200 stationary windows per cell (script default)
#   - Detection benchmark: 30 independent runs
#   - SE-CDT/CDT-MSW comparison: 30 seeds per scenario
#   - Adaptation benchmark: 30 runs on stepping, sudden, mixed
#   - Figures/tables regenerated after the experiments
#   - Thesis and presentation PDFs rebuilt from a clean LaTeX state
#
# Usage:
#   ./run_all.sh [options]
#
# Options:
#   --quick             Smoke test only (2 runs/seeds where supported)
#   --skip-benchmark    Skip experiments; regenerate plots/tables and PDFs only
#   --skip-h0           Skip H0 calibration
#   --skip-build        Skip LaTeX PDF compilation
#   --help              Show this help message
#
# Optional environment overrides:
#   PYTHON=python3
#   BENCHMARK_N_RUNS=30
#   BENCHMARK_PROPER_N_SEEDS=30
#   ADAPTATION_N_RUNS=30
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
BENCHMARK_N_RUNS="${BENCHMARK_N_RUNS:-30}"
BENCHMARK_PROPER_N_SEEDS="${BENCHMARK_PROPER_N_SEEDS:-30}"
ADAPTATION_N_RUNS="${ADAPTATION_N_RUNS:-30}"

QUICK_MODE=false
SKIP_BENCHMARK=false
SKIP_H0=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=true
            export QUICK_MODE=True
            BENCHMARK_N_RUNS=2
            BENCHMARK_PROPER_N_SEEDS=2
            ADAPTATION_N_RUNS=2
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-h0)
            SKIP_H0=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: ./run_all.sh [--quick] [--skip-benchmark] [--skip-h0] [--skip-build]"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            shift
            ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p results/logs results/plots results/tables results/raw
LOG_FILE="${PROJECT_ROOT}/results/logs/run_all_${TIMESTAMP}.log"

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

run_logged() {
    echo "+ $*" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

run_env_logged() {
    echo "+ $*" | tee -a "$LOG_FILE"
    env "$@" 2>&1 | tee -a "$LOG_FILE"
}

print_header "1. Environment"
if [ -f ".venv/bin/activate" ]; then
    # Use the local environment when it exists, but do not create one here.
    source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Python: $($PYTHON --version 2>&1)" | tee -a "$LOG_FILE"
echo "Detection runs: ${BENCHMARK_N_RUNS}" | tee -a "$LOG_FILE"
echo "Classification seeds/scenario: ${BENCHMARK_PROPER_N_SEEDS}" | tee -a "$LOG_FILE"
echo "Adaptation runs/scenario: ${ADAPTATION_N_RUNS}" | tee -a "$LOG_FILE"
print_success "Environment ready"

if [ "$SKIP_BENCHMARK" = false ]; then
    if [ "$SKIP_H0" = false ]; then
        print_header "2a. H0 Calibration"
        if [ "$QUICK_MODE" = true ]; then
            run_logged "$PYTHON" scripts/h0_calibration.py --quick
        else
            run_logged "$PYTHON" scripts/h0_calibration.py
        fi
        print_success "H0 calibration complete"
    fi

    print_header "2b. Detection Benchmark (N=${BENCHMARK_N_RUNS})"
    run_env_logged \
        "BENCHMARK_N_RUNS=${BENCHMARK_N_RUNS}" \
        "$PYTHON" main.py benchmark
    print_success "Detection benchmark complete"

    print_header "2c. SE-CDT vs CDT-MSW Classification (N=${BENCHMARK_PROPER_N_SEEDS})"
    run_env_logged \
        "BENCHMARK_PROPER_N_SEEDS=${BENCHMARK_PROPER_N_SEEDS}" \
        "$PYTHON" main.py compare
    print_success "Classification benchmark complete"

    print_header "2d. Type-Specific Adaptation (N=${ADAPTATION_N_RUNS})"
    for dtype in stepping sudden mixed; do
        echo "Scenario: ${dtype}" | tee -a "$LOG_FILE"
        run_logged "$PYTHON" main.py monitoring --drift_type "$dtype" --n_runs "$ADAPTATION_N_RUNS"
    done
    print_success "Adaptation benchmark complete"
else
    echo "Skipping benchmark execution." | tee -a "$LOG_FILE"
fi

print_header "3. Generate Tables and Figures"
run_logged "$PYTHON" main.py plot
print_success "Tables and figures regenerated"

if [ "$SKIP_BUILD" = false ]; then
    print_header "4. Build Thesis and Presentation"
    cd report/latex
    rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot main.fdb_latexmk main.fls
    rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.snm presentation.toc presentation.fdb_latexmk presentation.fls
    run_logged latexmk -pdf -interaction=nonstopmode main.tex
    run_logged latexmk -pdf -interaction=nonstopmode presentation.tex
    cd "$PROJECT_ROOT"
    print_success "PDF build complete"
fi

print_header "Execution Summary"
echo "Log file:      $LOG_FILE"
echo "Tables:        results/tables/"
echo "Plots:         results/plots/"
echo "Thesis PDF:    report/latex/main.pdf"
echo "Slides PDF:    report/latex/presentation.pdf"
