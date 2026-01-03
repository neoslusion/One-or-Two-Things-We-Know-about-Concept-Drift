#!/bin/bash
# ============================================================================
# Drift Detection Benchmark Runner
# ============================================================================
# This script runs the drift detection benchmark from the 
# experiments/drift_detection_benchmark folder using the virtual environment.
#
# Usage: ./run_benchmark.sh [options]
#   Options:
#     --help      Show this help message
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
BENCHMARK_DIR="${SCRIPT_DIR}/experiments/drift_detection_benchmark"
BACKUP_DIR="${SCRIPT_DIR}/experiments/backup"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: ./run_benchmark.sh"
            echo ""
            echo "Runs the drift detection benchmark using main.py"
            echo "Configuration is in experiments/drift_detection_benchmark/config.py"
            echo ""
            echo "Results are saved to experiments/drift_detection_benchmark/publication_figures/"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1 (ignoring)${NC}"
            shift
            ;;
    esac
done

# Check virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${VENV_DIR}${NC}"
    echo "Please create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"

# Check required packages
echo -e "${GREEN}Checking dependencies...${NC}"
python3 -c "import numpy, pandas, sklearn, scipy" 2>/dev/null || {
    echo -e "${RED}Missing dependencies. Please install with: pip install -r requirements.txt${NC}"
    exit 1
}

# Add backup directory to Python path (for shape_dd, ow_mmd, etc.)
export PYTHONPATH="${BACKUP_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"

# Run the benchmark
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Starting Drift Detection Benchmark${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Main file: ${BENCHMARK_DIR}/main.py"
echo -e "Config: ${BENCHMARK_DIR}/config.py"
echo ""

START_TIME=$(date +%s)

# Run main.py
cd "${SCRIPT_DIR}"
python3 -m experiments.drift_detection_benchmark.main

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Main Benchmark Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Duration: ${ELAPSED} seconds ($(echo "scale=1; ${ELAPSED}/60" | bc) minutes)"
echo -e "Results: ${BENCHMARK_DIR}/publication_figures/"
echo ""

# Generate SNR-Adaptive specific figures
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Generating SNR-Adaptive Figures${NC}"
echo -e "${BLUE}============================================${NC}"
python3 "${BENCHMARK_DIR}/visualizations/generate_all_figures.py"
echo -e "Results: report/latex/image/"
echo ""

# Deactivate virtual environment
deactivate

echo -e "${GREEN}Done!${NC}"
