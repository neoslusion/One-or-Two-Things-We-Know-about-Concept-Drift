#!/bin/bash
# =============================================================================
# FULL BENCHMARK & REPORT PIPELINE
# =============================================================================
# Chạy toàn bộ benchmark và cập nhật báo cáo LaTeX
# 
# Usage:
#   ./run_full_pipeline.sh           # Chạy đầy đủ (30 runs, ~50 phút)
#   ./run_full_pipeline.sh --quick   # Chạy nhanh (5 runs, ~15 phút)
#   ./run_full_pipeline.sh --skip-benchmark  # Chỉ build PDF
#
# Pipeline steps:
#   1. benchmark  - Window-based drift detection benchmark
#   2. compare    - SE-CDT vs CDT_MSW comparison
#   3. monitoring - Prequential accuracy evaluation
#   4. plot       - Generate all figures
#   5. Build PDF
# =============================================================================

set -e  # Exit on error

# Colors for output
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
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# =============================================================================
# Step 0: Environment Setup
# =============================================================================
print_header "Step 0: Environment Setup"

if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found. Run: python3 -m venv .venv"
    exit 1
fi

source .venv/bin/activate
print_success "Virtual environment activated"

# Create output directories
mkdir -p results/tables results/plots results/logs results/raw
print_success "Output directories ready"

# =============================================================================
# Step 1: Run Benchmark
# =============================================================================
if [ "$SKIP_BENCHMARK" = false ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="results/logs/pipeline_${TIMESTAMP}.log"
    
    # -------------------------------------------------------------------------
    # Step 1a: Window-based Drift Detection Benchmark
    # -------------------------------------------------------------------------
    print_header "Step 1a: Window-based Drift Detection Benchmark"
    
    if [ "$QUICK_MODE" = true ]; then
        echo "Mode: QUICK (5 runs)"
        python main.py benchmark --quick 2>&1 | tee "$LOG_FILE"
    else
        echo "Mode: FULL (30 runs) - Estimated time: 20-30 minutes"
        python main.py benchmark 2>&1 | tee "$LOG_FILE"
    fi
    
    print_success "Benchmark completed"
    
    # -------------------------------------------------------------------------
    # Step 1b: SE-CDT vs CDT_MSW Comparison
    # -------------------------------------------------------------------------
    print_header "Step 1b: SE-CDT vs CDT_MSW Comparison"
    
    python main.py compare 2>&1 | tee -a "$LOG_FILE"
    
    print_success "Comparison completed"
    
    # -------------------------------------------------------------------------
    # Step 1c: Monitoring / Prequential Evaluation
    # -------------------------------------------------------------------------
    print_header "Step 1c: Monitoring / Prequential Evaluation"
    
    python main.py monitoring 2>&1 | tee -a "$LOG_FILE"
    
    print_success "Monitoring evaluation completed"
    
    # -------------------------------------------------------------------------
    # Step 2: Generate All Figures
    # -------------------------------------------------------------------------
    print_header "Step 2: Generating All Figures"
    
    python main.py plot 2>&1 | tee -a "$LOG_FILE"
    
    print_success "Figures generated. Log: $LOG_FILE"
else
    print_warning "Skipping benchmark (--skip-benchmark flag)"
fi

# =============================================================================
# Step 3: Verify Generated Files
# =============================================================================
print_header "Step 3: Verifying Generated Files"

echo ""
echo "Tables in results/tables/:"
ls -la results/tables/*.tex 2>/dev/null || print_warning "No tables found"

echo ""
echo "Plots in results/plots/:"
ls results/plots/*.png 2>/dev/null | wc -l | xargs -I {} echo "{} PNG files generated"

# Check required tables
REQUIRED_TABLES=(
    "results/tables/table_I_comprehensive_performance.tex"
    "results/tables/table_II_f1_by_dataset.tex"
    "results/tables/table_se_cdt_aggregate.tex"
)

MISSING=false
for table in "${REQUIRED_TABLES[@]}"; do
    if [ ! -f "$table" ]; then
        print_error "Missing: $table"
        MISSING=true
    fi
done

if [ "$MISSING" = true ]; then
    print_error "Some required tables are missing!"
    exit 1
fi

print_success "All required files present"

# =============================================================================
# Step 4: Build LaTeX Report
# =============================================================================
print_header "Step 4: Building LaTeX Report"

cd report/latex

# Clean auxiliary files
rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot 2>/dev/null

echo "Running pdflatex (1/3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

echo "Running bibtex..."
bibtex main > /dev/null 2>&1 || true

echo "Running pdflatex (2/3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

echo "Running pdflatex (3/3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

if [ -f "main.pdf" ]; then
    PDF_SIZE=$(ls -lh main.pdf | awk '{print $5}')
    PDF_PAGES=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "?")
    print_success "PDF generated: main.pdf ($PDF_SIZE, $PDF_PAGES pages)"
else
    print_error "PDF generation failed!"
    exit 1
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Summary
# =============================================================================
print_header "Pipeline Complete!"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                         SUMMARY                                  ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║${NC}  Tables:   results/tables/*.tex"
echo -e "${GREEN}║${NC}  Figures:  results/plots/*.png"
echo -e "${GREEN}║${NC}  PDF:      report/latex/main.pdf"
echo -e "${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  To view PDF:"
echo -e "${GREEN}║${NC}    xdg-open report/latex/main.pdf"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
