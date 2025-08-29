#!/bin/bash

# Thesis PDF Build Script
# This script compiles the LaTeX thesis with proper bibliography handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LATEX_DIR="report/latex"
MAIN_FILE="main"
OUTPUT_DIR="output"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_status "Checking LaTeX dependencies..."
    
    if ! command_exists pdflatex; then
        print_error "pdflatex not found. Please install TeX Live or MiKTeX."
        exit 1
    fi
    
    if ! command_exists bibtex; then
        print_error "bibtex not found. Please install TeX Live or MiKTeX."
        exit 1
    fi
    
    print_success "All dependencies found."
}

# Function to clean auxiliary files
clean_aux_files() {
    print_status "Cleaning auxiliary files..."
    cd "$LATEX_DIR"
    
    # Remove auxiliary files
    rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc *.lof *.lot *.nav *.snm *.vrb
    
    print_success "Auxiliary files cleaned."
    cd - > /dev/null
}

# Function to compile LaTeX
compile_latex() {
    print_status "Starting LaTeX compilation..."
    cd "$LATEX_DIR"
    
    # First pass
    print_status "Running pdflatex (1st pass)..."
    pdflatex -interaction=nonstopmode "$MAIN_FILE.tex" > pdflatex_1st.log 2>&1
    if [ $? -ne 0 ] && ! grep -q "Output written on $MAIN_FILE.pdf" pdflatex_1st.log; then
        print_error "First pdflatex pass failed. Check main.log for details."
        cat main.log | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    # Run bibtex if .bib file exists
    if [ -f "references.bib" ]; then
        print_status "Running bibtex..."
        if ! bibtex "$MAIN_FILE" > /dev/null 2>&1; then
            print_warning "BibTeX failed, but continuing..."
        else
            print_success "BibTeX completed successfully."
        fi
    fi
    
    # Second pass
    print_status "Running pdflatex (2nd pass)..."
    pdflatex -interaction=nonstopmode "$MAIN_FILE.tex" > pdflatex_2nd.log 2>&1
    if [ $? -ne 0 ] && ! grep -q "Output written on $MAIN_FILE.pdf" pdflatex_2nd.log; then
        print_error "Second pdflatex pass failed. Check main.log for details."
        cat main.log | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    # Third pass (to resolve all cross-references)
    print_status "Running pdflatex (3rd pass)..."
    pdflatex -interaction=nonstopmode "$MAIN_FILE.tex" > pdflatex_3rd.log 2>&1
    if [ $? -ne 0 ] && ! grep -q "Output written on $MAIN_FILE.pdf" pdflatex_3rd.log; then
        print_error "Third pdflatex pass failed. Check main.log for details."
        cat main.log | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    cd - > /dev/null
    print_success "LaTeX compilation completed successfully."
}

# Function to check output
check_output() {
    if [ -f "$LATEX_DIR/$MAIN_FILE.pdf" ]; then
        local file_size=$(stat -c%s "$LATEX_DIR/$MAIN_FILE.pdf" 2>/dev/null || stat -f%z "$LATEX_DIR/$MAIN_FILE.pdf" 2>/dev/null)
        local file_size_mb=$((file_size / 1024 / 1024))
        print_success "PDF generated successfully: $LATEX_DIR/$MAIN_FILE.pdf (${file_size_mb}MB)"
        
        # Copy to output directory if specified
        if [ "$1" = "--output" ] && [ -n "$2" ]; then
            mkdir -p "$2"
            cp "$LATEX_DIR/$MAIN_FILE.pdf" "$2/"
            print_success "PDF copied to: $2/$MAIN_FILE.pdf"
        fi
    else
        print_error "PDF generation failed!"
        exit 1
    fi
}

# Function to show statistics
show_stats() {
    cd "$LATEX_DIR"
    if [ -f "$MAIN_FILE.log" ]; then
        local pages=$(grep -o "Output written on $MAIN_FILE.pdf ([0-9]* pages" "$MAIN_FILE.log" | grep -o "[0-9]*" | head -1)
        local warnings=$(grep -c "Warning" "$MAIN_FILE.log" 2>/dev/null || echo "0")
        local errors=$(grep -c "Error" "$MAIN_FILE.log" 2>/dev/null || echo "0")
        
        print_status "Compilation Statistics:"
        echo "  Pages: ${pages:-Unknown}"
        echo "  Warnings: $warnings"
        echo "  Errors: $errors"
        
        if [ "$warnings" -gt 0 ]; then
            print_warning "There were $warnings warnings. Check main.log for details."
        fi
    fi
    cd - > /dev/null
}

# Function to display help
show_help() {
    echo "Thesis PDF Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean, -c       Clean auxiliary files before building"
    echo "  --output, -o DIR  Copy PDF to specified output directory"
    echo "  --fast, -f        Fast build (skip bibliography)"
    echo "  --watch, -w       Watch mode (rebuild on file changes)"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                Build thesis PDF"
    echo "  $0 --clean        Clean and build"
    echo "  $0 -o build/      Build and copy PDF to build/ directory"
    echo "  $0 --watch        Watch for changes and rebuild automatically"
}

# Function for fast build (skip bibliography)
fast_build() {
    print_status "Starting fast build (skipping bibliography)..."
    cd "$LATEX_DIR"
    
    # Single pass
    print_status "Running pdflatex..."
    if ! pdflatex -interaction=nonstopmode "$MAIN_FILE.tex" > /dev/null 2>&1; then
        print_error "pdflatex failed. Check main.log for details."
        cat main.log | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    cd - > /dev/null
    print_success "Fast build completed."
}

# Function for watch mode
watch_mode() {
    print_status "Starting watch mode. Press Ctrl+C to stop."
    print_status "Watching for changes in $LATEX_DIR/*.tex and $LATEX_DIR/*.bib files..."
    
    # Check if inotifywait is available
    if ! command_exists inotifywait; then
        print_warning "inotifywait not found. Falling back to polling mode."
        
        # Polling mode
        while true; do
            sleep 2
            if find "$LATEX_DIR" -name "*.tex" -o -name "*.bib" -newer "$LATEX_DIR/$MAIN_FILE.pdf" 2>/dev/null | grep -q .; then
                print_status "Changes detected, rebuilding..."
                compile_latex
                check_output
            fi
        done
    else
        # inotify mode
        while true; do
            inotifywait -e modify,create,delete -r "$LATEX_DIR" --include='.*\.(tex|bib)$' -q
            print_status "Changes detected, rebuilding..."
            sleep 1  # Brief delay to allow file writes to complete
            compile_latex
            check_output
        done
    fi
}

# Main execution
main() {
    local clean_first=false
    local output_dir=""
    local fast_mode=false
    local watch_mode_enabled=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean|-c)
                clean_first=true
                shift
                ;;
            --output|-o)
                output_dir="$2"
                shift 2
                ;;
            --fast|-f)
                fast_mode=true
                shift
                ;;
            --watch|-w)
                watch_mode_enabled=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Header
    echo "=================================="
    echo "  Thesis PDF Build Script"
    echo "=================================="
    echo ""
    
    # Check if we're in the right directory
    if [ ! -d "$LATEX_DIR" ]; then
        print_error "LaTeX directory '$LATEX_DIR' not found!"
        print_error "Please run this script from the project root directory."
        exit 1
    fi
    
    # Check dependencies
    check_dependencies
    
    # Clean if requested
    if [ "$clean_first" = true ]; then
        clean_aux_files
    fi
    
    # Watch mode
    if [ "$watch_mode_enabled" = true ]; then
        watch_mode
        return
    fi
    
    # Build
    if [ "$fast_mode" = true ]; then
        fast_build
    else
        compile_latex
    fi
    
    # Check output and copy if needed
    if [ -n "$output_dir" ]; then
        check_output --output "$output_dir"
    else
        check_output
    fi
    
    # Show statistics
    show_stats
    
    echo ""
    print_success "Build completed successfully! 🎉"
}

# Run main function with all arguments
main "$@"
