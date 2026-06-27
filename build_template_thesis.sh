#!/bin/bash

# HCMUT Thesis Template PDF Build Script
# This script compiles the LaTeX thesis in the official school format with proper bibliography handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LATEX_DIR="report/HCMUT_Master_Thesis_Template"
MAIN_FILE="main"
TARGET_NAME="2370116_LePhucDuc_ThesisReport_HCMUT"
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
    rm -f chapters/*.aux ext_pages/*.aux
    
    print_success "Auxiliary files cleaned."
    cd - > /dev/null
}

# Function to compile LaTeX
compile_latex() {
    print_status "Starting LaTeX compilation for HCMUT Thesis Template..."
    cd "$LATEX_DIR"
    
    # First pass
    print_status "Running pdflatex (1st pass)..."
    set +e  # Temporarily disable exit on error for pdflatex
    pdflatex -interaction=nonstopmode -jobname="$TARGET_NAME" "$MAIN_FILE.tex" > pdflatex_1st.log 2>&1
    local exit_code=$?
    set -e  # Re-enable exit on error
    if [ $exit_code -ne 0 ] && ! grep -q "Output written on $TARGET_NAME.pdf" pdflatex_1st.log; then
        print_error "First pdflatex pass failed. Check $TARGET_NAME.log for details."
        cat "$TARGET_NAME.log" | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    # Run bibtex if .bib file exists
    if [ -f "main.bib" ]; then
        print_status "Running bibtex..."
        if ! bibtex "$TARGET_NAME" > /dev/null 2>&1; then
            print_warning "BibTeX failed, but continuing..."
        else
            print_success "BibTeX completed successfully."
        fi
    fi
    
    # Second pass
    print_status "Running pdflatex (2nd pass)..."
    set +e  # Temporarily disable exit on error for pdflatex
    pdflatex -interaction=nonstopmode -jobname="$TARGET_NAME" "$MAIN_FILE.tex" > pdflatex_2nd.log 2>&1
    local exit_code=$?
    set -e  # Re-enable exit on error
    if [ $exit_code -ne 0 ] && ! grep -q "Output written on $TARGET_NAME.pdf" pdflatex_2nd.log; then
        print_error "Second pdflatex pass failed. Check $TARGET_NAME.log for details."
        cat "$TARGET_NAME.log" | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    # Third pass (to resolve all cross-references)
    print_status "Running pdflatex (3rd pass)..."
    set +e  # Temporarily disable exit on error for pdflatex
    pdflatex -interaction=nonstopmode -jobname="$TARGET_NAME" "$MAIN_FILE.tex" > pdflatex_3rd.log 2>&1
    local exit_code=$?
    set -e  # Re-enable exit on error
    if [ $exit_code -ne 0 ] && ! grep -q "Output written on $TARGET_NAME.pdf" pdflatex_3rd.log; then
        print_error "Third pdflatex pass failed. Check $TARGET_NAME.log for details."
        cat "$TARGET_NAME.log" | tail -20
        cd - > /dev/null
        exit 1
    fi
    
    cd - > /dev/null
    print_success "LaTeX compilation completed successfully."
}

# Function to check output
check_output() {
    if [ -f "$LATEX_DIR/$TARGET_NAME.pdf" ]; then
        local file_size=$(stat -c%s "$LATEX_DIR/$TARGET_NAME.pdf" 2>/dev/null || stat -f%z "$LATEX_DIR/$TARGET_NAME.pdf" 2>/dev/null)
        local file_size_mb=$((file_size / 1024 / 1024))
        print_success "PDF generated successfully: $LATEX_DIR/$TARGET_NAME.pdf (${file_size_mb}MB)"
        
        # Copy to output directory if specified
        if [ "$1" = "--output" ] && [ -n "$2" ]; then
            mkdir -p "$2"
            cp "$LATEX_DIR/$TARGET_NAME.pdf" "$2/"
            print_success "PDF copied to: $2/$TARGET_NAME.pdf"
        fi
    else
        print_error "PDF generation failed!"
        exit 1
    fi
}

# Function to show statistics
show_stats() {
    cd "$LATEX_DIR"
    if [ -f "$TARGET_NAME.log" ]; then
        local pages=$(grep -o "Output written on $TARGET_NAME.pdf ([0-9]* pages" "$TARGET_NAME.log" | grep -o "([0-9]*" | grep -o "[0-9]*")
        local warnings=$(grep -c "Warning" "$TARGET_NAME.log" 2>/dev/null || echo "0")
        local errors=$(grep -c "Error" "$TARGET_NAME.log" 2>/dev/null || echo "0")
        
        print_status "Compilation Statistics:"
        echo "  Pages: ${pages:-Unknown}"
        echo "  Warnings: $warnings"
        echo "  Errors: $errors"
        
        if [ "$warnings" -gt 0 ]; then
            print_warning "There were $warnings warnings. Check $TARGET_NAME.log for details."
        fi
    fi
    cd - > /dev/null
}

# Main execution
main() {
    local clean_first=false
    local output_dir=""
    
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
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check if we're in the right directory
    if [ ! -d "$LATEX_DIR" ]; then
        print_error "LaTeX directory '$LATEX_DIR' not found!"
        exit 1
    fi
    
    # Check dependencies
    check_dependencies
    
    # Clean if requested
    if [ "$clean_first" = true ]; then
        clean_aux_files
    fi
    
    # Build
    compile_latex
    
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
