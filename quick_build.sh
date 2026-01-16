#!/bin/bash

# Quick Thesis Build Script
# Simple script for fast PDF generation

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Quick Thesis Build${NC}"
echo "========================"

# Change to latex directory
cd report/latex

# Clean previous build
echo -e "${BLUE}📁 Cleaning previous build...${NC}"
rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc *.lof *.lot

# Build PDF
echo -e "${BLUE}📝 Compiling LaTeX...${NC}"
pdflatex -interaction=nonstopmode main.tex > build1.log 2>&1
if [ $? -ne 0 ] && ! grep -q "Output written on main.pdf" build1.log; then
    echo -e "${RED}First pass failed${NC}"
    tail -10 main.log
    exit 1
fi

echo -e "${BLUE}📚 Processing bibliography...${NC}"
bibtex main > /dev/null 2>&1 || echo -e "${BLUE}⚠️  No bibliography to process${NC}"

echo -e "${BLUE}📝 Second pass...${NC}"
pdflatex -interaction=nonstopmode main.tex > build2.log 2>&1
if [ $? -ne 0 ] && ! grep -q "Output written on main.pdf" build2.log; then
    echo -e "${RED}Second pass failed${NC}"
    tail -10 main.log
    exit 1
fi

echo -e "${BLUE}📝 Final pass...${NC}"
pdflatex -interaction=nonstopmode main.tex > build3.log 2>&1
if [ $? -ne 0 ] && ! grep -q "Output written on main.pdf" build3.log; then
    echo -e "${RED}Final pass failed${NC}"
    tail -10 main.log
    exit 1
fi

# Check output
if [ -f "main.pdf" ]; then
    pages=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "Unknown")
    size=$(ls -lh main.pdf | awk '{print $5}')
    echo ""
    echo -e "${GREEN}Success! PDF generated${NC}"
    echo -e "   📄 Pages: $pages"
    echo -e "   📦 Size: $size"
    echo -e "   📍 Location: report/latex/main.pdf"
else
    echo -e "${RED}PDF generation failed!${NC}"
    exit 1
fi

cd - > /dev/null
echo ""
echo -e "${GREEN} Build completed!${NC}"
