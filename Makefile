# Thesis Makefile
# Simple make commands for building the thesis

# Variables
LATEX_DIR = report/latex
MAIN_FILE = main
BUILD_SCRIPT = ./build_thesis.sh
QUICK_SCRIPT = ./quick_build.sh

# Default target
.PHONY: help
help:
	@echo "📚 Thesis Build System"
	@echo "====================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build     - Full build with bibliography"
	@echo "  make quick     - Quick build for testing"
	@echo "  make clean     - Clean auxiliary files"
	@echo "  make watch     - Watch mode (rebuild on changes)"
	@echo "  make view      - Open PDF viewer"
	@echo "  make install   - Install LaTeX dependencies"
	@echo "  make help      - Show this help"
	@echo ""

# Full build
.PHONY: build
build:
	@echo "🔨 Building thesis..."
	$(BUILD_SCRIPT)

# Quick build
.PHONY: quick
quick:
	@echo "⚡ Quick build..."
	$(QUICK_SCRIPT)

# Clean build
.PHONY: clean
clean:
	@echo "🧹 Cleaning build files..."
	$(BUILD_SCRIPT) --clean

# Watch mode
.PHONY: watch
watch:
	@echo "👀 Starting watch mode..."
	$(BUILD_SCRIPT) --watch

# View PDF
.PHONY: view
view:
	@if [ -f "$(LATEX_DIR)/$(MAIN_FILE).pdf" ]; then \
		echo "👁️  Opening PDF..."; \
		if command -v xdg-open > /dev/null; then \
			xdg-open "$(LATEX_DIR)/$(MAIN_FILE).pdf"; \
		elif command -v open > /dev/null; then \
			open "$(LATEX_DIR)/$(MAIN_FILE).pdf"; \
		else \
			echo "❌ No PDF viewer found"; \
		fi; \
	else \
		echo "❌ PDF not found. Run 'make build' first."; \
	fi

# Install dependencies
.PHONY: install
install:
	@echo "📦 Installing LaTeX dependencies..."
	@if command -v apt-get > /dev/null; then \
		echo "Installing via apt-get..."; \
		sudo apt-get update; \
		sudo apt-get install -y texlive-full; \
	elif command -v brew > /dev/null; then \
		echo "Installing via Homebrew..."; \
		brew install --cask mactex; \
	elif command -v pacman > /dev/null; then \
		echo "Installing via pacman..."; \
		sudo pacman -S texlive-most; \
	else \
		echo "❌ Package manager not found. Please install LaTeX manually."; \
		echo "Visit: https://www.latex-project.org/get/"; \
	fi

# Check PDF exists
.PHONY: check
check:
	@if [ -f "$(LATEX_DIR)/$(MAIN_FILE).pdf" ]; then \
		pages=$$(pdfinfo "$(LATEX_DIR)/$(MAIN_FILE).pdf" 2>/dev/null | grep Pages | awk '{print $$2}' || echo "Unknown"); \
		size=$$(ls -lh "$(LATEX_DIR)/$(MAIN_FILE).pdf" | awk '{print $$5}'); \
		echo "✅ PDF exists: $$pages pages, $$size"; \
	else \
		echo "❌ PDF not found"; \
	fi

# Development commands
.PHONY: dev
dev: clean build view

# Fast development cycle
.PHONY: fast
fast: quick view

# Release build
.PHONY: release
release:
	@echo "🚀 Creating release build..."
	$(BUILD_SCRIPT) --clean --output release/
	@echo "📦 Release created in release/ directory"
