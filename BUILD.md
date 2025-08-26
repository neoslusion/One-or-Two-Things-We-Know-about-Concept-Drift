# Thesis Build Guide

This document explains how to build the LaTeX thesis PDF using the provided build tools.

## 🚀 Quick Start

The easiest way to build your thesis:

```bash
# Build the complete thesis
make build

# Or use the quick build script
./quick_build.sh
```

## 📋 Available Build Methods

### 1. Makefile (Recommended)

```bash
make help          # Show all available commands
make build         # Full build with bibliography
make quick         # Quick build for testing
make clean         # Clean auxiliary files
make watch         # Watch mode (rebuild on changes)
make view          # Open PDF viewer
make check         # Check if PDF exists and show info
```

### 2. Advanced Build Script

```bash
./build_thesis.sh --help          # Show help
./build_thesis.sh                 # Standard build
./build_thesis.sh --clean         # Clean and build
./build_thesis.sh --fast          # Fast build (skip bibliography)
./build_thesis.sh --watch         # Watch mode
./build_thesis.sh -o output/      # Build and copy to output directory
```

### 3. Quick Build Script

```bash
./quick_build.sh                  # Simple, fast build
```

## 🛠️ Build Process Explained

The full build process consists of:

1. **Clean** - Remove auxiliary files from previous builds
2. **First Pass** - Initial LaTeX compilation
3. **Bibliography** - Process references with BibTeX
4. **Second Pass** - Incorporate bibliography
5. **Third Pass** - Resolve all cross-references

## 📁 File Structure

```
project/
├── report/latex/           # LaTeX source files
│   ├── main.tex           # Main document
│   ├── titlepage.tex      # Title page
│   ├── references.bib     # Bibliography
│   └── chapters/          # Chapter files
├── build_thesis.sh        # Advanced build script
├── quick_build.sh         # Simple build script
├── Makefile              # Make commands
└── BUILD.md              # This guide
```

## 🔧 Troubleshooting

### Common Issues

1. **LaTeX not found**
   ```bash
   make install          # Install LaTeX dependencies
   ```

2. **Build fails with errors**
   ```bash
   make clean           # Clean and try again
   make build
   ```

3. **Missing packages**
   - Install `texlive-full` on Ubuntu/Debian
   - Install `MacTeX` on macOS
   - Check the log files in `report/latex/main.log`

### Build Artifacts

These files are automatically generated and ignored by git:
- `*.aux` - Auxiliary files
- `*.log` - Compilation logs
- `*.toc` - Table of contents
- `*.lof` - List of figures
- `*.lot` - List of tables
- `*.bbl`, `*.blg` - Bibliography files
- `*.synctex.gz` - Sync files

## 🎯 Development Workflow

### For Active Development

```bash
# Start watch mode in one terminal
make watch

# Edit your .tex files in another terminal/editor
# PDF will rebuild automatically on changes
```

### For Quick Testing

```bash
# Fast build cycle
make quick
make view
```

### For Final Release

```bash
# Clean build for final version
make clean
make build
make check         # Verify output
```

## 📊 Build Output

Successful build will show:
- ✅ Number of pages generated
- ✅ PDF file size
- ✅ Location of generated PDF
- ⚠️ Any warnings (usually safe to ignore)

Example output:
```
✅ Success! PDF generated
   📄 Pages: 53
   📦 Size: 312K
   📍 Location: report/latex/main.pdf
```

## 🔍 Advanced Features

### Watch Mode
Automatically rebuilds when you save changes:
```bash
make watch
# or
./build_thesis.sh --watch
```

### Custom Output Directory
```bash
./build_thesis.sh -o /path/to/output/
```

### Fast Build (Skip Bibliography)
For quick iterations when you haven't changed references:
```bash
make quick
# or
./build_thesis.sh --fast
```

## 📋 Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- pdflatex
- bibtex
- Standard Unix tools (bash, make)

## 🆘 Getting Help

If you encounter issues:

1. Check the build logs in `report/latex/main.log`
2. Run `make clean` and try again
3. Ensure all required packages are installed
4. Check that all `.tex` files are properly formatted

For LaTeX-specific issues, consult:
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [TeX Stack Exchange](https://tex.stackexchange.com/)
