#!/bin/bash

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Verify minimum version (numpy==1.26.4 requires Python ≥ 3.9; tested with 3.12.3)
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "Warning: Python $PY_VERSION detected. Requirements pinned for Python 3.10+ (tested with 3.12.3)."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Creating one first..."
    echo "Please run this script again after requirements.txt is created."
    exit 1
fi

