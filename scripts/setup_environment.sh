#!/bin/bash

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

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

