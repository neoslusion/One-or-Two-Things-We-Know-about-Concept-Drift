# PowerShell script for Windows environment setup

# Check if Python 3 is installed
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Python found: $pythonVersion"
} catch {
    Write-Host "Python 3 is not installed or not in PATH. Please install Python 3.7+ first." -ForegroundColor Red
    Write-Host "You can download Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv .venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create virtual environment. Please ensure Python venv module is available." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate virtual environment." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip." -ForegroundColor Yellow
}

# Install requirements
Write-Host "Installing required packages..." -ForegroundColor Green
if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully installed all required packages!" -ForegroundColor Green
    } else {
        Write-Host "Some packages failed to install. Please check the output above." -ForegroundColor Yellow
    }
} else {
    Write-Host "requirements.txt not found. Creating one first..." -ForegroundColor Yellow
    Write-Host "Please run this script again after requirements.txt is created." -ForegroundColor Red
    exit 1
}

Write-Host "Environment setup completed!" -ForegroundColor Green
Write-Host "To activate the virtual environment in the future, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
