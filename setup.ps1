# Enhanced AI Image Renamer Setup for Windows 11 (PowerShell)
Write-Host "Enhanced AI Image Renamer Setup for Windows 11" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Found $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "✗ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv venv_image_renamer
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv_image_renamer\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Check for NVIDIA GPU
Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    nvidia-smi | Out-Null
    Write-Host "✓ NVIDIA GPU detected, installing PyTorch with CUDA support..." -ForegroundColor Green
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
} catch {
    Write-Host "No NVIDIA GPU detected, installing CPU-only PyTorch..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Install other dependencies
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install transformers>=4.30.0 pillow>=9.0.0 accelerate>=0.20.0 sentencepiece>=0.1.99

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the enhanced image renamer:" -ForegroundColor Cyan
Write-Host "1. Double-click run.bat" -ForegroundColor White
Write-Host "2. Or run: .\run.ps1" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"
