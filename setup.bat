@echo off
echo Enhanced AI Image Renamer Setup for Windows 11
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv_image_renamer
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv_image_renamer\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Check for NVIDIA GPU and install appropriate PyTorch
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected, installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No NVIDIA GPU detected, installing CPU-only PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

if %errorlevel% neq 0 (
    echo Error: Failed to install PyTorch
    pause
    exit /b 1
)

REM Install other dependencies
echo Installing other dependencies...
pip install transformers>=4.30.0 pillow>=9.0.0 accelerate>=0.20.0 sentencepiece>=0.1.99
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo.
echo To run the enhanced image renamer:
echo 1. Double-click run.bat
echo 2. Or manually: activate venv_image_renamer\Scripts\activate.bat then python enhanced_image_renamer.py
echo.
pause
