@echo off
title Enhanced AI Image Renamer Setup v2
cls
echo Enhanced AI Image Renamer Setup for Windows 11
echo ================================================
echo.

REM --- Python Check ---
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in your system's PATH.
    echo Please install Python 3.8 or higher from https://python.org
    echo IMPORTANT: During installation, make sure to check the box that says "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM --- Virtual Environment ---
if not exist venv_image_renamer (
    echo Creating Python virtual environment...
    python -m venv venv_image_renamer
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create the virtual environment.
        echo Please check your Python installation and permissions.
        echo.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)
echo.

REM --- Activate Virtual Environment ---
echo Activating virtual environment...
call venv_image_renamer\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    echo.
    pause
    exit /b 1
)
echo.

REM --- Upgrade Pip ---
echo Upgrading pip package manager...
python -m pip install --upgrade pip --quiet
echo.

REM --- PyTorch Installation ---
echo Checking for NVIDIA GPU to install the correct PyTorch version...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support.
    echo This may take several minutes...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo NVIDIA GPU not detected by 'where nvidia-smi'.
    echo This might be a PATH issue. Installing CPU-only PyTorch as a fallback.
    echo This may take several minutes...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch.
    echo This could be a network issue or a problem with your Python environment.
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo PyTorch installed successfully.
echo.

REM --- Install Other Dependencies ---
echo Installing other required libraries from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install one or more dependencies from requirements.txt.
    echo Please check your internet connection and the file contents.
    echo.
    pause
    exit /b 1
)
echo.

echo ==================================
echo      SETUP COMPLETE!
echo ==================================
echo.
echo To run the program, you can now double-click on 'run.bat'.
echo.
pause
