@echo off
title Enhanced AI Image Renamer Setup for Python 3.13
cls
echo Enhanced AI Image Renamer Setup for Python 3.13
echo ==============================================
echo.
echo NOTE: Python 3.13 is very new and some packages may not have
echo pre-built wheels yet. This setup handles those compatibility issues.
echo.

REM --- Python Check ---
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in your system's PATH.
    echo Please install Python 3.8-3.12 from https://python.org for better compatibility.
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
echo.

REM --- Upgrade Build Tools ---
echo Installing/upgrading build tools for Python 3.13...
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade build
echo.

REM --- PyTorch Installation ---
echo Installing PyTorch (this may take several minutes)...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --prefer-binary
) else (
    echo Installing CPU-only PyTorch.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --prefer-binary
)
echo.

REM --- Install Dependencies One by One ---
echo Installing dependencies individually to handle Python 3.13 compatibility...
echo.

echo Installing Pillow...
pip install "pillow>=9.0.0" --prefer-binary
echo.

echo Installing transformers...
pip install "transformers>=4.30.0" --prefer-binary --no-build-isolation
echo.

echo Installing accelerate...
pip install "accelerate>=0.20.0" --prefer-binary --no-build-isolation
echo.

echo Installing sentencepiece...
pip install "sentencepiece>=0.1.99" --prefer-binary --no-build-isolation
if %errorlevel% neq 0 (
    echo Sentencepiece failed, trying alternative installation...
    pip install sentencepiece --no-build-isolation --force-reinstall
)
echo.

echo Installing additional packages...
pip install requests tqdm numpy --prefer-binary
echo.

echo ==================================
echo      SETUP COMPLETE!
echo ==================================
echo.
echo The setup is complete. Some packages may have shown warnings
echo due to Python 3.13 compatibility, but the core functionality
echo should work.
echo.
echo To run the program, double-click on 'run.bat'.
echo.
echo If you encounter import errors when running the program,
echo you can install missing packages individually using:
echo   pip install [package-name]
echo.
pause
