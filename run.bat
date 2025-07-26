@echo off
echo Starting Enhanced AI Image Renamer...
echo.

REM Check if virtual environment exists
if not exist "venv_image_renamer" (
    echo Virtual environment not found. Running setup first...
    call setup.bat
    if %errorlevel% neq 0 (
        echo Setup failed. Please check the error messages above.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv_image_renamer\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Run the application
python enhanced_image_renamer.py
if %errorlevel% neq 0 (
    echo Error: Failed to run the application
    pause
    exit /b 1
)

echo.
echo Application closed.
pause
