@echo off
REM Quick start script for Barkour Quadruped Simulation
REM This script activates the virtual environment and runs the simulation

echo ============================================================
echo BARKOUR QUADRUPED SIMULATION - QUICK START
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running simulation...
echo.

REM Run the simulation
python run_barkour_local.py

echo.
echo ============================================================
echo Simulation finished!
echo ============================================================
echo.

REM Keep window open
pause
