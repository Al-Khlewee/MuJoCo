@echo off
REM Setup script for Barkour Quadruped Simulation
REM This script creates a virtual environment and installs all dependencies

echo ============================================================
echo BARKOUR QUADRUPED SIMULATION - SETUP
echo ============================================================
echo.
echo This script will:
echo   1. Create a Python virtual environment
echo   2. Install all required packages
echo   3. Verify the installation
echo.
echo This may take 5-10 minutes depending on your internet speed.
echo.
pause

REM Check Python version
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.9, 3.10, or 3.11
    pause
    exit /b 1
)
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Ask user about GPU support
echo.
echo ============================================================
echo GPU CONFIGURATION
echo ============================================================
echo Do you have an NVIDIA GPU with CUDA support?
echo.
echo   1. Yes - Install GPU version (faster, requires CUDA)
echo   2. No  - Install CPU version (slower, but works everywhere)
echo.
set /p gpu_choice="Enter your choice (1 or 2): "

if "%gpu_choice%"=="1" (
    echo.
    echo Installing GPU version with CUDA 12 support...
    pip install -U "jax[cuda12]"
) else (
    echo.
    echo Installing CPU version...
    pip install -U jax
)
echo.

REM Install MuJoCo and MJX
echo Installing MuJoCo and MJX...
pip install mujoco mujoco_mjx
if errorlevel 1 (
    echo ERROR: Failed to install MuJoCo packages!
    pause
    exit /b 1
)
echo.

REM Install Brax and dependencies
echo Installing Brax and dependencies...
pip install brax ml_collections flax orbax-checkpoint
if errorlevel 1 (
    echo ERROR: Failed to install Brax packages!
    pause
    exit /b 1
)
echo.

REM Install video rendering
echo Installing video rendering packages...
pip install mediapy opencv-python etils
if errorlevel 1 (
    echo ERROR: Failed to install video packages!
    pause
    exit /b 1
)
echo.

REM Verify installation
echo ============================================================
echo VERIFYING INSTALLATION
echo ============================================================
echo.

echo Checking JAX...
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
if errorlevel 1 (
    echo WARNING: JAX verification failed!
)
echo.

echo Checking MuJoCo...
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
if errorlevel 1 (
    echo WARNING: MuJoCo verification failed!
)
echo.

echo Checking Brax...
python -c "import brax; print('Brax installed successfully')"
if errorlevel 1 (
    echo WARNING: Brax verification failed!
)
echo.

REM Check if MuJoCo Menagerie exists
echo Checking for MuJoCo Menagerie...
if exist "mujoco_menagerie\google_barkour_vb" (
    echo Found: mujoco_menagerie\google_barkour_vb
) else (
    echo.
    echo WARNING: MuJoCo Menagerie not found!
    echo You need to clone the repository:
    echo   git clone https://github.com/google-deepmind/mujoco_menagerie
    echo.
)

REM Check if policy exists
echo Checking for trained policy...
if exist "mjx_brax_quadruped_policy" (
    echo Found: mjx_brax_quadruped_policy
) else (
    echo.
    echo WARNING: Trained policy not found!
    echo You need the mjx_brax_quadruped_policy folder in this directory.
    echo.
)

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. Ensure you have these folders:
echo      - mjx_brax_quadruped_policy\
echo      - mujoco_menagerie\google_barkour_vb\
echo.
echo   2. Edit run_barkour_local.py to configure:
echo      - Velocity commands (X_VEL, Y_VEL, ANG_VEL)
echo      - Number of simulation steps (N_STEPS)
echo.
echo   3. Run the simulation:
echo      - Double-click run_simulation.bat
echo      - Or run: python run_barkour_local.py
echo.
echo For detailed instructions, see README_BARKOUR.md
echo.
pause
