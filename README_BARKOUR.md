# Barkour Quadruped Local Simulation Guide

This guide explains how to run the trained Barkour quadruped policy on your local Windows machine.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [File Structure](#file-structure)
4. [Configuration](#configuration)
5. [Running the Simulation](#running-the-simulation)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Hardware Requirements
- **Recommended**: NVIDIA GPU with CUDA support (for fast simulation)
- **Minimum**: CPU (will work but slower)
- **RAM**: At least 8 GB
- **Storage**: ~2 GB for dependencies

### Software Requirements
- **Python**: 3.9, 3.10, or 3.11 (Python 3.12 may have compatibility issues)
- **Windows 10/11**
- **CUDA Toolkit** (optional, for GPU acceleration): 11.8 or 12.x

---

## Installation

### Step 1: Create a Python Virtual Environment

```powershell
# Navigate to your project directory
cd c:\users\hatem\Desktop\MuJoCo

# Create virtual environment
python -m venv venv

# Activate the environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Install Required Packages

#### Option A: With GPU Support (CUDA 12.x)

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install JAX with CUDA support
pip install -U "jax[cuda12]"

# Install MuJoCo and MJX
pip install mujoco
pip install mujoco_mjx

# Install Brax and dependencies
pip install brax
pip install ml_collections
pip install flax
pip install orbax-checkpoint

# Install video rendering
pip install mediapy
pip install opencv-python

# Install additional dependencies
pip install etils
```

#### Option B: CPU Only (No GPU)

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install JAX CPU version
pip install -U jax

# Install MuJoCo and MJX
pip install mujoco
pip install mujoco_mjx

# Install Brax and dependencies
pip install brax
pip install ml_collections
pip install flax
pip install orbax-checkpoint

# Install video rendering
pip install mediapy
pip install opencv-python

# Install additional dependencies
pip install etils
```

### Step 3: Verify Installation

```powershell
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
python -c "import brax; print('Brax installed successfully')"
```

---

## File Structure

Ensure your directory structure looks like this:

```
c:\users\hatem\Desktop\MuJoCo\
├── mjx_brax_quadruped_policy/          # Trained policy files
│   ├── params                           # Policy parameters
│   └── ...
├── mujoco_menagerie/                    # Robot models
│   └── google_barkour_vb/
│       ├── scene_mjx.xml                # Main scene file
│       ├── assets/                      # Meshes and textures
│       └── ...
├── tutorial.ipynb                       # Original tutorial
├── run_barkour_local.py                 # ← The script you'll run
└── README_BARKOUR.md                    # ← This file
```

---

## Configuration

### Basic Configuration

Open `run_barkour_local.py` and modify the configuration section (lines 28-43):

```python
# File paths
BASE_DIR = Path(r"c:\users\hatem\Desktop\MuJoCo")
MODEL_PATH = BASE_DIR / "mjx_brax_quadruped_policy"
MENAGERIE_PATH = BASE_DIR / "mujoco_menagerie" / "google_barkour_vb"
OUTPUT_VIDEO = BASE_DIR / "barkour_simulation.mp4"

# Simulation parameters
X_VEL = 1.0      # Forward velocity (m/s) - range: [-0.6, 1.5]
Y_VEL = 0.0      # Sideways velocity (m/s) - range: [-0.8, 0.8]
ANG_VEL = -0.5   # Yaw rate (rad/s) - range: [-0.7, 0.7]

N_STEPS = 500    # Number of simulation steps
RENDER_EVERY = 2 # Render every N frames
RANDOM_SEED = 0  # Random seed for reproducibility
```

### Velocity Command Ranges

- **X_VEL (Forward/Backward)**:
  - Forward: 0.0 to 1.5 m/s
  - Backward: -0.6 to 0.0 m/s
  
- **Y_VEL (Sideways)**:
  - Right: -0.8 to 0.0 m/s
  - Left: 0.0 to 0.8 m/s
  
- **ANG_VEL (Rotation)**:
  - Clockwise: -0.7 to 0.0 rad/s
  - Counter-clockwise: 0.0 to 0.7 rad/s

---

## Running the Simulation

### Basic Usage

```powershell
# Activate virtual environment if not already active
.\venv\Scripts\Activate.ps1

# Run the simulation
python run_barkour_local.py
```

### Example Output

```
============================================================
BARKOUR QUADRUPED SIMULATION
============================================================

============================================================
CHECKING ENVIRONMENT
============================================================
✓ Model found: c:\users\hatem\Desktop\MuJoCo\mjx_brax_quadruped_policy
✓ Menagerie path found: c:\users\hatem\Desktop\MuJoCo\mujoco_menagerie\google_barkour_vb
✓ Scene file found: c:\users\hatem\Desktop\MuJoCo\mujoco_menagerie\google_barkour_vb\scene_mjx.xml
============================================================
All files found successfully!
============================================================

Configuring JAX...
JAX devices: [cuda(id=0)]
✓ GPU detected - using GPU acceleration

============================================================
LOADING MODEL AND ENVIRONMENT
============================================================
Creating environment...
✓ Environment created (action size: 12, obs size: 465)
Loading policy from: c:\users\hatem\Desktop\MuJoCo\mjx_brax_quadruped_policy
✓ Policy loaded successfully
Creating inference function...
✓ Inference function ready
JIT compiling environment functions (this may take a moment)...
✓ Environment functions compiled

============================================================
RUNNING SIMULATION
============================================================
Commands: x_vel=1.00, y_vel=0.00, ang_vel=-0.50
Steps: 500, Render every: 2
============================================================

Simulating...
  Step 100/500
  Step 200/500
  Step 300/500
  Step 400/500
  Step 500/500
✓ Simulation complete (501 frames)

============================================================
RENDERING VIDEO
============================================================
Rendering frames...
✓ Rendered 251 frames at 25.0 FPS
Saving video to: c:\users\hatem\Desktop\MuJoCo\barkour_simulation.mp4
✓ Video saved successfully!
  File: c:\users\hatem\Desktop\MuJoCo\barkour_simulation.mp4
  Size: 1.23 MB

============================================================
SIMULATION COMPLETE!
============================================================
```

---

## Troubleshooting

### Issue: Module Not Found Errors

**Problem**: `ModuleNotFoundError: No module named 'jax'` or similar

**Solution**:
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall the missing package
pip install jax
```

### Issue: CUDA/GPU Not Detected

**Problem**: Script runs on CPU instead of GPU

**Solution**:
1. Check CUDA installation:
   ```powershell
   nvidia-smi
   ```

2. Reinstall JAX with CUDA:
   ```powershell
   pip uninstall jax jaxlib
   pip install -U "jax[cuda12]"
   ```

3. Set XLA flags (already done in script, but can verify):
   ```powershell
   $env:XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
   ```

### Issue: File Not Found Errors

**Problem**: `FileNotFoundError: Scene file not found`

**Solution**:
1. Verify all paths in the configuration section
2. Check that `mujoco_menagerie` is cloned correctly:
   ```powershell
   cd c:\users\hatem\Desktop\MuJoCo
   git clone https://github.com/google-deepmind/mujoco_menagerie
   ```

### Issue: Video Not Saving

**Problem**: Video rendering fails

**Solution**:
```powershell
# Install/reinstall video dependencies
pip install --upgrade mediapy opencv-python
pip install imageio imageio-ffmpeg
```

### Issue: Slow Performance

**Problem**: Simulation runs very slowly

**Solutions**:
1. **Use GPU**: Ensure CUDA is properly installed
2. **Reduce steps**: Change `N_STEPS = 250` for faster testing
3. **Increase render interval**: Change `RENDER_EVERY = 5` to render fewer frames
4. **Close other applications**: Free up system resources

### Issue: Out of Memory

**Problem**: `RuntimeError: Out of memory`

**Solution**:
```python
# Reduce batch size by modifying environment (if needed)
# Or use CPU instead of GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

---

## Advanced Usage

### Running Multiple Simulations

Create a batch script to run multiple simulations with different commands:

```python
# run_multiple_simulations.py
import subprocess
import os

commands = [
    (1.0, 0.0, 0.0),   # Forward
    (1.0, 0.0, -0.5),  # Forward + turn left
    (0.0, 0.5, 0.0),   # Sideways right
    (-0.3, 0.0, 0.0),  # Backward
]

for i, (x_vel, y_vel, ang_vel) in enumerate(commands):
    print(f"\nRunning simulation {i+1}/{len(commands)}")
    print(f"Commands: x={x_vel}, y={y_vel}, ang={ang_vel}")
    
    # Modify the script to use these values
    # Then run it
    subprocess.run(["python", "run_barkour_local.py"])
```

### Custom Camera Views

Modify the render call in the script:

```python
# Change this line (around line 675):
video = env.render(rollout[::RENDER_EVERY], camera='track')

# To one of these options:
video = env.render(rollout[::RENDER_EVERY], camera='side')
video = env.render(rollout[::RENDER_EVERY], camera='front')
```

### Changing Simulation Length

```python
# Shorter test runs (10 seconds at 50 Hz)
N_STEPS = 500

# Longer runs (20 seconds)
N_STEPS = 1000

# Very short test (2 seconds)
N_STEPS = 100
```

### Higher Quality Video

```python
# Modify the render call to increase resolution:
video = env.render(
    rollout[::RENDER_EVERY], 
    camera='track',
    width=640,   # Default: 240
    height=480   # Default: 320
)
```

### Saving Trajectory Data

Add this code before rendering to save trajectory data:

```python
import pickle

# Save rollout data
trajectory_data = {
    'states': rollout,
    'commands': [X_VEL, Y_VEL, ANG_VEL],
    'n_steps': len(rollout)
}

with open('trajectory_data.pkl', 'wb') as f:
    pickle.dump(trajectory_data, f)
print("Trajectory data saved to trajectory_data.pkl")
```

---

## Environment Variables (Optional)

For better performance, you can set these environment variables:

```powershell
# In PowerShell (temporary, for current session):
$env:XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"

# Or add to your script permanently (already included in run_barkour_local.py)
```

---

## Command Examples

### Walk Forward
```python
X_VEL = 1.0
Y_VEL = 0.0
ANG_VEL = 0.0
```

### Walk Backward
```python
X_VEL = -0.5
Y_VEL = 0.0
ANG_VEL = 0.0
```

### Strafe Right
```python
X_VEL = 0.0
Y_VEL = -0.5
ANG_VEL = 0.0
```

### Circle Left (forward + turn)
```python
X_VEL = 0.8
Y_VEL = 0.0
ANG_VEL = 0.5
```

### Spin in Place
```python
X_VEL = 0.0
Y_VEL = 0.0
ANG_VEL = 0.7
```

---

## Additional Resources

- **MuJoCo Documentation**: https://mujoco.readthedocs.io/
- **MJX Documentation**: https://mujoco.readthedocs.io/en/stable/mjx.html
- **Brax Documentation**: https://github.com/google/brax
- **MuJoCo Menagerie**: https://github.com/google-deepmind/mujoco_menagerie
- **JAX Documentation**: https://jax.readthedocs.io/

---

## Support

If you encounter issues not covered in this guide:

1. Check the error message carefully
2. Verify all file paths are correct
3. Ensure all dependencies are installed
4. Try running with CPU first (remove GPU flags)
5. Check the original tutorial.ipynb for reference

---

**Happy Simulating! 🤖🐕**
