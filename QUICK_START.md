# BARKOUR QUADRUPED - QUICK REFERENCE

## Installation (One-time setup)

### Option 1: Automated Setup (Windows)
```
Double-click: setup.bat
```

### Option 2: Manual Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# For GPU:
pip install -U "jax[cuda12]"

# For CPU:
pip install -U jax
```

---

## Running the Simulation

### Quick Run (Windows)
```
Double-click: run_simulation.bat
```

### Manual Run
```powershell
.\venv\Scripts\Activate.ps1
python run_barkour_local.py
```

---

## Configuration (edit run_barkour_local.py)

### File Paths (lines 28-31)
```python
BASE_DIR = Path(r"c:\users\hatem\Desktop\MuJoCo")
MODEL_PATH = BASE_DIR / "mjx_brax_quadruped_policy"
MENAGERIE_PATH = BASE_DIR / "mujoco_menagerie" / "google_barkour_vb"
OUTPUT_VIDEO = BASE_DIR / "barkour_simulation.mp4"
```

### Velocity Commands (lines 34-36)
```python
X_VEL = 1.0      # Forward: [0, 1.5], Backward: [-0.6, 0]
Y_VEL = 0.0      # Left: [0, 0.8], Right: [-0.8, 0]
ANG_VEL = -0.5   # CCW: [0, 0.7], CW: [-0.7, 0]
```

### Simulation Settings (lines 38-40)
```python
N_STEPS = 500      # Duration (500 = 10 seconds at 50 Hz)
RENDER_EVERY = 2   # Render every N frames
RANDOM_SEED = 0    # For reproducibility
```

---

## Common Commands

| Behavior | X_VEL | Y_VEL | ANG_VEL |
|----------|-------|-------|---------|
| Forward  | 1.0   | 0.0   | 0.0     |
| Backward | -0.5  | 0.0   | 0.0     |
| Left     | 0.0   | 0.5   | 0.0     |
| Right    | 0.0   | -0.5  | 0.0     |
| Spin CCW | 0.0   | 0.0   | 0.5     |
| Spin CW  | 0.0   | 0.0   | -0.5    |
| Circle   | 0.8   | 0.0   | 0.5     |

---

## Troubleshooting

### GPU Not Detected
```powershell
# Check NVIDIA GPU
nvidia-smi

# Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install -U "jax[cuda12]"
```

### Module Not Found
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Reinstall packages
pip install -r requirements.txt
```

### Files Not Found
```powershell
# Clone MuJoCo Menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie

# Verify policy files exist
dir mjx_brax_quadruped_policy
```

### Slow Performance
- Use GPU instead of CPU
- Reduce N_STEPS (e.g., 250)
- Increase RENDER_EVERY (e.g., 5)

---

## Output

- **Video File**: `barkour_simulation.mp4`
- **Location**: Same directory as script
- **Format**: MP4, 25 FPS
- **Duration**: ~10 seconds for 500 steps

---

## File Structure Required

```
c:\users\hatem\Desktop\MuJoCo\
├── mjx_brax_quadruped_policy/     ← Trained policy (required)
├── mujoco_menagerie/              ← Robot models (required)
│   └── google_barkour_vb/
│       └── scene_mjx.xml
├── run_barkour_local.py           ← Main script
├── requirements.txt               ← Dependencies
├── setup.bat                      ← Setup wizard
├── run_simulation.bat             ← Quick runner
└── README_BARKOUR.md              ← Full documentation
```

---

## Performance Tips

1. **First run**: Takes 30-60 seconds (JIT compilation)
2. **Subsequent runs**: Much faster (~10 seconds)
3. **Use GPU**: 5-10x faster than CPU
4. **Batch simulations**: Run multiple commands in sequence

---

## Getting Help

1. Check `README_BARKOUR.md` for detailed docs
2. Review error messages carefully
3. Verify file paths are correct
4. Check `tutorial.ipynb` for reference

---

**Quick Start**: Double-click `setup.bat`, then `run_simulation.bat`
