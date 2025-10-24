# Training Configuration for Barkour Quadruped

This file explains how to use the training script and customize it for your hardware.

## Quick Start

```powershell
# Basic training (will take ~30-60 minutes on good GPU)
python train_barkour_local.py
```

## Hardware Requirements

### Minimum (CPU only)
- **RAM**: 16GB
- **Time**: 4-8 hours for 10M steps
- **Note**: Set `NUM_TIMESTEPS = 10_000_000` for faster testing

### Recommended (GPU)
- **GPU**: NVIDIA RTX 3060 or better with 12GB+ VRAM
- **RAM**: 16GB system RAM
- **Time**: 30-60 minutes for 100M steps

### Optimal (High-end GPU)
- **GPU**: RTX 4080/4090 or A100
- **RAM**: 32GB system RAM
- **Time**: 15-30 minutes for 100M steps

## Configuration Options

Edit these variables at the top of `train_barkour_local.py`:

### Training Duration
```python
NUM_TIMESTEPS = 100_000_000  # 100M steps (full training)
NUM_TIMESTEPS = 50_000_000   # 50M steps (faster, still good)
NUM_TIMESTEPS = 10_000_000   # 10M steps (quick test)
```

### GPU Memory Settings
If you get OUT_OF_MEMORY errors:

```python
# Reduce these values:
NUM_ENVS = 2048        # Instead of 4096 (uses less VRAM)
NUM_MINIBATCHES = 16   # Instead of 32
BATCH_SIZE = 128       # Instead of 256
```

### CPU Training
```python
USE_GPU = False          # Force CPU mode
NUM_ENVS = 256          # Much fewer environments
NUM_TIMESTEPS = 10_000_000  # Shorter training
```

## Training Process

The script will:

1. **Setup** (~10 seconds)
   - Create checkpoint directories
   - Initialize environment
   - Configure JAX/GPU

2. **JIT Compilation** (~30-60 seconds)
   - First evaluation compiles the code
   - Only happens once at start

3. **Training Loop** (main time)
   - Runs PPO algorithm
   - Saves checkpoints every N steps
   - Shows progress plots in real-time
   - Updates terminal with metrics

4. **Saving** (~5 seconds)
   - Saves final model
   - Saves training plots
   - Saves training summary

## Outputs

After training completes:

```
c:\users\hatem\Desktop\MuJoCo\
├── trained_barkour_policy/        ← Your trained model (use with inference script)
├── training_checkpoints/
│   └── barkour_joystick/
│       ├── step_10000000/         ← Checkpoint at 10M steps
│       ├── step_20000000/         ← Checkpoint at 20M steps
│       └── ...
└── training_logs/
    ├── training_progress.png      ← Training curves
    └── training_summary.txt       ← Final statistics
```

## Using Your Trained Model

After training, update `run_barkour_local.py`:

```python
# Change this line:
MODEL_PATH = BASE_DIR / "mjx_brax_quadruped_policy"  # Old

# To this:
MODEL_PATH = BASE_DIR / "trained_barkour_policy"     # Your trained model
```

Then run:
```powershell
python run_barkour_local.py
```

## Monitoring Training

### Real-time Plots
Two plots will update during training:
- **Left**: Episode reward over time (should increase)
- **Right**: Training speed (steps per second)

### Terminal Output
Shows every evaluation:
```
Step 10,000,000 / 100,000,000 (10.0%)
Reward: 25.43 ± 3.21
Elapsed time: 5.2 minutes
ETA: 47.3 minutes
```

### Checkpoints
Models are automatically saved at each evaluation checkpoint.

## Stopping and Resuming

### Stopping
- Press **Ctrl+C** to stop training gracefully
- The last checkpoint will be saved

### Resuming (Advanced)
To resume from a checkpoint, modify the training script:

```python
train_fn = functools.partial(
    ppo.train,
    # ... other params ...
    restore_checkpoint_path=CHECKPOINT_DIR / "step_50000000"  # Add this line
)
```

## Expected Results

### Training Progress
- **Steps 0-20M**: Reward increases from ~0 to ~15
- **Steps 20-50M**: Reward increases to ~25-30
- **Steps 50-100M**: Reward plateaus at ~30-40
- **Final reward**: 35-45 (similar to Colab notebook)

### Success Criteria
Your policy is well-trained if:
- Final reward > 30
- Robot completes 500-step episodes
- Walks smoothly with various velocity commands

## Troubleshooting

### Out of Memory
```python
# Reduce these settings:
NUM_ENVS = 2048
NUM_MINIBATCHES = 16
BATCH_SIZE = 128
```

### Training Too Slow
- Ensure GPU is being used (check terminal output)
- Close other programs
- Reduce `NUM_TIMESTEPS` for testing

### Poor Final Performance
- Train longer (increase `NUM_TIMESTEPS`)
- Check that reward is increasing in plots
- Ensure domain randomization is working

### GPU Not Detected
```powershell
# Check CUDA installation
nvidia-smi

# Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install -U "jax[cuda12]"
```

## Comparison with Google Colab

| Metric | Colab (A100) | Your Machine (estimate) |
|--------|--------------|-------------------------|
| Training time | ~6 minutes | 30-60 minutes (RTX 3080) |
| Steps per second | ~300,000 | 50,000-150,000 |
| GPU Memory | 40GB | 12-24GB |
| Final reward | ~35-40 | Should be similar |

## Advanced: Custom Rewards

To modify the reward function, edit the `get_config()` function:

```python
def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    tracking_lin_vel=1.5,     # Increase for better tracking
                    tracking_ang_vel=0.8,
                    orientation=-5.0,          # Increase penalty for falling
                    # ... other rewards ...
                )
            ),
            tracking_sigma=0.25,
        )
    )
    return default_config
```

## Tips for Best Results

1. **Use GPU**: Training on CPU is 10-20x slower
2. **Start small**: Test with 10M steps first
3. **Monitor progress**: Watch the plots and terminal
4. **Save checkpoints**: They're saved automatically
5. **Be patient**: Full training takes time but results are worth it

## Getting Help

If training fails:
1. Check error messages in terminal
2. Verify GPU is detected
3. Try reducing memory settings
4. Start with shorter training (10M steps)
5. Check `training_logs/` for details

---

**Ready to train?** Run:
```powershell
python train_barkour_local.py
```

Good luck! 🚀🤖
