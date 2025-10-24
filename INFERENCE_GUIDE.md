# Barkour Training & Inference Guide

## Overview
The `train_barkour_local.py` script now supports both **training** and **inference** modes!

## Usage Options

### 1. Train New Policy (Default Mode)
Train a policy from scratch and optionally test it after training:
```bash
python train_barkour_local.py
```
- Trains for 5M steps (configurable in script)
- Shows live progress plots
- Saves checkpoints automatically
- After training, asks if you want to test the policy

### 2. Inference-Only Mode
Test a previously trained model without retraining:
```bash
python train_barkour_local.py --inference-only
```
- Loads the trained model from `trained_barkour_policy/`
- Runs 4 different test scenarios
- Generates videos for each test
- No training required!

### 3. No-Prompt Training
Skip the confirmation prompt (useful for scripts):
```bash
python train_barkour_local.py --no-prompt
```

## What Inference Testing Does

When you run inference (either after training or with `--inference-only`), the script will:

1. **Load the trained model** from `trained_barkour_policy/`

2. **Run 4 test scenarios**:
   - Forward at 1.5 m/s
   - Forward-Right diagonal movement
   - Rotate in place (yaw)
   - Forward with left turn

3. **For each scenario**:
   - Simulates 500 steps (~10 seconds)
   - Collects reward statistics
   - Measures distance traveled
   - Generates a video

4. **Saves videos** to `training_logs/` folder:
   - `test_1_Forward_at_1.5_m_s.mp4`
   - `test_2_Forward-Right_diagonal.mp4`
   - `test_3_Rotate_in_place_(yaw).mp4`
   - `test_4_Forward_with_left_turn.mp4`

## Output Files

### Training Outputs
```
training_checkpoints/barkour_joystick/
  └── step_1000000/
  └── step_2000000/
  └── ...

trained_barkour_policy/
  └── params  (final trained model)

training_logs/
  └── training_progress.png
  └── training_summary.txt
```

### Inference Outputs
```
training_logs/
  └── test_1_Forward_at_1.5_m_s.mp4
  └── test_2_Forward-Right_diagonal.mp4
  └── test_3_Rotate_in_place_(yaw).mp4
  └── test_4_Forward_with_left_turn.mp4
```

## Complete Workflow Example

### Step 1: Train the policy
```bash
python train_barkour_local.py
```
- Wait for training to complete (~20-30 minutes on CPU for 5M steps)
- When prompted "Would you like to test the trained policy now? (y/n):", type **y**
- Watch the robot perform different locomotion tests
- Videos will be saved automatically

### Step 2: Test again later
If you want to test the same model again with different commands:
```bash
python train_barkour_local.py --inference-only
```

### Step 3: Use with run_barkour_local.py
The trained model is also compatible with your inference script:
```python
# In run_barkour_local.py, update:
MODEL_PATH = BASE_DIR / "trained_barkour_policy"
```

## Customizing Test Scenarios

Edit the `test_commands` list in the `run_inference_demo()` function:

```python
test_commands = [
    (x_vel, y_vel, ang_vel, "Description"),
    (1.5, 0.0, 0.0, "Fast forward"),
    (0.5, 0.5, 0.0, "Diagonal"),
    # Add your own commands here!
]
```

Where:
- `x_vel`: Forward/backward speed (-0.6 to 1.5 m/s)
- `y_vel`: Sideways speed (-0.8 to 0.8 m/s)
- `ang_vel`: Turning speed (-0.7 to 0.7 rad/s)

## Troubleshooting

### "No trained model found"
- Run training first: `python train_barkour_local.py`
- Or update `OUTPUT_MODEL_PATH` in the script

### "Video rendering failed"
- Make sure imageio is installed: `pip install imageio imageio-ffmpeg`
- Check that ffmpeg is available

### Inference is slow
- This is normal on CPU
- GPU would be faster but JAX on Windows doesn't support it
- Consider reducing `n_steps` in the `run_inference_demo()` function

## Performance Tips

### For Training:
- **CPU**: Set `NUM_TIMESTEPS = 5_000_000` (5M steps, ~30 min)
- **GPU** (if you have Colab): Set `NUM_TIMESTEPS = 100_000_000` (100M, ~6 min on A100)

### For Inference:
- Inference is fast even on CPU (~1-2 minutes for 4 scenarios)
- Each scenario takes 500 steps = ~10 seconds simulated time

## Comparing with Pre-trained Model

To compare your trained model with the pre-trained one:

1. Run inference on pre-trained:
```bash
# In run_barkour_local.py, use original MODEL_PATH
python run_barkour_local.py
```

2. Run inference on your trained model:
```bash
python train_barkour_local.py --inference-only
```

3. Compare the videos side-by-side!

## Next Steps

1. **Train longer**: Increase `NUM_TIMESTEPS` to 10M, 20M, or more
2. **Tune hyperparameters**: Adjust learning rate, batch size, etc.
3. **Modify rewards**: Change the reward function in `BarkourEnv`
4. **Add new behaviors**: Create custom test scenarios
5. **Deploy to Colab**: Use the `train_barkour_colab.ipynb` for GPU training

---

**Questions?** Check the inline comments in `train_barkour_local.py` for detailed explanations.
