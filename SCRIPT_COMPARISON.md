# Script Comparison: Inference vs Training

## Overview

You now have **two complementary scripts** for working with the Barkour quadruped:

### 1. `run_barkour_local.py` - **INFERENCE** (What you currently use)
- **Purpose**: Run a pre-trained policy
- **Input**: Trained model file (`mjx_brax_quadruped_policy`)
- **Output**: Simulation video
- **Time**: ~10-30 seconds
- **Use case**: Test and visualize trained policies

### 2. `train_barkour_local.py` - **TRAINING** (New script)
- **Purpose**: Train a policy from scratch
- **Input**: Environment definition, hyperparameters
- **Output**: Trained model file + checkpoints
- **Time**: 30-60 minutes (GPU) or 4-8 hours (CPU)
- **Use case**: Create your own policies, experiment with training

---

## Side-by-Side Comparison

| Feature | Inference Script | Training Script |
|---------|-----------------|-----------------|
| **File** | `run_barkour_local.py` | `train_barkour_local.py` |
| **Main function** | Load and execute policy | Train policy with PPO |
| **Key operation** | `model.load_params()` | `ppo.train()` |
| **Duration** | Seconds | Minutes to hours |
| **GPU required** | No (but faster with) | Highly recommended |
| **Memory usage** | Low (~2GB) | High (12-24GB VRAM) |
| **Parallelization** | Single rollout | 1000s of parallel envs |
| **Output** | Video (MP4) | Trained model + plots |
| **Interactive** | No | Yes (live plots) |

---

## Code Structure Comparison

### Inference Script Flow
```
1. Load environment
2. Load pre-trained policy ← Uses saved parameters
3. Reset environment
4. For 500 steps:
   - Get action from policy
   - Step environment
   - Store frames
5. Render video
```

### Training Script Flow
```
1. Load environment  
2. Initialize random policy ← Starts from scratch
3. For 100M steps:
   - Collect experience (parallel envs)
   - Compute advantages
   - Update policy (gradient descent)
   - Evaluate performance
   - Save checkpoint
4. Save final model
```

---

## Key Differences in Code

### 1. Model Loading vs Initialization

**Inference:**
```python
# Load pre-trained parameters
params = model.load_params(MODEL_PATH)
inference_fn = make_inference_fn(params)
```

**Training:**
```python
# Train from scratch
make_inference_fn, params, metrics = ppo.train(
    environment=env,
    num_timesteps=100_000_000,
    # ... many hyperparameters ...
)
```

### 2. Environment Usage

**Inference:**
```python
# Single environment instance
env = envs.get_environment('barkour')

# Single rollout
for i in range(500):
    action, _ = inference_fn(state.obs, rng)
    state = env.step(state, action)
```

**Training:**
```python
# Many parallel environments (e.g., 4096)
train_fn = functools.partial(
    ppo.train,
    num_envs=4096,  # ← Runs 4096 envs in parallel!
    # ...
)

# Training automatically manages parallel rollouts
```

### 3. Observation Handling

**Inference:**
```python
# Apply normalization from saved params
normalizer_params = params[0]  # Extract normalizer
def preprocess_observations_fn(obs, rng):
    return normalize_fn(obs, normalizer_params)
```

**Training:**
```python
# Build normalization statistics during training
train_fn = functools.partial(
    ppo.train,
    normalize_observations=True,  # ← Learns mean/std online
    # ...
)
```

### 4. Output

**Inference:**
```python
# Render and save video
video = env.render(rollout, camera='track')
imageio.mimwrite('barkour_simulation.mp4', video, fps=25)
```

**Training:**
```python
# Save trained model
model.save_params(OUTPUT_MODEL_PATH, params)

# Save checkpoints during training
def save_checkpoint(step, make_policy, params):
    orbax_checkpointer.save(path, params, force=True)
```

---

## Shared Components

Both scripts share:
- ✅ **BarkourEnv class** - Same environment definition
- ✅ **Reward functions** - Same reward structure
- ✅ **Network architecture** - Same (128,128,128,128) layers
- ✅ **Domain randomization** - Training uses it, inference doesn't need it

---

## When to Use Each Script

### Use `run_barkour_local.py` (Inference) when:
- ✅ You want to test a trained policy quickly
- ✅ You want to visualize robot behavior
- ✅ You're experimenting with different velocity commands
- ✅ You need to demo the robot
- ✅ You don't have much time or GPU resources

### Use `train_barkour_local.py` (Training) when:
- ✅ You want to train your own policy from scratch
- ✅ You want to experiment with different rewards
- ✅ You want to modify the environment
- ✅ You need to understand the learning process
- ✅ You have time and GPU resources
- ✅ You want to reproduce Colab notebook results

---

## Workflow: Typical Usage

### Standard Workflow
```
1. Use TRAINING script to create policy
   └─→ trained_barkour_policy/
   
2. Use INFERENCE script to test policy  
   └─→ barkour_simulation.mp4
```

### Experimentation Workflow
```
1. Modify rewards in TRAINING script
2. Train new policy (30-60 min)
3. Test with INFERENCE script
4. If not satisfied, go back to step 1
```

### Quick Testing Workflow
```
1. Use pre-trained policy with INFERENCE script
2. Test different velocity commands
3. Generate demo videos
```

---

## Resource Requirements

### Inference Script
```
CPU: Any modern CPU
RAM: 4-8 GB
GPU: Optional (makes it faster)
Time: 10-30 seconds
Storage: ~1 MB (video output)
```

### Training Script
```
CPU: Multi-core recommended
RAM: 16-32 GB
GPU: NVIDIA with 12+ GB VRAM (highly recommended)
Time: 30 min - 6 hours depending on hardware
Storage: ~500 MB (checkpoints + logs)
```

---

## Converting Between Scripts

### From Training Output to Inference Input

After training completes:

1. Training produces: `trained_barkour_policy/`
2. Update inference script:
   ```python
   MODEL_PATH = BASE_DIR / "trained_barkour_policy"
   ```
3. Run inference as normal

### Testing Training Checkpoints

You can test intermediate checkpoints:

```python
# In run_barkour_local.py
MODEL_PATH = BASE_DIR / "training_checkpoints" / "barkour_joystick" / "step_50000000"
```

---

## Modification Guide

### Want to change velocity ranges?

**Inference**: Change X_VEL, Y_VEL, ANG_VEL at runtime
**Training**: Modify `sample_command()` in environment

### Want to change rewards?

**Inference**: No effect (policy already trained)
**Training**: Modify `get_config()` and retrain

### Want different robot behavior?

**Inference**: Use different pre-trained policy
**Training**: Modify rewards and retrain

### Want to use different robot?

**Both scripts**: Change MENAGERIE_PATH to different robot model

---

## Summary

| Aspect | Inference Script | Training Script |
|--------|-----------------|-----------------|
| **Analogy** | Playing a video game with AI | Teaching AI to play |
| **Complexity** | Simple (~650 lines) | Complex (~900 lines) |
| **Flexibility** | Test only | Create + test |
| **Speed** | Fast | Slow |
| **Learning** | None | PPO algorithm |
| **Dependencies** | Trained model | None (trains from scratch) |

---

## Quick Commands

```powershell
# Run inference (fast)
python run_barkour_local.py

# Train from scratch (slow but creates your own model)
python train_barkour_local.py

# Both together:
# 1. Train
python train_barkour_local.py
# 2. Update MODEL_PATH in run_barkour_local.py
# 3. Test
python run_barkour_local.py
```

---

**You now have the complete toolkit for both using AND creating quadruped policies!** 🎉
