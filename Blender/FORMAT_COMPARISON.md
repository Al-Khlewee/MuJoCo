# Format Comparison: Policy File vs Animation Data

## Overview
This document explains the conversion from a trained MJX/Brax policy to Blender-compatible animation data.

## File Types

### 1. `mjx_brax_quadruped_policy` (Binary Policy File)
- **Type**: Binary file (Python pickle/Flax format)
- **Content**: Neural network weights and parameters
- **Size**: ~500 KB - 5 MB
- **Use**: Run policy to generate robot actions

**Structure:**
```python
{
  'params': {
    'policy': {
      'hidden_layers': [...],  # Neural network weights
      'output_layer': [...],
    },
    'value_function': {...},
  },
  'normalizer_params': {...},
}
```

**Cannot be directly used in Blender!**

---

### 2. `barkour_animation.txt` (Animation Data)
- **Type**: JSON text file
- **Content**: Joint positions over time
- **Size**: 100 KB - 10 MB (depends on duration/FPS)
- **Use**: Animate robot in Blender

**Structure:**
```json
{
  "Format": "qpos_trajectory",
  "Robot": "google_barkour_vb",
  "Duration": 5.0,
  "FPS": 60,
  "JointNames": ["floating_base", "fl_hx", ...],
  "Command": [1.0, 0.0, 0.0],
  "Frames": [
    {
      "dt": 0.016667,
      "base_pos": [0.0, 0.0, 0.3],
      "base_quat": [1.0, 0.0, 0.0, 0.0],
      "joints": [0.1, -0.5, 1.2, ...]
    },
    ...
  ]
}
```

**Can be directly loaded in Blender!**

---

### 3. `unitree_g1_run.txt` (Reference Format)
- **Type**: JSON text file
- **Content**: Direct joint position format
- **Use**: Same as animation data, different robot

**Structure:**
```json
{
  "Format": "direct_qpos",
  "JointNames": [...],
  "Labels": [...],
  "Loop": "wrap",
  "Frames": [
    [0.01666, x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
  ]
}
```

---

## Conversion Process

```
┌─────────────────────────────────┐
│  mjx_brax_quadruped_policy      │
│  (Binary Neural Network)        │
└─────────────────────────────────┘
                │
                │ convert_policy_to_animation.py
                │
                ├─► Load policy weights
                ├─► Create Brax environment
                ├─► Run simulation loop:
                │     - Get observations
                │     - Policy → Actions
                │     - Step physics
                │     - Record joint positions
                │
                ▼
┌─────────────────────────────────┐
│  barkour_animation.txt          │
│  (JSON Joint Trajectories)      │
└─────────────────────────────────┘
                │
                │ animate_barkour.py (Blender)
                │
                ├─► Parse JSON
                ├─► Find robot objects
                ├─► Apply keyframes:
                │     - Base position/rotation
                │     - Joint rotations
                │
                ▼
┌─────────────────────────────────┐
│  Animated Robot in Blender      │
│  (Keyframed Animation)          │
└─────────────────────────────────┘
```

## Data Transformation

### Policy Execution
```python
# Input: Robot state (observations)
obs = [joint_positions, joint_velocities, base_orientation, ...]

# Policy neural network
action = policy_network(obs)

# Output: Joint torques/positions
action = [torque_fl_hx, torque_fl_hy, ...]
```

### Simulation Step
```python
# Apply actions to physics
new_state = environment.step(state, action)

# Extract joint positions
qpos = new_state.pipeline_state.q
# qpos = [base_x, base_y, base_z, qw, qx, qy, qz, joint_angles...]
```

### Frame Recording
```python
frame = {
    "dt": 0.02,
    "base_pos": qpos[0:3],     # [x, y, z]
    "base_quat": qpos[3:7],    # [qw, qx, qy, qz]
    "joints": qpos[7:],        # All joint angles
}
```

## Key Differences

| Aspect | Policy File | Animation File |
|--------|-------------|----------------|
| Format | Binary (Pickle) | Text (JSON) |
| Content | NN Weights | Joint Positions |
| Size | ~1-5 MB | ~0.1-10 MB |
| Purpose | Generate actions | Replay motion |
| Editable | No | Yes |
| Platform | Python/JAX | Any (JSON) |

## Why Convert?

### 1. **Platform Independence**
- Policy requires JAX/Brax (Python only)
- Animation data works in any software (Blender, Unity, etc.)

### 2. **Determinism**
- Policy may have randomness
- Animation is exact replay

### 3. **Performance**
- Policy requires neural network evaluation
- Animation is simple keyframe playback

### 4. **Editing**
- Can't edit neural network weights easily
- Can manually edit joint positions in JSON

### 5. **Visualization**
- Policy needs physics simulator
- Animation just needs 3D software

## Example: Same Motion, Different Formats

### Policy (Conceptual)
```python
# Hidden in neural network weights
if forward_command:
    move_legs_in_walking_pattern()
```

### Animation (Explicit)
```json
{
  "Frames": [
    {"joints": [0.0, -0.3, 0.6, ...]},  # Frame 1
    {"joints": [0.1, -0.4, 0.7, ...]},  # Frame 2
    {"joints": [0.2, -0.5, 0.8, ...]},  # Frame 3
    ...
  ]
}
```

## Format Comparison Table

| Feature | `mjx_brax_quadruped_policy` | `barkour_animation.txt` | `unitree_g1_run.txt` |
|---------|----------------------------|------------------------|---------------------|
| Robot | Barkour VB | Barkour VB | Unitree G1 |
| Format | Binary | JSON (trajectory) | JSON (direct) |
| Joints | 12 (quadruped) | 12 (quadruped) | 37 (humanoid) |
| Base | Floating | Floating | Floating |
| Editable | ❌ No | ✅ Yes | ✅ Yes |
| Blender | ❌ No | ✅ Yes | ✅ Yes |

## Summary

**To animate in Blender, you must convert:**
- ❌ Cannot use: `mjx_brax_quadruped_policy` directly
- ✅ Must use: `barkour_animation.txt` (generated from policy)
- ✅ Reference: `unitree_g1_run.txt` (same idea, different robot)

**Conversion tool:**
```bash
python convert_policy_to_animation.py \
  --policy mjx_brax_quadruped_policy \
  --output barkour_animation.txt
```

This converts the "brain" (policy) into "recorded motion" (animation data).
