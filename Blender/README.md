# Blender Animation Suite for MuJoCo Robots

Complete toolkit for importing and animating MuJoCo robots in Blender using trained policies.

## 📁 Files in This Directory

### Core Scripts

1. **`mujoco_barkour_importer.py`** ⭐
   - Import Barkour VB robot into Blender
   - Parses MuJoCo XML files
   - Imports STL meshes with proper hierarchy
   - **Use**: Run in Blender to import robot model

2. **`convert_policy_to_animation.py`** ⭐
   - Converts trained policy to animation data
   - Runs simulation to extract joint trajectories
   - Outputs JSON file compatible with Blender
   - **Use**: Run in terminal with Python/JAX

3. **`animate_barkour.py`** ⭐
   - Animates imported robot using animation data
   - Applies keyframes to joints and base
   - Supports looping and playback control
   - **Use**: Run in Blender after importing robot

### Helpers

4. **`quick_test_animation.py`**
   - Generates simple test animation (sine wave)
   - No policy required
   - Good for testing the pipeline
   - **Use**: Quick testing without training

### Documentation

5. **`README_ANIMATION.md`** 📖
   - Complete workflow guide
   - Step-by-step instructions
   - Troubleshooting tips
   - **Read this first!**

6. **`FORMAT_COMPARISON.md`** 📖
   - Explains policy vs animation formats
   - Conversion process details
   - Data structure comparison

7. **`README.md`** (this file)
   - Overview of all files
   - Quick start guide

### Data Files

8. **`unitree_g1_run.txt`**
   - Reference animation data (Unitree G1 humanoid)
   - Example of the format
   - Can be used with appropriate importer

9. **`Bar.blend`**
   - Your Blender project file
   - Contains imported robots and scenes

10. **`barkour_animation.txt`** (generated)
    - Animation data from policy
    - Created by `convert_policy_to_animation.py`

## 🚀 Quick Start

### Option A: Use Trained Policy (Realistic Motion)

1. **Convert policy to animation:**
   ```bash
   cd c:\users\hatem\Desktop\MuJoCo\Blender
   python convert_policy_to_animation.py \
     --policy ../mjx_brax_quadruped_policy \
     --output barkour_animation.txt \
     --duration 5.0 --fps 60
   ```

2. **Import robot in Blender:**
   - Open Blender
   - Scripting workspace
   - Open `mujoco_barkour_importer.py`
   - Run Script
   - File → Import → MuJoCo Barkour VB Robot
   - Select `../mujoco_menagerie/google_barkour_vb/barkour_vb.xml`

3. **Animate robot:**
   - Open `animate_barkour.py`
   - Run Script
   - Sidebar (N key) → Barkour Robot tab
   - Load Animation Data → Select `barkour_animation.txt`
   - Press Spacebar to play!

### Option B: Test Animation (Simple Motion)

1. **Generate test animation:**
   ```bash
   python quick_test_animation.py
   ```

2. **Follow steps 2-3 from Option A**
   - Use `barkour_test_animation.txt` instead

## 📊 Workflow Diagram

```
Training                Conversion              Blender
────────                ──────────              ───────

train_barkour_local.py
        │
        ▼
mjx_brax_quadruped_policy ──→ convert_policy_to_animation.py
(Neural Network)                        │
                                        ▼
                              barkour_animation.txt
                                        │
                                        ▼
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        mujoco_barkour_importer.py              animate_barkour.py
                    │                                       │
                    ▼                                       ▼
            Robot in Blender  ──────────────→   Animated Robot!
```

## 🎯 Use Cases

### 1. Visualize Trained Policies
- Train policy with `train_barkour_local.py`
- Convert to animation
- Render high-quality videos in Blender

### 2. Create Demonstrations
- Generate different locomotion gaits
- Combine multiple animations
- Add camera movements and effects

### 3. Debug Training
- Visualize policy behavior at different training checkpoints
- Compare before/after training
- Identify issues in gait patterns

### 4. Export to Other Software
- Create animations in Blender
- Export to Unity, Unreal, Maya, etc.
- Use for presentations, papers, websites

## 🔧 Technical Details

### Supported Robots
- ✅ **Barkour VB** (quadruped) - Fully supported
- 🔄 **Unitree G1** (humanoid) - Reference format available
- 🔄 **Other MuJoCo robots** - Requires adapter scripts

### Requirements
- **Blender**: 3.0+
- **Python**: 3.9+
- **Libraries**: JAX, Brax, MuJoCo
- **Storage**: ~500 MB for models and data

### Animation Format
```json
{
  "Format": "qpos_trajectory",
  "Robot": "google_barkour_vb",
  "JointNames": [...],
  "Frames": [
    {
      "dt": 0.016667,
      "base_pos": [x, y, z],
      "base_quat": [qw, qx, qy, qz],
      "joints": [...]
    }
  ]
}
```

## 📚 Learn More

### Documentation
- `README_ANIMATION.md` - Complete workflow guide
- `FORMAT_COMPARISON.md` - Format details
- Comments in each script

### External Resources
- [Blender Documentation](https://docs.blender.org/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Brax GitHub](https://github.com/google/brax)

## 🐛 Common Issues

### "Policy file not found"
→ Check path to `mjx_brax_quadruped_policy`

### "No robot objects found"
→ Import robot first with `mujoco_barkour_importer.py`

### "STL import not available"
→ Enable STL addon in Blender Preferences

### "JAX/Brax not installed"
→ `pip install jax jaxlib brax mujoco`

## 💡 Tips

1. **Start with test animation** to verify pipeline
2. **Use low FPS (30)** for faster conversion
3. **Enable loop** for continuous playback
4. **Clear scene** before importing multiple robots
5. **Save Blender file often**

## 🎬 Advanced Features

### Multiple Animations
Load different animation data to compare:
- Different training checkpoints
- Different commands (walk, turn, etc.)
- Different robots

### Custom Camera
Set up camera to follow robot:
- Add camera
- Add Track To constraint
- Target: Robot base

### Rendering
Export high-quality videos:
- Set resolution (1920×1080)
- Choose format (MP4, PNG sequence)
- Render → Render Animation

## 📝 File Sizes

| File | Type | Size |
|------|------|------|
| `mujoco_barkour_importer.py` | Script | ~30 KB |
| `convert_policy_to_animation.py` | Script | ~12 KB |
| `animate_barkour.py` | Script | ~15 KB |
| `mjx_brax_quadruped_policy` | Binary | ~2 MB |
| `barkour_animation.txt` | JSON | ~500 KB - 5 MB |
| `Bar.blend` | Blender | ~10 MB - 100 MB |

## ✅ Checklist

Before starting:
- [ ] Blender installed (3.0+)
- [ ] Python environment set up
- [ ] JAX/Brax installed
- [ ] MuJoCo Menagerie downloaded
- [ ] Trained policy available (or use test)

## 🤝 Contributing

Found a bug or have an improvement?
- Add better joint mapping
- Support more robots
- Improve animation interpolation
- Add IK constraints

## 📄 License

MIT License - Free to use and modify

---

**Created**: October 24, 2025  
**Author**: AI Assistant  
**Version**: 1.0.0

**Questions?** Check `README_ANIMATION.md` for detailed instructions!
