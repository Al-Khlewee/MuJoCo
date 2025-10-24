# Barkour Robot Animation in Blender

Complete workflow for animating the Barkour quadruped robot in Blender using trained MJX/Brax policies.

## 📋 Overview

This workflow allows you to:
1. **Import** the Barkour robot model into Blender
2. **Convert** a trained policy into animation data
3. **Animate** the robot in Blender with the simulation data

## 🔧 Prerequisites

### Software Requirements
- **Blender** 3.0 or later ([Download](https://www.blender.org/download/))
- **Python** 3.9+ with the following packages:
  ```bash
  pip install jax jaxlib brax mujoco
  ```

### Files Needed
- `mujoco_barkour_importer.py` - Import Barkour into Blender
- `convert_policy_to_animation.py` - Convert policy to animation data
- `animate_barkour.py` - Animate robot in Blender
- Trained policy file (e.g., `mjx_brax_quadruped_policy`)
- MuJoCo Menagerie Barkour model (`mujoco_menagerie/google_barkour_vb/`)

## 📖 Complete Workflow

### Step 1: Convert Policy to Animation Data

First, convert your trained policy into animation data that Blender can use.

```bash
cd c:\users\hatem\Desktop\MuJoCo\Blender

python convert_policy_to_animation.py \
  --policy ../mjx_brax_quadruped_policy \
  --output barkour_animation.txt \
  --duration 5.0 \
  --fps 60 \
  --vx 1.0 \
  --vy 0.0 \
  --vyaw 0.0
```

**Parameters:**
- `--policy`: Path to your trained policy file
- `--output`: Output animation data file name
- `--duration`: Simulation duration in seconds (default: 5.0)
- `--fps`: Frames per second for output (default: 60)
- `--vx`: Forward velocity command (default: 1.0)
- `--vy`: Lateral velocity command (default: 0.0)
- `--vyaw`: Yaw rotation rate (default: 0.0)

**Output:**
- Creates `barkour_animation.txt` with joint positions over time
- Format compatible with Blender animation script

### Step 2: Import Robot into Blender

1. **Open Blender**
2. **Open Scripting Workspace**:
   - Top menu: Click "Scripting"
3. **Load Importer Script**:
   - Click "Open" button in Text Editor
   - Navigate to: `c:\users\hatem\Desktop\MuJoCo\Blender\mujoco_barkour_importer.py`
   - Click "Open Text Block"
4. **Run Importer**:
   - Click "Run Script" button (▶️)
   - This registers the importer addon

5. **Import Robot**:
   - Method A - Via Menu:
     - File → Import → MuJoCo Barkour VB Robot (.xml)
     - Navigate to: `mujoco_menagerie/google_barkour_vb/barkour_vb.xml`
     - Click "Import MuJoCo Barkour VB Robot"
   
   - Method B - Via Sidebar:
     - Press `N` to open sidebar
     - Click "Barkour Robot" tab
     - Click "Import from XML"
     - Select `barkour_vb.xml`

6. **Import Options**:
   - ✅ Import Visual Meshes (recommended)
   - ❌ Import Collision Meshes (optional)
   - ❌ Clear Scene (optional)

**Result:**
- Barkour robot appears in viewport
- Robot parts organized as empties and meshes
- Ready for animation

### Step 3: Animate Robot

1. **Load Animator Script** (in Blender):
   - In Scripting workspace
   - Click "Open" in Text Editor
   - Navigate to: `c:\users\hatem\Desktop\MuJoCo\Blender\animate_barkour.py`
   - Click "Open Text Block"

2. **Run Animator Script**:
   - Click "Run Script" button (▶️)
   - This registers the animator addon

3. **Load Animation Data**:
   - Press `N` to open sidebar
   - Click "Barkour Robot" tab
   - Click "Load Animation Data"
   - Select `barkour_animation.txt` (created in Step 1)
   - Enable "Loop Animation" (optional)
   - Click "Animate Barkour Robot"

4. **Play Animation**:
   - Press `SPACEBAR` to play
   - Or use playback controls in sidebar
   - Adjust timeline as needed

## 📁 File Structure

```
c:\users\hatem\Desktop\MuJoCo\
├── Blender/
│   ├── mujoco_barkour_importer.py    ← Import robot into Blender
│   ├── animate_barkour.py            ← Animate robot
│   ├── convert_policy_to_animation.py ← Convert policy to animation
│   ├── barkour_animation.txt         ← Generated animation data
│   └── Bar.blend                     ← Your Blender file
│
├── mujoco_menagerie/
│   └── google_barkour_vb/
│       ├── barkour_vb.xml            ← Robot model
│       └── assets/                   ← STL meshes
│
└── mjx_brax_quadruped_policy         ← Trained policy (binary)
```

## 🎬 Animation Data Format

The generated animation file (`barkour_animation.txt`) contains:

```json
{
  "Format": "qpos_trajectory",
  "Robot": "google_barkour_vb",
  "Duration": 5.0,
  "FPS": 60,
  "JointNames": ["floating_base", "fl_hx", "fl_hy", ...],
  "Command": [1.0, 0.0, 0.0],
  "Frames": [
    {
      "dt": 0.02,
      "base_pos": [0.0, 0.0, 0.3],
      "base_quat": [1.0, 0.0, 0.0, 0.0],
      "joints": [0.1, -0.5, 1.2, ...]
    },
    ...
  ]
}
```

## 🎯 Joint Mapping

Barkour VB has 12 actuated joints (4 legs × 3 joints):

### Leg Joint Structure
Each leg has 3 joints:
- **hx**: Hip abduction/adduction (X-axis)
- **hy**: Hip flexion/extension (Y-axis)
- **kn**: Knee flexion/extension (Y-axis)

### Joint Order
1. **Front Left (fl)**: `fl_hx`, `fl_hy`, `fl_kn`
2. **Front Right (fr)**: `fr_hx`, `fr_hy`, `fr_kn`
3. **Hind Left (hl)**: `hl_hx`, `hl_hy`, `hl_kn`
4. **Hind Right (hr)**: `hr_hx`, `hr_hy`, `hr_kn`

Plus a floating base for position and orientation.

## 🔄 Different Animation Commands

Create different locomotion behaviors:

### Walk Forward
```bash
python convert_policy_to_animation.py \
  --policy ../mjx_brax_quadruped_policy \
  --output walk_forward.txt \
  --vx 1.0 --vy 0.0 --vyaw 0.0
```

### Walk Sideways
```bash
python convert_policy_to_animation.py \
  --policy ../mjx_brax_quadruped_policy \
  --output walk_sideways.txt \
  --vx 0.0 --vy 1.0 --vyaw 0.0
```

### Turn in Place
```bash
python convert_policy_to_animation.py \
  --policy ../mjx_brax_quadruped_policy \
  --output turn_in_place.txt \
  --vx 0.0 --vy 0.0 --vyaw 1.0
```

### Circle Motion
```bash
python convert_policy_to_animation.py \
  --policy ../mjx_brax_quadruped_policy \
  --output circle.txt \
  --vx 0.5 --vy 0.0 --vyaw 0.5
```

## 🎨 Blender Tips

### Camera Setup
1. Add a camera: Add → Camera
2. Position camera to view robot
3. Set camera to follow robot:
   - Select camera
   - Add constraint: Track To
   - Target: Base empty of robot

### Lighting
1. Add lights for better visualization
2. Recommended: HDRI environment lighting
   - Shading workspace
   - World properties → Environment Texture

### Rendering
1. Set output properties:
   - Resolution: 1920×1080
   - Frame rate: 60 FPS
   - Format: MP4 or PNG sequence
2. Render animation: Render → Render Animation

### Materials
- Robot is imported with colored materials
- Customize in Shading workspace
- Adjust metallic/roughness for realistic look

## 🐛 Troubleshooting

### "Policy file not found"
- Check path to `mjx_brax_quadruped_policy`
- Use absolute paths if relative paths fail

### "No robot objects found in scene"
- Make sure you imported the robot first
- Check that objects start with "Body_"

### "Import error: STL format not available"
- Enable STL addon in Blender:
  - Edit → Preferences → Add-ons
  - Search "STL"
  - Enable "Import-Export: STL format"

### Animation looks wrong
- Check joint mapping in `animate_barkour.py`
- Verify robot was imported correctly
- Try re-importing with visual meshes only

### Simulation crashes
- Reduce duration: `--duration 3.0`
- Lower FPS: `--fps 30`
- Check JAX/Brax installation

## 📚 References

- **MuJoCo Menagerie**: https://github.com/google-deepmind/mujoco_menagerie
- **Brax Documentation**: https://github.com/google/brax
- **Blender API**: https://docs.blender.org/api/current/

## 💡 Advanced Usage

### Export for Other Software

The animation data format can be adapted for other 3D software:
- **Unity**: Convert to FBX with animation
- **Unreal Engine**: Export as Alembic
- **Maya**: Use Python API similar to Blender

### Custom Joint Control

Edit `animate_barkour.py` to customize:
- Joint rotation axes
- Animation curves (linear, bezier, etc.)
- Constraints between joints
- IK solvers for legs

### Multiple Robots

Animate multiple robots:
1. Import robot multiple times
2. Rename object hierarchies
3. Apply different animation data to each

## ✅ Summary

**Complete workflow:**
1. ✅ Train policy → `mjx_brax_quadruped_policy`
2. ✅ Convert to animation → `python convert_policy_to_animation.py`
3. ✅ Import robot → Run `mujoco_barkour_importer.py` in Blender
4. ✅ Animate → Run `animate_barkour.py` in Blender
5. ✅ Render → Create final video

**Result:** Animated Barkour robot performing trained locomotion behaviors in Blender! 🎉

---

**Created**: October 24, 2025  
**Author**: AI Assistant  
**License**: MIT
