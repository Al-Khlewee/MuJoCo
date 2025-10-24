# Complete Blender Animation Workflow - Step by Step

## 🎯 Goal
Animate the Barkour robot in Blender using your trained `mjx_brax_quadruped_policy`.

## 📋 What You Have

### Already in Your Workspace:
- ✅ `mjx_brax_quadruped_policy` - Trained policy file
- ✅ `mujoco_menagerie/google_barkour_vb/` - Robot model files
- ✅ `Blender/` folder with all necessary scripts

### Scripts Created:
1. `convert_policy_to_animation.py` - Convert policy → animation
2. `mujoco_barkour_importer.py` - Import robot into Blender  
3. `animate_barkour.py` - Animate the robot
4. `quick_test_animation.py` - Quick test (no policy needed)

## 🚀 Method 1: Using Your Trained Policy (Recommended)

### Step 1: Convert Policy to Animation Data

Open PowerShell and run:

```powershell
cd c:\users\hatem\Desktop\MuJoCo\Blender

python convert_policy_to_animation.py `
  --policy ..\mjx_brax_quadruped_policy `
  --output barkour_walk_forward.txt `
  --duration 5.0 `
  --fps 60 `
  --vx 1.0 `
  --vy 0.0 `
  --vyaw 0.0
```

**What this does:**
- Loads your trained policy
- Runs a 5-second simulation
- Records joint positions at 60 FPS
- Saves to `barkour_walk_forward.txt`

**Expected output:**
```
======================================================================
MJX/Brax Policy to Blender Animation Converter
======================================================================

Loading policy from: ..\mjx_brax_quadruped_policy
✓ Policy loaded successfully
Creating Barkour environment...
✓ Environment created: barkour_vb_joystick
Building inference function...
✓ Inference function ready

Running simulation...
  Duration: 5.0s
  FPS: 60
  Command: vx=1.00, vy=0.00, vyaw=0.00
  Progress: 100.0% (300 frames)
✓ Simulation complete: 300 frames generated

Saving animation data...
  Output: barkour_walk_forward.txt
  Frames: 300
  Duration: 5.0s
✓ Animation data saved successfully
  File size: 1.2 MB

======================================================================
✅ SUCCESS!
======================================================================
```

### Step 2: Open Blender

1. Launch Blender
2. Switch to **Scripting** workspace (top menu bar)

### Step 3: Import Robot into Blender

**In Blender:**

1. **Load Importer Script:**
   - In Text Editor (Scripting workspace)
   - Click "Open" button
   - Navigate to: `c:\users\hatem\Desktop\MuJoCo\Blender\mujoco_barkour_importer.py`
   - Click "Open Text Block"

2. **Run Importer:**
   - Click the "▶️ Run Script" button
   - Console should show: "MuJoCo Barkour VB Importer registered"

3. **Import Robot:**
   - Go to: **File → Import → MuJoCo Barkour VB Robot (.xml)**
   - Navigate to: `c:\users\hatem\Desktop\MuJoCo\mujoco_menagerie\google_barkour_vb\`
   - Select: `barkour_vb.xml`
   - Settings:
     - ✅ Import Visual Meshes
     - ❌ Import Collision Meshes
     - ❌ Clear Scene
   - Click **"Import MuJoCo Barkour VB Robot"**

4. **Verify Import:**
   - Robot should appear in viewport
   - Check outliner - should see "Body_*" objects
   - You should see the blue/orange Barkour robot

### Step 4: Load Animation

**Still in Blender:**

1. **Load Animator Script:**
   - In Text Editor, click "Open"
   - Navigate to: `c:\users\hatem\Desktop\MuJoCo\Blender\animate_barkour.py`
   - Click "Open Text Block"

2. **Run Animator:**
   - Click "▶️ Run Script" button
   - Console should show: "Barkour Robot Animator registered"

3. **Apply Animation:**
   - Press `N` to open sidebar (if not visible)
   - Click **"Barkour Robot"** tab in sidebar
   - Click **"Load Animation Data"**
   - Select: `barkour_walk_forward.txt`
   - Enable: ✅ Loop Animation
   - Click **"Animate Barkour Robot"**

4. **Watch Animation:**
   - Console shows: "✅ Animation Complete!"
   - Timeline should show 300 frames
   - Press **SPACEBAR** to play animation
   - Robot should walk forward!

## 🧪 Method 2: Quick Test (Without Policy)

If you don't have the trained policy yet or want to test the pipeline:

### Step 1: Generate Test Animation

```powershell
cd c:\users\hatem\Desktop\MuJoCo\Blender
python quick_test_animation.py
```

This creates `barkour_test_animation.txt` with a simple walking pattern.

### Step 2-4: Same as Method 1

Follow Steps 2-4 from Method 1, but use `barkour_test_animation.txt` instead.

## 🎨 Different Motion Commands

Create different locomotion behaviors:

### Walk Forward (default)
```powershell
python convert_policy_to_animation.py `
  --policy ..\mjx_brax_quadruped_policy `
  --output walk_forward.txt `
  --vx 1.0 --vy 0.0 --vyaw 0.0
```

### Walk Backward
```powershell
python convert_policy_to_animation.py `
  --policy ..\mjx_brax_quadruped_policy `
  --output walk_backward.txt `
  --vx -0.5 --vy 0.0 --vyaw 0.0
```

### Strafe Left
```powershell
python convert_policy_to_animation.py `
  --policy ..\mjx_brax_quadruped_policy `
  --output strafe_left.txt `
  --vx 0.0 --vy 0.5 --vyaw 0.0
```

### Turn in Place
```powershell
python convert_policy_to_animation.py `
  --policy ..\mjx_brax_quadruped_policy `
  --output turn_in_place.txt `
  --vx 0.0 --vy 0.0 --vyaw 1.0
```

### Circle Motion
```powershell
python convert_policy_to_animation.py `
  --policy ..\mjx_brax_quadruped_policy `
  --output circle.txt `
  --vx 0.5 --vy 0.0 --vyaw 0.5
```

## 📹 Rendering Video in Blender

After animating:

1. **Switch to Layout workspace**
2. **Set up camera:**
   - Add → Camera
   - Position to view robot
   - Select camera, press Ctrl+Alt+Numpad 0 to set as active

3. **Configure output:**
   - Properties panel → Output Properties
   - Resolution: 1920 × 1080
   - Frame Rate: 60 FPS
   - Output: Choose location and filename
   - File Format: FFmpeg video → H.264

4. **Render:**
   - Render → Render Animation
   - Wait for completion
   - Find video in output location

## 🐛 Troubleshooting

### "Module 'jax' not found"
```powershell
pip install jax jaxlib brax mujoco
```

### "Policy file not found"
Check path:
```powershell
ls ..\mjx_brax_quadruped_policy
```

### "No robot objects found"
- Make sure you imported the robot first
- Check Outliner for "Body_*" objects
- Re-import if needed

### "Animation looks weird"
- Try different command values
- Check console for errors
- Verify FPS matches (usually 60)

### "Blender crashes"
- Close other programs
- Reduce animation duration
- Lower FPS to 30

## 📊 File Overview

```
c:\users\hatem\Desktop\MuJoCo\
│
├── mjx_brax_quadruped_policy          ← Your trained policy
│
├── mujoco_menagerie\
│   └── google_barkour_vb\
│       ├── barkour_vb.xml            ← Robot model
│       └── assets\                    ← STL files
│
└── Blender\
    ├── convert_policy_to_animation.py ← Step 1
    ├── mujoco_barkour_importer.py    ← Step 3
    ├── animate_barkour.py            ← Step 4
    ├── quick_test_animation.py       ← Quick test
    │
    ├── barkour_walk_forward.txt      ← Generated (Step 1)
    ├── barkour_test_animation.txt    ← Generated (test)
    │
    └── Bar.blend                      ← Your Blender file
```

## ✅ Success Checklist

- [ ] Converted policy to animation data
- [ ] Opened Blender
- [ ] Loaded importer script
- [ ] Imported Barkour robot
- [ ] Loaded animator script
- [ ] Applied animation
- [ ] Played animation successfully
- [ ] (Optional) Rendered video

## 🎯 Next Steps

After mastering the basic workflow:

1. **Experiment with commands** - Try different vx/vy/vyaw values
2. **Create sequences** - Combine multiple animations
3. **Add effects** - Lighting, camera movements, environments
4. **Export animations** - For presentations, papers, websites
5. **Try other robots** - Adapt scripts for Unitree G1, etc.

## 📚 Additional Resources

- `README_ANIMATION.md` - Detailed workflow guide
- `FORMAT_COMPARISON.md` - Technical format details
- `README.md` - File overview

## 💡 Tips for Best Results

1. **Start small**: 2-3 seconds, 30 FPS for testing
2. **Check console**: Watch for errors during conversion
3. **Save often**: Save Blender file after importing
4. **Use naming**: Name animations by command (walk_forward, turn_left)
5. **Keep originals**: Don't overwrite animation files

## 🎉 Final Result

You should see:
- Barkour robot walking/moving in Blender
- Smooth animation following your trained policy
- Realistic quadruped locomotion
- Ready to render high-quality videos!

---

**Questions?** Check the documentation files or console output for errors.

**Success?** Try different commands and create amazing robot animations! 🚀
