# Barkour Animation Troubleshooting Guide

## Issue: "Robotic parts weren't correctly organized when it played"

This guide helps diagnose and fix common animation issues in Blender.

---

## Common Problems & Solutions

### 1. Parts Flying Apart / Not Connected

**Symptom**: Robot parts separate or float away during animation.

**Cause**: Bodies are not properly parented in hierarchy.

**Solution**:
1. Check the Outliner (top-right panel in Blender)
2. Verify the hierarchy looks like this:
   ```
   Body_torso (root)
   ├── Body_leg_front_left
   │   └── Body_upper_leg_front_left
   │       └── Body_lower_leg_front_left
   ├── Body_leg_front_right
   │   └── Body_upper_leg_front_right
   │       └── Body_lower_leg_3
   ├── Body_leg_hind_left
   │   └── Body_upper_leg_hind_left
   │       └── Body_lower_leg_2
   └── Body_leg_hind_right
       └── Body_upper_leg_hind_right
           └── Body_lower_leg_4
   ```
3. If the hierarchy is wrong, **re-import** the robot using `mujoco_barkour_importer.py`

---

### 2. Parts Rotated Incorrectly / Strange Angles

**Symptom**: Legs point in wrong directions, joints bend backwards, or robot looks deformed.

**Cause**: Initial rotations from import are interfering with animation rotations.

**Solution**:
The updated `animate_barkour.py` now includes `initialize_joint_rotations()` which resets all joint bodies to zero rotation before animating. This should be automatic.

**Manual Fix** (if needed):
1. Select all Body_* empties (except Body_torso)
2. Press `Alt+R` to clear rotations
3. Run the animation script again

---

### 3. Base/Torso Not Moving

**Symptom**: Legs animate but the body stays in place.

**Cause**: Base object not identified correctly.

**Check Console Output**:
```
Found: Body_torso
  → Identified as base object
```

If you don't see this, the script couldn't find the torso.

**Solution**:
- Ensure an object named `Body_torso` exists
- Or manually rename your root body to `Body_torso`

---

### 4. Some Joints Work, Others Don't

**Symptom**: Some legs animate correctly, others stay stiff.

**Cause**: Body name mismatch between animation script and imported objects.

**Check Console Output**:
```
Verifying joint body mappings...
  Found 12/12 joint bodies:
    fl_hx → leg_front_left ✓
    fl_hy → upper_leg_front_left ✓
    ...
  ✓ All joint bodies found!
```

If you see **MISSING** warnings:
```
⚠ WARNING: 4 bodies not found:
  fr_kn → lower_leg_3 ✗ MISSING
```

**Solution**:
1. Check what bodies actually exist in Outliner
2. Update the `get_joint_to_body_mapping()` function in `animate_barkour.py`
3. Match the body names to what's actually in your scene

**Common Name Variations**:
- Lower legs might be named: `lower_leg_front_left` vs `lower_leg` vs `lower_leg_1`
- Check the console during import to see exact names created

---

### 5. Animation Plays But Robot Doesn't Move

**Symptom**: Timeline plays, but robot stays frozen.

**Possible Causes**:

**A) No Keyframes Applied**
- Check: Select Body_torso → Graph Editor → should see animation curves
- Solution: Run the animation script again

**B) Wrong FPS Setting**
- Animation data is 60 FPS, but scene might be 24 FPS
- Solution: Script now sets `bpy.context.scene.render.fps = 60` automatically

**C) Timeline Not Set Correctly**
- Check: Timeline should show frames 1-300
- Solution: Script sets `frame_start=1, frame_end=num_frames`

---

### 6. Robot Moves But Looks Jittery/Choppy

**Symptom**: Animation is not smooth.

**Cause**: Interpolation mode is set to constant instead of linear.

**Solution**:
1. Select all Body_* objects
2. Open Graph Editor
3. Select all keyframes (press `A`)
4. Press `T` → Linear
5. Or press `T` → Bezier for smooth interpolation

---

### 7. Coordinate System Issues

**Symptom**: Robot is upside down, sideways, or moving in wrong direction.

**Cause**: MuJoCo uses Z-up, Blender uses Z-up but with different conventions.

**Current Status**: The conversion script should handle this correctly.

**If Still Wrong**:
Check the `position_to_blender()` and `quaternion_to_blender()` functions in the importer.

---

## Diagnostic Checklist

Run through this checklist to identify the issue:

### ✓ Pre-Animation Checks

- [ ] **Robot imported successfully**
  - See Body_* empties in Outliner
  - Meshes visible in 3D viewport
  
- [ ] **Proper hierarchy exists**
  - Body_torso at root
  - Legs properly nested (leg → upper_leg → lower_leg)
  
- [ ] **Animation file generated**
  - `barkour_policy_animation.txt` exists
  - File size ~7000+ lines (300 frames)
  - JSON format is valid

### ✓ During Animation Script

Watch the console output for:

```
Searching for robot objects...
  Found: Body_torso
  Found: Body_leg_front_left
  Found: Body_upper_leg_front_left
  ...
Total objects found: 17

Verifying joint body mappings...
  Found 12/12 joint bodies:
    fl_hx → leg_front_left ✓
  ✓ All joint bodies found!

Applying animation...
  Total frames: 300
  FPS: 60
  
  Initializing joint rotations to rest pose...
  Initialized 12 joint bodies
  
    Progress: 16.7%
    Progress: 33.3%
    ...
    Progress: 100.0%
✓ Animation applied successfully
```

**Red Flags**:
- "ERROR: No robot objects found" → Re-import robot
- "WARNING: X bodies not found" → Name mismatch, check mapping
- "Failed to parse animation data" → Regenerate animation file

### ✓ Post-Animation Checks

- [ ] **Timeline set correctly**
  - Frame range: 1-300
  - FPS: 60
  
- [ ] **Keyframes exist**
  - Select Body_torso → see orange keyframe markers on timeline
  - Select Body_leg_front_left → see keyframe markers
  
- [ ] **Animation plays**
  - Press SPACEBAR
  - See frame counter advancing
  - Robot should move!

---

## Advanced Debugging

### Check Keyframe Data in Console

Add this at the end of `animate_barkour.py`'s execute():

```python
# Debug: Print keyframe info
for obj in bpy.data.objects:
    if obj.animation_data and obj.animation_data.action:
        print(f"{obj.name}: {len(obj.animation_data.action.fcurves)} fcurves")
```

Expected output:
```
Body_torso: 7 fcurves (3 location + 4 quaternion)
Body_leg_front_left: 1 fcurves (1 rotation axis)
Body_upper_leg_front_left: 1 fcurves
...
```

### Check Actual Rotation Values

Select a leg body, scrub timeline to frame 50, check properties panel:
- Rotation mode should be: **XYZ Euler**
- Rotation values should be non-zero (e.g., Y: 0.463 radians = ~26.5°)

### Export Animation Data for Inspection

Add to `convert_policy_to_animation.py`:

```python
# After generating frames
print(f"\nFirst frame joint angles:")
print(animation_data['Frames'][0]['joints'])
print(f"\nFrame 50 joint angles:")
print(animation_data['Frames'][50]['joints'])
```

Check that joint angles are changing over time (not all zeros).

---

## Quick Fix: Start Fresh

If nothing works, try a clean slate:

1. **Delete everything**:
   ```python
   # In Blender Python console
   bpy.ops.object.select_all(action='SELECT')
   bpy.ops.object.delete()
   ```

2. **Re-import robot**:
   - Run `mujoco_barkour_importer.py`
   - Select `barkour_vb.xml`
   - Import

3. **Verify import succeeded**:
   - Check Outliner for Body_* objects
   - Check hierarchy is correct

4. **Re-run animation**:
   - Run updated `animate_barkour.py`
   - Load `barkour_policy_animation.txt`
   - Watch console for errors

5. **Test**:
   - Press SPACEBAR
   - Robot should animate!

---

## Still Not Working?

### Get Detailed Diagnostics

Add this to `animate_barkour.py` at the end of `apply_animation()`:

```python
# Detailed diagnostics
print("\n" + "="*70)
print("DIAGNOSTIC REPORT")
print("="*70)

print("\nObjects with animation data:")
for obj in bpy.data.objects:
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        print(f"  {obj.name}:")
        print(f"    Action: {action.name}")
        print(f"    Keyframes: {len(action.fcurves)} curves")
        for fc in action.fcurves:
            print(f"      {fc.data_path}[{fc.array_index}]: {len(fc.keyframe_points)} keyframes")

print("\nBody hierarchy:")
def print_hierarchy(obj, indent=0):
    print("  " * indent + f"├── {obj.name}")
    for child in obj.children:
        print_hierarchy(child, indent + 1)

for obj in bpy.data.objects:
    if obj.parent is None and obj.name.startswith('Body_'):
        print_hierarchy(obj)

print("="*70)
```

Send this output to debug further.

---

## Contact Information

If you've tried all these solutions and it still doesn't work, please provide:

1. **Console output** from running the animation script (full text)
2. **List of objects** in your scene (from Outliner)
3. **Sample of animation data** (first 2-3 frames from the .txt file)
4. **Blender version** (Help → About Blender)

This will help diagnose the specific issue with your setup.

---

## Summary of Latest Fixes (v1.1)

✅ **Added**: `initialize_joint_rotations()` - Resets all joints to zero before animating
✅ **Added**: `verify_joint_mappings()` - Checks that all required bodies exist  
✅ **Fixed**: Base object detection now specifically looks for "torso"
✅ **Fixed**: Joint-to-body mapping uses correct MuJoCo body names
✅ **Fixed**: Proper use of euler angle indices (`[0]`, `[1]`, `[2]`)
✅ **Improved**: Better console output for debugging

**Version**: 1.1  
**Date**: October 24, 2025
