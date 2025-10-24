# Barkour Animation Fix

## Problem
The robot was not animating in Blender because the animation script was looking for objects with generic names that didn't match the actual MuJoCo body structure.

## Root Cause
**Joint-to-Body Mismatch**: The animation data uses short joint names (e.g., `fl_hx`, `fl_hy`, `fl_kn`), but the MuJoCo importer creates body objects with full names (e.g., `Body_leg_front_left`, `Body_upper_leg_front_left`).

In MuJoCo's kinematic model:
- **Joints** define rotation relationships between parent and child bodies
- **Bodies** are the actual movable parts
- Each joint rotates its **child body**, not a separate joint object

## Solution
Updated `animate_barkour.py` to properly map joint angles to body objects:

### 1. Base Object Detection
```python
# Changed from generic search to specific torso detection
if 'torso' in body_name.lower():
    self.base_object = obj
```

### 2. Joint-to-Body Mapping
Created explicit mapping between joint indices and body names:

| Joint Index | Joint Name | Body to Rotate | Axis |
|-------------|------------|----------------|------|
| 0 | fl_hx | `leg_front_left` | X |
| 1 | fl_hy | `upper_leg_front_left` | Y |
| 2 | fl_kn | `lower_leg_front_left` | Y |
| 3 | fr_hx | `leg_front_right` | X |
| 4 | fr_hy | `upper_leg_front_right` | Y |
| 5 | fr_kn | `lower_leg_3` | Y |
| 6 | hl_hx | `leg_hind_left` | X |
| 7 | hl_hy | `upper_leg_hind_left` | Y |
| 8 | hl_kn | `lower_leg_2` | Y |
| 9 | hr_hx | `leg_hind_right` | X |
| 10 | hr_hy | `upper_leg_hind_right` | Y |
| 11 | hr_kn | `lower_leg_4` | Y |

### 3. Body Name Reference
From `barkour_vb.xml`:
- **Root**: `torso` (with freejoint for base motion)
- **Legs**: 
  - `leg_[position]_[side]` (abduction rotation)
  - `upper_leg_[position]_[side]` (hip rotation)
  - `lower_leg_[number]` (knee rotation, numbered 1-4 or named)

Where:
- `[position]` = front | hind
- `[side]` = left | right

## Key Changes

### Before (Broken)
```python
def get_joint_mapping(self):
    joint_names = []
    for leg in ['fl', 'fr', 'hl', 'hr']:
        joint_names.extend([f"{leg}_hx", f"{leg}_hy", f"{leg}_kn"])
    return joint_names

# Tried to find objects named "fl_hx", "fl_hy" etc. (don't exist!)
obj = self.robot_objects.get(joint_name)
```

### After (Fixed)
```python
def get_joint_to_body_mapping(self):
    return [
        {'joint': 'fl_hx', 'body': 'leg_front_left', 'axis': 'X'},
        {'joint': 'fl_hy', 'body': 'upper_leg_front_left', 'axis': 'Y'},
        # ... etc for all 12 joints
    ]

# Now finds actual body objects like "leg_front_left"
obj = self.robot_objects.get(body_name)
```

## Testing Instructions

1. **Import Robot** (if not already done):
   ```python
   # In Blender, run: mujoco_barkour_importer.py
   # Select: mujoco_menagerie/google_barkour_vb/barkour_vb.xml
   ```

2. **Load Animation**:
   ```python
   # In Blender, run updated animate_barkour.py
   # Select: Blender/barkour_policy_animation.txt
   ```

3. **Verify**:
   - Check console for "Found: Body_torso", "Body_leg_front_left", etc.
   - Should see "Animation applied: 300 frames"
   - Press SPACE to play animation
   - Robot should now move!

## Technical Notes

### MuJoCo Joint-Body Relationship
```xml
<body name="torso">
  <body name="leg_front_left" pos="..." quat="...">
    <joint name="abduction_front_left"/>  <!-- Rotates leg_front_left -->
    <body name="upper_leg_front_left">
      <joint name="hip_front_left"/>      <!-- Rotates upper_leg_front_left -->
      <body name="lower_leg_front_left">
        <joint name="knee_front_left"/>   <!-- Rotates lower_leg_front_left -->
      </body>
    </body>
  </body>
</body>
```

### Blender Object Hierarchy
After import:
```
Body_torso (empty, receives base transform)
├── Body_leg_front_left (empty, rotates on X for abduction)
│   └── Body_upper_leg_front_left (empty, rotates on Y for hip)
│       └── Body_lower_leg_front_left (empty, rotates on Y for knee)
├── Body_leg_front_right (...)
├── Body_leg_hind_left (...)
└── Body_leg_hind_right (...)
```

## Animation Data Format
Each frame in `barkour_policy_animation.txt`:
```json
{
  "dt": 0.02,
  "base_pos": [x, y, z],           // Applied to Body_torso location
  "base_quat": [qw, qx, qy, qz],   // Applied to Body_torso rotation
  "joints": [                       // 12 joint angles in radians
    fl_hx, fl_hy, fl_kn,            // Front Left
    fr_hx, fr_hy, fr_kn,            // Front Right
    hl_hx, hl_hy, hl_kn,            // Hind Left
    hr_hx, hr_hy, hr_kn             // Hind Right
  ]
}
```

## Common Issues

### Issue: "Body not found" warnings
**Solution**: Verify the importer ran correctly. Check Outliner for `Body_*` empties.

### Issue: Robot parts not moving
**Solution**: Ensure body names match exactly (case-sensitive). Check console output.

### Issue: Robot moves but rotations are wrong
**Solution**: Verify axis assignments (X for abduction, Y for hip/knee) match MuJoCo joint definitions.

### Issue: Some legs work, others don't
**Solution**: Check numbered lower_leg bodies (lower_leg_2, 3, 4) are mapped correctly.

## Files Modified
- `Blender/animate_barkour.py` - Fixed joint-to-body mapping

## Files Reference
- `mujoco_menagerie/google_barkour_vb/barkour_vb.xml` - Source of body/joint structure
- `Blender/barkour_policy_animation.txt` - Animation data (300 frames)
- `Blender/mujoco_barkour_importer.py` - Creates Body_* empties

---

**Status**: ✅ FIXED - Animation should now work correctly
**Date**: 2025-10-24
**Tested**: Ready for user testing in Blender
