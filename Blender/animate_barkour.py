"""
Blender Barkour Robot Animator
================================

This script animates the Barkour VB robot in Blender using simulation data
generated from a trained policy.

Author: AI Assistant
Date: October 24, 2025
License: MIT

Usage:
1. Import Barkour robot using mujoco_barkour_importer.py
2. Run this script in Blender's scripting panel
3. Select the animation data file (e.g., barkour_animation.txt)
4. Click "Animate Robot" to apply keyframes

Features:
- Reads JSON animation data from policy simulation
- Applies keyframes to robot joints and base
- Supports looping animations
- Real-time preview in viewport
"""

import bpy
import json
import math
from pathlib import Path
from mathutils import Vector, Quaternion, Euler
from bpy.props import StringProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper


bl_info = {
    "name": "Barkour Robot Animator",
    "author": "AI Assistant",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Barkour Robot",
    "description": "Animate Barkour robot using policy simulation data",
    "category": "Animation",
}


# ============================================================================
# Animation Data Parser
# ============================================================================

class AnimationDataParser:
    """Parse animation data from JSON file."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.joint_names = []
        self.frames = []
        self.fps = 60
        self.duration = 0
        
    def parse(self):
        """Load and parse the animation file."""
        try:
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
            
            # Extract metadata
            self.joint_names = self.data.get('JointNames', [])
            self.frames = self.data.get('Frames', [])
            self.fps = self.data.get('FPS', 60)
            self.duration = self.data.get('Duration', 0)
            
            print(f"Loaded animation data:")
            print(f"  Format: {self.data.get('Format', 'unknown')}")
            print(f"  Robot: {self.data.get('Robot', 'unknown')}")
            print(f"  Frames: {len(self.frames)}")
            print(f"  Duration: {self.duration}s")
            print(f"  FPS: {self.fps}")
            print(f"  Joints: {len(self.joint_names)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading animation file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_frame_data(self, frame_index):
        """Get data for a specific frame."""
        if frame_index < 0 or frame_index >= len(self.frames):
            return None
        
        return self.frames[frame_index]


# ============================================================================
# Barkour Robot Animator
# ============================================================================

class BarkourRobotAnimator:
    """Animate Barkour robot in Blender."""
    
    def __init__(self, animation_data):
        self.animation_data = animation_data
        self.robot_objects = {}
        self.base_object = None
        self.rest_pose_map = {}
        
    def find_robot_objects(self):
        """Find robot objects in the scene."""
        print("\nSearching for robot objects...")
        
        # Look for body empties created by importer
        for obj in bpy.data.objects:
            if obj.type == 'EMPTY' and obj.name.startswith('Body_'):
                body_name = obj.name.replace('Body_', '')
                self.robot_objects[body_name] = obj
                print(f"  Found: {obj.name}")
                
                # Identify base/root object (torso is the root body)
                if 'torso' in body_name.lower():
                    self.base_object = obj
                    print(f"    → Identified as base object")
        
        print(f"\nTotal objects found: {len(self.robot_objects)}")
        
        if not self.base_object:
            # If no explicit base found, try to find torso
            torso_obj = self.robot_objects.get('torso')
            if torso_obj:
                self.base_object = torso_obj
                print(f"Using torso as base object")
            elif self.robot_objects:
                self.base_object = list(self.robot_objects.values())[0]
                print(f"Using {self.base_object.name} as base object")
        
        # Verify joint body mappings
        self.verify_joint_mappings()
        
        return len(self.robot_objects) > 0
    
    def verify_joint_mappings(self):
        """Verify that all required joint bodies exist."""
        print("\nVerifying joint body mappings...")
        
        joint_to_body_mapping = self.get_joint_to_body_mapping()
        missing_bodies = []
        found_bodies = []
        
        for joint_info in joint_to_body_mapping:
            body_name = joint_info['body']
            joint_name = joint_info['joint']
            
            if body_name in self.robot_objects:
                found_bodies.append(f"{joint_name} → {body_name} ✓")
            else:
                missing_bodies.append(f"{joint_name} → {body_name} ✗ MISSING")
        
        if found_bodies:
            print(f"  Found {len(found_bodies)}/{len(joint_to_body_mapping)} joint bodies:")
            for msg in found_bodies[:3]:  # Show first 3
                print(f"    {msg}")
            if len(found_bodies) > 3:
                print(f"    ... and {len(found_bodies) - 3} more")
        
        if missing_bodies:
            print(f"\n  ⚠ WARNING: {len(missing_bodies)} bodies not found:")
            for msg in missing_bodies:
                print(f"    {msg}")
            print(f"\n  Available bodies: {list(self.robot_objects.keys())}")
        else:
            print(f"  ✓ All joint bodies found!")
    
    def apply_animation(self, loop=True):
        """Apply animation keyframes to robot."""
        
        if not self.find_robot_objects():
            print("ERROR: No robot objects found in scene!")
            print("Please import the robot first using mujoco_barkour_importer.py")
            return False
        
        # Set scene FPS
        bpy.context.scene.render.fps = self.animation_data.fps
        
        # Calculate total frames
        num_frames = len(self.animation_data.frames)
        
        print("\nApplying animation...")
        print(f"  Total frames: {num_frames}")
        print(f"  FPS: {self.animation_data.fps}")
        
        # Set timeline
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = num_frames
        
        # Clear existing animation data
        self.clear_animation()
        
        # Initialize joint rest quaternions
        self.initialize_joint_rotations()
        
        # Apply keyframes
        for frame_idx, frame_data in enumerate(self.animation_data.frames):
            blender_frame = frame_idx + 1  # Blender frames start at 1
            
            # Set base position and orientation
            if self.base_object and 'base_pos' in frame_data and 'base_quat' in frame_data:
                self.set_base_transform(
                    self.base_object,
                    frame_data['base_pos'],
                    frame_data['base_quat'],
                    blender_frame
                )
            
            # Set joint angles
            if 'joints' in frame_data:
                self.set_joint_angles(frame_data['joints'], blender_frame)
            
            # Progress indicator
            if (frame_idx + 1) % 50 == 0 or frame_idx == num_frames - 1:
                progress = (frame_idx + 1) / num_frames * 100
                print(f"    Progress: {progress:.1f}%")
        
        print(f"✓ Animation applied successfully")
        
        # Set loop mode if requested
        if loop:
            print(f"  Animation set to loop")
        
        return True
    
    def initialize_joint_rotations(self):
        """Initialize all joint body rest quaternions without altering pose."""
        print("\n  Initializing joint rotation modes...")
        
        joint_to_body_mapping = self.get_joint_to_body_mapping()
        initialized_count = 0
        self.rest_pose_map.clear()
        
        for joint_info in joint_to_body_mapping:
            body_name = joint_info['body']
            obj = self.robot_objects.get(body_name)
            
            if obj:
                obj.rotation_mode = 'QUATERNION'
                rest_quat = obj.rotation_quaternion.copy()
                self.rest_pose_map[body_name] = rest_quat
                initialized_count += 1
            else:
                print(f"    WARNING: Could not find object for body '{body_name}'")
        
        print(f"  Stored rest quaternions for {initialized_count} joint bodies")
    
    def set_base_transform(self, obj, position, quaternion, frame):
        """Set base object position and rotation with keyframe."""
        
        # MuJoCo uses Z-up, Blender also uses Z-up, so we can use directly
        # Position: (x, y, z) - keep as-is
        obj.location = Vector(position)
        obj.keyframe_insert(data_path="location", frame=frame)
        
        # Rotation: MuJoCo quaternion (qw, qx, qy, qz) -> Blender (w, x, y, z)
        qw, qx, qy, qz = quaternion
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Quaternion((qw, qx, qy, qz))
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    
    def set_joint_angles(self, joint_angles, frame):
        """Set joint angles with quaternion keyframes."""
        joint_to_body_mapping = self.get_joint_to_body_mapping()
        joint_count = min(len(joint_angles), len(joint_to_body_mapping))
        
        for i in range(joint_count):
            angle = joint_angles[i]
            joint_info = joint_to_body_mapping[i]
            body_name = joint_info['body']
            axis_local = joint_info['axis'].copy()
            
            obj = self.robot_objects.get(body_name)
            if not obj:
                if frame == 1:
                    print(f"  Warning: Body '{body_name}' not found for joint {i}")
                continue
            
            rest_quat = self.rest_pose_map.get(body_name)
            if rest_quat is None:
                rest_quat = obj.rotation_quaternion.copy()
                self.rest_pose_map[body_name] = rest_quat
            else:
                rest_quat = rest_quat.copy()
            
            obj.rotation_mode = 'QUATERNION'
            rot_delta = Quaternion(axis_local, angle)
            obj.rotation_quaternion = rest_quat @ rot_delta
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    
    def get_joint_to_body_mapping(self):
        """
        Get mapping of joint indices to body names and rotation axes.
        
    Joint data follows MJCF actuator order:
    0-2:  FL (front left):  abduction, hip, knee
    3-5:  HL (hind left):   abduction, hip, knee
    6-8:  FR (front right): abduction, hip, knee
    9-11: HR (hind right):  abduction, hip, knee
        
        MuJoCo body structure and joint axes:
        - abduction joint (hx): rotates leg_* body around X-axis (left-right splay)
        - hip joint (hy): rotates upper_leg_* body around Y-axis (forward-back swing)
        - knee joint (kn): rotates lower_leg_* body around Y-axis (knee bend)
        """
        
        axis_z = Vector((0.0, 0.0, 1.0))

        return [
            # Front Left Leg (indices 0-2)
            {'joint': 'fl_hx', 'body': 'leg_front_left', 'axis': axis_z.copy()},
            {'joint': 'fl_hy', 'body': 'upper_leg_front_left', 'axis': axis_z.copy()},
            {'joint': 'fl_kn', 'body': 'lower_leg_front_left', 'axis': axis_z.copy()},

            # Hind Left Leg (indices 3-5)
            {'joint': 'hl_hx', 'body': 'leg_hind_left', 'axis': axis_z.copy()},
            {'joint': 'hl_hy', 'body': 'upper_leg_hind_left', 'axis': axis_z.copy()},
            {'joint': 'hl_kn', 'body': 'lower_leg_hind_left', 'axis': axis_z.copy()},

            # Front Right Leg (indices 6-8)
            {'joint': 'fr_hx', 'body': 'leg_front_right', 'axis': axis_z.copy()},
            {'joint': 'fr_hy', 'body': 'upper_leg_front_right', 'axis': axis_z.copy()},
            {'joint': 'fr_kn', 'body': 'lower_leg_front_right', 'axis': axis_z.copy()},

            # Hind Right Leg (indices 9-11)
            {'joint': 'hr_hx', 'body': 'leg_hind_right', 'axis': axis_z.copy()},
            {'joint': 'hr_hy', 'body': 'upper_leg_hind_right', 'axis': axis_z.copy()},
            {'joint': 'hr_kn', 'body': 'lower_leg_hind_right', 'axis': axis_z.copy()},
        ]
    
    def clear_animation(self):
        """Clear all animation data from robot objects."""
        
        for obj in self.robot_objects.values():
            if obj.animation_data:
                obj.animation_data_clear()


# ============================================================================
# Blender Operator
# ============================================================================

class ANIM_OT_animate_barkour(bpy.types.Operator, ImportHelper):
    """Animate Barkour Robot from Simulation Data"""
    bl_idname = "anim.animate_barkour"
    bl_label = "Animate Barkour Robot"
    bl_options = {'REGISTER', 'UNDO'}
    
    # File selection
    filename_ext = ".txt"
    filter_glob: StringProperty(
        default="*.txt;*.json",
        options={'HIDDEN'},
    )
    
    loop_animation: BoolProperty(
        name="Loop Animation",
        description="Enable animation looping",
        default=True,
    )
    
    def execute(self, context):
        """Execute the animation operation."""
        
        filepath = self.filepath
        
        if not Path(filepath).exists():
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}
        
        print("\n" + "="*70)
        print("Barkour Robot Animator")
        print("="*70)
        
        # Parse animation data
        parser = AnimationDataParser(filepath)
        if not parser.parse():
            self.report({'ERROR'}, "Failed to parse animation data")
            return {'CANCELLED'}
        
        # Create animator
        animator = BarkourRobotAnimator(parser)
        
        # Apply animation
        try:
            success = animator.apply_animation(loop=self.loop_animation)
            
            if success:
                self.report({'INFO'}, f"Animation applied: {len(parser.frames)} frames")
                
                # Set playback to start
                bpy.context.scene.frame_set(1)
                
                print("\n" + "="*70)
                print("✅ Animation Complete!")
                print("="*70)
                print("\nPress SPACEBAR to play animation")
                print("="*70 + "\n")
                
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Failed to apply animation")
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Animation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def draw(self, context):
        """Draw the operator panel."""
        layout = self.layout
        
        box = layout.box()
        box.label(text="Animation Options:", icon='ANIM')
        box.prop(self, "loop_animation")


# ============================================================================
# UI Panel
# ============================================================================

class VIEW3D_PT_barkour_animator_panel(bpy.types.Panel):
    """Creates a Panel in the 3D Viewport sidebar"""
    bl_label = "Barkour Animator"
    bl_idname = "VIEW3D_PT_barkour_animator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Barkour Robot'
    
    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Animate Robot", icon='ARMATURE_DATA')
        
        col = box.column(align=True)
        col.operator("anim.animate_barkour", text="Load Animation Data", icon='ANIM_DATA')
        
        layout.separator()
        
        # Playback controls
        playback_box = layout.box()
        playback_box.label(text="Playback:", icon='PLAY')
        
        row = playback_box.row(align=True)
        row.operator("screen.animation_play", text="Play", icon='PLAY')
        row.operator("screen.animation_cancel", text="Stop", icon='PAUSE')
        
        playback_box.prop(context.scene, "frame_current", text="Frame")
        
        layout.separator()
        
        # Info
        info_box = layout.box()
        info_box.label(text="Workflow:", icon='INFO')
        col = info_box.column(align=True)
        col.label(text="1. Import Barkour robot")
        col.label(text="2. Convert policy to animation")
        col.label(text="3. Load animation data")
        col.label(text="4. Press Play!")


# ============================================================================
# Registration
# ============================================================================

classes = (
    ANIM_OT_animate_barkour,
    VIEW3D_PT_barkour_animator_panel,
)


def register():
    """Register addon."""
    for cls in classes:
        bpy.utils.register_class(cls)
    print("Barkour Robot Animator registered")


def unregister():
    """Unregister addon."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Barkour Robot Animator unregistered")


# ============================================================================
# Script Mode
# ============================================================================

if __name__ == "__main__":
    register()
    
    print("\n" + "="*70)
    print("Barkour Robot Animator - Ready!")
    print("="*70)
    print("\nUsage:")
    print("1. Via Menu: View3D > Sidebar (N key) > Barkour Robot tab")
    print("2. Click 'Load Animation Data'")
    print("3. Select animation file (e.g., barkour_animation.txt)")
    print("4. Click 'Animate Barkour Robot'")
    print("\n" + "="*70 + "\n")
