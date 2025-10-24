"""
MuJoCo Barkour VB Robot Importer for Blender
=============================================

This script imports the Google Barkour VB quadruped robot model from MuJoCo XML files into Blender.
It parses the XML structure, imports STL meshes, and reconstructs the hierarchical body structure.

Barkour VB is a research quadruped robot designed for agile locomotion and parkour movements.

Author: AI Assistant
Date: October 18, 2025
License: MIT

Features:
- Parse MuJoCo XML files (barkour_vb.xml, barkour_vb_mjx.xml)
- Import STL meshes with proper transformations
- Reconstruct kinematic tree hierarchy
- Support for visual and collision geometries
- Blender UI panel for easy configuration
- Based on G1 importer architecture

Usage:
1. Install as Blender add-on or run as script
2. Open Blender > File > Import > MuJoCo Barkour VB Robot
3. Select barkour_vb.xml and configure options
4. Click Import
"""

import bpy
import xml.etree.ElementTree as ET
import os
from mathutils import Vector, Quaternion
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper

bl_info = {
    "name": "MuJoCo Barkour VB Robot Importer",
    "author": "AI Assistant",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > MuJoCo Barkour VB Robot",
    "description": "Import Google Barkour VB quadruped robot from MuJoCo XML files",
    "category": "Import-Export",
    "warning": "Requires STL import addon enabled in Preferences",
}


# ============================================================================
# Utility Functions
# ============================================================================

def quaternion_to_blender(quat_mujoco):
    """Convert MuJoCo quaternion (w, x, y, z) to Blender quaternion."""
    if quat_mujoco is None or len(quat_mujoco) != 4:
        return Quaternion((1, 0, 0, 0))
    
    w, x, y, z = quat_mujoco
    return Quaternion((w, x, y, z))


def position_to_blender(pos_mujoco):
    """Convert MuJoCo position to Blender position."""
    if pos_mujoco is None or len(pos_mujoco) != 3:
        return Vector((0, 0, 0))
    
    x, y, z = pos_mujoco
    return Vector((x, y, z))


def parse_vector(string, default=None):
    """Parse a space-separated string of numbers into a list of floats."""
    if string is None:
        return default
    try:
        return [float(x) for x in string.split()]
    except (ValueError, AttributeError):
        return default


def create_empty_at_location(name, location, rotation, parent=None):
    """Create an empty object at a specific location with rotation."""
    bpy.ops.object.select_all(action='DESELECT')
    
    if parent is None:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
    else:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    
    empty = bpy.context.active_object
    empty.name = name
    empty.rotation_mode = 'QUATERNION'
    empty.rotation_quaternion = rotation
    empty.empty_display_size = 0.05
    
    if parent:
        empty.parent = parent
        empty.location = location
        bpy.context.view_layer.update()
    else:
        empty.location = location
    
    print(f"Created empty '{name}' at {location}")
    return empty


def import_stl_mesh(filepath, name, location, rotation, parent=None, material=None):
    """
    Import an STL file and position it correctly.
    
    Args:
        filepath: Path to STL file
        name: Name for the imported object
        location: Vector position
        rotation: Quaternion rotation
        parent: Optional parent object
        material: Optional material to apply
    
    Returns:
        Imported mesh object or None if import fails
    """
    if not os.path.exists(filepath):
        print(f"    ✗ STL file not found: {filepath}")
        return None
    
    bpy.ops.object.select_all(action='DESELECT')
    objects_before = set(bpy.context.scene.objects)
    
    try:
        abs_filepath = os.path.abspath(filepath)
        print(f"    → Attempting to import: {os.path.basename(abs_filepath)}")
        
        # Try new API first (Blender 4.x+), then fall back to old API (Blender 3.x)
        import_result = None
        try:
            import_result = bpy.ops.wm.stl_import(filepath=abs_filepath)
            print(f"       Used Blender 4.x API (wm.stl_import)")
        except AttributeError:
            import_result = bpy.ops.import_mesh.stl(filepath=abs_filepath)
            print(f"       Used Blender 3.x API (import_mesh.stl)")
        
        if import_result and import_result != {'FINISHED'}:
            print(f"    ⚠ Import returned unexpected result: {import_result}")
        
    except AttributeError as e:
        print(f"    ✗ STL import operator not available!")
        print(f"      Please enable 'Import-Export: STL format' in Preferences → Add-ons")
        return None
    except Exception as e:
        print(f"    ✗ Error importing STL {os.path.basename(filepath)}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Find newly imported object
    objects_after = set(bpy.context.scene.objects)
    new_objects = objects_after - objects_before
    
    if not new_objects:
        print(f"    ✗ No object created when importing {os.path.basename(filepath)}")
        return None
    
    mesh_obj = list(new_objects)[0]
    mesh_obj.name = name
    mesh_obj.rotation_mode = 'QUATERNION'
    
    # Set parent and transform
    if parent:
        mesh_obj.parent = parent
        mesh_obj.location = location
        mesh_obj.rotation_quaternion = rotation
        bpy.context.view_layer.update()
    else:
        mesh_obj.location = location
        mesh_obj.rotation_quaternion = rotation
    
    # Apply material
    if material:
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = material
        else:
            mesh_obj.data.materials.append(material)
    
    print(f"    ✓ Imported mesh '{name}' successfully")
    return mesh_obj


def create_material(name, color):
    """Create a material with a specific color."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Create Principled BSDF
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 0)
    node_bsdf.inputs['Base Color'].default_value = color
    node_bsdf.inputs['Metallic'].default_value = 0.3
    node_bsdf.inputs['Roughness'].default_value = 0.5
    
    # Create output
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 0)
    
    # Link nodes
    links = mat.node_tree.links
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    
    return mat


# ============================================================================
# MuJoCo XML Parser
# ============================================================================

class MuJoCoParser:
    """Parse MuJoCo XML files and extract robot structure."""
    
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.xml_dir = os.path.dirname(xml_path)
        self.tree = None
        self.root = None
        self.meshes = {}
        self.materials = {}
        self.mesh_dir = "assets"
        self.default_classes = {}  # Store default class definitions
        
    def parse(self):
        """Parse the XML file."""
        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
            
            # Get mesh directory from compiler settings
            compiler = self.root.find('compiler')
            if compiler is not None:
                mesh_dir = compiler.get('meshdir', 'assets')
                self.mesh_dir = mesh_dir
                print(f"Compiler meshdir: {self.mesh_dir}")
            else:
                print(f"No compiler meshdir specified, using default: {self.mesh_dir}")
            
            # Parse assets
            self._parse_assets()
            
            # Parse default classes to extract mesh assignments
            self._parse_defaults()
            
            return True
        except Exception as e:
            print(f"Error parsing XML: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_assets(self):
        """Parse the asset section for meshes and materials."""
        asset_section = self.root.find('asset')
        if asset_section is None:
            return
        
        # Parse meshes
        for mesh_elem in asset_section.findall('mesh'):
            mesh_name = mesh_elem.get('name')
            mesh_file = mesh_elem.get('file')
            
            if mesh_file:
                if not mesh_name:
                    mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
                
                # Build full path
                # Check if mesh_file already contains path (like "assets/file.stl")
                if os.path.dirname(mesh_file):
                    # File path already includes directory (e.g., "assets/mesh.stl")
                    mesh_path = os.path.join(self.xml_dir, mesh_file)
                else:
                    # Just filename, use mesh_dir from compiler
                    mesh_path = os.path.join(self.xml_dir, self.mesh_dir, mesh_file)
                
                self.meshes[mesh_name] = mesh_path
                print(f"  Registered mesh: {mesh_name} -> {mesh_path}")
        
        # Parse materials
        for mat_elem in asset_section.findall('material'):
            mat_name = mat_elem.get('name')
            rgba = parse_vector(mat_elem.get('rgba'), [0.7, 0.7, 0.7, 1.0])
            
            if mat_name:
                self.materials[mat_name] = rgba
    
    def _parse_defaults(self):
        """Parse default class definitions to extract mesh assignments."""
        default_section = self.root.find('default')
        if default_section is None:
            return
        
        # Recursively parse all default classes
        self._parse_default_recursive(default_section, "")
        
        print(f"  Parsed {len(self.default_classes)} default classes with mesh definitions")
    
    def _parse_default_recursive(self, default_elem, parent_class=""):
        """Recursively parse default elements to build class hierarchy."""
        for child_default in default_elem.findall('default'):
            class_name = child_default.get('class', '')
            
            if class_name:
                # Check if this default has a geom with mesh attribute
                geom_elem = child_default.find('geom')
                if geom_elem is not None:
                    mesh_name = geom_elem.get('mesh')
                    if mesh_name:
                        self.default_classes[class_name] = mesh_name
                        print(f"    Class '{class_name}' -> mesh '{mesh_name}'")
                
                # Recursively parse nested defaults
                self._parse_default_recursive(child_default, class_name)
    
    def get_worldbody(self):
        """Get the worldbody element."""
        return self.root.find('worldbody')
    
    def get_mesh_path(self, mesh_name):
        """Get full path to a mesh file."""
        return self.meshes.get(mesh_name)
    
    def get_material_color(self, material_name):
        """Get material RGBA color."""
        return self.materials.get(material_name, [0.7, 0.7, 0.7, 1.0])


# ============================================================================
# Robot Importer
# ============================================================================

class MuJoCoBarkourImporter:
    """Import MuJoCo Barkour VB robot into Blender."""
    
    def __init__(self, xml_path, import_visual=True, import_collision=True, 
                 clear_scene=False):
        self.xml_path = xml_path
        self.import_visual = import_visual
        self.import_collision = import_collision
        self.clear_scene = clear_scene
        self.parser = MuJoCoParser(xml_path)
        self.created_objects = []
        self.blender_materials = {}
        
    def import_robot(self):
        """Main import function."""
        print(f"\n{'='*70}")
        print(f"Starting Barkour VB import from: {self.xml_path}")
        print(f"{'='*70}\n")
        
        # Parse XML
        if not self.parser.parse():
            print("ERROR: Failed to parse XML file")
            return False
        
        print(f"Parsed XML successfully")
        print(f"Found {len(self.parser.meshes)} mesh definitions")
        print(f"Found {len(self.parser.materials)} materials\n")
        
        # Clear scene if requested
        if self.clear_scene:
            print("Clearing scene...")
            self._clear_scene()
        
        # Create materials
        self._create_materials()
        print(f"Created {len(self.blender_materials)} Blender materials\n")
        
        # Import robot structure
        worldbody = self.parser.get_worldbody()
        if worldbody is None:
            print("ERROR: No worldbody found in XML")
            return False
        
        print("Importing robot structure...\n")
        
        # Process all bodies in worldbody
        body_count = 0
        for body_elem in worldbody.findall('body'):
            self._import_body(body_elem, parent_obj=None)
            body_count += 1
        
        print(f"\n{'='*70}")
        print(f"Import completed!")
        print(f"Created {len(self.created_objects)} objects from {body_count} root bodies")
        print(f"{'='*70}\n")
        
        # Select imported objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in self.created_objects:
            obj.select_set(True)
        
        if self.created_objects:
            bpy.context.view_layer.objects.active = self.created_objects[0]
        
        return True
    
    def _clear_scene(self):
        """Clear all mesh and empty objects from the scene."""
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type in ['MESH', 'EMPTY']:
                obj.select_set(True)
        
        bpy.ops.object.delete(use_global=False)
        print("Scene cleared (meshes and empties removed)")
    
    def _create_materials(self):
        """Create Blender materials from parsed materials."""
        for mat_name, rgba in self.parser.materials.items():
            material = create_material(f"Barkour_{mat_name}", rgba)
            self.blender_materials[mat_name] = material
        
        # Create default material (Barkour blue/orange theme)
        default_material = create_material("Barkour_default", [0.615, 0.811, 0.929, 1.0])
        self.blender_materials['default'] = default_material
    
    def _import_body(self, body_elem, parent_obj=None):
        """Recursively import a body and its children."""
        body_name = body_elem.get('name', 'unnamed_body')
        
        # Get position and rotation
        pos_str = body_elem.get('pos', '0 0 0')
        quat_str = body_elem.get('quat', '1 0 0 0')
        
        pos = position_to_blender(parse_vector(pos_str, [0, 0, 0]))
        quat = quaternion_to_blender(parse_vector(quat_str, [1, 0, 0, 0]))
        
        print(f"  Creating body: {body_name} at pos={pos}, quat={quat}")
        
        # Create empty for body
        body_obj = create_empty_at_location(
            name=f"Body_{body_name}",
            location=pos,
            rotation=quat,
            parent=parent_obj
        )
        self.created_objects.append(body_obj)
        
        # Import geometries (meshes)
        geom_count = 0
        for geom_elem in body_elem.findall('geom'):
            result = self._import_geometry(geom_elem, body_obj)
            if result:
                geom_count += 1
        
        if geom_count > 0:
            print(f"    -> Imported {geom_count} geometries for {body_name}")
        
        # Recursively import child bodies
        for child_body_elem in body_elem.findall('body'):
            self._import_body(child_body_elem, parent_obj=body_obj)
        
        return body_obj
    
    def _import_geometry(self, geom_elem, parent_obj):
        """Import a geometry element."""
        # Check geometry type
        geom_type = geom_elem.get('type', 'mesh')
        
        # Get class to determine if visual or collision
        geom_class = geom_elem.get('class', '')
        is_visual = 'visual' in geom_class.lower() or geom_class == '' or '/' in geom_class
        is_collision = 'collision' in geom_class.lower()
        
        # Check if we should import this geometry
        if is_visual and not self.import_visual:
            print(f"    ⊘ Skipping visual geometry (import_visual=False)")
            return None
        if is_collision and not self.import_collision:
            print(f"    ⊘ Skipping collision geometry (import_collision=False)")
            return None
        
        # Only import mesh geometries (skip primitives for now)
        if geom_type != 'mesh':
            print(f"    ⊘ Skipping non-mesh geometry (type={geom_type})")
            return None
        
        # Get mesh reference - either directly or from class
        mesh_name = geom_elem.get('mesh')
        
        # If no direct mesh attribute, look up from class defaults
        if not mesh_name and geom_class:
            mesh_name = self.parser.default_classes.get(geom_class)
            if mesh_name:
                print(f"    → Mesh name '{mesh_name}' resolved from class '{geom_class}'")
        
        if not mesh_name:
            print(f"    ✗ Geometry has no mesh reference (class={geom_class})")
            return None
        
        mesh_path = self.parser.get_mesh_path(mesh_name)
        if not mesh_path:
            print(f"    ✗ Mesh '{mesh_name}' not found in assets")
            return None
        
        print(f"    → Importing geometry: {mesh_name} (class={geom_class})")
        
        # Get position and rotation (relative to parent body)
        pos_str = geom_elem.get('pos', '0 0 0')
        quat_str = geom_elem.get('quat', '1 0 0 0')
        
        pos = position_to_blender(parse_vector(pos_str, [0, 0, 0]))
        quat = quaternion_to_blender(parse_vector(quat_str, [1, 0, 0, 0]))
        
        # Get material - Barkour has RGBA directly in geom
        rgba_str = geom_elem.get('rgba')
        if rgba_str:
            rgba = parse_vector(rgba_str, [0.615, 0.811, 0.929, 1.0])
            mat_name = f"barkour_{mesh_name}"
            if mat_name not in self.blender_materials:
                material = create_material(f"Barkour_{mesh_name}", rgba)
                self.blender_materials[mat_name] = material
            else:
                material = self.blender_materials[mat_name]
        else:
            mat_name = geom_elem.get('material', 'default')
            material = self.blender_materials.get(mat_name, self.blender_materials['default'])
        
        # Generate object name
        geom_class_suffix = "_visual" if is_visual else "_collision"
        obj_name = f"{mesh_name}{geom_class_suffix}"
        
        # Import mesh
        mesh_obj = import_stl_mesh(
            filepath=mesh_path,
            name=obj_name,
            location=pos,
            rotation=quat,
            parent=parent_obj,
            material=material
        )
        
        if mesh_obj:
            self.created_objects.append(mesh_obj)
            
            # Set display settings for collision meshes
            if is_collision:
                mesh_obj.display_type = 'WIRE'
                mesh_obj.show_in_front = True
            
            return mesh_obj
        else:
            print(f"    ✗ Failed to import mesh for {mesh_name}")
        
        return None


# ============================================================================
# Blender Operator
# ============================================================================

class IMPORT_OT_mujoco_barkour(bpy.types.Operator, ImportHelper):
    """Import Google Barkour VB Robot from MuJoCo XML"""
    bl_idname = "import_scene.mujoco_barkour"
    bl_label = "Import MuJoCo Barkour VB Robot"
    bl_options = {'REGISTER', 'UNDO'}
    
    # File selection
    filename_ext = ".xml"
    filter_glob: StringProperty(
        default="*.xml",
        options={'HIDDEN'},
    )
    
    import_visual: BoolProperty(
        name="Import Visual Meshes",
        description="Import visual geometry meshes",
        default=True,
    )
    
    import_collision: BoolProperty(
        name="Import Collision Meshes",
        description="Import collision geometry meshes",
        default=False,
    )
    
    clear_scene: BoolProperty(
        name="Clear Scene",
        description="Remove all existing objects before import",
        default=False,
    )
    
    def execute(self, context):
        """Execute the import operation."""
        xml_path = self.filepath
        
        if not os.path.exists(xml_path):
            self.report({'ERROR'}, f"File not found: {xml_path}")
            return {'CANCELLED'}
        
        # Create importer
        importer = MuJoCoBarkourImporter(
            xml_path=xml_path,
            import_visual=self.import_visual,
            import_collision=self.import_collision,
            clear_scene=self.clear_scene
        )
        
        # Import robot
        try:
            success = importer.import_robot()
            if success:
                self.report({'INFO'}, f"Successfully imported Barkour VB robot from {os.path.basename(xml_path)}")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Failed to import robot")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def draw(self, context):
        """Draw the operator panel."""
        layout = self.layout
        
        box = layout.box()
        box.label(text="Import Options:", icon='IMPORT')
        box.prop(self, "import_visual")
        box.prop(self, "import_collision")
        box.prop(self, "clear_scene")


# ============================================================================
# UI Panel
# ============================================================================

class VIEW3D_PT_mujoco_barkour_panel(bpy.types.Panel):
    """Creates a Panel in the 3D Viewport sidebar"""
    bl_label = "MuJoCo Barkour VB Importer"
    bl_idname = "VIEW3D_PT_mujoco_barkour"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Barkour Robot'
    
    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Import Barkour VB Robot", icon='ARMATURE_DATA')
        
        col = box.column(align=True)
        col.operator("import_scene.mujoco_barkour", text="Import from XML", icon='IMPORT')
        
        layout.separator()
        
        info_box = layout.box()
        info_box.label(text="Info:", icon='INFO')
        col = info_box.column(align=True)
        col.label(text="• Google Research quadruped")
        col.label(text="• Designed for parkour/agility")
        col.label(text="• Uses STL mesh files")
        col.label(text="• Location: google_barkour_vb folder")


# ============================================================================
# Registration
# ============================================================================

classes = (
    IMPORT_OT_mujoco_barkour,
    VIEW3D_PT_mujoco_barkour_panel,
)


def menu_func_import(self, context):
    """Add to import menu."""
    self.layout.operator(IMPORT_OT_mujoco_barkour.bl_idname, text="MuJoCo Barkour VB Robot (.xml)")


def register():
    """Register addon."""
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    print("MuJoCo Barkour VB Importer registered")


def unregister():
    """Unregister addon."""
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("MuJoCo Barkour VB Importer unregistered")


# ============================================================================
# Script Mode
# ============================================================================

if __name__ == "__main__":
    register()
    
    print("\n" + "="*70)
    print("MuJoCo Barkour VB Robot Importer - Ready!")
    print("="*70)
    print("\nUsage:")
    print("1. Via Menu: File > Import > MuJoCo Barkour VB Robot (.xml)")
    print("2. Via Sidebar: View3D > Sidebar (N key) > Barkour Robot tab")
    print("\n" + "="*70 + "\n")
