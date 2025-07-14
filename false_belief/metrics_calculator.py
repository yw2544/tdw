#!/usr/bin/env python3
"""
Metrics Calculator Module for TDW Room Environment
Calculates occlusion ratio and visibility ratio based on compute_occ_vis.py implementation
Modified to work within a single TDW session without recreating scenes
"""
import math
import numpy as np
from collections import Counter
from pathlib import Path
from shapely.geometry import Polygon
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from camera_manager import render_scene_for_metrics

def get_object_bounds(model_name: str, rotation: dict = {"x": 0, "y": 0, "z": 0}, scale: float = 1.0):
    """Get object's real bounding box dimensions from TDW ModelLibrarian"""
    try:
        librarian = ModelLibrarian("models_core.json")
        record = librarian.get_record(model_name)
        bounds = record.bounds  # dict of left, right, front, back, bottom, top
        
        # Calculate original dimensions (apply scale)
        width = (bounds['right']['x'] - bounds['left']['x']) * scale
        depth = (bounds['front']['z'] - bounds['back']['z']) * scale
        height = (bounds['top']['y'] - bounds['bottom']['y']) * scale
        
        # Apply Y-axis rotation effect
        y_rotation = rotation.get('y', 0) % 360
        if y_rotation in [90, 270]:
            width, depth = depth, width  # 90-degree rotation swaps width and depth
        elif y_rotation not in [0, 180]:
            # Other angles need bounding box calculation
            angle_rad = math.radians(y_rotation)
            new_width = abs(width * math.cos(angle_rad)) + abs(depth * math.sin(angle_rad))
            new_depth = abs(width * math.sin(angle_rad)) + abs(depth * math.cos(angle_rad))
            width, depth = new_width, new_depth
            
        return (width, depth, height)
    except Exception as e:
        print(f"Failed to get object bounds for {model_name}: {e}")
        return (1.0, 1.0, 1.0)  # Default values

def find_target_color_in_segmentation(seg_array: np.ndarray) -> np.ndarray | None:
    """Find target object color from segmentation image"""
    if len(seg_array.shape) != 3:
        return None
    
    # Find all non-black pixels
    non_black_mask = np.any(seg_array > 0, axis=2)
    
    if not np.any(non_black_mask):
        return None
    
    # Get all non-black pixel colors
    non_black_pixels = seg_array[non_black_mask]
    
    # Find most common color (should be target object color)
    pixel_tuples = [tuple(pixel) for pixel in non_black_pixels]
    
    color_counts = Counter(pixel_tuples)
    most_common_color_tuple = color_counts.most_common(1)[0][0]
    
    # Convert back to numpy array
    target_color = np.array(most_common_color_tuple, dtype=np.uint8)
    
    return target_color

def compute_specific_color_area(seg_array: np.ndarray, target_color: np.ndarray) -> int:
    """Calculate pixel area of specific color from segmentation image"""
    if len(seg_array.shape) != 3:
        return 0
    
    mask = np.all(seg_array == target_color, axis=-1)
    return int(np.sum(mask))

def compute_occlusion_ratio(c: Controller, target_obj_data: dict, all_objects_data: list, 
                          camera_config: dict, output_dir: Path, view_name: str):
    """
    Calculate occlusion ratio for target object by temporarily hiding/showing occlusion objects
    Works within the same TDW session without recreating scenes
    """
    print(f"    Computing occlusion ratio for {target_obj_data['model']}...")
    
    # Get occlusion object IDs
    occlusion_ids = [obj_data["id"] for obj_data in all_objects_data if obj_data["type"] == "occlusion"]
    
    # 1) Hide occlusion objects to render baseline scene
    print(f"      Rendering baseline scene (hiding occlusion objects)...")
    
    # Hide occlusion objects
    hide_commands = []
    for obj_id in occlusion_ids:
        hide_commands.append({
            "$type": "hide_object",
            "id": obj_id
        })
    
    if hide_commands:
        c.communicate(hide_commands)
    
    # Render baseline scene
    color1, seg1 = render_scene_for_metrics(c, camera_config, output_dir, f"{view_name}_baseline")
    
    # Find target color in segmentation
    target_color = find_target_color_in_segmentation(seg1)
    if target_color is not None:
        area1 = compute_specific_color_area(seg1, target_color)
    else:
        print(f"      Warning: Target object color not found in baseline")
        area1 = 0
    
    # 2) Show occlusion objects again to render occluded scene
    print(f"      Rendering occluded scene (showing occlusion objects)...")
    
    # Show occlusion objects
    show_commands = []
    for obj_id in occlusion_ids:
        show_commands.append({
            "$type": "show_object",
            "id": obj_id
        })
    
    if show_commands:
        c.communicate(show_commands)
    
    # Render occluded scene
    color2, seg2 = render_scene_for_metrics(c, camera_config, output_dir, f"{view_name}_occluded")
    
    # Use same target color to calculate area
    if target_color is not None:
        area2 = compute_specific_color_area(seg2, target_color)
    else:
        area2 = 0
    
    # 3) Calculate occlusion ratio
    if area1 > 0:
        occlusion_ratio = area2 / area1
    else:
        occlusion_ratio = 0
    
    print(f"      Baseline area: {area1} pixels, Occluded area: {area2} pixels")
    print(f"      Occlusion ratio: {occlusion_ratio:.3f}")
    
    return occlusion_ratio

def compute_visibility_ratio(obj_data: dict, camera_config: dict, fov_deg=90, samples=64):
    """
    Calculate visibility ratio based on topdown view geometry
    Based on compute_occ_vis.py implementation
    
    Steps:
    1. Construct target object bounding box polygon B in XZ plane
    2. Construct camera view sector F (vertex at camera position, angle fov_deg)
    3. Calculate intersection I = B ∩ F, return I.area / B.area
    """
    # 1) Get target object real dimensions
    x0, z0 = obj_data["position"]["x"], obj_data["position"]["z"]
    width, depth, height = get_object_bounds(obj_data["model"], obj_data["rotation"], obj_data["scale"])
    hx, hz = width/2, depth/2
    
    # Target object rectangular area
    B = Polygon([
        (x0-hx, z0-hz),
        (x0+hx, z0-hz), 
        (x0+hx, z0+hz),
        (x0-hx, z0+hz)
    ])
    
    # 2) Construct camera view sector F
    cx, cz = camera_config["position"]["x"], camera_config["position"]["z"]
    lx, lz = camera_config["look_at"]["x"], camera_config["look_at"]["z"]
    
    # Camera direction vector
    look_direction = np.array([lx-cx, lz-cz], dtype=float)
    look_direction /= np.linalg.norm(look_direction)
    
    # Half of view angle
    half_fov = math.radians(fov_deg / 2)
    
    # Rotation function
    def rotate_vector(v, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array([v[0]*cos_a - v[1]*sin_a, v[0]*sin_a + v[1]*cos_a])
    
    # View boundary vectors
    left_edge = rotate_vector(look_direction, +half_fov)
    right_edge = rotate_vector(look_direction, -half_fov)
    
    # Calculate sector radius (enough to cover target object)
    corners = np.array(B.exterior.coords[:-1])  # Remove duplicate last point
    distances = np.linalg.norm(corners - np.array([cx, cz]), axis=1)
    radius = distances.max() * 1.5  # Leave some margin
    
    # Construct sector polygon: camera position + sampled points on arc
    sector_points = [(cx, cz)]  # Sector vertex
    
    # Sample points between left and right boundaries
    for i in range(samples + 1):
        angle = -half_fov + i * (2 * half_fov / samples)
        direction = rotate_vector(look_direction, angle)
        point = (cx + direction[0] * radius, cz + direction[1] * radius)
        sector_points.append(point)
    
    F = Polygon(sector_points)
    
    # 3) Calculate intersection and return ratio
    intersection = B.intersection(F)
    
    if B.area > 0:
        visibility_ratio = intersection.area / B.area
        return visibility_ratio
    else:
        return 0.0

def compute_metrics_for_view(c: Controller, camera_position: dict, target_obj_data: dict, 
                           all_objects_data: list, output_dir: Path, view_name: str):
    """
    Compute both occlusion and visibility ratios for all objects from a specific camera view
    Works within the same TDW session
    """
    camera_config = {
        "position": camera_position,
        "look_at": target_obj_data["position"],
        "field_of_view": 90
    }
    
    metrics = {}
    
    print(f"  Computing metrics for {view_name}...")
    
    # For each object, compute appropriate metrics
    for obj_data in all_objects_data:
        model_name = obj_data["model"]
        is_target = obj_data["type"] == "target"
        
        print(f"    Processing {model_name} ({'target' if is_target else 'occlusion'})...")
        
        # Add object actual size
        width, depth, height = get_object_bounds(obj_data["model"], obj_data["rotation"], obj_data["scale"])
        obj_data["actual_size"] = {"width": float(width), "depth": float(depth), "height": float(height)}
        
        # Compute visibility ratio for all objects
        visibility_ratio = compute_visibility_ratio(obj_data, camera_config, fov_deg=90)
        
        obj_metrics = {
            "is_target": is_target,
            "visibility_ratio": float(visibility_ratio)
        }
        
        # Compute occlusion ratio only for target object
        if is_target:
            occlusion_ratio = compute_occlusion_ratio(c, obj_data, all_objects_data, camera_config, output_dir, view_name)
            obj_metrics["occlusion_ratio"] = float(occlusion_ratio)
        
        metrics[model_name] = obj_metrics
    
    return metrics 