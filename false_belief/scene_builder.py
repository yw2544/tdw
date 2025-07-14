#!/usr/bin/env python3
"""
Scene Builder Module for TDW Room Environment
Handles room scene construction with grid lines, objects, and colored cubes
"""
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
import random
import math

def get_grid_position(grid_x, grid_z):
    """
    Convert grid coordinates to world coordinates
    Grid coordinates range: 0-9 (corresponding to 10x10 grid)
    World coordinates: -4.5 to 4.5
    """
    if not (0 <= grid_x <= 9) or not (0 <= grid_z <= 9):
        raise ValueError(f"Grid coordinates must be in range 0-9, got: ({grid_x}, {grid_z})")
    
    world_x = -4.5 + grid_x * 1.0
    world_z = -4.5 + grid_z * 1.0
    return {"x": world_x, "y": 0, "z": world_z}

def get_center_5x5_positions():
    """
    Get all positions in the center 5x5 grid (grid coordinates 2-6)
    """
    positions = []
    for x in range(2, 7):  # 2,3,4,5,6
        for z in range(2, 7):  # 2,3,4,5,6
            positions.append((x, z))
    return positions

def get_outer_positions():
    """
    Get all grid positions outside the 5x5 center area
    """
    positions = []
    for x in range(10):
        for z in range(10):
            if not (2 <= x <= 6 and 2 <= z <= 6):  # Not in center 5x5 area
                positions.append((x, z))
    return positions

def create_room_base(c: Controller, room_size: int = 12, grid_size: int = 10):
    """
    Create basic room with grid lines
    """
    commands = [
        TDWUtils.create_empty_room(room_size + 2, room_size + 2),
        {"$type": "set_screen_size", "width": 512, "height": 512},
    ]
    
    # Grid line positions: from -5 to 5, one line every meter (11 lines)
    grid_line_positions = []
    for i in range(grid_size + 1):
        pos = -5.0 + i * 1.0  # -5, -4, -3, ..., 4, 5
        grid_line_positions.append(pos)
    
    # Create vertical grid lines (along Z-axis direction)
    for x_pos in grid_line_positions:
        obj_id = c.get_unique_id()
        
        commands.append({
            "$type": "load_primitive_from_resources",
            "primitive_type": "Cube",
            "id": obj_id,
            "position": {
                "x": x_pos, 
                "y": 0.01,
                "z": 0
            },
            "rotation": {"x": 0, "y": 0, "z": 0}
        })
        
        # Scale to thin line
        commands.append({
            "$type": "scale_object",
            "id": obj_id,
            "scale_factor": {"x": 0.02, "y": 0.02, "z": grid_size}
        })
        
        # Set to dark gray
        commands.append({
            "$type": "set_color",
            "id": obj_id,
            "color": {"r": 0.3, "g": 0.3, "b": 0.3, "a": 1.0}
        })
    
    # Create horizontal grid lines (along X-axis direction)  
    for z_pos in grid_line_positions:
        obj_id = c.get_unique_id()
        
        commands.append({
            "$type": "load_primitive_from_resources",
            "primitive_type": "Cube",
            "id": obj_id,
            "position": {
                "x": 0, 
                "y": 0.01,
                "z": z_pos
            },
            "rotation": {"x": 0, "y": 0, "z": 0}
        })
        
        # Scale to thin line
        commands.append({
            "$type": "scale_object", 
            "id": obj_id,
            "scale_factor": {"x": grid_size, "y": 0.02, "z": 0.02}
        })
        
        # Set to dark gray
        commands.append({
            "$type": "set_color",
            "id": obj_id,
            "color": {"r": 0.3, "g": 0.3, "b": 0.3, "a": 1.0}
        })
    
    return commands

def add_target_and_occlusion_objects(c: Controller, target_objects: list, occlusion_objects: list, 
                                    num_occlusion_objects: int = 1):
    """
    Add target and occlusion objects to the scene
    Returns: (commands, target_obj_data, occlusion_objects_data, object_positions)
    """
    # Random selection of objects
    selected_target = random.choice(target_objects)
    selected_occlusion = random.sample(occlusion_objects, min(num_occlusion_objects, len(occlusion_objects)))
    
    # Get center 5x5 positions
    center_positions = get_center_5x5_positions()
    
    # Randomly assign object positions in center 5x5 area
    total_objects = 1 + len(selected_occlusion)  # target + occlusion objects
    if total_objects > len(center_positions):
        raise ValueError(f"Number of objects ({total_objects}) exceeds center 5x5 area capacity ({len(center_positions)})")
    
    object_positions = random.sample(center_positions, total_objects)
    target_position = object_positions[0]
    occlusion_positions = object_positions[1:]
    
    target_world_pos = get_grid_position(target_position[0], target_position[1])
    
    commands = []
    
    # Add target object
    target_id = c.get_unique_id()
    target_rotation = {"x": 0, "y": random.randint(0, 360), "z": 0}
    commands.append(c.get_add_object(
        model_name=selected_target,
        position=target_world_pos,
        rotation=target_rotation,
        object_id=target_id
    ))
    
    # Collect target object data
    target_obj_data = {
        "id": target_id,
        "model": selected_target,
        "position": target_world_pos,
        "rotation": target_rotation,
        "scale": 1.0,
        "grid_position": target_position,
        "type": "target"
    }
    
    # Collect all objects data
    all_objects_data = [target_obj_data]
    
    # Add occlusion objects
    for i, (occlusion_model, pos) in enumerate(zip(selected_occlusion, occlusion_positions)):
        world_pos = get_grid_position(pos[0], pos[1])
        occlusion_id = c.get_unique_id()
        occlusion_rotation = {"x": 0, "y": random.randint(0, 360), "z": 0}
        
        commands.append(c.get_add_object(
            model_name=occlusion_model,
            position=world_pos,
            rotation=occlusion_rotation,
            object_id=occlusion_id
        ))
        
        # Collect occlusion object data
        occlusion_obj_data = {
            "id": occlusion_id,
            "model": occlusion_model,
            "position": world_pos,
            "rotation": occlusion_rotation,
            "scale": 1.0,
            "grid_position": pos,
            "type": "occlusion"
        }
        all_objects_data.append(occlusion_obj_data)
    
    return commands, target_obj_data, all_objects_data, {
        "target_position": target_position,
        "occlusion_positions": occlusion_positions
    }

def add_colored_cubes(c: Controller, colored_positions: list, colors: list):
    """
    Add colored cubes to specified grid positions
    """
    commands = []
    
    for i, (grid_x, grid_z) in enumerate(colored_positions):
        pos = get_grid_position(grid_x, grid_z)
        color_info = colors[i % len(colors)]
        
        # Create colored cube
        cube_id = c.get_unique_id()
        
        commands.append({
            "$type": "load_primitive_from_resources",
            "primitive_type": "Cube",
            "id": cube_id,
            "position": {
                "x": pos["x"], 
                "y": 0.05,  # Slightly above ground to avoid overlapping with grid lines
                "z": pos["z"]
            },
            "rotation": {"x": 0, "y": 0, "z": 0}
        })
        
        # Scale cube to fill grid cell (slightly smaller to leave space for grid lines)
        commands.append({
            "$type": "scale_object",
            "id": cube_id,
            "scale_factor": {"x": 0.9, "y": 0.1, "z": 0.9}
        })
        
        # Set color
        commands.append({
            "$type": "set_color",
            "id": cube_id,
            "color": color_info["rgb"]
        })
    
    return commands

def can_see_target(agent_pos, target_pos, occlusion_positions):
    """
    Check if agent can directly see target (not blocked by occlusion objects)
    Simplified version: check if agent-to-target line passes through any occlusion object's grid
    """
    agent_x, agent_z = agent_pos
    target_x, target_z = target_pos
    
    for occ_x, occ_z in occlusion_positions:
        if is_on_line_between(agent_x, agent_z, target_x, target_z, occ_x, occ_z):
            return False
    
    return True

def is_on_line_between(x1, z1, x2, z2, px, pz):
    """
    Check if point (px, pz) is on the line between points (x1, z1) and (x2, z2)
    """
    # Check if on same line (horizontal, vertical or diagonal)
    if x1 == x2:  # Vertical line
        return px == x1 and min(z1, z2) < pz < max(z1, z2)
    elif z1 == z2:  # Horizontal line
        return pz == z1 and min(x1, x2) < px < max(x1, x2)
    elif abs(x2 - x1) == abs(z2 - z1):  # Diagonal line
        # Check if on diagonal
        dx = 1 if x2 > x1 else -1
        dz = 1 if z2 > z1 else -1
        steps = abs(x2 - x1)
        for i in range(1, steps):
            if x1 + i * dx == px and z1 + i * dz == pz:
                return True
    
    return False

def get_agent_view_angle(agent_pos, target_pos):
    """
    Calculate the angle agent looks toward target (radians)
    """
    agent_x, agent_z = agent_pos
    target_x, target_z = target_pos
    
    dx = target_x - agent_x
    dz = target_z - agent_z
    
    angle = math.atan2(dz, dx)
    return angle

def is_in_view_range(agent_pos, target_pos, test_pos, view_angle_degrees=90):
    """
    Check if test_pos is within agent's view range
    Agent looks toward target, with view_angle_degrees field of view
    """
    agent_x, agent_z = agent_pos
    target_x, target_z = target_pos
    test_x, test_z = test_pos
    
    # Calculate angle agent looks toward target
    target_angle = math.atan2(target_z - agent_z, target_x - agent_x)
    
    # Calculate angle agent looks toward test_pos
    test_angle = math.atan2(test_z - agent_z, test_x - agent_x)
    
    # Calculate angle difference
    angle_diff = abs(target_angle - test_angle)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    
    # Check if within view range
    half_view_angle = math.radians(view_angle_degrees / 2)
    return angle_diff <= half_view_angle

def can_agent_see_position(agent_pos, test_pos, occlusion_positions):
    """
    Check if agent can directly see test_pos (not blocked by occlusion objects)
    """
    agent_x, agent_z = agent_pos
    test_x, test_z = test_pos
    
    # Check if agent-to-test_pos line passes through any occlusion object's grid
    for occ_x, occ_z in occlusion_positions:
        if is_on_line_between(agent_x, agent_z, test_x, test_z, occ_x, occ_z):
            return False
    
    return True

def is_valid_colored_position(agent_pos, target_pos, test_pos, occlusion_positions, view_angle_degrees=90):
    """
    Check if test_pos is a valid colored grid position:
    1. Within agent's view range
    2. Agent can directly see the position (not blocked by occlusion objects)
    """
    # Check if within view range
    if not is_in_view_range(agent_pos, target_pos, test_pos, view_angle_degrees):
        return False
    
    # Check if agent can directly see the position
    if not can_agent_see_position(agent_pos, test_pos, occlusion_positions):
        return False
    
    return True

def select_agent_and_colored_positions(target_position, occlusion_positions, num_colored_grids=5):
    """
    Select agent position and colored grid positions
    Returns: (agent_position, colored_positions)
    """
    outer_positions = get_outer_positions()
    
    # Select agent position that can see target in outer area
    valid_agent_positions = []
    for pos in outer_positions:
        if can_see_target(pos, target_position, occlusion_positions):
            valid_agent_positions.append(pos)
    
    if not valid_agent_positions:
        print("Warning: No agent position found that can see target, using random position")
        valid_agent_positions = outer_positions
    
    agent_position = random.choice(valid_agent_positions)
    
    # Select colored grid positions from remaining grids that agent can see and are within view range
    used_positions = set([target_position] + occlusion_positions + [agent_position])
    available_positions = [(x, z) for x in range(10) for z in range(10) if (x, z) not in used_positions]
    
    # Filter grids that agent can see and are within 90-degree view range
    valid_colored_positions = []
    for pos in available_positions:
        if is_valid_colored_position(agent_position, target_position, pos, occlusion_positions, view_angle_degrees=90):
            valid_colored_positions.append(pos)
    
    if len(valid_colored_positions) < num_colored_grids:
        print(f"Warning: Only {len(valid_colored_positions)} valid colored positions found, adjusting number")
        num_colored_grids = len(valid_colored_positions)
    
    if num_colored_grids > 0:
        colored_positions = random.sample(valid_colored_positions, num_colored_grids)
    else:
        colored_positions = []
    
    return agent_position, colored_positions 