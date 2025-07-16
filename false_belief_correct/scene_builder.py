#!/usr/bin/env python3
"""
Scene Builder Module for TDW Room Environment
Handles room scene construction with grid lines, objects, and colored cubes
"""
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
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

def get_center_6x6_positions():
    """
    Get all positions in the center 6x6 grid (grid coordinates 2-7) 
    """
    positions = []
    for x in range(2, 8):  # 2,3,4,5,6,7
        for z in range(2, 8):  # 2,3,4,5,6,7
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

def get_cardinal_directions():
    """
    Get the four cardinal directions: East, South, West, North
    """
    return [
        {"x": 0, "y": 0, "z": 0},    # East
        {"x": 0, "y": 90, "z": 0},   # South  
        {"x": 0, "y": 180, "z": 0},  # West
        {"x": 0, "y": 270, "z": 0}   # North
    ]

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

def add_target_and_chairs(c: Controller, target_objects: list, chair_objects: list):
    """
    Add one target object and two chairs to the scene in center 5x5 area
    Returns: (commands, target_obj_data, all_objects_data, object_positions)
    """
    # Random selection of objects
    selected_target = random.choice(target_objects)
    selected_chairs = random.sample(chair_objects, 2)  # Select 2 chairs
    
    # Get center 5x5 positions
    center_positions = get_center_5x5_positions()
    
    # Select 3 positions for target + 2 chairs
    object_positions = random.sample(center_positions, 3)
    target_position = object_positions[0]
    chair_positions = object_positions[1:3]
    
    target_world_pos = get_grid_position(target_position[0], target_position[1])
    
    commands = []
    cardinal_directions = get_cardinal_directions()
    
    # Add target object
    target_id = c.get_unique_id()
    target_rotation = random.choice(cardinal_directions)
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
    
    # Add chair objects
    for i, (chair_model, pos) in enumerate(zip(selected_chairs, chair_positions)):
        world_pos = get_grid_position(pos[0], pos[1])
        chair_id = c.get_unique_id()
        chair_rotation = random.choice(cardinal_directions)
        
        commands.append(c.get_add_object(
            model_name=chair_model,
            position=world_pos,
            rotation=chair_rotation,
            object_id=chair_id
        ))
        
        # Collect chair object data
        chair_obj_data = {
            "id": chair_id,
            "model": chair_model,
            "position": world_pos,
            "rotation": chair_rotation,
            "scale": 1.0,
            "grid_position": pos,
            "type": "chair"
        }
        all_objects_data.append(chair_obj_data)
    
    return commands, target_obj_data, all_objects_data, {
        "target_position": target_position,
        "chair_positions": chair_positions
    }

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

def find_agent_view_position(c: Controller, object_manager: "ObjectManager", all_objects_data):
    """
    遍历中间6x6格子，每个格子东南西北四个方向1.5米水平看，
    检测能够看到所有三个物体的地方，保存位置和方向
    
    Args:
        c: TDW Controller
        object_manager: Object manager instance
        all_objects_data: 包含所有物体数据的列表
        
    Returns:
        dict: {
            "position": {"x": float, "y": float, "z": float},
            "grid_position": (int, int),
            "direction": str,  # "east", "south", "west", "north"
            "look_at": {"x": float, "y": float, "z": float}
        } or None if no valid position found
    """
    # 获取中间6x6格子位置
    center_6x6_positions = get_center_6x6_positions()
    
    # 四个方向：东南西北
    directions = [
        {"name": "east", "angle": 0, "look_offset": {"x": 1, "z": 0}},
        {"name": "south", "angle": 90, "look_offset": {"x": 0, "z": -1}},
        {"name": "west", "angle": 180, "look_offset": {"x": -1, "z": 0}},
        {"name": "north", "angle": 270, "look_offset": {"x": 0, "z": 1}}
    ]
    
    # 获取所有物体的ID
    object_ids = [obj["id"] for obj in all_objects_data]
    
    print(f"Looking for agent position that can see all {len(object_ids)} objects...")
    
    # 遍历每个6x6格子位置
    for grid_x, grid_z in center_6x6_positions:
        agent_world_pos = get_grid_position(grid_x, grid_z)
        agent_world_pos["y"] = 1.5  # 1.5米高度
        
        # 遍历四个方向
        for direction in directions:
            # 计算朝向点（距离当前位置2米的点）
            look_at = {
                "x": agent_world_pos["x"] + direction["look_offset"]["x"] * 2,
                "y": 1.5,
                "z": agent_world_pos["z"] + direction["look_offset"]["z"] * 2
            }
            
            print(f"  Testing grid({grid_x}, {grid_z}) looking {direction['name']}...")
            
            # 检查是否能看到所有物体
            can_see_all = True
            visibility_results = {}
            
            for obj_data in all_objects_data:
                obj_id = obj_data["id"]
                obj_model = obj_data["model"]
                obj_type = obj_data["type"]
                
                # 使用can_camera_see_object函数检测
                visibility = can_camera_see_object(
                    c, object_manager, agent_world_pos, look_at, obj_id
                )
                
                visibility_results[obj_id] = {
                    "model": obj_model,
                    "type": obj_type,
                    "visible": visibility["can_see"],
                    "direction": visibility["object_direction"]
                }
                
                if not visibility["can_see"]:
                    can_see_all = False
                    print(f"    Cannot see {obj_model} ({obj_type})")
                    break
                else:
                    print(f"    ✓ Can see {obj_model} ({obj_type}) - {visibility['object_direction']}")
            
            # 如果能看到所有物体，返回这个位置
            if can_see_all:
                print(f"  ✓ Found valid position: grid({grid_x}, {grid_z}) looking {direction['name']}")
                return {
                    "position": agent_world_pos,
                    "grid_position": (grid_x, grid_z),
                    "direction": direction["name"],
                    "look_at": look_at,
                    "visibility_results": visibility_results
                }
            else:
                print(f"    ✗ Cannot see all objects from this position/direction")
    
    print("  ✗ No valid agent position found that can see all objects")
    return None

def select_agent_and_colored_positions_new(all_objects_data, object_manager: "ObjectManager", c: Controller, num_colored_grids=5):
    """
    新的agent和彩色格子位置选择逻辑
    Returns: (agent_info, colored_positions)
    """
    # 找到能看到所有物体的agent位置
    agent_info = find_agent_view_position(c, object_manager, all_objects_data)
    
    if agent_info is None:
        raise ValueError("Cannot find a valid agent position that can see all objects")
    
    # 选择彩色格子位置：在中间6x6格子内随机选择，且在agent视野内
    center_6x6_positions = get_center_6x6_positions()
    
    # 排除已占用的位置
    occupied_positions = set()
    for obj_data in all_objects_data:
        occupied_positions.add(tuple(obj_data["grid_position"]))
    occupied_positions.add(agent_info["grid_position"])
    
    # 找到可用的位置
    available_positions = [pos for pos in center_6x6_positions if pos not in occupied_positions]
    
    # 过滤出在agent视野内的位置
    valid_colored_positions = []
    for pos in available_positions:
        pos_world = get_grid_position(pos[0], pos[1])
        
        # 检查是否在agent视野内（简化版本，检查角度）
        agent_pos = agent_info["position"]
        look_at = agent_info["look_at"]
        
        # 计算agent朝向
        agent_dx = look_at["x"] - agent_pos["x"]
        agent_dz = look_at["z"] - agent_pos["z"]
        agent_angle = math.atan2(agent_dz, agent_dx)
        
        # 计算到测试位置的角度
        test_dx = pos_world["x"] - agent_pos["x"]
        test_dz = pos_world["z"] - agent_pos["z"]
        test_angle = math.atan2(test_dz, test_dx)
        
        # 计算角度差
        angle_diff = abs(test_angle - agent_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # 90度视野范围内
        if angle_diff <= math.radians(45):  # 45度是90度视野的一半
            valid_colored_positions.append(pos)
    
    # 随机选择彩色格子位置
    if len(valid_colored_positions) < num_colored_grids:
        print(f"Warning: Only {len(valid_colored_positions)} valid colored positions found, adjusting number")
        num_colored_grids = len(valid_colored_positions)
    
    if num_colored_grids > 0:
        colored_positions = random.sample(valid_colored_positions, num_colored_grids)
    else:
        colored_positions = []
    
    return agent_info, colored_positions

def select_occluder_position_in_agent_view(agent_info, all_objects_data):
    """
    选择在agent视线范围内的6x6格子位置用于放置occluder
    
    Args:
        agent_info: agent信息，包含位置、朝向等
        all_objects_data: 现有物体数据列表
        
    Returns:
        tuple: (grid_x, grid_z) 或 None 如果没有找到合适位置
    """
    # 获取中间6x6格子位置
    center_6x6_positions = get_center_6x6_positions()
    
    # 排除已占用的位置
    occupied_positions = set()
    for obj_data in all_objects_data:
        occupied_positions.add(tuple(obj_data["grid_position"]))
    occupied_positions.add(agent_info["grid_position"])
    
    # 找到可用的位置
    available_positions = [pos for pos in center_6x6_positions if pos not in occupied_positions]
    
    # 过滤出在agent视野内的位置
    valid_occluder_positions = []
    for pos in available_positions:
        pos_world = get_grid_position(pos[0], pos[1])
        
        # 检查是否在agent视野内（使用与彩色格子相同的逻辑）
        agent_pos = agent_info["position"]
        look_at = agent_info["look_at"]
        
        # 计算agent朝向
        agent_dx = look_at["x"] - agent_pos["x"]
        agent_dz = look_at["z"] - agent_pos["z"]
        agent_angle = math.atan2(agent_dz, agent_dx)
        
        # 计算到测试位置的角度
        test_dx = pos_world["x"] - agent_pos["x"]
        test_dz = pos_world["z"] - agent_pos["z"]
        test_angle = math.atan2(test_dz, test_dx)
        
        # 计算角度差
        angle_diff = abs(test_angle - agent_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # 90度视野范围内
        if angle_diff <= math.radians(45):  # 45度是90度视野的一半
            valid_occluder_positions.append(pos)
    
    if valid_occluder_positions:
        print(f"  Found {len(valid_occluder_positions)} valid occluder positions in agent view (6x6 grid)")
        # 随机选择一个位置
        selected_position = random.choice(valid_occluder_positions)
        print(f"  Selected occluder position: grid{selected_position}")
        return selected_position
    else:
        print("  ⚠️ No valid occluder positions found in agent view (6x6 grid)")
        return None

def extract_coordinate_scalar(point, axis):
    """
    Extract coordinate value from a point object
    Args:
        point: A 3D coordinate point (could be array or dict-like)
        axis: 0 for x, 1 for y, 2 for z
    Returns:
        float: The coordinate value
    """
    if hasattr(point, '__getitem__'):
        if isinstance(point, dict):
            coords = ['x', 'y', 'z']
            return point.get(coords[axis], 0.0)
        else:
            # Assume it's an array-like object
            return float(point[axis])
    else:
        # If it's a single value, assume it's the requested coordinate
        return float(point)

class ObjectManager:
    """
    Simplified object manager to track objects in the scene
    """
    def __init__(self):
        self.objects_static = {}
        self.transforms = {}
        self.bounds = {}
    
    def add_object(self, object_id, model_name, position, rotation, scale=1.0):
        """Add object to tracking"""
        self.objects_static[object_id] = model_name
        self.transforms[object_id] = type('Transform', (), {
            'position': [position["x"], position["y"], position["z"]]
        })()
        
        # Create a simple bounds object
        # In real TDW, this would come from actual object bounds
        # For now, create a simplified version
        bounds_dict = {
            'left': [position["x"] - 0.5, position["y"], position["z"]],
            'right': [position["x"] + 0.5, position["y"], position["z"]],
            'front': [position["x"], position["y"], position["z"] + 0.5],
            'back': [position["x"], position["y"], position["z"] - 0.5],
            'top': [position["x"], position["y"] + 1.0, position["z"]],
            'bottom': [position["x"], position["y"], position["z"]],
            'center': [position["x"], position["y"], position["z"]]
        }
        
        self.bounds[object_id] = type('Bounds', (), bounds_dict)()

def can_camera_see_object(c: Controller, object_manager: "ObjectManager", camera_position, camera_look_at, target_object_id, HORIZONTAL_FOV=90):
    """
    检测摄像机在指定位置和朝向是否可以看到目标物体
    要求物体完全在视野内且无遮挡才算看见
    
    Args:
        c: TDW Controller
        object_manager: Object manager instance
        camera_position: 摄像机位置 {"x": float, "y": float, "z": float}
        camera_look_at: 摄像机朝向的目标点 {"x": float, "y": float, "z": float}
        target_object_id: 目标物体ID
        HORIZONTAL_FOV: 水平视野角度（默认90度）
        
    Returns:
        dict: 包含可见性、图片路径和物体朝向的信息
            {
                "can_see": bool,
                "image_path": str or None,
                "object_direction": str or None  # "front", "back", "left", "right"
            }
    """
    result = {
        "can_see": False,
        "image_path": None,
        "object_direction": None
    }
    
    # 获取目标物体的位置和边界
    if target_object_id not in object_manager.objects_static:
        print(f"Object {target_object_id} not found")
        return result
        
    target_transform = object_manager.transforms[target_object_id]
    target_bounds = object_manager.bounds[target_object_id]
    target_position = target_transform.position
    
    # 计算摄像机的朝向角度
    cam_dx = camera_look_at["x"] - camera_position["x"]
    cam_dz = camera_look_at["z"] - camera_position["z"]
    camera_angle = math.degrees(math.atan2(cam_dx, cam_dz))
    if camera_angle < 0:
        camera_angle += 360
    
    # 边界框的每个面都是三维坐标点，从中提取对应轴的坐标值
    # 找到所有边界点中各轴的最值
    all_boundary_points = [
        target_bounds.left, target_bounds.right, target_bounds.front, 
        target_bounds.back, target_bounds.top, target_bounds.bottom
    ]
    
    # 提取所有边界点的x坐标，找最小值和最大值
    x_coords = [extract_coordinate_scalar(point, 0) for point in all_boundary_points]
    min_x = min(x_coords)
    max_x = max(x_coords)
    
    # 提取所有边界点的z坐标，找最小值和最大值  
    z_coords = [extract_coordinate_scalar(point, 2) for point in all_boundary_points]
    min_z = min(z_coords)
    max_z = max(z_coords)
    
    half_width = (max_x - min_x) / 2
    half_depth = (max_z - min_z) / 2
    
    # target_position 是三维坐标数组 [x, y, z]
    target_x = target_bounds.center[0]  # x 坐标
    target_z = target_bounds.center[2]  # z 坐标
    
    corner_points = [
        [target_x - half_width, target_z - half_depth],  # 左后
        [target_x + half_width, target_z - half_depth],  # 右后
        [target_x - half_width, target_z + half_depth],  # 左前
        [target_x + half_width, target_z + half_depth],  # 右前
    ]
    
    # 检查所有角点是否都在视野内
    for corner_x, corner_z in corner_points:
        # 计算角点相对于摄像机的角度
        corner_dx = corner_x - camera_position["x"]
        corner_dz = corner_z - camera_position["z"]
        corner_angle = math.degrees(math.atan2(corner_dx, corner_dz))
        if corner_angle < 0:
            corner_angle += 360
        
        # 计算角度差
        angle_diff = abs(corner_angle - camera_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        # 如果任何一个角点超出视野范围，则物体不完全可见
        if angle_diff > HORIZONTAL_FOV / 2:
            return result
    
    # 如果所有角点都在视野内，则物体可见
    result["can_see"] = True
    
    # 计算物体相对于摄像机的方向
    obj_dx = target_x - camera_position["x"]
    obj_dz = target_z - camera_position["z"]
    obj_angle = math.degrees(math.atan2(obj_dx, obj_dz))
    if obj_angle < 0:
        obj_angle += 360
    
    # 确定物体朝向
    angle_to_obj = obj_angle - camera_angle
    if angle_to_obj < 0:
        angle_to_obj += 360
    
    if 315 <= angle_to_obj or angle_to_obj < 45:
        result["object_direction"] = "front"
    elif 45 <= angle_to_obj < 135:
        result["object_direction"] = "right"
    elif 135 <= angle_to_obj < 225:
        result["object_direction"] = "back"
    else:
        result["object_direction"] = "left"
    
    return result

def select_occluder_position_in_agent_view(agent_info, all_objects_data):
    """
    选择在agent视线范围内的6x6格子位置用于放置occluder
    
    Args:
        agent_info: agent信息，包含位置、朝向等
        all_objects_data: 现有物体数据列表
        
    Returns:
        tuple: (grid_x, grid_z) 或 None 如果没有找到合适位置
    """
    # 获取中间6x6格子位置
    center_6x6_positions = get_center_6x6_positions()
    
    # 排除已占用的位置
    occupied_positions = set()
    for obj_data in all_objects_data:
        occupied_positions.add(tuple(obj_data["grid_position"]))
    occupied_positions.add(agent_info["grid_position"])
    
    # 找到可用的位置
    available_positions = [pos for pos in center_6x6_positions if pos not in occupied_positions]
    
    # 过滤出在agent视野内的位置
    valid_occluder_positions = []
    for pos in available_positions:
        pos_world = get_grid_position(pos[0], pos[1])
        
        # 检查是否在agent视野内（使用与彩色格子相同的逻辑）
        agent_pos = agent_info["position"]
        look_at = agent_info["look_at"]
        
        # 计算agent朝向
        agent_dx = look_at["x"] - agent_pos["x"]
        agent_dz = look_at["z"] - agent_pos["z"]
        agent_angle = math.atan2(agent_dz, agent_dx)
        
        # 计算到测试位置的角度
        test_dx = pos_world["x"] - agent_pos["x"]
        test_dz = pos_world["z"] - agent_pos["z"]
        test_angle = math.atan2(test_dz, test_dx)
        
        # 计算角度差
        angle_diff = abs(test_angle - agent_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # 90度视野范围内
        if angle_diff <= math.radians(45):  # 45度是90度视野的一半
            valid_occluder_positions.append(pos)
    
    if valid_occluder_positions:
        print(f"  Found {len(valid_occluder_positions)} valid occluder positions in agent view (6x6 grid)")
        # 随机选择一个位置
        selected_position = random.choice(valid_occluder_positions)
        print(f"  Selected occluder position: grid{selected_position}")
        return selected_position
    else:
        print("  ⚠️ No valid occluder positions found in agent view (6x6 grid)")
        return None