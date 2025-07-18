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
    # if not (0 <= grid_x <= 9) or not (0 <= grid_z <= 9):
    #     raise ValueError(f"Grid coordinates must be in range 0-9, got: ({grid_x}, {grid_z})")
    
    world_x = -4.5 + grid_x * 1.0
    world_z = -4.5 + grid_z * 1.0
    return {"x": world_x, "y": 0, "z": world_z}
def get_center_3x3_positions():
    """
    Get all positions in the center 5x5 grid (grid coordinates 2-6)
    """
    positions = []
    for x in range(3, 6):  # 2,3,4,5,6
        for z in range(3, 6):  # 2,3,4,5,6
            positions.append((x, z))
    return positions
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
        {"$type": "set_screen_size", "width": 1024, "height": 1024},
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

def add_target_and_chairs(c: Controller, chair_objects: list):
    """
    Add three chair objects to the scene in center 5x5 area
    All chairs use the same model but with different colors
    Returns: (commands, target_obj_data, all_objects_data, object_positions)
    """
    # Use the same chair model for all chairs
    selected_chair_model = chair_objects[0] if chair_objects else "chair_generic"
    
    # Get center 5x5 positions
    center_positions = get_center_5x5_positions()
    
    # Select 3 positions ensuring no same row or column
    def select_positions_no_overlap(positions, count=3):
        """选择指定数量的位置，确保没有相同的行或列"""
        max_attempts = 100
        for attempt in range(max_attempts):
            selected = random.sample(positions, count)
            
            # 检查是否有相同的行或列
            rows = [pos[0] for pos in selected]
            cols = [pos[1] for pos in selected]
            
            # 如果行数和列数都没有重复，则返回这个选择
            if len(set(rows)) == len(rows) and len(set(cols)) == len(cols):
                return selected
        
        # 如果尝试多次都失败，返回随机选择（作为后备方案）
        print("Warning: Could not find positions without row/column overlap, using random selection")
        return random.sample(positions, count)
    
    object_positions = select_positions_no_overlap(center_positions, 3)
    
    print(f"Selected positions: {object_positions}")
    
    commands = []
    cardinal_directions = get_cardinal_directions()
    all_objects_data = []
    
    # Define colors for the three chairs: red, green, blue
    chair_colors = [
        {"rgb": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}, "name": "red"},
        {"rgb": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0}, "name": "green"},
        {"rgb": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}, "name": "blue"}
    ]
    
    chair_types = ["target", "chair", "chair"]
    
    # Add three chairs with different colors
    for i, (position, color_info, chair_type) in enumerate(zip(object_positions, chair_colors, chair_types)):
        world_pos = get_grid_position(position[0], position[1])
        chair_id = c.get_unique_id()
        chair_rotation = random.choice(cardinal_directions)
        
        # Add chair object
        commands.append(c.get_add_object(
            model_name=selected_chair_model,
            position=world_pos,
            rotation=chair_rotation,
            object_id=chair_id
        ))
        
        # Set chair color
        commands.append({
            "$type": "set_color",
            "id": chair_id,
            "color": color_info["rgb"]
        })
        
        # Collect chair object data
        chair_obj_data = {
            "id": chair_id,
            "model": color_info["name"]+ " chair",
            "position": world_pos,
            "rotation": chair_rotation,
            "scale": 1.0,
            "grid_position": position,
            "type": chair_type,
            "color": color_info["name"]
        }
        all_objects_data.append(chair_obj_data)
    
    # First chair is the target
    target_obj_data = all_objects_data[0]
    
    return commands, target_obj_data, all_objects_data, {
        "target_position": object_positions[0],
        "chair_positions": object_positions[1:3]
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
    # center_6x6_positions = get_center_6x6_positions()
    center_6x6_positions = [(5,-2)]
    
    # 四个方向：东南西北
    directions = [
        # {"name": "east", "angle": 0, "look_offset": {"x": 1, "z": 0}},
        # {"name": "south", "angle": 90, "look_offset": {"x": 0, "z": -1}},
        # {"name": "west", "angle": 180, "look_offset": {"x": -1, "z": 0}},
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
                # visibility = can_camera_see_object(
                #     c, object_manager, agent_world_pos, look_at, obj_id
                # )
                # print(visibility)
                visibility = {
                    "can_see": True,
                    "image_path": None,
                    "object_direction":  'front'
                }
                
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

def select_colored_positions(all_objects_data, object_manager: "ObjectManager", c: Controller, agent_info, num_colored_grids=5):
    """
    先构造问题，再验证答案的逻辑，并保存旋转后的图片
    采用更好的架构：先分类，再统一审核
    Returns: (colored_grid_infos, question_data)
    """
    
    # 获取目标物体（红色椅子）
    target_obj = next(obj for obj in all_objects_data if obj["type"] == "target")
    print(target_obj)
    
    # 随机选择物体旋转角度和目标方向
    rotation_angles = [90, 180, 270]
    target_rotation = random.choice(rotation_angles)
    rotation_desc = f"rotated {target_rotation} degrees clockwise"
    
    # 保存原始旋转角度，然后应用新的旋转
    print(target_obj["rotation"], rotation_desc)
    
    # 应用旋转到目标物体
    rotation_command = {
        "$type": "rotate_object_by",
        "angle": target_rotation,
        "axis": "yaw",
        "id": target_obj["id"],
        "is_world": True
    }
    target_obj['rotation']['y']=(target_obj['rotation']['y']+target_rotation) % 360
    object_manager.update_rotation(target_obj["id"],target_obj['rotation'])
    c.communicate([rotation_command])
    
    orientation_options = ["facing forward", "facing backward", "facing left", "facing right"]
    target_orientation = random.choice(orientation_options)
    orientation_desc = target_orientation
    
    color_options = ["red", "yellow", "blue", "green", "purple"]
    
    # 构造prompt模板
    prompt_template = (
        f"An object {rotation_desc}. "
        f"You need to find the best colored grid cell to place a camera so that from the camera view, "
        f"the object appears to be oriented {orientation_desc}. "
        f"Available colored grid options are: {', '.join(color_options)}. "
        f"Please answer with the color name (e.g., red, blue, yellow, green) and your orientation."
    )
    
    # 获取所有可用位置
    all_positions = [(x, z) for x in range(10) for z in range(10)]
    occupied_positions = set()
    for obj_data in all_objects_data:
        occupied_positions.add(tuple(obj_data["grid_position"]))
    occupied_positions.add(agent_info["grid_position"])
    
    available_positions = [pos for pos in all_positions if pos not in occupied_positions]
    
    # 第一步：分类所有可用位置为正确答案和错误答案
    correct_grids = []
    wrong_grids = []
    
    test_directions = [
        {"offset": {"x": 0, "z": 1}, "name": "north"},
        {"offset": {"x": 1, "z": 0}, "name": "east"}, 
        {"offset": {"x": 0, "z": -1}, "name": "south"},
        {"offset": {"x": -1, "z": 0}, "name": "west"}
    ]
    
    direction_mapping = {
        "front": "facing forward",
        "back": "facing backward", 
        "left": "facing left",
        "right": "facing right"
    }
    
    print(f"Classifying {len(available_positions)} positions into correct/wrong answers...")
    
    # 遍历所有可用位置进行分类
    for pos in available_positions:
        grid_x, grid_z = pos
        is_correct_position = False
        best_direction_info = None
        
        # 测试该位置的四个朝向
        for direction in test_directions:
            cam_pos = get_grid_position(grid_x, grid_z)
            cam_pos["y"] = 1.5
            
            look_at = {
                "x": cam_pos["x"] + direction["offset"]["x"],
                "y": 1.5,
                "z": cam_pos["z"] + direction["offset"]["z"]
            }
            
            # 检查能否看到目标物体
            visibility = can_camera_see_object(c, object_manager, cam_pos, look_at, target_obj["id"])
            
            if visibility["can_see"]:
                obj_direction = visibility["object_direction"]
                
                # 检查是否匹配目标方向
                if direction_mapping.get(obj_direction) == target_orientation:
                    is_correct_position = True
                    best_direction_info = {
                        "camera_position": cam_pos,
                        "look_at": look_at,
                        "object_direction": obj_direction,
                        "camera_direction": direction["name"]
                    }
                    break  # 找到正确方向就不需要继续测试
        
        # 根据测试结果分类
        grid_info = {
            "grid_position": pos,
            "is_correct_answer": is_correct_position
        }
        
        if is_correct_position:
            grid_info.update(best_direction_info)
            correct_grids.append(grid_info)
        else:
            # 对于错误答案，使用默认的朝向设置
            cam_pos = get_grid_position(grid_x, grid_z)
            cam_pos["y"] = 1.5
            grid_info.update({
                "camera_position": cam_pos,
                "look_at": {"x": cam_pos["x"], "y": 1.5, "z": cam_pos["z"] + 2},
                "object_direction": "unknown",
                "camera_direction": "north"
            })
            wrong_grids.append(grid_info)
    
    print(f"Found {len(correct_grids)} correct positions and {len(wrong_grids)} wrong positions")
    
    # 第二步：对所有候选位置进行agent视野范围验证
    def validate_agent_visibility(grid_info):
        """检查位置是否在agent视野内"""
        pos = grid_info["grid_position"]
        pos_world = get_grid_position(pos[0], pos[1])
        
        agent_pos = agent_info["position"]
        look_at_agent = agent_info["look_at"]
        
        agent_dx = look_at_agent["x"] - agent_pos["x"]
        agent_dz = look_at_agent["z"] - agent_pos["z"]
        agent_angle = math.atan2(agent_dz, agent_dx)
        
        test_dx = pos_world["x"] - agent_pos["x"]
        test_dz = pos_world["z"] - agent_pos["z"]
        test_angle = math.atan2(test_dz, test_dx)
        
        angle_diff = abs(test_angle - agent_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        agent_grid_pos = agent_info["grid_position"]
        
        # 获取遮挡物位置
        occlusion_positions = [obj["grid_position"] for obj in all_objects_data if obj["type"] == "occlusion"]
        
        # 使用现有的can_see_target函数检查可见性
        return can_see_target(agent_grid_pos, pos, occlusion_positions) and  angle_diff <= math.radians(45)
    
    # 验证正确答案的可见性
    valid_correct_grids = [grid for grid in correct_grids if validate_agent_visibility(grid)]
    print(f"Valid correct grids (in agent view): {len(valid_correct_grids)}")
    
    if not valid_correct_grids:
        raise NotImplementedError(prompt_template)
    
    # 验证错误答案的可见性
    valid_wrong_grids = [grid for grid in wrong_grids if validate_agent_visibility(grid)]
    print(f"Valid wrong grids (in agent view): {len(valid_wrong_grids)}")
    
    # 第三步：选择最终的网格（1个正确 + 3个错误）
    all_grid_infos = []
    correct_colors = []
    
    # 选择一个正确答案
    selected_correct = random.choice(valid_correct_grids)
    selected_correct["color"] = color_options[0]  # 红色
    correct_colors.append(color_options[0])
    all_grid_infos.append(selected_correct)
    print(f"Selected correct answer: {selected_correct['grid_position']} - {selected_correct['color']}")
    
    # 选择3个错误答案
    if len(valid_wrong_grids) < 3:
        print(f"Warning: Only {len(valid_wrong_grids)} wrong grids available, need 3")
    
    selected_wrong = random.sample(valid_wrong_grids, min(3, len(valid_wrong_grids)))
    
    for i, wrong_grid in enumerate(selected_wrong):
        color_name = color_options[len(all_grid_infos)]
        wrong_grid["color"] = color_name
        all_grid_infos.append(wrong_grid)
        print(f"Selected wrong answer {i+1}: {wrong_grid['grid_position']} - {wrong_grid['color']}")
    
    # 创建问题数据
    question_data = {
        "prompt": prompt_template,
        "rotation_desc": rotation_desc,
        "target_rotation": target_rotation,
        "orientation_desc": orientation_desc, 
        "target_orientation": target_orientation,
        "correct_answers": correct_colors,
        "all_options": color_options,
    }
    
    return all_grid_infos, question_data

def add_occlusion_object(c: Controller, object_manager: "ObjectManager", all_objects_data, occlusion_objects):
    """
    Add occlusion object in agent view
    
    Args:
        c: TDW Controller
        object_manager: Object manager instance
        agent_info: Agent information
        all_objects_data: List of existing objects
        occlusion_objects: List of available occlusion object models
        
    Returns:
        dict: Occlusion object data or None if no position found
    """
    occlusion_position = select_occluder_position_in_agent_view( all_objects_data)
    
    if occlusion_position:
        selected_occlusion = random.choice(occlusion_objects)
        occlusion_world_pos = get_grid_position(occlusion_position[0], occlusion_position[1])
        
        occlusion_id = c.get_unique_id()
        cardinal_directions = get_cardinal_directions()
        occlusion_rotation = random.choice(cardinal_directions)
        
        occlusion_command = c.get_add_object(
            model_name=selected_occlusion,
            position=occlusion_world_pos,
            rotation=occlusion_rotation,
            object_id=occlusion_id
        )
        
        c.communicate([occlusion_command])
        
        print(f"🚧 Added occlusion object: {selected_occlusion} at grid{occlusion_position}")
        
        # Create object data
        occlusion_obj_data = {
            "id": occlusion_id,
            "model": selected_occlusion,
            "position": occlusion_world_pos,
            "rotation": occlusion_rotation,
            "scale": 1.0,
            "grid_position": occlusion_position,
            "type": "occlusion"
        }
        
        # Add to object manager
        object_manager.add_object(
            occlusion_id, selected_occlusion, occlusion_world_pos, 
            occlusion_rotation, "occlusion", 1.0
        )
        
        return occlusion_obj_data
    else:
        print("⚠️ No available positions for occlusion object")
        return None

def select_occluder_position_in_agent_view(all_objects_data):
    """
    选择在agent视线范围内的6x6格子位置用于放置occluder
    
    Args:
        agent_info: agent信息，包含位置、朝向等
        all_objects_data: 现有物体数据列表
        
    Returns:
        tuple: (grid_x, grid_z) 或 None 如果没有找到合适位置
    """
    # 获取中间6x6格子位置
    center_6x6_positions = get_center_3x3_positions()
    
    # 排除已占用的位置
    occupied_positions = set()
    for obj_data in all_objects_data:
        occupied_positions.add(tuple(obj_data["grid_position"]))
    
    # 找到可用的位置
    available_positions = [pos for pos in center_6x6_positions if pos not in occupied_positions]
    
    if available_positions:
        print(f"  Found {len(available_positions)} valid occluder positions in agent view (6x6 grid)")
        # 随机选择一个位置
        selected_position = random.choice(available_positions)
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
        self.rotations = {}  # 添加旋转信息存储
        self.target_id = None
        self.all_ids = []
        self.occlusion_id = None
        
    
    def add_object(self, object_id, model_name, position, rotation, object_type = None, scale=1.0):
        """Add object to tracking"""
        self.objects_static[object_id] = model_name
        self.transforms[object_id] = type('Transform', (), {
            'position': [position["x"], position["y"], position["z"]]
        })()
        
        # 存储旋转信息
        self.rotations[object_id] = rotation
        
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
        if object_type=="occlusion":
            self.occlusion_id = object_id
        elif object_type=="target":
            self.target_id = object_id
        self.all_ids.append(object_id)
    
    def update_rotation(self, object_id, rotation):
        """Update object rotation"""
        if object_id in self.rotations:
            self.rotations[object_id] = rotation
def _ray_intersects_box_2d(ray_start, ray_end_2d, box_bounds):
    """
    检测射线是否与边界框相交（仅考虑x,z平面的2D投影）
    
    Args:
        ray_start: 射线起点 {"x": float, "y": float, "z": float}
        ray_end_2d: 射线终点 [x, z] (仅2D)
        box_bounds: 边界框信息
        
    Returns:
        bool: True表示相交
    """
    # 计算边界框在x,z平面的最小和最大坐标
    all_boundary_points = [
        box_bounds.left, box_bounds.right, box_bounds.front,
        box_bounds.back, box_bounds.top, box_bounds.bottom
    ]
    
    # 找x,z轴的最值
    x_coords = [extract_coordinate_scalar(point, 0) for point in all_boundary_points]
    z_coords = [extract_coordinate_scalar(point, 2) for point in all_boundary_points]
    
    box_min_2d = [min(x_coords), min(z_coords)]
    box_max_2d = [max(x_coords), max(z_coords)]
    
    # 使用线段与2D AABB的相交测试
    return _line_segment_intersects_aabb_2d(
        [ray_start["x"], ray_start["z"]], 
        ray_end_2d, 
        box_min_2d, 
        box_max_2d
    )

def _line_segment_intersects_aabb_2d(start, end, box_min, box_max):
    """
    检测线段是否与轴对齐边界框(AABB)相交（仅2D x,z平面）
    """
    # 计算线段方向
    direction = [end[i] - start[i] for i in range(2)]  # 仅x,z两个维度
    
    t_min = 0.0
    t_max = 1.0
    
    for i in range(2):  # 仅检查x,z轴
        if abs(direction[i]) < 1e-8:  # 射线平行于某个轴
            if start[i] < box_min[i] or start[i] > box_max[i]:
                return False
        else:
            t1 = (box_min[i] - start[i]) / direction[i]
            t2 = (box_max[i] - start[i]) / direction[i]
            
            if t1 > t2:
                t1, t2 = t2, t1
                
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            
            if t_min > t_max:
                return False
                
    return True
def _is_object_between_camera_and_point_2d(camera_pos, target_point_2d, obstacle_pos):
    """
    检查障碍物是否在摄像机和目标点之间（仅考虑x,z平面）
    """
    # 计算摄像机到目标点的距离（仅x,z平面）
    cam_to_target_dist = math.sqrt(
        (target_point_2d[0] - camera_pos["x"])**2 + 
        (target_point_2d[1] - camera_pos["z"])**2
    )
    
    # 计算摄像机到障碍物的距离（仅x,z平面）
    obstacle_x = extract_coordinate_scalar(obstacle_pos, 0)  # x 坐标
    obstacle_z = extract_coordinate_scalar(obstacle_pos, 2)  # z 坐标
    cam_to_obstacle_dist = math.sqrt(
        (obstacle_x - camera_pos["x"])**2 + 
        (obstacle_z - camera_pos["z"])**2
    )
    
    # 障碍物必须在摄像机和目标之间才能造成遮挡
    return cam_to_obstacle_dist < cam_to_target_dist
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
        big_transform = object_manager.transforms[object_manager.occlusion_id]
        big_bounds = object_manager.bounds[object_manager.occlusion_id]
        
        # 检查到所有角点的射线是否被遮挡（仅在x,z平面上）
        for corner_x, corner_z in corner_points:
            # 仅使用x,z坐标进行2D遮挡检测
            corner_2d = [corner_x, corner_z]
            
            # 检查射线是否与大物体的边界框相交（仅x,z平面）
            if _ray_intersects_box_2d(camera_position, corner_2d, big_bounds):
                # 进一步检查遮挡物是否真的在摄像机和角点之间
                if _is_object_between_camera_and_point_2d(camera_position, corner_2d, big_transform.position):
                    return result
    # 如果所有角点都在视野内，则物体可见
    result["can_see"] = True
    
    # 直接从object_manager获取物体的旋转角度
    # object_rotation_y = 0  # 默认值
    
    # 从all_objects_data中查找目标物体的旋转信息
    # 这需要object_manager维护物体的旋转信息
    object_rotation_y = object_manager.rotations[target_object_id]['y']
    
    # 计算摄像机相对于物体的角度
    obj_dx = target_x - camera_position["x"]
    obj_dz = target_z - camera_position["z"]
    camera_to_obj_angle = math.degrees(math.atan2(obj_dx, obj_dz))
    if camera_to_obj_angle < 0:
        camera_to_obj_angle += 360
    
    # 计算摄像机相对于物体正面的角度
    relative_camera_angle = (camera_to_obj_angle - object_rotation_y) % 360
    
    # 根据相对角度确定物体在摄像机视野中的朝向
    if 315 <= relative_camera_angle or relative_camera_angle < 45:
        result["object_direction"] = "front"   
    elif 45 <= relative_camera_angle < 135:
        result["object_direction"] = "left"    
    elif 135 <= relative_camera_angle < 225:
        result["object_direction"] = "back"     
    else:
        result["object_direction"] = "right"     
    
    return result


