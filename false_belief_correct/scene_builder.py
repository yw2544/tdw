def is_point_in_fov(camera_position, camera_look_at, point, horizontal_fov=90):
    """
    判断某个点是否在摄像机的水平视野范围内。
    camera_position: 摄像机位置 {"x": float, "y": float, "z": float}
    camera_look_at: 摄像机朝向的目标点 {"x": float, "y": float, "z": float}
    point: 目标点 [x, z] 或 (x, z)
    horizontal_fov: 视野角度（度）
    """
    cam_dx = camera_look_at["x"] - camera_position["x"]
    cam_dz = camera_look_at["z"] - camera_position["z"]
    camera_angle = math.degrees(math.atan2(cam_dx, cam_dz))
    if camera_angle < 0:
        camera_angle += 360

    point_dx = point[0] - camera_position["x"]
    point_dz = point[1] - camera_position["z"]
    point_angle = math.degrees(math.atan2(point_dx, point_dz))
    if point_angle < 0:
        point_angle += 360

    angle_diff = abs(point_angle - camera_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return angle_diff <= horizontal_fov / 2
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
from pathlib import Path

def get_center(n):
    """
    Get all positions in the center n x n grid using actual world coordinates
    
    Args:
        n: Size of the center grid (e.g., 3 for 3x3, 5 for 5x5)
        
    Returns:
        List of (x, z) world coordinate tuples
    """
    positions = []
    start = -(n // 2)
    end = start + n
    
    for x in range(start, end):
        for z in range(start, end):
            positions.append((x, z))
    
    return positions


def create_room_base(c: Controller, room_size: int = 12):
    """
    Create basic room with grid lines
    """
    commands = [
        TDWUtils.create_empty_room(room_size+2, room_size+2),
        {"$type": "set_screen_size", "width": 512, "height": 512},
    ]
    
    return commands

def add_target_and_chairs(c: Controller, chair_objects: list, object_manager: "ObjectManager"):
    """
    Add three chair objects to the scene and register them in object manager
    Returns: (commands, target_obj_data, positions)
    """
    # Use the same chair model for all chairs
    selected_chair_model = chair_objects[0] if chair_objects else "chair_generic"
    
    # Get center 5x5 positions using world coordinates
    center_positions = get_center(5)
    
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
    chair_colors = [
        {"rgb": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}, "name": "red"},
        {"rgb": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0}, "name": "green"},
        {"rgb": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}, "name": "blue"}
    ]
    chair_types = ["target", "chair", "chair"]
    
    # Add chairs and register in object manager
    for i, (position, color_info, chair_type) in enumerate(zip(object_positions, chair_colors, chair_types)):
        world_pos = {"x": position[0], "y": 0, "z": position[1]}
        chair_id = c.get_unique_id()
        chair_rotation = random.choice(cardinal_directions)
        
        # TDW commands
        commands.append(c.get_add_object(
            model_name=selected_chair_model,
            position=world_pos,
            rotation=chair_rotation,
            object_id=chair_id
        ))
        commands.append({
            "$type": "set_color",
            "id": chair_id,
            "color": color_info["rgb"]
        })
        
        # Register in object manager
        object_manager.add_object(
            chair_id, 
            color_info["name"] + " chair",
            world_pos,
            chair_rotation,
            chair_type,
            1.0,
            color=color_info["name"]
        )
    
    target_obj_data = object_manager.get_target_object()
    
    return commands, target_obj_data, {
        "target_position": object_positions[0],
        "chair_positions": object_positions[1:3]
    }

def add_colored_cubes(c: Controller, object_manager: "ObjectManager", colored_positions: list, selected_colors: list):
    """
    Add colored cubes to specified grid positions
    如果传入object_manager，则只保存cube的id到object_manager.colored_cube_ids
    """
    colors = {
        "red": {"r": 0.8, "g": 0.1, "b": 0.1, "a": 0.8},
        "yellow":  {"r": 0.9, "g": 0.9, "b": 0.1, "a": 0.8},
        "blue": {"r": 0.1, "g": 0.3, "b": 0.8, "a": 0.8},
        "green":  {"r": 0.1, "g": 0.7, "b": 0.1, "a": 0.8},
        "purple":  {"r": 0.6, "g": 0.1, "b": 0.8, "a": 0.8}
    }
    
    commands = []
    colored_cube_ids = []
    for i, (grid_x, grid_z) in enumerate(colored_positions):
        pos = (grid_x, grid_z)
        cube_id = c.get_unique_id()
        commands.append({
            "$type": "load_primitive_from_resources",
            "primitive_type": "Cube",
            "id": cube_id,
            "position": {
                "x": pos[0], 
                "y": 0.05,
                "z": pos[1]
            },
            "rotation": {"x": 0, "y": 0, "z": 0}
        })
        commands.append({
            "$type": "scale_object",
            "id": cube_id,
            "scale_factor": {"x": 0.9, "y": 0.1, "z": 0.9}
        })
        commands.append({
            "$type": "set_color",
            "id": cube_id,
            "color": colors[selected_colors[i]]
        })
        colored_cube_ids.append(cube_id)
    
    object_manager.colored_cube_ids.extend(colored_cube_ids)
    return commands


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


        
def find_agent_view_position(c: Controller, object_manager: "ObjectManager"):
    """
    Find agent position that can see all objects
    """
    center_6x6_positions = [(0, -6)]
    directions = [
        {"name": "north", "angle": 270, "look_offset": {"x": 0, "z": 1}}
    ]
    
    all_objects = object_manager.get_all_objects()
    print(f"Looking for agent position that can see all {len(all_objects)} objects...")
    
    for world_x, world_z in center_6x6_positions:
        agent_world_pos = {"x": world_x, "y": 1.5, "z": world_z}
        
        for direction in directions:
            look_at = {
                "x": agent_world_pos["x"] + direction["look_offset"]["x"] * 2,
                "y": 1.5,
                "z": agent_world_pos["z"] + direction["look_offset"]["z"] * 2
            }
            
            print(f"  Testing position({world_x}, {world_z}) looking {direction['name']}...")
            
            can_see_all = True
            visibility_results = {}
            
            for obj_data in all_objects:
                visibility = {
                    "can_see": True,
                    "image_path": None,
                    "object_direction": 'front'
                }
                
                visibility_results[obj_data["id"]] = {
                    "model": obj_data["model"],
                    "type": obj_data["type"],
                    "visible": visibility["can_see"],
                    "direction": visibility["object_direction"]
                }
                
                if not visibility["can_see"]:
                    can_see_all = False
                    break
            
            if can_see_all:
                print(f"  ✓ Found valid position: ({world_x}, {world_z}) looking {direction['name']}")
                return {
                    "position": agent_world_pos,
                    "direction": direction["name"],
                    "look_at": look_at,
                    "visibility_results": visibility_results
                }
    
    return None

def select_colored_positions(object_manager: "ObjectManager", agent_info, rotation_desc=None):
    """
    Select colored positions using object manager
    """
    # 获取目标物体
    target_obj = object_manager.get_target_object()
        
    orientation_options = ["facing forward", "facing backward", "facing left", "facing right"]
    target_orientation_idx = random.randint(0, 3)
    while True:
        orientation_desc = orientation_options[target_orientation_idx]
        
        color_options = ["red", "yellow", "blue", "green"]
        
        # 构造prompt模板
        if rotation_desc is not None:
            prompt = (
                f"An object {rotation_desc}. "
                f"You need to find the best colored grid cell to place a camera so that from the camera view, "
                f"the object appears to be oriented {orientation_desc}. "
                f"Available colored grid options are: {', '.join(color_options)}. "
                f"Please answer with the color name (e.g., red, blue, yellow, green) and your orientation."
            )
        else:
            prompt = (
                f"You need to find the best colored grid cell to place a camera so that from the camera view, "
                f"the object appears to be oriented {orientation_desc}. "
                f"Available colored grid options are: {', '.join(color_options)}. "
                f"Please answer with the color name (e.g., red, blue, yellow, green) and your orientation."
            )
        
        
        available_positions = object_manager.get_available_positions(8)
        
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
            world_x, world_z = pos
            is_correct_position = False
            best_direction_info = None
            
            # 测试该位置的四个朝向
            for direction in test_directions:
                cam_pos = {"x": world_x, "y": 1.5, "z": world_z}
                
                look_at = {
                    "x": cam_pos["x"] + direction["offset"]["x"],
                    "y": 1.5,
                    "z": cam_pos["z"] + direction["offset"]["z"]
                }
                
                # 检查能否看到目标物体
                visibility = can_camera_see_object(object_manager, cam_pos, look_at, target_obj["id"])
                
                if visibility["can_see"]:
                    obj_direction = visibility["object_direction"]
                    
                    # 检查是否匹配目标方向
                    if direction_mapping.get(obj_direction) == orientation_desc:
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
                "position": pos,
                "is_correct_answer": is_correct_position
            }
            
            if is_correct_position:
                grid_info.update(best_direction_info)
                correct_grids.append(grid_info)
            else:
                # 对于错误答案，使用默认的朝向设置
                cam_pos = {"x": world_x, "y": 1.5, "z": world_z}
                grid_info.update({
                    "camera_position": cam_pos,
                    "look_at": {"x": cam_pos["x"], "y": 1.5, "z": cam_pos["z"] + 2},
                    "object_direction": "unknown",
                    "camera_direction": "north"
                })
                wrong_grids.append(grid_info)
        
        print(f"Found {len(correct_grids)} correct positions and {len(wrong_grids)} wrong positions")
        
        valid_correct_grids = [grid for grid in correct_grids if is_visible(object_manager,agent_info["position"],agent_info['look_at'], grid["position"])]
        print(f"Valid correct grids (in agent view): {len(valid_correct_grids)}")
    
        if not valid_correct_grids:
            target_orientation_idx = (target_orientation_idx + 1) % 4
        else:
            break
    
    # 验证错误答案的可见性
    valid_wrong_grids = [grid for grid in wrong_grids if is_visible(object_manager,agent_info["position"],agent_info['look_at'], grid["position"])]
    print(f"Valid wrong grids (in agent view): {len(valid_wrong_grids)}")
    
    # 第三步：选择最终的网格（1个正确 + 3个错误）
    all_grid_infos = []
    correct_colors = []
    
    # 随机分配颜色给正确和错误答案，保证唯一
    total_needed = 4
    color_choices = random.sample(color_options, total_needed)
    # 正确答案
    selected_correct = random.choice(valid_correct_grids)
    selected_correct["color"] = color_choices[0]
    # 根据正确答案的look_at参数，计算摄像机朝向的度数（0/90/180/270）
    look_vec = (
        selected_correct["look_at"]["x"] - selected_correct["camera_position"]["x"],
        selected_correct["look_at"]["z"] - selected_correct["camera_position"]["z"]
    )
    if look_vec == (0, 1):
        selected_correct["look_at_degree"] = 0
    elif look_vec == (1, 0):
        selected_correct["look_at_degree"] = 90
    elif look_vec == (0, -1):
        selected_correct["look_at_degree"] = 180
    elif look_vec == (-1, 0):
        selected_correct["look_at_degree"] = 270
    else:
        raise ValueError
    correct_colors.append(color_choices[0])
    all_grid_infos.append(selected_correct)
    print(f"Selected correct answer: {selected_correct['position']} - {selected_correct['color']}")

    # 错误答案
    if len(valid_wrong_grids) < 3:
        print(f"Warning: Only {len(valid_wrong_grids)} wrong grids available, need 3")

    selected_wrong = random.sample(valid_wrong_grids, min(3, len(valid_wrong_grids)))
    for i, wrong_grid in enumerate(selected_wrong):
        color_name = color_choices[i+1]
        wrong_grid["color"] = color_name
        all_grid_infos.append(wrong_grid)
        print(f"Selected wrong answer {i+1}: {wrong_grid['position']} - {wrong_grid['color']}")
    
    # 创建问题数据
    question_data = {
        "prompt": prompt,
        "orientation_desc": orientation_desc, 
        "answer": selected_correct["color"],
        "rotation": selected_correct["look_at_degree"],
        "all_options": color_options,
    }
    
    return all_grid_infos, question_data

def select_occluder_position_in_agent_view(object_manager: "ObjectManager", agent_info):
    """
    Select occluder position between target and agent to ensure occlusion of target only
    
    Args:
        object_manager: Object manager instance
        agent_info: Agent information including position
        
    Returns:
        tuple: (world_x, world_z) 或 None 如果没有找到合适位置
    """
    # 获取目标物体位置
    target_obj = object_manager.get_target_object()
    if not target_obj:
        return None
    
    # 获取所有其他物体（非目标物体）
    other_objects = [obj for obj in object_manager.get_all_objects() if obj["type"] != "target"]
    
    target_pos = target_obj["position"]
    agent_pos = agent_info["position"]
    
    # 计算从agent到target的方向向量
    dx = target_pos["x"] - agent_pos["x"]
    dz = target_pos["z"] - agent_pos["z"]
    
    # 计算距离
    distance = math.sqrt(dx**2 + dz**2)
    if distance < 2.0:  # 如果距离太近，不放置遮挡物
        print("  ⚠️ Target too close to agent, skipping occlusion object")
        return None
    
    # 在agent和target之间选择几个候选位置
    candidate_positions = []
    # 在30%到70%的距离之间选择位置
    for i in range(30):  # 0.3~0.8, 步长0.05
        ratio = 0.3 + i * 0.02
        grid_x = agent_pos["x"] + dx * ratio
        grid_z = agent_pos["z"] + dz * ratio
        candidate_positions.append((grid_x, grid_z))
    

    # 检查哪些位置可用且不会遮挡其他物体（target除外）
    valid_positions = []
    for pos in candidate_positions:
        # 检查该位置的1x1 bound是否与现有物体真实size有重叠
        px, pz = pos
        overlap = False
        # 构造候选位置的1x1区域AABB
        candidate_aabb = {
            "left_top": {"x": px - 0.5, "z": pz + 0.5},
            "right_bottom": {"x": px + 0.5, "z": pz - 0.5}
        }
        for obj in object_manager.get_all_objects():
            # 获取object的size_info
            size_info = obj.get("size")
            if not size_info:
                # 没有size_info时，默认1x1区域
                obj_x = obj["position"]["x"]
                obj_z = obj["position"]["z"]
                obj_aabb = {
                    "left_top": {"x": obj_x - 0.5, "z": obj_z + 0.5},
                    "right_bottom": {"x": obj_x + 0.5, "z": obj_z - 0.5}
                }
            else:
                obj_aabb = size_info
            # 检查AABB重叠
            if not (
                candidate_aabb["left_top"]["x"] > obj_aabb["right_bottom"]["x"] or
                candidate_aabb["right_bottom"]["x"] < obj_aabb["left_top"]["x"] or
                candidate_aabb["left_top"]["z"] < obj_aabb["right_bottom"]["z"] or
                candidate_aabb["right_bottom"]["z"] > obj_aabb["left_top"]["z"]
            ):
                overlap = True
                break
        if overlap:
            continue

        # 检查该遮挡物位置是否会遮挡到其他非target物体
        will_occlude_other = False
        for obj in other_objects:
            obj_pos = obj["position"]
            # 只判断x,z
            # 构造一个临时的遮挡物bound
            temp_bound = {
                'left': [px - 0.5, 0, pz],
                'right': [px + 0.5, 0, pz],
                'front': [px, 0, pz + 0.5],
                'back': [px, 0, pz - 0.5],
                'top': [px, 1.0, pz],
                'bottom': [px, 0, pz],
                'center': [px, 0, pz]
            }
            # 判断agent到该物体的视线是否被遮挡
            if _ray_intersects_box_2d(agent_pos, [obj_pos["x"], obj_pos["z"]], type('Bounds', (), temp_bound)()):
                # 遮挡物必须在agent和物体之间
                if _is_object_between_camera_and_point_2d(agent_pos, [obj_pos["x"], obj_pos["z"]], [px, 0, pz]):
                    will_occlude_other = True
                    break
        if will_occlude_other:
            continue

        valid_positions.append(pos)

    if valid_positions:
        selected_position = random.choice(valid_positions)
        print(f"  Selected occluder position between agent and target: {selected_position}")
        print(f"  This position will occlude target but not other objects")
        return selected_position
    else:
        print("  ⚠️ No valid occluder positions found that only occlude target")
        return None

def add_occlusion_object(c: Controller, object_manager: "ObjectManager", occlusion_objects, agent_info):
    """
    Add occlusion object using object manager
    """
    occlusion_position = select_occluder_position_in_agent_view(object_manager, agent_info)
    
    if occlusion_position:
        selected_occlusion = random.choice(occlusion_objects)
        occlusion_world_pos = {"x": occlusion_position[0], "y": 0, "z": occlusion_position[1]}
        
        occlusion_id = c.get_unique_id()
        occlusion_rotation = {"x": 0, "y": 180, "z": 0}
        
        # TDW commands
        occlusion_command = c.get_add_object(
            model_name=selected_occlusion,
            position=occlusion_world_pos,
            rotation=occlusion_rotation,
            object_id=occlusion_id
        )
        c.communicate([occlusion_command])
        
        # Register in object manager
        object_manager.add_object(
            occlusion_id, 
            selected_occlusion, 
            selected_occlusion, 
            occlusion_world_pos, 
            occlusion_rotation, 
            "occlusion", 
            1.0
        )
        
        return object_manager.get_objects_by_type("occlusion")[-1]  # 返回最新添加的遮挡物
    
    return None


def remove_occlusion_object(c: Controller, object_manager: "ObjectManager"):
    """
    从场景中移除已添加的遮挡物（occlusion object）。
    """
    if object_manager.occlusion_id is not None:
        occlusion_id = object_manager.occlusion_id
        # 先移除occupied_positions
        if occlusion_id in object_manager.transforms and occlusion_id in object_manager.bounds:
            # 获取遮挡物的bound
            bounds = object_manager.bounds[occlusion_id]
            # 计算占用的格子
            x_coords = [bounds.left[0], bounds.right[0]]
            z_coords = [bounds.front[2], bounds.back[2]]
            min_x = round(min(x_coords))
            max_x = round(max(x_coords))
            min_z = round(min(z_coords))
            max_z = round(max(z_coords))
            for x in range(min_x, max_x + 1):
                for z in range(min_z, max_z + 1):
                    if (x, z) in object_manager.occupied_positions:
                        object_manager.occupied_positions.remove((x, z))
        c.communicate([{
            "$type": "destroy_object",
            "id": occlusion_id
        }])
        # 可选：从object_manager中移除相关信息
        object_manager.remove_object(occlusion_id)


class ObjectManager:
    """
    Unified object manager to track all objects in the scene
    """
    def __init__(self):
        self.objects_static = {}
        self.transforms = {}
        self.bounds = {}
        self.rotations = {}
        self.target_id = None
        self.all_ids = []
        self.occlusion_id = None
        self.occupied_positions = set()
        self.colored_cube_ids = []
        self.objects_data = []  # 存储完整的物体数据

    def remove_object(self, object_id):
        """
        移除指定object_id的所有相关数据。
        """
        # 移除静态数据
        self.objects_static.pop(object_id, None)
        self.transforms.pop(object_id, None)
        self.bounds.pop(object_id, None)
        self.rotations.pop(object_id, None)
        # 移除objects_data中的条目
        self.objects_data = [obj for obj in self.objects_data if obj["id"] != object_id]
        # 从all_ids移除
        if object_id in self.all_ids:
            self.all_ids.remove(object_id)
        # 如果是target或occlusion，重置id
        if self.target_id == object_id:
            self.target_id = None
        if self.occlusion_id == object_id:
            self.occlusion_id = None
        # 从colored_cube_ids移除
        if object_id in self.colored_cube_ids:
            self.colored_cube_ids.remove(object_id)
        # occupied_positions无法直接移除，需根据实际情况维护
        
    def add_object(self, object_id, model_name, name, position, rotation, object_type=None, scale=1.0, size_info=None, color=None):
        """Add object to tracking with complete data"""
        # 存储静态数据
        self.objects_static[object_id] = model_name
        self.transforms[object_id] = type('Transform', (), {
            'position': [position["x"], position["y"], position["z"]]
        })()
        self.rotations[object_id] = rotation
        
        # 创建边界框
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
        
        # 计算占用位置
        if not size_info:
            # 没有size信息时，构造1x1 bounding box的size_info
            cx = position["x"]
            cz = position["z"]
            size_info = {
                "left_top": {"x": cx - 0.5, "z": cz + 0.5},
                "right_bottom": {"x": cx + 0.5, "z": cz - 0.5}
            }
        left = round(size_info["left_top"]["x"])
        right = round(size_info["right_bottom"]["x"])
        top = round(size_info["left_top"]["z"])
        bottom = round(size_info["right_bottom"]["z"])
        for x in range(left, right + 1):
            for z in range(bottom, top + 1):
                # if x==0 and z==-3:
                #     a=0
                self.occupied_positions.add((x, z))
        
        # 存储完整物体数据
        obj_data = {
            "id": object_id,
            "model": model_name,
            "name": name,
            "position": position,
            "rotation": rotation,
            "scale": scale,
            "type": object_type or "object",
            "size": size_info
        }
        if color:
            obj_data["color"] = color
            
        self.objects_data.append(obj_data)
        
        # 设置特殊ID
        if object_type == "occlusion":
            self.occlusion_id = object_id
        elif object_type == "target":
            self.target_id = object_id
        self.all_ids.append(object_id)
    
    def get_all_objects(self):
        """获取所有物体数据"""
        return self.objects_data
    
    def get_objects_by_type(self, object_type):
        """根据类型获取物体"""
        return [obj for obj in self.objects_data if obj["type"] == object_type]
    
    def get_target_object(self):
        """获取目标物体"""
        targets = self.get_objects_by_type("target")
        return targets[0] if targets else None
    
    def add_agent_position(self, agent_position):
        """添加agent占用的位置"""
        agent_x = int(round(agent_position["x"]))
        agent_z = int(round(agent_position["z"]))
        self.occupied_positions.add((agent_x, agent_z))
    
    def get_available_positions(self, world_range=10):
        """获取所有可用的世界坐标位置"""
        available_positions = []
        for x in range(-world_range//2, world_range//2 + 1):
            for z in range(-world_range//2, world_range//2 + 1):
                if (x, z) not in self.occupied_positions:
                    available_positions.append((x, z))
        return available_positions
    
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
    x_coords = [point[0] for point in all_boundary_points]
    z_coords = [point[2] for point in all_boundary_points]
    
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
    obstacle_x = obstacle_pos[0]
    obstacle_z = obstacle_pos[2]  # z 坐标
    cam_to_obstacle_dist = math.sqrt(
        (obstacle_x - camera_pos["x"])**2 + 
        (obstacle_z - camera_pos["z"])**2
    )
    
    # 障碍物必须在摄像机和目标之间才能造成遮挡
    return cam_to_obstacle_dist < cam_to_target_dist
def can_camera_see_object( object_manager: "ObjectManager", camera_position, camera_look_at, target_object_id, HORIZONTAL_FOV=90):
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
    x_coords = [point[0] for point in all_boundary_points]
    min_x = min(x_coords)
    max_x = max(x_coords)
    
    # 提取所有边界点的z坐标，找最小值和最大值  
    z_coords = [point[2] for point in all_boundary_points]
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
        # 判断角点是否在视野内
        if not is_point_in_fov(camera_position, camera_look_at, [corner_x, corner_z], HORIZONTAL_FOV):
            return result
        # 检查遮挡
        if object_manager.occlusion_id and object_manager.occlusion_id in object_manager.transforms:
            big_transform = object_manager.transforms[object_manager.occlusion_id]
            big_bounds = object_manager.bounds[object_manager.occlusion_id]
            corner_2d = [corner_x, corner_z]
            if _ray_intersects_box_2d(camera_position, corner_2d, big_bounds):
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

def is_visible(object_manager: "ObjectManager", camera_position: dict, camera_look_at, target_position: tuple) -> bool:
    """
    判断从camera_position到target_position的视线是否被occlusion object遮挡。
    只判断遮挡，不考虑target的bound，只判断target_position点本身。
    返回True表示可见，False表示被遮挡。
    """
    # 判断点是否在视野内
    # 这里假设camera_look_at为正前方2米
    
    if not is_point_in_fov(camera_position, camera_look_at, target_position):
        return False

    # 如果没有遮挡物，直接可见
    if not object_manager.occlusion_id or object_manager.occlusion_id not in object_manager.transforms:
        return True

    # 获取遮挡物的边界
    occlusion_transform = object_manager.transforms[object_manager.occlusion_id]
    occlusion_bounds = object_manager.bounds[object_manager.occlusion_id]

    # 只考虑x,z平面
    ray_start = camera_position
    ray_end_2d = [target_position[0], target_position[1]]

    # 检查射线是否与遮挡物的bound相交
    if _ray_intersects_box_2d(ray_start, ray_end_2d, occlusion_bounds):
        # 检查遮挡物是否在camera和target之间
        if _is_object_between_camera_and_point_2d(ray_start, ray_end_2d, occlusion_transform.position):
            return False

    return True