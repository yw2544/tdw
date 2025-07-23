
#!/usr/bin/env python3
"""
Room Environment - 10x10 grid containing target objects, chairs, agent and colored grid cameras
Based on original false_belief but with corrected logic:
- 1 target object + 2 chairs in center 5x5
- Agent position found by traversing 6x6 center grid to see all 3 objects
- Before/after occlusion photo sequence
"""
import os
from tdw.controller import Controller
from pathlib import Path
import random
import json
from datetime import datetime
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
# Import the new modules
from scene_builder import (
    create_room_base, add_target_and_chairs, add_colored_cubes,
    select_colored_positions,  ObjectManager,
    find_agent_view_position, add_occlusion_object, remove_occlusion_object
)
from camera_manager import capture_topdown_view
occlusion_objects = [
    "fridge_large", 
    # "dining_room_table", 
    # "cabinet_24_wood_beach_honey", 
    # "cabinet_36_white_wood",
    # "5ft_wood_shelving", 
    # "metal_lab_shelf"
]

def create_room_environment_correct(num_colored_grids=5, output_dir="./room_env_correct"):
    """
    Create corrected room environment
    
    Parameters:
    num_colored_grids: Number of colored grids (default 5)
    output_dir: Output directory
    """
    # Object pool definitions - updated according to requirements
    target_objects = [
        # 'blue_club_chair',
        'white_club_chair', 
        # 'yellow_side_chair',
        # 'red_side_chair', 
        # 'green_side_chair'
    ]
    # Define 5 colors (red, yellow, blue, green, purple)
    
    
    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"Creating corrected room environment...")
    print(f"Output directory: {out_dir}")
    
    # Environment parameters
    room_size = 12
    
    # Create controller
    c = Controller()
    
    # Create object manager to track objects
    object_manager = ObjectManager()
    
    try:
        # Create basic room scene with grid
        print("\nCreating room and grid...")
        room_commands = create_room_base(c, room_size)
        
        # Add target and chair objects (1 target + 2 chairs)
        print("Adding target and chair objects...")
        obj_commands, target_obj_data, all_objects_data, positions = add_target_and_chairs(
            c, target_objects
        )
        
        # Add objects to object manager
        for obj_data in all_objects_data:
            object_manager.add_object(
                obj_data["id"], 
                obj_data["model"], 
                obj_data["position"], 
                obj_data["rotation"], 
                obj_data['type'],
                obj_data["scale"]
            )
        
        # Send scene creation commands first
        print("\nSending scene creation commands...")
        all_commands = room_commands + obj_commands
        agent_info = find_agent_view_position(c, object_manager, all_objects_data)
        agent_camera = ThirdPersonCamera(
            avatar_id="agent_cam",
            position=agent_info["position"],
            look_at=agent_info["look_at"],
            field_of_view=90
        )
        
        agent_capture = ImageCapture(
            avatar_ids=["agent_cam"], 
            path=str(out_dir),
            png=True
        )
        
        c.add_ons.extend([agent_camera, agent_capture])
        c.communicate(all_commands)
        print("Initial scene creation completed!")
         # Add occlusion object before colored cubes
        # occlusion_obj_data = add_occlusion_object(c, object_manager, all_objects_data,occlusion_objects)
        
        # if occlusion_obj_data:
        #     all_objects_data.append(occlusion_obj_data)
        
        # Find agent position that can see all objects
        print("Finding agent position that can see all objects...")
        colored_grid_infos, question_data = select_colored_positions(
            all_objects_data, object_manager, c,agent_info,  num_colored_grids
        )
        
        # Print configuration details
        target_world_pos = target_obj_data["position"]
        agent_world_pos = agent_info["position"]
        
        print(f"\n🎯 Selected target object: {target_obj_data['model']}")
        print(f"🪑 Selected chair objects: {[obj['model'] for obj in all_objects_data if obj['type'] == 'chair']}")
        print(f"\n📍 Target object position: grid{positions['target_position']} = world({target_world_pos['x']}, {target_world_pos['z']})")
        print(f"🪑 Chair positions: {[f'grid{pos}' for pos in positions['chair_positions']]}")
        print(f"🤖 Agent position: grid{agent_info['position']} = world({agent_world_pos['x']}, {agent_world_pos['z']}, {agent_world_pos['y']})")
        print(f"👁️ Agent looking {agent_info['direction']} (can see all objects)")
        print(f"🎨 Colored grid positions: {len(colored_grid_infos)} positions")
        print(f"\n❓ Generated question: {question_data}")
        # print(f"\n❓ Generated question: {question_data['prompt_1']}")
        # print(f"\n❓ Generated question: {question_data['prompt_2']}")
        # print(f"✅ Correct answers: {question_data['correct_answers']}")
        
        for i, grid_info in enumerate(colored_grid_infos):
            pos = grid_info["position"]
            color_name = grid_info["color"]
            is_correct = "✓" if grid_info["is_correct_answer"] else "✗"
            print(f"  Grid{i+1}: {pos} - {color_name} {is_correct}")
        
       
        # Add colored cubes
        if colored_grid_infos:
            print("Adding colored cube grids...")
            colored_positions = [info["position"] for info in colored_grid_infos]
            cube_commands = add_colored_cubes(c, colored_positions, colors)
            c.communicate(cube_commands)
            
            images = agent_capture.get_pil_images()
            if "agent_cam" in images and "_img" in images["agent_cam"]:
                img = images["agent_cam"]["_img"]
                img.save(out_dir / "prompt_2.png")
                print("✓ Agent view (with colored cubes) saved")
        else:
            print("No colored grids to add")
        
        
        # Capture topdown view (after occlusion object is added)
        topdown_path = out_dir.with_name(out_dir.stem + "_topdown.png")
        capture_topdown_view(c, topdown_path)
        
        # Capture agent view (after occlusion)
        print("Capturing agent view (after occlusion)...")
        # Use the same camera setup
        c.communicate([])
        
        view_metrics= {}

        
        # All metrics computed within same TDW session, no need to rebuild scene
        print(f"\n✅ All metrics computed successfully!")
       
        
        print(f"\n🎉 Room environment creation completed!")
        print(f"📊 Environment statistics:")
        print(f"  - Target objects: 1 ({target_obj_data['model']})")
        print(f"  - Occlusion objects: {len([obj for obj in all_objects_data if obj['type'] == 'occlusion'])}")
        print(f"  - Agent position: grid{agent_info['position']} (1.5m high)")
        print(f"  - Colored grids: {len(colored_positions)}")
        print(f"  - Generated images: {2 + len(colored_positions)}")
        print(f"  - Save location: {out_dir}")
        
        print(f"\n📁 Generated files:")
        for file in sorted(out_dir.glob("*.png")):
            print(f"  - {file.name}")
        
        print(f"\n🎯 Scene configuration details:")
        print(f"  - Target: {target_obj_data['model']} at grid{positions['target_position']}")
        print(f"  - Occlusion: {[obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion']}")
        print(f"  - Agent: grid{agent_info['position']} looking at target (can see directly, 90-degree FOV)")
        print(f"  - Colored cameras: {len(colored_positions)} cameras, agent can see and within 90-degree view, all looking at target (90-degree FOV)")
        
        # Add object actual size information
        print(f"\n📏 Object size information:")
        for obj_data in all_objects_data:
            if "actual_size" in obj_data:
                size = obj_data["actual_size"]
                print(f"  - {obj_data['model']}: {size['width']:.2f} x {size['depth']:.2f} x {size['height']:.2f} meters")
        
        # Generate JSON output
        print(f"\n💾 Generating JSON output...")
        
        # Prepare environment configuration data
        # environment_config = {
        #     "scene_info": {
        #         "grid_spacing": 1.0,
        #         "room_size": room_size,
        #         "timestamp": datetime.now().isoformat()
        #     },
        #     "objects": all_objects_data,
        #     "agent": {
        #         "position": agent_world_pos,
        #         "position": list(agent_info["position"]),
        #         "field_of_view": 90,
        #         "can_see_target": True
        #     },
        #     "views": view_metrics,
        #     "colored_grids": [
        #         {
        #             "position": list(pos),
        #             "world_position": get_grid_position(pos[0], pos[1]),
        #             "color": colors[i % len(colors)]["name"]
        #         }
        #         for i, pos in enumerate(colored_positions)
        #     ]
        # }
        
        # # Save JSON file
        # json_path = out_dir / "scene_data.json"
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(environment_config, f, indent=2, ensure_ascii=False)
        
        # print(f"  ✓ Scene data saved: {json_path}")
        
        # Output metrics summary
        print(f"\n📊 Metrics summary:")
        for view_name, view_info in view_metrics.items():
            print(f"  {view_name}:")
            for obj_model, obj_metrics in view_info["metrics"].items():
                if obj_metrics["is_target"]:
                    print(f"    - {obj_model} (target): occlusion={obj_metrics['occlusion_ratio']:.3f}, visibility={obj_metrics['visibility_ratio']:.3f}")
                else:
                    print(f"    - {obj_model}: visibility={obj_metrics['visibility_ratio']:.3f}")
        
        # Return scene configuration information
        occlusion_positions = [obj["position"] for obj in all_objects_data if obj["type"] == "occlusion"]
        return {
            "target_object": target_obj_data['model'],
            "target_position": positions["target_position"],
            "occlusion_objects": [obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion'],
            "occlusion_positions": occlusion_positions,
            "agent_position": agent_info["position"],
            "colored_positions": colored_positions,
            "view_metrics": view_metrics,
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        try:
            c.communicate({"$type": "terminate"})
        except:
            pass
        
def generate_qa(c: Controller, object_manager: "ObjectManager",  agent_info, agent_capture, out_dir , rotation_desc=None):
    colored_grid_infos, question_data = select_colored_positions(
        object_manager,  agent_info, rotation_desc
    )
    
    agent_world_pos = agent_info["position"]
    
    print(f"🪑 Selected objects: {[obj['model'] for obj in object_manager.get_all_objects() if obj['type'] != 'target']}")
    print(f"🤖 Agent position: world({agent_world_pos['x']}, {agent_world_pos['z']}, {agent_world_pos['y']})")
    print(f"👁️ Agent looking {agent_info['direction']} (can see all objects)")
    print(f"🎨 Colored grid positions: {len(colored_grid_infos)} positions")
    
    print(f"\n❓ Generated question: {question_data}")
    
    for i, grid_info in enumerate(colored_grid_infos):
        pos = grid_info["position"]
        color_name = grid_info["color"]
        is_correct = "✓" if grid_info["is_correct_answer"] else "✗"
        print(f"  Grid{i+1}: {pos} - {color_name} {is_correct}")
    
    
    # Add colored cubes
    if colored_grid_infos:
        print("Adding colored cube grids...")
        colored_positions = [info["position"] for info in colored_grid_infos]
        colors = [info["color"] for info in colored_grid_infos]
        cube_commands = add_colored_cubes(c, object_manager, colored_positions, colors)
        c.communicate(cube_commands)
        
        images = agent_capture.get_pil_images()
        if "agent_cam" in images and "_img" in images["agent_cam"]:
            img = images["agent_cam"]["_img"]
            img.save(out_dir)
            print("✓ Agent view (with colored cubes) saved")
    else:
        print("No colored grids to add")
    remove_colored_cubes(c, object_manager)
    return question_data

def rotate_object_qa(c: Controller, object_manager: "ObjectManager",agent_capture, out_dir):
    target_obj = object_manager.get_target_object()
    if not target_obj:
        raise ValueError("No target object found")
    
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
    
    # 保存旋转后的图片 - 使用现有的agent camera
    print(f"Capturing image after {rotation_desc}...")
    prompt = (
        f"An object {rotation_desc}."
        f"Please answer with the object that rotates."
    )
    # 从现有的ImageCapture addon获取图片
    images = agent_capture.get_pil_images()
    if "agent_cam" in images and "_img" in images["agent_cam"]:
        img = images["agent_cam"]["_img"]
        img.save(out_dir)
        print(f"✓ Rotation image saved: {out_dir}")
    return prompt, rotation_desc, target_obj['name']

def create_room_from_json(json_path, output_dir="./data"):
    """
    根据JSON文件创建房间环境
    
    Parameters:
    json_path: JSON文件路径
    output_dir: 输出目录
    """

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取JSON文件: {e}")
        return None
    
    # 获取对象信息
    if "objects" not in data:
        print("❌ JSON文件中没有找到objects字段")
        return None
    
    objects_data = data["objects"]
    # room_size_data = data.get("room_size", [5, 5])
    room_size_data = [12, 12]
    room_size = max(room_size_data)  # 增加一些边界空间
    
    # 输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"从JSON文件创建房间环境...")
    print(f"JSON文件: {json_path}")
    print(f"输出目录: {out_dir}")
    print(f"找到 {len(objects_data)} 个对象")
    
    # 创建控制器
    c = Controller()
    
    # 创建对象管理器
    object_manager = ObjectManager()
    # get_occlusion_object_sizes(c,occlusion_objects)
    try:
        # 创建基础房间
        print("\n创建房间和网格...")
        room_commands = create_room_base(c, room_size)
        
        # 准备对象添加命令
        obj_commands = []
        
        print("\n添加JSON中的对象...")
        for i, obj in enumerate(objects_data):
            obj_id = obj["object_id"]
            model_name = obj["model"]
            position = obj["pos"]
            rotation = obj["rot"]
            name = obj["name"]
            # 获取缩放信息
            scale = 1.0
            if "attributes" in obj and "scale" in obj["attributes"]:
                scale = obj["attributes"]["scale"]
            
            # 使用c.get_add_object方法添加对象
            obj_commands.append(c.get_add_object(
                model_name=model_name,
                position=position,
                rotation=rotation,
                object_id=obj_id
            ))
            
            # 如果有缩放设置
            if scale != 1.0:
                obj_commands.append({
                    "$type": "scale_object",
                    "id": obj_id,
                    "scale_factor": {"x": scale, "y": scale, "z": scale}
                })
            
            # 如果有颜色设置
            if "attributes" in obj and "color" in obj["attributes"]:
                color_info = obj["attributes"]["color"]
                obj_commands.append({
                    "$type": "set_color",
                    "id": obj_id,
                    "color": {
                        "r": color_info["r"],
                        "g": color_info["g"], 
                        "b": color_info["b"],
                        "a": color_info["a"]
                    }
                })
            
            # 计算size信息
            size_info = None
            if "size" in obj:
                size = obj["size"]
                width, depth = size[0], size[1]
                left = position["x"] - width / 2
                right = position["x"] + width / 2
                top = position["z"] + depth / 2
                bottom = position["z"] - depth / 2
                
                size_info = {
                    "width": width,
                    "depth": depth,
                    "left_top": {"x": left, "z": top},
                    "right_bottom": {"x": right, "z": bottom}
                }
            
            # 直接添加到对象管理器
            # 根据has_orientation属性决定物体类型
            object_type = "object"  # 默认类型
            if "attributes" in obj and obj["attributes"].get("has_orientation", False) and object_manager.target_id is None:
                object_type = "target"  # 有朝向的物体设为目标
            
            object_manager.add_object(
                obj_id, 
                model_name,
                name, 
                position, 
                rotation, 
                object_type, 
                scale,
                size_info
            )
            
            print(f"  ✓ 添加对象: {model_name} (ID: {obj_id}) at ({position['x']:.2f}, {position['z']:.2f}) - {object_type}")
        if object_manager.target_id == None:
            print('没有target')
            return None
        # 发送所有命令
        print("\n发送场景创建命令...")
        all_commands = room_commands + obj_commands
        agent_info = find_agent_view_position(c, object_manager)
        agent_camera = ThirdPersonCamera(
            avatar_id="agent_cam",
            position=agent_info["position"],
            look_at=agent_info["look_at"],
            field_of_view=90
        )
        
        top_camera = ThirdPersonCamera(
            avatar_id="top_cam",
            position={"x": 0, "y": 10, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0}
        )
        
        agent_capture = ImageCapture(
            avatar_ids=["agent_cam","top_cam"], 
            path=str(out_dir),
            png=True
        )
        
        
        c.add_ons.extend([agent_camera, top_camera, agent_capture])
        c.communicate(all_commands)
        print("Initial scene creation completed!")
        
        if add_occlusion_object(c, object_manager, occlusion_objects, agent_info) == None:
            return None
        
        question_data_2 = generate_qa(c,object_manager,agent_info,agent_capture, out_dir / "prompt_2.png")
        
        remove_occlusion_object(c, object_manager)
        
        prompt_1, rotation_desc, answer = rotate_object_qa(c,object_manager,agent_capture, out_dir / "prompt_1.png")
                
        if add_occlusion_object(c, object_manager, occlusion_objects, agent_info) == None:
            return None 
        
        question_data_3 = generate_qa(c,object_manager,agent_info,agent_capture, out_dir / "prompt_3.png",rotation_desc)
    
        return {
            "prompt_1": prompt_1,
            "image_1": str(out_dir / "prompt_1.png"),
            "answer_1": answer,
            "prompt_2": question_data_2['prompt'],
            "image_2": str(out_dir / "prompt_2.png"),
            "answer_2": question_data_2['answer'],
            "answer_2_rotation": question_data_2["rotation"],
            "prompt_3": question_data_3['prompt'],
            "image_3": str(out_dir / "prompt_3.png"),
            "answer_3": question_data_3['answer'],
            "answer_3_rotation": question_data_3["rotation"],   
        }
        
        
        
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 清理
        try:
            c.communicate({"$type": "terminate"})
        except:
            pass
        
def remove_colored_cubes(c: Controller, object_manager=None):
    """
    删除所有 colored cubes。
    优先使用 object_manager.colored_cube_ids，如果没有则根据 colored_positions 查找。
    """
    colored_cube_ids = object_manager.colored_cube_ids
    
    # 删除所有找到的cube
    for oid in colored_cube_ids:
        c.communicate([{"$type": "destroy_object", "id": oid}])
    object_manager.colored_cube_ids = []


def test_room_environment():
    # Create room environment
    # scene_config = create_room_environment_correct(
    #     num_colored_grids=5,      # 5 colored grids
    #     output_dir="./room_env_captures"
    # )
    results = []
    data_dir = './data'
    os.makedirs(data_dir,exist_ok=True)
    parent_dir = r'D:\WechatFile\xwechat_files\wxid_phyyn4ok8cyh22_4a2a\msg\file\2025-07\test'
    for idx, subfolder in enumerate(os.listdir(parent_dir)):
        subfolder_path = os.path.join(parent_dir, subfolder)
        if os.path.isdir(subfolder_path):
            json_path = os.path.join(subfolder_path, "meta_data.json")
            if os.path.exists(json_path):
                print(f"[{idx}] Processing: {json_path}")
                result = create_room_from_json(json_path, output_dir=f'{data_dir}/{idx}')
                if result:
                    results.append(result)
            
    # 输出到json文件
    output_json_path=data_dir+'/results.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"All results saved to {output_json_path}")

    # json_path = r"D:\WechatFile\xwechat_files\wxid_phyyn4ok8cyh22_4a2a\msg\file\2025-07\test\run02\meta_data.json"  # 或者你的JSON文件路径
    
    # # 创建房间环境
    # result = create_room_from_json(
    #     json_path=json_path,
    #     output_dir="./json_captures"
    # )
    # print(result)
    

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(51)
    
    # Run room environment test
    test_room_environment()


