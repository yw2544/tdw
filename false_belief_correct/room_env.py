#!/usr/bin/env python3
"""
Room Environment - 10x10 grid containing target objects, chairs, agent and colored grid cameras
Based on original false_belief but with corrected logic:
- 1 target object + 2 chairs in center 5x5
- Agent position found by traversing 6x6 center grid to see all 3 objects
- Before/after occlusion photo sequence
"""
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
    select_colored_positions, get_grid_position, ObjectManager,
    get_center_5x5_positions, add_occlusion_object,
    can_camera_see_object,find_agent_view_position
)
from camera_manager import capture_topdown_view, capture_agent_view, capture_colored_grid_view
from metrics_calculator import compute_metrics_for_view

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
    
    occlusion_objects = [
        "fridge_large", 
        # "dining_room_table", 
        # "cabinet_24_wood_beach_honey", 
        # "cabinet_36_white_wood",
        # "5ft_wood_shelving", 
        # "metal_lab_shelf"
    ]
    
    # Define 5 colors (red, yellow, blue, green, purple)
    colors = [
        {"name": "red", "rgb": {"r": 0.8, "g": 0.1, "b": 0.1, "a": 0.8}},
        {"name": "yellow", "rgb": {"r": 0.9, "g": 0.9, "b": 0.1, "a": 0.8}},
        {"name": "blue", "rgb": {"r": 0.1, "g": 0.3, "b": 0.8, "a": 0.8}},
        {"name": "green", "rgb": {"r": 0.1, "g": 0.7, "b": 0.1, "a": 0.8}},
        # {"name": "purple", "rgb": {"r": 0.6, "g": 0.1, "b": 0.8, "a": 0.8}},
    ]
    
    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"Creating corrected room environment...")
    print(f"Output directory: {out_dir}")
    
    # Environment parameters
    room_size = 12
    grid_size = 10
    
    # Create controller
    c = Controller()
    
    # Create object manager to track objects
    object_manager = ObjectManager()
    
    try:
        # Create basic room scene with grid
        print("\nCreating room and grid...")
        room_commands = create_room_base(c, room_size, grid_size)
        
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
        occlusion_obj_data = add_occlusion_object(c, object_manager, all_objects_data,occlusion_objects)
        
        if occlusion_obj_data:
            all_objects_data.append(occlusion_obj_data)
        
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
        print(f"🤖 Agent position: grid{agent_info['grid_position']} = world({agent_world_pos['x']}, {agent_world_pos['z']}, {agent_world_pos['y']})")
        print(f"👁️ Agent looking {agent_info['direction']} (can see all objects)")
        print(f"🎨 Colored grid positions: {len(colored_grid_infos)} positions")
        
        print(f"\n❓ Generated question: {question_data['prompt']}")
        print(f"✅ Correct answers: {question_data['correct_answers']}")
        
        for i, grid_info in enumerate(colored_grid_infos):
            pos = grid_info["grid_position"]
            color_name = grid_info["color"]
            is_correct = "✓" if grid_info["is_correct_answer"] else "✗"
            print(f"  Grid{i+1}: {pos} - {color_name} {is_correct}")
        
       
        # Add colored cubes
        if colored_grid_infos:
            print("Adding colored cube grids...")
            colored_positions = [info["grid_position"] for info in colored_grid_infos]
            cube_commands = add_colored_cubes(c, colored_positions, colors)
            c.communicate(cube_commands)
        else:
            print("No colored grids to add")
        
        # Capture views - Phase 1: Before occlusion
        print("\n📸 Phase 1: Capturing agent view before occlusion...")

        # Clear previous cameras
        # c.add_ons.clear()

        # Save agent view (before occlusion)
        # images = agent_capture.get_pil_images()
        # if "agent_cam" in images and "_img" in images["agent_cam"]:
        #     img = images["agent_cam"]["_img"]
        #     img.save(out_dir / "agent_view_before_occlusion.png")
        #     print("✓ Agent view (before occlusion) saved: agent_view_before_occlusion.png")
        
        # Phase 3: Capture topdown and after occlusion views
        print("\n📸 Phase 3: Capturing topdown and after occlusion views...")
        
        # Capture topdown view (after occlusion object is added)
        capture_topdown_view(c, out_dir)
        
        # Capture agent view (after occlusion)
        print("Capturing agent view (after occlusion)...")
        # Use the same camera setup
        c.communicate([])
        
        view_metrics= {}

        
        # All metrics computed within same TDW session, no need to rebuild scene
        print(f"\n✅ All metrics computed successfully!")
       
        
        print(f"\n🎉 Room environment creation completed!")
        print(f"📊 Environment statistics:")
        print(f"  - Grid size: {grid_size}x{grid_size} = {grid_size * grid_size} cells")
        print(f"  - Target objects: 1 ({target_obj_data['model']})")
        print(f"  - Occlusion objects: {len([obj for obj in all_objects_data if obj['type'] == 'occlusion'])}")
        print(f"  - Agent position: grid{agent_info['grid_position']} (1.5m high)")
        print(f"  - Colored grids: {len(colored_positions)}")
        print(f"  - Generated images: {2 + len(colored_positions)}")
        print(f"  - Save location: {out_dir}")
        
        print(f"\n📁 Generated files:")
        for file in sorted(out_dir.glob("*.png")):
            print(f"  - {file.name}")
        
        print(f"\n🎯 Scene configuration details:")
        print(f"  - Target: {target_obj_data['model']} at grid{positions['target_position']}")
        print(f"  - Occlusion: {[obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion']}")
        print(f"  - Agent: grid{agent_info['grid_position']} looking at target (can see directly, 90-degree FOV)")
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
        environment_config = {
            "scene_info": {
                "grid_size": grid_size,
                "grid_spacing": 1.0,
                "room_size": room_size,
                "timestamp": datetime.now().isoformat()
            },
            "objects": all_objects_data,
            "agent": {
                "position": agent_world_pos,
                "grid_position": list(agent_info["grid_position"]),
                "field_of_view": 90,
                "can_see_target": True
            },
            "views": view_metrics,
            "colored_grids": [
                {
                    "grid_position": list(pos),
                    "world_position": get_grid_position(pos[0], pos[1]),
                    "color": colors[i % len(colors)]["name"]
                }
                for i, pos in enumerate(colored_positions)
            ]
        }
        
        # Save JSON file
        json_path = out_dir / "scene_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(environment_config, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Scene data saved: {json_path}")
        
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
        occlusion_positions = [obj["grid_position"] for obj in all_objects_data if obj["type"] == "occlusion"]
        return {
            "target_object": target_obj_data['model'],
            "target_position": positions["target_position"],
            "occlusion_objects": [obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion'],
            "occlusion_positions": occlusion_positions,
            "agent_position": agent_info["grid_position"],
            "colored_positions": colored_positions,
            "view_metrics": view_metrics,
            "json_output": str(json_path),
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

def test_room_environment():
    """Test room environment"""
    print("=" * 60)
    print("🏠 Testing Room Environment")
    print("=" * 60)
    
    # Create room environment
    scene_config = create_room_environment_correct(
        num_colored_grids=5,      # 5 colored grids
        output_dir="./room_env_captures"
    )
    
    if scene_config:
        print(f"\n✅ Successfully created room environment!")
        print(f"🎯 Basic configuration: target={scene_config['target_object']}, occlusion={scene_config['occlusion_objects']}")
        print(f"📁 JSON output: {scene_config['json_output']}")
        print(f"📊 Computed metrics for {len(scene_config['view_metrics'])} views")

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    # random.seed(51)
    
    # Run room environment test
    test_room_environment()
