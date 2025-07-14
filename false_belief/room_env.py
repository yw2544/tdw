#!/usr/bin/env python3
"""
Room Environment - 10x10 grid containing target objects, occlusion objects, agent and colored grid cameras
Based on colorful_grid.py, adds object pools and camera systems
Integrates occlusion ratio and visibility ratio calculations, outputs detailed JSON data
"""
from tdw.controller import Controller
from pathlib import Path
import random
import json
from datetime import datetime

# Import the new modules
from scene_builder import (
    create_room_base, add_target_and_occlusion_objects, add_colored_cubes,
    select_agent_and_colored_positions, get_grid_position
)
from camera_manager import capture_topdown_view, capture_agent_view, capture_colored_grid_view
from metrics_calculator import compute_metrics_for_view

def create_room_environment(num_occlusion_objects=1, num_colored_grids=5, output_dir="./room_env"):
    """
    Create room environment
    
    Parameters:
    num_occlusion_objects: Number of occlusion objects (default 1)
    num_colored_grids: Number of colored grids (default 5)
    output_dir: Output directory
    """
    # Object pool definitions
    target_objects = [
        'blue_club_chair',
        'white_club_chair', 
        'yellow_side_chair',
        'red_side_chair', 
        'green_side_chair'
    ]
    
    occlusion_objects = [
        "fridge_large", 
        "dining_room_table", 
        "cabinet_24_wood_beach_honey", 
        "cabinet_36_white_wood",
        "5ft_wood_shelving", 
        "metal_lab_shelf"
    ]
    
    # Define 5 colors (red, yellow, blue, green, purple)
    colors = [
        {"name": "red", "rgb": {"r": 0.8, "g": 0.1, "b": 0.1, "a": 0.8}},
        {"name": "yellow", "rgb": {"r": 0.9, "g": 0.9, "b": 0.1, "a": 0.8}},
        {"name": "blue", "rgb": {"r": 0.1, "g": 0.3, "b": 0.8, "a": 0.8}},
        {"name": "green", "rgb": {"r": 0.1, "g": 0.7, "b": 0.1, "a": 0.8}},
        {"name": "purple", "rgb": {"r": 0.6, "g": 0.1, "b": 0.8, "a": 0.8}},
    ]
    
    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"Creating room environment...")
    print(f"Output directory: {out_dir}")
    
    # Environment parameters
    room_size = 12
    grid_size = 10
    
    # Create controller
    c = Controller()
    
    try:
        # Create basic room scene with grid
        print("\nCreating room and grid...")
        room_commands = create_room_base(c, room_size, grid_size)
        
        # Add target and occlusion objects
        print("Adding target and occlusion objects...")
        obj_commands, target_obj_data, all_objects_data, positions = add_target_and_occlusion_objects(
            c, target_objects, occlusion_objects, num_occlusion_objects
        )
        
        # Select agent and colored positions
        print("Selecting agent and colored grid positions...")
        agent_position, colored_positions = select_agent_and_colored_positions(
            positions["target_position"], positions["occlusion_positions"], num_colored_grids
        )
        
        # Print configuration details
        target_world_pos = get_grid_position(positions["target_position"][0], positions["target_position"][1])
        agent_world_pos = get_grid_position(agent_position[0], agent_position[1])
        agent_world_pos["y"] = 1.5  # 1.5 meters high
        
        print(f"\n🎯 Selected target object: {target_obj_data['model']}")
        print(f"🚧 Selected occlusion objects: {[obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion']}")
        print(f"\n📍 Target object position: grid{positions['target_position']} = world({target_world_pos['x']}, {target_world_pos['z']})")
        print(f"🤖 Agent position: grid{agent_position} = world({agent_world_pos['x']}, {agent_world_pos['z']}, {agent_world_pos['y']})")
        print(f"🎨 Colored grid positions: {len(colored_positions)} positions (agent can see and within 90-degree view)")
        
        for i, pos in enumerate(colored_positions):
            world_pos = get_grid_position(pos[0], pos[1])
            color_name = colors[i % len(colors)]["name"]
            print(f"  Grid{i+1}: {pos} = world({world_pos['x']}, {world_pos['z']}) - {color_name}")
        
        # Add colored cubes
        if colored_positions:
            print("Adding colored cube grids...")
            cube_commands = add_colored_cubes(c, colored_positions, colors)
        else:
            print("No colored grids to add (no available positions within agent's view)")
            cube_commands = []
        
        # Send all scene creation commands
        print("\nSending scene creation commands...")
        all_commands = room_commands + obj_commands + cube_commands
        resp = c.communicate(all_commands)
        print("Scene creation completed!")
        
        # Capture views
        print("\n📸 Capturing views...")
        
        # Capture topdown view
        capture_topdown_view(c, out_dir)
        
        # Capture agent view
        capture_agent_view(c, agent_world_pos, target_world_pos, out_dir)
        
        # Capture colored grid views and compute metrics
        view_metrics = {}
        
        if colored_positions:
            print("\n📸 Capturing colored grid views and computing metrics...")
        else:
            print("\n📸 No colored grid views to capture")
        
        for i, (grid_x, grid_z) in enumerate(colored_positions):
            cam_pos = get_grid_position(grid_x, grid_z)
            cam_pos["y"] = 1.5  # 1.5 meters high
            
            color_name = colors[i % len(colors)]["name"]
            view_name = f"view_{i+1}_{color_name}"
            
            print(f"  Capturing {color_name} grid view...")
            
            # Capture view image
            capture_colored_grid_view(c, (grid_x, grid_z), target_world_pos, color_name, i+1, out_dir)
            
            # Compute metrics for this view
            metrics = compute_metrics_for_view(
                c, cam_pos, target_obj_data, all_objects_data, out_dir, view_name
            )
            
            # Collect view data
            view_info = {
                "view_name": view_name,
                "camera_position": cam_pos,
                "grid_position": [grid_x, grid_z],
                "color": color_name,
                "field_of_view": 90,
                "metrics": metrics
            }
            
            view_metrics[view_name] = view_info
        
        # All metrics computed within same TDW session, no need to rebuild scene
        print(f"\n✅ All metrics computed successfully!")
        
        print(f"\n🎉 Room environment creation completed!")
        print(f"📊 Environment statistics:")
        print(f"  - Grid size: {grid_size}x{grid_size} = {grid_size * grid_size} cells")
        print(f"  - Target objects: 1 ({target_obj_data['model']})")
        print(f"  - Occlusion objects: {len([obj for obj in all_objects_data if obj['type'] == 'occlusion'])}")
        print(f"  - Agent position: grid{agent_position} (1.5m high)")
        print(f"  - Colored grids: {len(colored_positions)}")
        print(f"  - Generated images: {2 + len(colored_positions)}")
        print(f"  - Save location: {out_dir}")
        
        print(f"\n📁 Generated files:")
        for file in sorted(out_dir.glob("*.png")):
            print(f"  - {file.name}")
        
        print(f"\n🎯 Scene configuration details:")
        print(f"  - Target: {target_obj_data['model']} at grid{positions['target_position']}")
        print(f"  - Occlusion: {[obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion']}")
        print(f"  - Agent: grid{agent_position} looking at target (can see directly, 90-degree FOV)")
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
                "grid_position": list(agent_position),
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
        return {
            "target_object": target_obj_data['model'],
            "target_position": positions["target_position"],
            "occlusion_objects": [obj['model'] for obj in all_objects_data if obj['type'] == 'occlusion'],
            "occlusion_positions": positions["occlusion_positions"],
            "agent_position": agent_position,
            "colored_positions": colored_positions,
            "view_metrics": view_metrics,
            "json_output": str(json_path)
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
    scene_config = create_room_environment(
        num_occlusion_objects=1,  # 1 occlusion object
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
    # random.seed(42)
    
    # Run room environment test
    test_room_environment()
