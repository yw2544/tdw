#!/usr/bin/env python3
"""
Camera Manager Module for TDW Room Environment
Handles different camera views and image capture operations
"""
from tdw.controller import Controller
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from pathlib import Path

def capture_topdown_view(c: Controller, output_dir: Path):
    """
    Capture topdown view of the scene
    Returns: PIL Image
    """
    print("Capturing topdown view...")
    
    # Clear previous addons
    c.add_ons.clear()
    
    # Create topdown camera
    top_camera = ThirdPersonCamera(
        avatar_id="top_cam",
        position={"x": 0, "y": 10, "z": 0},
        look_at={"x": 0, "y": 0, "z": 0}
    )
    
    top_capture = ImageCapture(
        avatar_ids=["top_cam"], 
        path=str(output_dir),
        png=True
    )
    
    c.add_ons.extend([top_camera, top_capture])
    c.communicate([])
    
    # Save topdown view
    images = top_capture.get_pil_images()
    if "top_cam" in images and "_img" in images["top_cam"]:
        img = images["top_cam"]["_img"]
        img.save(output_dir / "room_topdown.png")
        print("✓ Topdown view saved: room_topdown.png")
        return img
    else:
        print("✗ Failed to capture topdown view")
        return None

def capture_agent_view(c: Controller, agent_world_pos: dict, target_world_pos: dict, output_dir: Path):
    """
    Capture agent's view looking at target
    Returns: PIL Image
    """
    print("Capturing agent view...")
    
    # Clear previous cameras
    c.add_ons.clear()
    
    agent_camera = ThirdPersonCamera(
        avatar_id="agent_cam",
        position=agent_world_pos,
        look_at=target_world_pos,
        field_of_view=90
    )
    
    agent_capture = ImageCapture(
        avatar_ids=["agent_cam"], 
        path=str(output_dir),
        png=True
    )
    
    c.add_ons.extend([agent_camera, agent_capture])
    c.communicate([])
    
    # Save agent view
    images = agent_capture.get_pil_images()
    if "agent_cam" in images and "_img" in images["agent_cam"]:
        img = images["agent_cam"]["_img"]
        img.save(output_dir / "agent_view.png")
        print("✓ Agent view saved: agent_view.png")
        return img
    else:
        print("✗ Failed to capture agent view")
        return None

def capture_colored_grid_view(c: Controller, grid_info: dict, view_index: int, output_dir: Path):
    """
    Capture view from a colored grid position
    Returns: PIL Image
    """
    cam_pos = grid_info["camera_position"]
    look_at = grid_info["look_at"]
    color_name = grid_info["color"]
    
    cam_id = f"colored_cam_{view_index}"
    filename = f"colored_view_{view_index}_{color_name}.png"
    
    print(f"  Capturing {color_name} grid view...")
    
    # Clear previous cameras
    c.add_ons.clear()
    
    colored_camera = ThirdPersonCamera(
        avatar_id=cam_id,
        position=cam_pos,
        look_at=look_at,
        field_of_view=90
    )
    
    colored_capture = ImageCapture(
        avatar_ids=[cam_id], 
        path=str(output_dir),
        png=True
    )
    
    c.add_ons.extend([colored_camera, colored_capture])
    c.communicate([])
    
    # Save colored grid view
    images = colored_capture.get_pil_images()
    if cam_id in images and "_img" in images[cam_id]:
        img = images[cam_id]["_img"]
        img.save(output_dir / filename)
        print(f"    ✓ {color_name} grid view saved: {filename}")
        return img
    else:
        print(f"    ✗ Failed to capture {color_name} grid view")
        return None

def setup_camera_for_metrics(c: Controller, camera_position: dict, target_position: dict, field_of_view: int = 90):
    """
    Setup camera for metrics calculation
    Returns: camera addon
    """
    # Clear previous addons
    c.add_ons.clear()
    
    camera = ThirdPersonCamera(
        avatar_id="metrics_cam",
        position=camera_position,
        look_at=target_position,
        field_of_view=field_of_view
    )
    
    c.add_ons.append(camera)
    return camera

def render_scene_for_metrics(c: Controller, camera_config: dict, output_dir: Path, prefix: str):
    """
    Render scene specifically for metrics calculation
    Returns: (color_img, seg_array)
    """
    # Clear previous addons
    c.add_ons.clear()
    
    # Create camera
    camera = ThirdPersonCamera(
        avatar_id="metrics_cam",
        position=camera_config["position"],
        look_at=camera_config["look_at"],
        field_of_view=camera_config.get("field_of_view", 90)
    )
    
    # Create image capture (default generates color images)
    capture = ImageCapture(
        avatar_ids=["metrics_cam"],
        path=str(output_dir),
        png=True
    )
    
    # Add to controller
    c.add_ons.extend([camera, capture])
    
    # Render
    c.communicate([])
    
    # Get images
    images = capture.get_pil_images()
    
    if "metrics_cam" not in images:
        raise RuntimeError("Failed to get camera images")
    
    # Save color image
    color_img = None
    for key in images["metrics_cam"]:
        if "img" in key or "color" in key:
            color_img = images["metrics_cam"][key]
            break
    
    if color_img is None:
        for key in images["metrics_cam"]:
            if "_id" not in key:
                color_img = images["metrics_cam"][key]
                break
    
    if color_img is not None:
        color_path = output_dir / f"{prefix}.png"
        color_img.save(color_path)
        print(f"  ✓ Color image saved: {color_path}")
    else:
        raise RuntimeError("Failed to get color image")
    
    # Now separately render segmentation image
    c.add_ons.clear()
    c.add_ons.extend([camera])
    
    # Create capture with segmentation
    seg_capture = ImageCapture(
        avatar_ids=["metrics_cam"],
        path=str(output_dir),
        png=True,
        pass_masks=["_id"]  # Request ID segmentation image
    )
    
    c.add_ons.append(seg_capture)
    
    # Render again for segmentation
    c.communicate([])
    
    seg_images = seg_capture.get_pil_images()
    
    # Save segmentation image
    if "metrics_cam" in seg_images and "_id" in seg_images["metrics_cam"]:
        seg_img = seg_images["metrics_cam"]["_id"]
        seg_path = output_dir / f"{prefix}_segmentation.png"
        seg_img.save(seg_path)
        print(f"  ✓ Segmentation image saved: {seg_path}")
        
        # Convert to numpy array for calculation
        import numpy as np
        seg_array = np.array(seg_img)
        return color_img, seg_array
    else:
        raise RuntimeError("Failed to get segmentation image")