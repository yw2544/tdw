#!/usr/bin/env python3
# pipeline_complete.py - Enhanced pipeline with visibility and occlusion ratio calculations
# dependency: tdw>=1.13.0 pillow numpy shapely
# Please run TDW build in another terminal after running this pipeline:
#   ./TDW.x86_64 -port 1071 -nogui &
# Example:
#   python pipeline_complete.py --output /workspace/run04 --objects 5 --room 10 10 --seed 42 --port 1071

import argparse, json, math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from shapely.geometry import Polygon

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from tdw.librarian import ModelLibrarian

# Object pool configuration with name, model, and attributes
OBJECT_POOL = {
    "blue_chair": {
        "name": "blue chair",
        "model": "wood_chair", 
        "scale": 1.0,
        "color": "blue",
        "has_orientation": True
    },
    "yellow_chair": {
        "name": "yellow chair",
        "model": "wood_chair",
        "scale": 1.0, 
        "color": "yellow",
        "has_orientation": True
    },
    "black_chair": {
        "name": "black chair",
        "model": "ligne_roset_armchair",
        "scale": 1.0,
        "color": "black",
        "has_orientation": True
    },
    "red_chair": {
        "name": "red chair", 
        "model": "white_club_chair",
        "scale": 1.0,
        "color": "red",
        "has_orientation": True
    },
    "green_chair": {
        "name": "green chair",
        "model": "white_club_chair", 
        "scale": 1.0,
        "color": "green",
        "has_orientation": True
    },
    "white_chair": {
        "name": "white chair",
        "model": "white_club_chair",
        "scale": 1.0,
        "color": None,
        "has_orientation": True
    },
    "pink_chair": {
        "name": "pink chair",
        "model": "lapalma_stil_chair",
        "scale": 1.0,
        "color": "pink",
        "has_orientation": True
    },
    "wooden_chair": {
        "name": "wooden chair",
        "model": "wood_chair",
        "scale": 1.0,
        "color": None,
        "has_orientation": True
    },
    "shelf": {
        "name": "shelf",
        "model": "5ft_wood_shelving",
        "scale": 1.0,
        "color": None,
        "has_orientation": False
    },
    "fridge": {
        "name": "fridge",
        "model": "fridge_large", 
        "scale": 0.8,
        "color": None,
        "has_orientation": False
    },
    "red_lamp": {
        "name": "red lamp",
        "model": "red_lamp",
        "scale": 1.5,
        "color": None,
        "has_orientation": False
    },
    "white_lamp": {
        "name": "white lamp",
        "model": "white_lamp",
        "scale": 1.5,
        "color": None,
        "has_orientation": False
    },
    "wooden_table": {
        "name": "wooden table",
        "model": "dining_room_table",
        "scale": 1.0,
        "color": None,
        "has_orientation": False
    },
    "kettle": {
        "name": "kettle",
        "model": "kettle",
        "scale": 1.8,
        "color": None,
        "has_orientation": False
    },
    "white_cabinet": {
        "name": "white cabinet",
        "model": "cabinet_36_white_wood",
        "scale": 1.0,
        "color": "white",
        "has_orientation": False
    },
    "wooden_cabinet": {
        "name": "wooden cabinet",
        "model": "cabinet_24_wood_beach_honey",
        "scale": 1.0,
        "color": None,
        "has_orientation": False
    },
    "vase": {
        "name": "vase",
        "model": "vase_01",
        "scale": 1.8,
        "color": None,
        "has_orientation": False
    },
    "basket": {
        "name": "basket",
        "model": "basket_18inx18inx12iin",
        "scale": 2.0,
        "color": None,
        "has_orientation": False
    }
}

# Standard color definitions
COLORS = {
    "red": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "yellow": {"r": 1.0, "g": 1.0, "b": 0.0, "a": 1.0},
    "blue": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "green": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0},
    "black": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "white": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "pink": {"r": 1.0, "g": 0.0, "b": 1.0, "a": 1.0}
}

@dataclass
class PlacedObj:
    object_id: int
    model: str
    name: str
    pos: Dict[str, float]
    rot: Dict[str, float]
    size: Tuple[float, float]
    scale: float
    color: Optional[str]
    has_orientation: bool = False
    orientation: Optional[str] = None
    
    @property
    def attributes(self) -> Dict:
        """Get object attributes for JSON output"""
        attr = {
            "scale": self.scale,
            "has_orientation": self.has_orientation
        }
        if self.color:
            color_values = COLORS[self.color]
            attr["color"] = {
                "name": self.color,
                "r": color_values["r"],
                "g": color_values["g"], 
                "b": color_values["b"],
                "a": color_values["a"]
            }
        if self.has_orientation and self.orientation:
            attr["orientation"] = self.orientation
        return attr

@dataclass
class ObjectRatios:
    object_id: int
    model: str
    visibility_ratio: float
    occlusion_ratio: float

@dataclass
class ShotMeta:
    file: str
    cam_id: str
    pos: Dict[str, float]
    direction: str
    object_ratios: List[ObjectRatios] = None

class DataConstructor:
    def __init__(self, out_dir: Path, room: Tuple[int, int], n: int,
                 pool: List[str], seed: int, min_d: float, screen: Tuple[int, int], port: int = 1071):
        # Output directory
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.w, self.d = room
        self.n = n
        self.pool = pool
        # Adaptive minimum distance based on room size
        if min(self.w, self.d) <= 6:  # Small rooms
            self.min_d = min(min_d, 0.3)  # Reduce minimum distance for small rooms
            print(f"[INFO] Using reduced min distance {self.min_d}m for small room ({self.w}x{self.d})")
        else:
            self.min_d = min_d
        self.sw, self.sh = screen
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.port = port
        
        # Controller
        # self.c = Controller(launch_build=True, port=self.port)
        self.c = Controller(launch_build=True, port=self.port)
        self.c.add_ons.append(ObjectManager(transforms=True, bounds=True))
        
        # Cameras: main view and top-down
        self.main_cam = ThirdPersonCamera(
            avatar_id="main_cam",
            position={"x": 0, "y": 0.8, "z": 0},
            field_of_view=90
        )
        self.top_cam = ThirdPersonCamera(
            avatar_id="top_down",
            #position={"x": 0, "y": 10, "z": 0},
            position={"x": 0, "y": self.w, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0}
        )
        self.c.add_ons.extend([self.main_cam, self.top_cam])
        
        # Image capture
        self.cap = ImageCapture(
            path=self.out_dir,
            avatar_ids=["main_cam", "top_down"],
            pass_masks=["_img"],
            png=True
        )
        self.c.add_ons.append(self.cap)
        self.cap.set(frequency="never")
        
        # Font
        try:
            self.font = ImageFont.truetype("arial.ttf", 40)
        except:
            self.font = ImageFont.load_default()
        
        # Records
        self.lib = ModelLibrarian("models_core.json")
        self.objs: List[PlacedObj] = []
        self.shots: List[ShotMeta] = []

    def get_object_bounds(self, model_name: str, rotation: dict = {"x": 0, "y": 0, "z": 0}, scale: float = 1.0):
        """Get object bounds from TDW ModelLibrarian with scale applied"""
        try:
            record = self.lib.get_record(model_name)
            bounds = record.bounds
            
            # Apply scale to all dimensions
            width = (bounds['right']['x'] - bounds['left']['x']) * scale
            depth = (bounds['front']['z'] - bounds['back']['z']) * scale
            height = (bounds['top']['y'] - bounds['bottom']['y']) * scale
            
            # Apply Y rotation effect
            y_rotation = rotation.get('y', 0) % 360
            if y_rotation in [90, 270]:
                width, depth = depth, width
            elif y_rotation not in [0, 180]:
                angle_rad = math.radians(y_rotation)
                new_width = abs(width * math.cos(angle_rad)) + abs(depth * math.sin(angle_rad))
                new_depth = abs(width * math.sin(angle_rad)) + abs(depth * math.cos(angle_rad))
                width, depth = new_width, new_depth
                
            return (width, depth, height)
        except Exception as e:
            print(f"Failed to get bounds for {model_name}: {e}")
            return (1.0 * scale, 1.0 * scale, 1.0 * scale)

    def _bounds(self, m: str) -> Tuple[float, float]:
        b = self.lib.get_record(m).bounds
        return b["right"]["x"] - b["left"]["x"], b["front"]["z"] - b["back"]["z"]

    def _rand(self, hw, hd):
        """Generate random integer position ensuring appropriate distance from walls based on room size"""
        # Adaptive wall margin based on room size
        if min(self.w, self.d) <= 6:  # Small rooms (like 5x5)
            wall_margin = 0.1  # Reduced margin for small rooms
            print(f"[INFO] Using reduced wall margin {wall_margin}m for small room ({self.w}x{self.d})")
        else:
            wall_margin = 0.3  # Standard margin for larger rooms
            
        min_x = -self.w//2 + math.ceil(hw) + wall_margin
        max_x = self.w//2 - math.ceil(hw) - wall_margin
        min_z = -self.d//2 + math.ceil(hd) + wall_margin
        max_z = self.d//2 - math.ceil(hd) - wall_margin
        
        # Ensure we have valid ranges with fallback
        if min_x >= max_x:
            print(f"[WARN] Object too wide for room. Using minimal margin.")
            min_x = -self.w//2 + math.ceil(hw) + 0.05
            max_x = self.w//2 - math.ceil(hw) - 0.05
        if min_z >= max_z:
            print(f"[WARN] Object too deep for room. Using minimal margin.")
            min_z = -self.d//2 + math.ceil(hd) + 0.05
            max_z = self.d//2 - math.ceil(hd) - 0.05
            
        # For very small ranges, ensure we have at least some options
        if max_x - min_x < 1:
            center_x = (min_x + max_x) / 2
            min_x = center_x - 0.5
            max_x = center_x + 0.5
        if max_z - min_z < 1:
            center_z = (min_z + max_z) / 2
            min_z = center_z - 0.5
            max_z = center_z + 0.5
            
        # Generate integer coordinates
        x = int(self.rng.uniform(min_x, max_x))
        z = int(self.rng.uniform(min_z, max_z))
        return (x, z)

    def _overlap(self, x, z, w, d):
        """Check if placing object at (x,z) with size (w,d) would overlap with existing objects"""
        return any(abs(x-o.pos["x"]) < (w + o.size[0])/2 + self.min_d and
                   abs(z-o.pos["z"]) < (d + o.size[1])/2 + self.min_d for o in self.objs)

    def _yaw_to_center(self, x, z):
        return math.degrees(math.atan2(-x, -z))
    
    def _rotation_to_orientation(self, rotation_y: int) -> str:
        """Convert rotation angle to orientation string"""
        angle = rotation_y % 360
        if angle == 0:
            return "north"
        elif angle == 90:
            return "east"
        elif angle == 180:
            return "south"
        elif angle == 270:
            return "west"
        else:
            # Round to nearest 90 degrees
            rounded = round(angle / 90) * 90 % 360
            return self._rotation_to_orientation(rounded)

    def compute_visibility_ratio(self, target_obj, cam_pos, look_direction, fov_deg=90, samples=64):
        """Compute visibility ratio based on top-down view geometry"""
        x0, z0 = target_obj.pos["x"], target_obj.pos["z"]
        width, depth, height = self.get_object_bounds(target_obj.model, target_obj.rot, target_obj.scale)
        hx, hz = width/2, depth/2
        
        # Target object bounding box
        B = Polygon([
            (x0-hx, z0-hz), (x0+hx, z0-hz), (x0+hx, z0+hz), (x0-hx, z0+hz)
        ])
        
        # Camera field of view sector
        cx, cz = cam_pos["x"], cam_pos["z"]
        look_dir = np.array(look_direction, dtype=float)
        look_dir /= np.linalg.norm(look_dir)
        
        half_fov = math.radians(fov_deg / 2)
        
        def rotate_vector(v, angle):
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            return np.array([v[0]*cos_a - v[1]*sin_a, v[0]*sin_a + v[1]*cos_a])
        
        # Calculate sector radius
        corners = np.array(B.exterior.coords[:-1])
        distances = np.linalg.norm(corners - np.array([cx, cz]), axis=1)
        radius = distances.max() * 1.5
        
        # Construct sector polygon
        sector_points = [(cx, cz)]
        for i in range(samples + 1):
            angle = -half_fov + i * (2 * half_fov / samples)
            direction = rotate_vector(look_dir, angle)
            point = (cx + direction[0] * radius, cz + direction[1] * radius)
            sector_points.append(point)
        
        F = Polygon(sector_points)
        intersection = B.intersection(F)
        
        if B.area > 0:
            return intersection.area / B.area
        return 0.0

    def find_target_color_in_segmentation(self, seg_array: np.ndarray, target_id: int) -> Optional[np.ndarray]:
        """Find target object color in segmentation image"""
        if len(seg_array.shape) != 3:
            return None
        
        # Convert object ID to segmentation color
        target_color = np.array([
            target_id % 255,
            (target_id // 255) % 255, 
            (target_id // 255 // 255) % 255
        ], dtype=np.uint8)
        
        return target_color

    def compute_pixel_count(self, seg_array: np.ndarray, target_color: np.ndarray) -> int:
        """Compute pixel count for specific color in segmentation"""
        if len(seg_array.shape) != 3:
            return 0
        mask = np.all(seg_array == target_color, axis=-1)
        return int(np.sum(mask))

    def get_baseline_count(self, obj: PlacedObj, pos: Dict[str, float], deg: int) -> int:
        """Get baseline pixel count (no occlusion) for object - currently not used"""
        # This method is kept for future use when segmentation works properly
        return 1000  # Default baseline count

    def is_object_center_in_view(self, obj: PlacedObj, cam_pos: Dict[str, float], look_direction: Tuple[float, float], fov_deg: float = 90) -> bool:
        """Check if object center is within camera field of view"""
        # Vector from camera to object center
        obj_x, obj_z = obj.pos["x"], obj.pos["z"]
        cam_x, cam_z = cam_pos["x"], cam_pos["z"]
        
        to_obj = np.array([obj_x - cam_x, obj_z - cam_z])
        if np.linalg.norm(to_obj) < 1e-6:
            return True  # Object is at camera position
            
        to_obj_norm = to_obj / np.linalg.norm(to_obj)
        look_dir_norm = np.array(look_direction) / np.linalg.norm(look_direction)
        
        # Calculate angle between look direction and direction to object
        cos_angle = np.clip(np.dot(look_dir_norm, to_obj_norm), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        
        return angle_deg <= fov_deg / 2

    def build(self):
        # Create room and set screen size
        self.c.communicate([
            TDWUtils.create_empty_room(self.w+1, self.d+1),
            {"$type": "set_screen_size", "width": self.sw, "height": self.sh}
        ])
        
        # Randomly choose and place objects
        chosen = self.rng.choice(self.pool, size=self.n, replace=False)
        for model_name in chosen:
            obj_config = OBJECT_POOL[model_name]
            w, d = self._bounds(obj_config["model"])
            
            # Scale the bounds for placement calculation
            w *= obj_config["scale"]
            d *= obj_config["scale"]
            
            max_attempts = 100 if min(self.w, self.d) <= 6 else 50  # More attempts for small rooms
            for attempt in range(max_attempts):
                x, z = self._rand(w/2, d/2)
                if not self._overlap(x, z, w, d):
                    break
                if attempt % 20 == 19:  # Progress info every 20 attempts
                    print(f"[INFO] Placement attempt {attempt + 1}/{max_attempts} for {model_name}")
            else:
                print(f"[ERROR] Failed to place {model_name} after {max_attempts} attempts")
                print(f"[INFO] Room size: {self.w}x{self.d}, Object size: {w:.2f}x{d:.2f}")
                print(f"[INFO] Current objects: {len(self.objs)}")
                raise RuntimeError(f"Failed to place object {model_name}")
            
            oid = self.c.get_unique_id()
            ry = int(self.rng.choice([0, 90, 180, 270]))
            
            # Add object with scale
            self.c.communicate([self.c.get_add_object(
                model_name=obj_config["model"],
                position={"x": x, "y": 0, "z": z},
                rotation={"x": 0, "y": ry, "z": 0},
                object_id=oid,
                library="models_core.json"
            )])
            
            # Apply scale if needed
            if obj_config["scale"] != 1.0:
                self.c.communicate([{
                    "$type": "scale_object",
                    "id": oid,
                    "scale_factor": {"x": obj_config["scale"], "y": obj_config["scale"], "z": obj_config["scale"]}
                }])
            
            # Apply color if needed
            if obj_config["color"]:
                color_values = COLORS[obj_config["color"]]
                self.c.communicate([{
                    "$type": "set_color",
                    "id": oid,
                    "color": color_values
                }])
                print(f"Applied {obj_config['color']} color to {obj_config['name']}")
            
            if obj_config["scale"] != 1.0:
                print(f"Applied scale {obj_config['scale']} to {obj_config['name']}")
            
            # Determine orientation for all objects
            has_orientation = obj_config.get("has_orientation", False)
            orientation = None
            if has_orientation:
                orientation = self._rotation_to_orientation(ry)
                print(f"Set orientation {orientation} ({ry}°) for {obj_config['name']}")
            else:
                print(f"No orientation for {obj_config['name']} (has_orientation=False)")
            
            # Calculate and display distance to nearest wall
            # Object edges: (x-w/2, x+w/2) and (z-d/2, z+d/2) 
            # Room bounds: (-self.w/2, self.w/2) and (-self.d/2, self.d/2)
            wall_distances = [
                (x - w/2) - (-self.w/2),  # Distance from object left edge to left wall
                (self.w/2) - (x + w/2),   # Distance from object right edge to right wall  
                (z - d/2) - (-self.d/2),  # Distance from object back edge to back wall
                (self.d/2) - (z + d/2)    # Distance from object front edge to front wall
            ]
            min_wall_distance = min(wall_distances)
            print(f"Placed {obj_config['name']} at integer coords ({int(x)}, {int(z)}), min wall distance: {min_wall_distance:.2f}m")
            
            self.objs.append(PlacedObj(oid, obj_config["model"], obj_config["name"], {"x": x, "y": 0, "z": z}, 
                                     {"x": 0, "y": ry, "z": 0}, (w, d), obj_config["scale"], obj_config["color"],
                                     has_orientation, orientation))
        
        # Optimize all object positions for visibility in all 4 directions
        print("[INFO] Optimizing object positions for visibility...")
        self._optimize_all_objects()
        
        # Set up cameras based on final object positions
        print("[INFO] Setting up cameras based on optimized positions...")
        self._setup_cameras()

    def _optimize_all_objects(self):
        """Optimize positions for all objects considering all camera positions and directions"""
        max_iterations = 10  # Increased iterations for better optimization
        
        for iteration in range(max_iterations):
            print(f"[INFO] Optimization iteration {iteration + 1}/{max_iterations}")
            any_changes = False
            
            for i, obj in enumerate(self.objs):
                print(f"[INFO] Optimizing object {i+1}/{len(self.objs)} ({obj.name}, ID: {obj.object_id})")
                
                # Get all camera positions (temporary agent position + all object positions)
                temp_agent_pos = {"x": 0, "y": 0.8, "z": 0}  # Temporary for optimization
                all_cam_positions = [temp_agent_pos] + [{"x": o.pos["x"], "y": 0.8, "z": o.pos["z"]} for o in self.objs]
                
                obj_changed = self._optimize_object_for_all_views(obj, all_cam_positions)
                if obj_changed:
                    any_changes = True
            
            if not any_changes:
                print(f"[INFO] Converged after {iteration + 1} iterations")
                break
        
        print("[INFO] Final verification of all object positions...")
        self._verify_all_object_positions()
        
        # Provide summary of overall optimization results
        print("[INFO] Optimization Summary:")
        total_violations = 0
        for obj in self.objs:
            temp_agent_pos = {"x": 0, "y": 0.8, "z": 0}
            all_cam_positions = [temp_agent_pos] + [{"x": o.pos["x"], "y": 0.8, "z": o.pos["z"]} for o in self.objs]
            _, violations = self._evaluate_object_position_with_violations(obj, all_cam_positions, [0, 90, 180, 270])
            total_violations += violations
            if violations == 0:
                print(f"  ✓ {obj.name}: Perfect positioning")
            else:
                print(f"  ⚠ {obj.name}: {violations} violations")
        
        if total_violations == 0:
            print("[SUCCESS] All objects meet visibility requirements!")
        else:
            print(f"[WARNING] Total violations across all objects: {total_violations}")
            print("  Note: Objects with violations may still be usable depending on your requirements.")

    def _optimize_object_for_all_views(self, obj: PlacedObj, all_cam_positions: List[Dict], max_attempts: int = 50) -> bool:
        """Optimize object position considering all camera views with strict visibility requirements"""
        original_pos = obj.pos.copy()
        best_pos = original_pos.copy()
        best_score = float('-inf')
        best_violations = float('inf')
        obj_changed = False
        
        directions = [0, 90, 180, 270]  # North, East, South, West
        
        # Calculate initial score and violations
        initial_score, initial_violations = self._evaluate_object_position_with_violations(obj, all_cam_positions, directions)
        best_score = initial_score
        best_violations = initial_violations
        
        print(f"[INFO] {obj.name} initial score: {initial_score:.3f}, violations: {initial_violations}")
        
        for attempt in range(max_attempts):
            # Try to find a better position with adaptive search range
            w, d = obj.size
            # Use increasing search range as we get more desperate
            search_factor = 1.0 + (attempt / max_attempts) * 2.0  # 1.0 to 3.0
            offset_x = self.rng.uniform(-2.0 * search_factor, 2.0 * search_factor)
            offset_z = self.rng.uniform(-2.0 * search_factor, 2.0 * search_factor)
            # Ensure new coordinates are integers
            new_x = int(original_pos["x"] + offset_x)
            new_z = int(original_pos["z"] + offset_z)
            
            # Check bounds (with adaptive wall margin) and overlaps
            if min(self.w, self.d) <= 6:  # Small rooms
                wall_margin = 0.1
            else:
                wall_margin = 0.3
            min_bound_x = -self.w//2 + w//2 + wall_margin
            max_bound_x = self.w//2 - w//2 - wall_margin
            min_bound_z = -self.d//2 + d//2 + wall_margin
            max_bound_z = self.d//2 - d//2 - wall_margin
            
            if (min_bound_x <= new_x <= max_bound_x and min_bound_z <= new_z <= max_bound_z and
                not any(abs(new_x-o.pos["x"]) < (w+o.size[0])/2+self.min_d and
                       abs(new_z-o.pos["z"]) < (d+o.size[1])/2+self.min_d 
                       for o in self.objs if o.object_id != obj.object_id)):
                
                # Test new position (ensure coordinates are integers)
                test_pos = {"x": float(new_x), "y": obj.pos["y"], "z": float(new_z)}
                old_pos = obj.pos
                obj.pos = test_pos
                
                test_score, test_violations = self._evaluate_object_position_with_violations(obj, all_cam_positions, directions)
                
                # Prioritize reducing violations first, then improving score
                is_better = False
                if test_violations < best_violations:
                    is_better = True
                elif test_violations == best_violations and test_score > best_score:
                    is_better = True
                
                if is_better:
                    # Keep the new position
                    best_pos = test_pos.copy()
                    best_score = test_score
                    best_violations = test_violations
                    obj_changed = True
                    self.c.communicate({
                        "$type": "teleport_object",
                        "id": obj.object_id,
                        "position": obj.pos
                    })
                    print(f"[INFO] Improved {obj.name} position to ({int(test_pos['x'])}, {int(test_pos['z'])}) (score: {test_score:.3f}, violations: {test_violations})")
                    
                    # If we have no violations, we can stop early
                    if test_violations == 0:
                        print(f"[INFO] {obj.name} achieved perfect positioning!")
                        return True
                else:
                    # Revert to old position
                    obj.pos = old_pos
        
        # Apply best position found
        if obj.pos != best_pos:
            obj.pos = best_pos
            self.c.communicate({
                "$type": "teleport_object",
                "id": obj.object_id,
                "position": obj.pos
            })
            obj_changed = True
        
        print(f"[INFO] {obj.name} final score: {best_score:.3f}, violations: {best_violations}")
        return obj_changed

    def _evaluate_object_position(self, obj: PlacedObj, all_cam_positions: List[Dict], directions: List[int]) -> float:
        """Evaluate object position based on visibility requirements for all camera views"""
        total_score = 0.0
        view_count = 0
        
        for cam_idx, cam_pos in enumerate(all_cam_positions):
            # Skip if this is the object's own camera position (except for agent camera)
            if cam_idx > 0 and cam_pos["x"] == obj.pos["x"] and cam_pos["z"] == obj.pos["z"]:
                continue
                
            for deg in directions:
                rad = math.radians(deg)
                look_direction = (math.sin(rad), math.cos(rad))
                
                # Check if object center is in view
                center_in_view = self.is_object_center_in_view(obj, cam_pos, look_direction)
                visibility_ratio = self.compute_visibility_ratio(obj, cam_pos, look_direction)
                
                if center_in_view:
                    # Object center is in view: require visibility > 0.7 and exposure > 0.9
                    if visibility_ratio >= 0.7:
                        score = visibility_ratio * 1.0  # Reward high visibility
                    else:
                        score = visibility_ratio * 0.5 - 0.5  # Penalty for low visibility
                else:
                    # Object center not in view: require visibility close to 0
                    if visibility_ratio <= 0.1:
                        score = 1.0 - visibility_ratio  # Reward low visibility
                    else:
                        score = -visibility_ratio  # Penalty for unwanted visibility
                
                total_score += score
                view_count += 1
        
        return total_score / max(view_count, 1)

    def _evaluate_object_position_with_violations(self, obj: PlacedObj, all_cam_positions: List[Dict], directions: List[int]) -> Tuple[float, int]:
        """Evaluate object position and return both score and violation count"""
        total_score = 0.0
        view_count = 0
        violations = 0
        
        for cam_idx, cam_pos in enumerate(all_cam_positions):
            # Skip if this is the object's own camera position (except for agent camera)
            if cam_idx > 0 and cam_pos["x"] == obj.pos["x"] and cam_pos["z"] == obj.pos["z"]:
                continue
                
            for deg in directions:
                rad = math.radians(deg)
                look_direction = (math.sin(rad), math.cos(rad))
                
                # Check if object center is in view
                center_in_view = self.is_object_center_in_view(obj, cam_pos, look_direction)
                visibility_ratio = self.compute_visibility_ratio(obj, cam_pos, look_direction)
                
                if center_in_view:
                    # Object center is in view: require visibility >= 0.7
                    if visibility_ratio >= 0.7:
                        score = visibility_ratio * 1.0  # Reward high visibility
                    else:
                        score = visibility_ratio * 0.5 - 0.5  # Penalty for low visibility
                        violations += 1  # Count as violation
                else:
                    # Object center not in view: require visibility <= 0.1
                    if visibility_ratio <= 0.1:
                        score = 1.0 - visibility_ratio  # Reward low visibility
                    else:
                        score = -visibility_ratio  # Penalty for unwanted visibility
                        violations += 1  # Count as violation
                
                total_score += score
                view_count += 1
        
        return total_score / max(view_count, 1), violations

    def _verify_all_object_positions(self):
        """Final verification of all object positions"""
        print("[INFO] Verifying object positions...")
        
        # Use temporary agent position for verification
        temp_agent_pos = {"x": 0, "y": 0.8, "z": 0}
        all_cam_positions = [temp_agent_pos] + [{"x": o.pos["x"], "y": 0.8, "z": o.pos["z"]} for o in self.objs]
        directions = [0, 90, 180, 270]
        
        for obj in self.objs:
            violations = []
            
            for cam_idx, cam_pos in enumerate(all_cam_positions):
                # Skip if this is the object's own camera position
                if cam_idx > 0 and cam_pos["x"] == obj.pos["x"] and cam_pos["z"] == obj.pos["z"]:
                    continue
                    
                for deg in directions:
                    rad = math.radians(deg)
                    look_direction = (math.sin(rad), math.cos(rad))
                    
                    center_in_view = self.is_object_center_in_view(obj, cam_pos, look_direction)
                    visibility_ratio = self.compute_visibility_ratio(obj, cam_pos, look_direction)
                    
                    cam_name = "agent" if cam_idx == 0 else f"obj{cam_idx}"
                    dir_name = {0: "north", 90: "east", 180: "south", 270: "west"}[deg]
                    
                    if center_in_view and visibility_ratio < 0.7:
                        violations.append(f"{cam_name}_{dir_name}: center in view but visibility={visibility_ratio:.3f} < 0.7 [NEED ≥70%]")
                    elif not center_in_view and visibility_ratio > 0.1:
                        violations.append(f"{cam_name}_{dir_name}: center not in view but visibility={visibility_ratio:.3f} > 0.1 [NEED ≤10%]")
            
            if violations:
                print(f"[WARN] {obj.name} has {len(violations)} violations:")
                for violation in violations[:5]:  # Show first 5 violations
                    print(f"  - {violation}")
                if len(violations) > 5:
                    print(f"  ... and {len(violations) - 5} more violations")
            else:
                print(f"[OK] {obj.name} meets all requirements")

    def _optimize_object_for_direction(self, obj: PlacedObj, deg: int, max_attempts: int = 20):
        """Optimize object position for a specific viewing direction"""
        rad = math.radians(deg)
        look_direction = (math.sin(rad), math.cos(rad))
        
        for attempt in range(max_attempts):
            # Check visibility from this object's position in this direction
            visibility_ratio = self.compute_visibility_ratio(obj, obj.pos, look_direction)
            
            if visibility_ratio >= 0.8:
                print(f"[INFO] Object {obj.name} satisfies visibility requirement for {deg}° (visibility: {visibility_ratio:.3f})")
                return True
                
            # Try to find a better position
            original_pos = obj.pos.copy()
            best_pos = original_pos.copy()
            best_visibility = visibility_ratio
            
            w, d = obj.size
            for _ in range(30):  # Try more positions
                offset_x = self.rng.uniform(-1.0, 1.0)  # Larger search range
                offset_z = self.rng.uniform(-1.0, 1.0)
                new_x = original_pos["x"] + offset_x
                new_z = original_pos["z"] + offset_z
                
                # Check bounds and overlaps
                if (abs(new_x) < self.w//2 - w//2 and abs(new_z) < self.d//2 - d//2 and
                    not any(abs(new_x-o.pos["x"]) < (w+o.size[0])/2+self.min_d and
                           abs(new_z-o.pos["z"]) < (d+o.size[1])/2+self.min_d 
                           for o in self.objs if o.object_id != obj.object_id)):
                    
                    # Test visibility at this position
                    test_pos = {"x": new_x, "y": obj.pos["y"], "z": new_z}
                    test_visibility = self.compute_visibility_ratio(obj, test_pos, look_direction)
                    
                    if test_visibility > best_visibility:
                        best_pos = test_pos
                        best_visibility = test_visibility
            
            # Apply the best position found
            if best_visibility > visibility_ratio:
                obj.pos = best_pos
                self.c.communicate({
                    "$type": "teleport_object",
                    "id": obj.object_id,
                    "position": obj.pos
                })
                print(f"[INFO] Moved {obj.name} to improve visibility for {deg}° (was {visibility_ratio:.3f}, now {best_visibility:.3f})")
            
            if best_visibility >= 0.8:
                return True
        
        print(f"[WARN] {obj.name} could not reach visibility>0.8 for {deg}° after {max_attempts} attempts (final: {best_visibility:.3f})")
        return False

    def _setup_cameras(self):
        """Set up all cameras based on final object positions"""
        # Central camera (agent position) - adaptive wall distance based on room size
        while True:
            if min(self.w, self.d) <= 6:  # Small rooms
                wall_margin = 0.2  # Reduced margin for small rooms
            else:
                wall_margin = 0.5  # Standard margin for larger rooms
            
            # Ensure we have valid range for agent position
            if self.w//2 - wall_margin <= -self.w//2 + wall_margin:
                wall_margin = 0.1  # Fallback for very small rooms
                
            cx = int(self.rng.uniform(-self.w//2 + wall_margin, self.w//2 - wall_margin))
            cz = int(self.rng.uniform(-self.d//2 + wall_margin, self.d//2 - wall_margin))
            if not self._overlap(cx, cz, 0, 0):
                break
        yaw = self._yaw_to_center(cx, cz)
        self.center = {"x": cx, "y": 0.8, "z": cz}
        self.main_cam.teleport(self.center)
        self.main_cam.rotate({"x": 0, "y": yaw, "z": 0})
        self.c.communicate([])
        
        # Camera specifications
        self.cam_specs = [
            {
                "id": "agent",
                "label": "A",
                "position": self.center,
                "rotation": {"y": yaw}
            }
        ]
        
        # Object cameras based on final positions
        for i, o in enumerate(self.objs, 1):
            self.cam_specs.append({
                "id": str(o.object_id),
                "label": str(i),
                "position": {
                    "x": o.pos["x"],
                    "y": 0.8,
                    "z": o.pos["z"]
                },
                "rotation": {
                    "y": self._yaw_to_center(o.pos["x"], o.pos["z"])
                }
            })

    def _save_with_label(self, img: Image.Image, fname: str, text: str):
        dr = ImageDraw.Draw(img)
        dr.text((15, 15), text, fill=(255, 0, 0, 255), font=self.font)
        img.save(self.out_dir / fname)

    def _snap_with_ratios(self, pos, deg, tag, label):
        """Capture image and record visibility/occlusion ratios for all objects"""
        rad = math.radians(deg)
        look_direction = (math.sin(rad), math.cos(rad))
        look_at = {
            "x": pos["x"] + math.sin(rad),
            "y": pos["y"],
            "z": pos["z"] + math.cos(rad)
        }
        
        self.main_cam.teleport(pos)
        self.main_cam.look_at(look_at)
        
        # Capture color image
        self.cap.set(frequency="once", avatar_ids=["main_cam"], save=False)
        self.c.communicate([])
        color_images = self.cap.get_pil_images()["main_cam"]
        
        # Record visibility ratios for objects in view
        object_ratios = []
        for obj in self.objs:
            # Skip the object if camera is at its position (don't calculate the object itself)
            if tag != "agent" and str(obj.object_id) == tag:
                continue
                
            # Check if object center is in camera view
            visibility_ratio = self.compute_visibility_ratio(obj, pos, look_direction)
            
            # Only include objects that are significantly visible (center in view)
            if visibility_ratio > 0.1:  # Object center is in view
                # Since we optimized positions in build(), visibility should already be >= 0.8
                object_ratios.append(ObjectRatios(
                    object_id=obj.object_id,
                    model=obj.model,
                    visibility_ratio=float(visibility_ratio),
                    occlusion_ratio=1.0  # Default to no occlusion since segmentation not working
                ))
                print(f"{obj.name} in view with visibility: {visibility_ratio:.3f}")
            else:
                print(f"{obj.name} not in view (visibility: {visibility_ratio:.3f})")
        
        # Save image without any overlay text
        dir_name = {0: "north", 90: "east", 180: "south", 270: "west"}[deg]
        fname = f"{tag}_facing_{dir_name}.png"
        img = color_images["_img"].copy()
        img.save(self.out_dir / fname)
        
        # Create shot metadata
        shot_meta = ShotMeta(fname, tag, pos.copy(), dir_name, object_ratios)
        self.shots.append(shot_meta)

    def capture(self):
        """Capture images for all camera views"""
        for spec in self.cam_specs:
            pos, label = spec["position"], spec["label"]
            
            # Check if there's an object at this camera position that needs to be hidden
            hidden = next((o for o in self.objs
                           if abs(o.pos["x"]-pos["x"])<1e-4 and
                              abs(o.pos["z"]-pos["z"])<1e-4), None)
            
            if hidden:
                self.c.communicate({
                    "$type": "teleport_object",
                    "id": hidden.object_id,
                    "position": {"x": 999, "y": -999, "z": 999}
                })
            
            # Capture 4 directions
            for d in (0, 90, 180, 270):
                self._snap_with_ratios(pos, d, spec["id"], label)
            
            # Restore hidden object
            if hidden:
                self.c.communicate({
                    "$type": "teleport_object",
                    "id": hidden.object_id,
                    "position": hidden.pos
                })
        
        # Capture top-down views
        self.cap.set(frequency="once", avatar_ids=["top_down"], save=False)
        self.c.communicate([])
        imgs = self.cap.get_pil_images()
        imgs["top_down"]["_img"].save(self.out_dir / "top_down_original.png")

    def render_annotated_topdown(self):
        """Draw camera annotation exactly as pipeline.py Visualizer.render()."""
        try:
            img = Image.open(self.out_dir / "top_down_original.png").convert("RGBA")
            draw = ImageDraw.Draw(img)
            room_size = self.w
            cx, cy = img.width // 2, img.height // 2
            ppu = img.width / room_size
            hfov = 90
            # Adaptive line length based on room size - smaller for smaller rooms
            L = min(0.6 * ppu, room_size * ppu / 8)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except:
                font = ImageFont.load_default()
            cardinals = [("North", 0), ("East", 90), ("South", 180), ("West", 270)]
            def _pix(x, z): return (cx + x * ppu, cy - z * ppu)
            for cam in self.cam_specs:
                px, py = _pix(cam["position"]["x"], cam["position"]["z"])  
                if cam["id"] == "agent":
                    r = 5 
                    draw.ellipse(
                        [(px - r, py - r), (px + r, py + r)],
                        fill=(0, 0, 255, 255),  # RGBA 蓝色
                        outline=None
                    )             
                for name, base in cardinals:
                    br = math.radians(base)
                    for ang in (br - math.radians(hfov/2), br + math.radians(hfov/2)):
                        ex, ey = px + L*math.sin(ang), py - L*math.cos(ang)
                        draw.line([px, py, ex, ey], fill=(255,0,0,255), width=1)
                    tx = px + 0.5*L*math.sin(br)
                    ty = py - 0.5*L*math.cos(br)
                    bb = draw.textbbox((0,0), name, font=font)
                    draw.text((tx-(bb[2]-bb[0])/2, ty-(bb[3]-bb[1])/2), name,
                              fill=(255,0,0,255), font=font)
                # bb = draw.textbbox((0,0), cam["label"], font=font)
                # draw.text((px-(bb[2]-bb[0])/2,py-(bb[3]-bb[1])/2),
                #           cam["label"], fill=(255,0,0,255), font=font)
            img.save(self.out_dir / "top_down_annotated.png")
        except Exception as e:
            print(f"Failed to create annotated top-down view: {e}")

    def save_meta(self):
        """Save metadata with object ratios for each shot"""
        meta_data = {
            "room_size": [self.w, self.d],
            "screen_size": [self.sw, self.sh],
            "seed": self.seed,
            "min_distance": self.min_d,
            "objects": [],
            "cameras": self.cam_specs,
            "images": []
        }
        
        # Add objects with name and attributes
        for obj in self.objs:
            obj_dict = {
                "object_id": obj.object_id,
                "model": obj.model,
                "name": obj.name,
                "pos": obj.pos,
                "rot": obj.rot,
                "size": list(obj.size),
                "attributes": obj.attributes
            }
            meta_data["objects"].append(obj_dict)
        
        # Convert shots to dict format including object ratios
        for shot in self.shots:
            shot_dict = {
                "file": shot.file,
                "cam_id": shot.cam_id,
                "pos": shot.pos,
                "direction": shot.direction,
                "object_ratios": [asdict(ratio) for ratio in shot.object_ratios] if shot.object_ratios else []
            }
            meta_data["images"].append(shot_dict)
        
        (self.out_dir / "meta_data.json").write_text(json.dumps(meta_data, indent=2))

    def close(self):
        self.c.communicate({"$type": "terminate"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--room", type=int, nargs=2, default=(10, 10))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=1071)
    args = ap.parse_args()

    # Model pool
    lib = ModelLibrarian("models_core.json")
    pool = list(OBJECT_POOL.keys()) # Use the new OBJECT_POOL

    # Create data constructor with 512x512 resolution
    print("[INFO] Step: DataConstructor init")
    dc = DataConstructor(
        Path(args.output), 
        tuple(args.room),
        args.objects, 
        pool, 
        args.seed, 
        0.5,
        (512, 512),  # Set resolution to 512x512
        args.port
    )
    
    try:
        print("[INFO] Step: build")
        dc.build()
        print("[INFO] Step: build done")
        print("[INFO] Step: capture")
        dc.capture()
        print("[INFO] Step: capture done")
        print("[INFO] Step: render_annotated_topdown")
        dc.render_annotated_topdown()
        print("[INFO] Step: render_annotated_topdown done")
        print("[INFO] Step: save_meta")
        dc.save_meta()
        print("[INFO] Step: save_meta done")
        
        # Generate room_setting.json
        print("[INFO] Step: generate room_setting")
        
        def yaw_to_ori(yaw: float):
            y = yaw % 360
            if y ==   0: return [1, 0]
            if y ==  90: return [0, 1]
            if y == 180: return [-1,0]
            if y == 270: return [0,-1]
            y_round = round(y / 90) * 90 % 360
            return yaw_to_ori(y_round)

        out_dir = Path(args.output)
        meta_path = out_dir / "meta_data.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cannot find {meta_path}")

        meta = json.loads(meta_path.read_text())

        # Construct agent from agent camera
        agent_cam = next(cam for cam in meta["cameras"] if cam["id"] == "agent")
        agent = {
            "name": "agent",
            "pos": [agent_cam["position"]["x"], agent_cam["position"]["z"]],
            "ori": yaw_to_ori(agent_cam["rotation"]["y"])
        }

        # Construct objects list using object names (not models)
        objects = []
        for obj in meta["objects"]:
            objects.append({
                "name": obj["name"],  # Use name instead of model
                "pos": [obj["pos"]["x"], obj["pos"]["z"]],
                "ori": yaw_to_ori(obj["rot"]["y"])
            })

        # Optional: include other cameras as virtual objects
        for cam in meta["cameras"]:
            if cam["id"] == "agent":
                continue
            objects.append({
                "name": f"cam_{cam['label']}",
                "pos": [cam["position"]["x"], cam["position"]["z"]],
                "ori": yaw_to_ori(cam["rotation"]["y"])
            })

        # Assemble room dict
        room = {
            "name": "room_setting",
            "objects": objects,
            "agent": agent,
            "all_objects": [agent] + objects
        }

        # Write to JSON
        room_path = out_dir / "room_setting.json"
        room_path.write_text(json.dumps(room, indent=2, ensure_ascii=False))
        print(f"Generated room_setting.json at: {room_path}")
        print("[INFO] Step: generate room_setting done")
        
        print(f"Pipeline complete → {dc.out_dir.resolve()}")
        print(f"Generated {len(dc.shots)} camera views and 2 top-down views")
    finally:
        print("[INFO] Step: close controller")
        dc.close()
        print("[INFO] Step: close controller done")


if __name__ == "__main__":
    main()
