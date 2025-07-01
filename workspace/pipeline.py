#!/usr/bin/env python3
# pipeline3_pruned.py  –  Pruned: 4 views per camera + top-down (original & annotated) + collage
# 依赖: tdw>=1.13.0  pillow  numpy
# 运行前请手动启动 TDW build:
#   ./TDW.x86_64 -port 1071 -nogui &
# 用法示例:
#   python pipeline3_pruned.py --output /workspace/run04 --objects 5 --room 10 10 --seed 42 --cols 4 --thumb 512

import argparse, json, math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from tdw.librarian import ModelLibrarian

@dataclass
class PlacedObj:
    oid:int; model:str; pos:Dict[str,float]; rot:Dict[str,float]; size:Tuple[float,float]
@dataclass
class ShotMeta:
    file:str; cam_id:str; pos:Dict[str,float]; direction:str

class DataConstructor:
    def __init__(self, out_dir:Path, room:Tuple[int,int], n:int,
                 pool:List[str], seed:int, min_d:float, screen:Tuple[int,int], port:int = 1071):
        # 输出目录
        self.out_dir = Path(out_dir); self.out_dir.mkdir(exist_ok=True, parents=True)
        self.w, self.d = room; self.n = n; self.pool = pool
        self.min_d = min_d; self.sw, self.sh = screen
        self.rng = np.random.RandomState(seed); self.seed = seed
        self.port = port
        # Controller 不自动 launch，先占端口
        self.c = Controller(launch_build=True, port=self.port)
        self.c.add_ons.append(ObjectManager(transforms=True, bounds=True))
        # 摄像机：主视和俯视
        self.main_cam = ThirdPersonCamera(avatar_id="main_cam",
                                          position={"x":0,"y":0.8,"z":0},
                                          field_of_view=90)
        self.top_cam  = ThirdPersonCamera(avatar_id="top_down",
                                          position={"x":0,"y":10,"z":0},
                                          look_at={"x":0,"y":0,"z":0})
        self.c.add_ons.extend([self.main_cam, self.top_cam])
        # 捕获但不自动写入：手动保存需要的图
        self.cap = ImageCapture(path=self.out_dir,
                                avatar_ids=["main_cam","top_down"],
                                pass_masks=["_img"],
                                png=False)
        self.c.add_ons.append(self.cap)
        self.cap.set(frequency="never")
        # 字体
        try:
            self.font = ImageFont.truetype("arial.ttf", 40)
        except:
            self.font = ImageFont.load_default()
        # 记录
        self.lib = ModelLibrarian("models_core.json")
        self.objs: List[PlacedObj] = []
        self.shots: List[ShotMeta] = []

    # ... (_bounds, _rand, _overlap, _yaw_to_center 同原) ...
    def _bounds(self, m:str)->Tuple[float,float]:
        b = self.lib.get_record(m).bounds
        return b["right"]["x"] - b["left"]["x"], b["front"]["z"] - b["back"]["z"]
    def _rand(self, hw, hd):
        return (int(self.rng.randint(-self.w//2+math.ceil(hw), self.w//2-math.ceil(hw))),
                int(self.rng.randint(-self.d//2+math.ceil(hd), self.d//2-math.ceil(hd))))
    def _overlap(self, x, z, w, d):
        return any(abs(x-o.pos["x"])<(w+o.size[0])/2+self.min_d and
                   abs(z-o.pos["z"])<(d+o.size[1])/2+self.min_d for o in self.objs)
    def _yaw_to_center(self, x, z):
        return math.degrees(math.atan2(-x, -z))

    def build(self):
        # 建房间 & 屏幕
        self.c.communicate([
            TDWUtils.create_empty_room(self.w+1, self.d+1),
            {"$type":"set_screen_size","width":self.sw,"height":self.sh}
        ])
        # 随机选对象
        chosen = self.rng.choice(self.pool, size=self.n, replace=False)
        for model in chosen:
            w,d = self._bounds(model)
            for _ in range(50):
                x,z = self._rand(w/2, d/2)
                if not self._overlap(x,z,w,d): break
            else:
                raise RuntimeError("放置失败")
            oid = self.c.get_unique_id(); ry = int(self.rng.choice([0,90,180,270]))
            self.c.communicate([ self.c.get_add_object(
                model_name=model,
                position={"x":x,"y":0,"z":z},
                rotation={"x":0,"y":ry,"z":0},
                object_id=oid,
                library="models_core.json"
            )])
            self.objs.append(PlacedObj(oid, model, {"x":x,"y":0,"z":z}, {"x":0,"y":ry,"z":0}, (w,d)))
        # 中心相机
        while True:
            cx,cz = self._rand(0,0)
            if not self._overlap(cx,cz,0,0): break
        yaw = self._yaw_to_center(cx,cz)
        self.center = {"x":cx,"y":0.8,"z":cz}
        self.main_cam.teleport(self.center); self.main_cam.rotate({"x":0,"y":yaw,"z":0})
        self.c.communicate([])
        # 相机列表
        self.cam_specs = [
            {
                "id": "central",
                "label": "C",
                "position": self.center,
                "rotation": {"y": yaw}
            }
        ]
        # 每个 object 的相机
        for i, o in enumerate(self.objs, 1):
            self.cam_specs.append({
                "id":    f"obj{i}",
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

    def _save_with_label(self, img:Image.Image, fname:str, text:str):
        dr = ImageDraw.Draw(img)
        dr.text((15,15), text, fill=(255,0,0,255), font=self.font)
        img.save(self.out_dir/fname)

    def _snap(self, pos, deg, tag, label):
        rad = math.radians(deg)
        self.main_cam.teleport(pos)
        self.main_cam.look_at({
            "x": pos["x"] + math.sin(rad),
            "y": pos["y"],
            "z": pos["z"] + math.cos(rad)
        })
        # self.c.communicate([])
        self.cap.set(frequency="once", avatar_ids=["main_cam"], save=False)
        self.c.communicate([])
        dir_name = {0:"north",90:"east",180:"south",270:"west"}[deg]
        fname = f"{tag}_facing_{dir_name}.png"
        overlay = dir_name.capitalize() if tag=="central" else label
        img = self.cap.get_pil_images()["main_cam"]["_img"].copy()
        self._save_with_label(img, fname, overlay)
        self.shots.append(ShotMeta(fname, tag, pos.copy(), dir_name))

    def capture(self):
        
        
        for spec in self.cam_specs:
            pos, label = spec["position"], spec["label"]
            hidden = next((o for o in self.objs
                           if abs(o.pos["x"]-pos["x"])<1e-4 and
                              abs(o.pos["z"]-pos["z"])<1e-4), None)
            if hidden:
                self.c.communicate({"$type":"teleport_object","id":hidden.oid,
                                     "position":{"x":999,"y":-999,"z":999}})
            for d in (0,90,180,270):
                self._snap(pos, d, spec["id"], label)
            if hidden:
                self.c.communicate({"$type":"teleport_object","id":hidden.oid,
                                     "position":hidden.pos})
        # top-down 
       
        self.cap.set(frequency="once", avatar_ids=["top_down"], save=False)
        self.c.communicate([])
        imgs = self.cap.get_pil_images()
        imgs["top_down"]["_img"].save(self.out_dir/"top_down_original.png")
        

    def save_meta(self):
        (self.out_dir/"meta_data.json").write_text(json.dumps({
            "room_size":[self.w,self.d],
            "screen_size":[self.sw,self.sh],
            "seed":self.seed,
            "min_distance":self.min_d,
            "objects":[asdict(o) for o in self.objs],
            "cameras":self.cam_specs,
            "images":[asdict(s) for s in self.shots]
        }, indent=2))

    def close(self):
        self.c.communicate({"$type":"terminate"})

class Visualizer:
    def __init__(self, meta:Path, out_dir:Path, hfov:float=90):
        self.meta = json.loads(Path(meta).read_text())
        self.out  = Path(out_dir)
        self.img  = Image.open(self.out/"top_down_original.png").convert("RGBA")
        self.drw  = ImageDraw.Draw(self.img)
        self.cx   = self.img.width // 2
        self.cy   = self.img.height// 2
        self.ppu  = self.img.width / self.meta["room_size"][0]
        self.hfov = hfov
        try:
            self.font = ImageFont.truetype("arial.ttf", 28)
        except:
            self.font = ImageFont.load_default()

    # ... (_pix same) ...
    def _pix(self, x, z): return (self.cx + x*self.ppu, self.cy - z*self.ppu)

    def render(self):
        L = 1.2 * self.ppu
        cardinals = [("North",0),("East",90),("South",180),("West",270)]
        for cam in self.meta["cameras"]:
            px, py = self._pix(cam["position"]["x"], cam["position"]["z"])
            self.drw.ellipse([px-6,py-6,px+6,py+6], fill=(0,0,255,255))
            for name, base in cardinals:
                br = math.radians(base)
                for ang in (br - math.radians(self.hfov/2), br + math.radians(self.hfov/2)):
                    ex, ey = px + L*math.sin(ang), py - L*math.cos(ang)
                    self.drw.line([px,py,ex,ey], fill=(255,0,0,255), width=2)
                tx = px + 0.5*L*math.sin(br); ty = py - 0.5*L*math.cos(br)
                bb = self.drw.textbbox((0,0), name, font=self.font)
                self.drw.text((tx-(bb[2]-bb[0])/2, ty-(bb[3]-bb[1])/2), name,
                              fill=(255,0,0,255), font=self.font)
            bb = self.drw.textbbox((0,0), cam["label"], font=self.font)
            self.drw.text((px-(bb[2]-bb[0])/2,py-(bb[3]-bb[1])/2),
                          cam["label"], fill=(255,0,0,255), font=self.font)
        self.img.save(self.out/"top_down_annotated.png")


def collage(out_dir:Path, cols:int, thumb:int):
    shots = sorted(list(out_dir.glob("central_facing_*.png")) +
                   list(out_dir.glob("obj*_facing_*.png")))
    if not shots: return
    thumbs = [Image.open(p).resize((thumb,thumb)) for p in shots]
    rows = (len(thumbs)+cols-1)//cols
    canvas = Image.new("RGB", (cols*thumb, rows*thumb), (0,0,0))
    for i, t in enumerate(thumbs):
        r,c = divmod(i, cols)
        canvas.paste(t, (c*thumb, r*thumb))
    canvas.save(out_dir/"collage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--room",   type=int, nargs=2, default=(10,10))
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--cols",   type=int, default=4)
    ap.add_argument("--thumb",  type=int, default=512)
    ap.add_argument("--port",   type=int, default=1071)
    args = ap.parse_args()

    lib  = ModelLibrarian("models_core.json")
    pool = [r.name for r in lib.records if "chair" in r.name.lower()]

    dc = DataConstructor(Path(args.output), tuple(args.room),
                         args.objects, pool, args.seed, 0.5,
                         (2048,2048), args.port)
    dc.build(); dc.capture(); dc.save_meta(); dc.close()

    viz = Visualizer(Path(args.output)/"meta_data.json", Path(args.output))
    viz.render()
    collage(Path(args.output), args.cols, args.thumb)

    print(f"✅ pipeline pruned complete → {Path(args.output).resolve()}")

if __name__=="__main__":
    main()
