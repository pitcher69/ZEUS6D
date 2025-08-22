#!/usr/bin/env python3
"""
Full standalone script that:
1. Modifies trimesh_utils.get_obj_diameter signature.
2. Prepares custom pyrender offscreen renderer for macOS.
3. Loads and recenters a mesh.
4. Renders synthetic views using pyrender.

Usage:
    python full_render.py \
        /path/to/model.obj \
        /path/to/poses.npy \
        /path/to/output_dir \
        0 False 0.6 1.0
"""

import os
import argparse
import numpy as np
from PIL import Image
import trimesh
import pyrender
from pyrender.offscreen import OffscreenRenderer
from pyrender.platforms.pyglet_platform import PygletPlatform
from pyrender.renderer import Renderer

# --- Section 1: Patch trimesh_utils.get_obj_diameter ---
# Replicating src/utils/trimesh_utils.py functionality inline

def as_mesh(scene_or_mesh):
    import trimesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        return trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
             for m in scene_or_mesh.geometry.values()]
        )
    else:
        return scene_or_mesh

def get_obj_diameter(mesh):
    # Changed signature: expect mesh object, not mesh_path
    extents = mesh.extents * 2
    return np.linalg.norm(extents)

# --- Section 2: Configure OffscreenRenderer for macOS architecture ---
os.environ["PYOPENGL_PLATFORM"] = "pyglet"
import pyrender.platforms
pyrender.platforms.egl = None

def _safe_create(self):
    self._platform = PygletPlatform(self.viewport_width, self.viewport_height)
    self._platform.init_context()  # creates and activates the context
    self._renderer = Renderer(viewport_width=self.viewport_width,
                              viewport_height=self.viewport_height)

OffscreenRenderer._create = _safe_create

# --- Section 3: Main rendering functionality ---

def render_views(
    mesh,
    obj_poses,
    output_dir,
    img_size,
    intrinsic,
    light_intensity=0.6,
    re_center_transform=np.eye(4),
):
    """Render and save RGB views for each pose."""
    # Create scene and lighting
    cam_pose = np.eye(4)
    cam_pose[1, 1] = -1
    cam_pose[2, 2] = -1

    ambient = np.array([0.02, 0.02, 0.02, 1.0])
    if light_intensity != 0.6:
        ambient = np.array([1.0, 1.0, 1.0, 1.0])
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=ambient)

    light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                               innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    scene.add(light, pose=cam_pose)

    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                       znear=0.05, zfar=100000)
    scene.add(camera, pose=cam_pose)

    renderer = OffscreenRenderer(viewport_width=img_size[1], viewport_height=img_size[0])
    cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

    os.makedirs(output_dir, exist_ok=True)
    for idx in range(obj_poses.shape[0]):
        scene.set_pose(cad_node, obj_poses[idx] @ re_center_transform)
        rgb, _ = renderer.render(scene, pyrender.constants.RenderFlags.RGBA)
        img = Image.fromarray(rgb.astype(np.uint8))
        img.save(os.path.join(output_dir, f"{idx:06d}.png"))

# --- Section 4: Command-line interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone synthetic view renderer.")
    parser.add_argument("cad_path", help="Path to 3D model file (.obj, .ply, etc.)")
    parser.add_argument("obj_pose", help="Path to numpy file of object poses (.npy)")
    parser.add_argument("output_dir", help="Directory for saved renders")
    parser.add_argument("gpus_devices", help="CUDA_VISIBLE_DEVICES index")
    parser.add_argument("disable_output", help="Unused flag for compatibility")
    parser.add_argument("light_intensity", type=float, default=0.6,
                        help="Spot light intensity")
    parser.add_argument("radius", type=float, default=1.0,
                        help="Scale factor for object distance")
    args = parser.parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_devices

    # Load poses and adjust units
    poses = np.load(args.obj_pose)
    poses[:, :3, 3] /= 1000.0
    if args.radius != 1.0:
        poses[:, :3, 3] *= args.radius

    # Determine camera intrinsics & image size
    is_tless = "tless" in args.output_dir
    if is_tless:
        intrinsic = np.array([1075.6509, 0.0, 360,
                              0.0, 1073.9035, 270,
                              0.0, 0.0, 1.0]).reshape(3,3)
        img_size = [540, 720]
    else:
        intrinsic = np.array([[572.4114, 0.0, 325.2611],
                              [0.0, 573.5704, 242.0490],
                              [0.0, 0.0, 1.0]])
        img_size = [480, 640]

    # Load and recenter mesh
    mesh = trimesh.load_mesh(args.cad_path)
    mesh = as_mesh(mesh)
    centroid = mesh.bounding_box.centroid
    transform = np.eye(4)
    transform[:3, 3] = -centroid
    print(f"Object center at {centroid}")

    # Scale mesh if diameter is large
    diameter = get_obj_diameter(mesh)
    if diameter > 100:
        mesh.apply_scale(0.001)

    # Convert to pyrender.Mesh
    if is_tless:
        mesh.visual.face_colors = np.ones((len(mesh.faces),3))*0.4
        mesh.visual.vertex_colors = mesh.visual.face_colors
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        mesh = pyrender.Mesh.from_trimesh(mesh)

    # Render all views
    render_views(
        mesh=mesh,
        obj_poses=poses,
        output_dir=args.output_dir,
        img_size=tuple(img_size),
        intrinsic=intrinsic,
        light_intensity=args.light_intensity,
        re_center_transform=transform
    )