
import os
import json
from glob import glob
from dino import *
from genpc import process_point_cloud
# === Import or define your functions here ===
# from your_module import process_point_cloud, extract_dino_pca_only

# === Paths
rgb_dir = "./data/rgb"
depth_dir = "./data/depth"
mask_dir = "./data/mask"
scene_camera_path = "./data/scene_camera.json"

pointcloud_output_dir = "./output/point_cloud"
pca_output_dir = "./output/dino_feature_pca"

mask_suffix = "_000000.png"

# === Step 1: Extract frame IDs from RGB filenames
rgb_files = glob(os.path.join(rgb_dir, "*.png"))
frame_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in rgb_files])  # e.g., '000620'
total = len(frame_ids)
def dino_pca():    
    # === Step 2: Load scene_camera.json
    with open(scene_camera_path, "r") as f:
        scene_camera_dict = json.load(f)
    
    # === Step 3: Loop over all frames
    for i, fid_padded in enumerate(frame_ids, 1):
        fid = fid_padded.lstrip("0") or "0"
    
        # === File paths
        rgb_path   = os.path.join(rgb_dir, f"{fid_padded}.png")
        depth_path = os.path.join(depth_dir, f"{fid_padded}.png")
        mask_path  = os.path.join(mask_dir, f"{fid_padded}{mask_suffix}")
    
        # === Run point cloud generation
        pointcloud_path = process_point_cloud(
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
            output_dir=pointcloud_output_dir
            )
    
        # === Run DINO+PCA
        extract_dino_pca_only(
            pointcloud_path=pointcloud_path,
            image_path=rgb_path,
            mask_path=mask_path,
            output_pca_dir=pca_output_dir,
            scene_camera_dict=scene_camera_dict
        )
    
        print(f"✅ [{i}/{total}] Done frame {fid_padded}")
