
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, AutoModel
import open3d as o3d

# === Paths ===
pointcloud_path = "/content/data/query_5000_scaled.ply"
image_folder = "/content/output/renders"
obj_pose_path = "/content/cnos/src/poses/predefined_poses/obj_poses_level0.npy"
cam_pose_path = "/content/cnos/src/poses/predefined_poses/cam_poses_level0.npy"

# === Camera intrinsics (YCBV) ===
intrinsic = np.array([[572.4114, 0.0, 325.2611],
                      [0.0, 573.57043, 242.04899],
                      [0.0, 0.0, 1.0]])
image_size = (480, 640)
patch_grid_size = (16,16)

# === Load DINO Giant ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
model = AutoModel.from_pretrained("facebook/dinov2-giant").eval().to(device)

# === Load data ===
pcd = o3d.io.read_point_cloud(pointcloud_path)
points = np.asarray(pcd.points)  # (5000, 3)
num_points = points.shape[0]
obj_poses = np.load(obj_pose_path)  # (42, 4, 4)
cam_poses = np.load(cam_pose_path)  # (42, 4, 4)
features_list = [[] for _ in range(num_points)]

# === Utility ===
def compute_T_obj_to_cam(obj_T_world, cam_T_world):
    world_T_cam = np.linalg.inv(cam_T_world)
    return world_T_cam @ obj_T_world

def project_points(points, T_obj_cam, intrinsic):
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])  # (N,4)
    cam_pts = (T_obj_cam @ pts_h.T).T[:, :3]  # (N,3)
    z = cam_pts[:, 2]
    valid = z > 0
    u = (cam_pts[:, 0] * intrinsic[0, 0]) / z + intrinsic[0, 2]
    v = (cam_pts[:, 1] * intrinsic[1, 1]) / z + intrinsic[1, 2]
    return np.stack([u, v], axis=1), valid

def get_patch_indices(uv, image_shape, patch_grid):
    h_img, w_img = image_shape
    h_p, w_p = patch_grid
    u_norm = np.clip(uv[:, 0] / w_img, 0, 1)
    v_norm = np.clip(uv[:, 1] / h_img, 0, 1)
    u_idx = (u_norm * w_p).astype(int)
    v_idx = (v_norm * h_p).astype(int)
    return np.clip(v_idx * w_p + u_idx, 0, w_p * h_p - 1)

def extract_dino_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", do_resize=True, size=518).to(device)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[0, 1:, :].cpu()  # [num_patches, 1536]
def features():
    # === Main loop ===
    print("🚀 Sampling features from rendered views...")
    for i in tqdm(range(len(obj_poses))):
        image_path = os.path.join(image_folder, f"{i:06d}.png")
        T_obj_cam = compute_T_obj_to_cam(obj_poses[i], cam_poses[i])
        uv, valid_mask = project_points(points, T_obj_cam, intrinsic)
    
        valid_uv = uv[valid_mask]
        valid_indices = np.where(valid_mask)[0]
    
        # Only keep in-image points
        in_bounds = (valid_uv[:, 0] >= 0) & (valid_uv[:, 0] < image_size[1]) & \
                    (valid_uv[:, 1] >= 0) & (valid_uv[:, 1] < image_size[0])
        valid_uv = valid_uv[in_bounds]
        valid_indices = valid_indices[in_bounds]
    
        if len(valid_uv) == 0:
            continue
    
        dino_feat = extract_dino_features(image_path)
        patch_ids = get_patch_indices(valid_uv, image_size, patch_grid_size)
        sampled_feats = dino_feat[patch_ids]
    
        for pt_idx, feat in zip(valid_indices, sampled_feats):
            features_list[pt_idx].append(feat.numpy())
    
    # === Aggregate features per point ===
    final_features = np.zeros((num_points, 1536), dtype=np.float32)
    for i, feats in enumerate(features_list):
        if len(feats) > 0:
            final_features[i] = np.mean(feats, axis=0)
    
    # === Save ===
    out_path = "/content/output/query_5000_dino.npy"
    np.save(out_path, final_features)
    print(f"✅ Saved: {out_path} with shape {final_features.shape}")
