
import os
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize  # 🔸 added for L2 normalization

def extract_dino_pca_only(pointcloud_path, image_path, mask_path, output_pca_dir,
                          scene_camera_dict=None):
    os.makedirs(output_pca_dir, exist_ok=True)

    # === Frame ID
    frame_id = os.path.splitext(os.path.basename(image_path))[0]        # '000620'
    frame_id_nopad = frame_id.lstrip("0") or "0"                        # '620'

    # === Intrinsics
    if scene_camera_dict and frame_id_nopad in scene_camera_dict:
        cam_K = scene_camera_dict[frame_id_nopad]["cam_K"]
    else:
        cam_K = [1066.778, 0.0, 312.9869,
                 0.0, 1067.487, 241.3109,
                 0.0, 0.0, 1.0]
    intrinsic = np.array(cam_K).reshape(3, 3)

    # === Load point cloud
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    points = np.asarray(pcd.points)
    assert points.shape[0] == 1000, f"Expected 1000 points, got {points.shape[0]}"

    # === Load image
    rgb = Image.open(image_path).convert("RGB")
    image_size = rgb.size[::-1]  # (height, width)
    patch_size = 14

    # === Apply mask if provided
    if mask_path:
        print("🎭 Applying mask on RGB image")
        mask = Image.open(mask_path).convert("L")
        rgb_np = np.array(rgb)
        mask_np = np.array(mask)
        rgb_np[mask_np == 0] = 0
        rgb = Image.fromarray(rgb_np)

    # === Load DINOv2 model (cached for reuse)
    print("🔍 Loading DINOv2 Giant...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device).eval()

    # === Project function
    def project(points, intrinsic):
        z = points[:, 2]
        u = (points[:, 0] * intrinsic[0, 0]) / z + intrinsic[0, 2]
        v = (points[:, 1] * intrinsic[1, 1]) / z + intrinsic[1, 2]
        return np.stack([u, v], axis=1), z > 0

    def get_patch_indices(uv, image_shape, patch_size):
        h_p, w_p = image_shape[0] // patch_size, image_shape[1] // patch_size
        u_idx = np.clip((uv[:, 0] / patch_size).astype(int), 0, w_p - 1)
        v_idx = np.clip((uv[:, 1] / patch_size).astype(int), 0, h_p - 1)
        return v_idx * w_p + u_idx

    # === DINO feature extraction
    print(f"📸 Extracting DINO features for frame {frame_id}")
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    dino_feat = output.last_hidden_state[0, 1:, :].cpu()  # [tokens, 1536]

    # === Project and map to DINO
    uv, valid = project(points, intrinsic)
    patch_ids = get_patch_indices(uv[valid], image_size, patch_size)
    patch_ids = np.clip(patch_ids, 0, dino_feat.shape[0] - 1)

    sampled_feats = np.zeros((points.shape[0], dino_feat.shape[1]), dtype=np.float32)
    sampled_feats[valid] = dino_feat[patch_ids].numpy()

    # === PCA to 64D
    print("🎛️  Reducing with PCA → 64D")
    pca = PCA(n_components=64)
    reduced_target = pca.fit_transform(sampled_feats)

    # ✅ Apply L2 normalization
    normalized_target = normalize(reduced_target, norm='l2', axis=1)

    # === Save normalized PCA result
    save_path = os.path.join(output_pca_dir, f"target_{frame_id}_pca64.npy")
    np.save(save_path, normalized_target)
    print("✅ Saved:", save_path, normalized_target.shape)

    return save_path
