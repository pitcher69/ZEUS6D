
import open3d as o3d
import cv2
import numpy as np
import os

def process_point_cloud(rgb_path, depth_path, mask_path, output_dir="/content/output/point_cloud"):
    os.makedirs(output_dir, exist_ok=True)

    # === Load images ===
    color_raw = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    assert color_raw is not None, "Failed to load RGB image"
    assert depth_raw is not None, "Failed to load depth image"
    assert mask_raw is not None, "Failed to load mask image"

    # === Resize RGB to match depth ===
    color_resized = cv2.resize(color_raw, (depth_raw.shape[1], depth_raw.shape[0]))

    # === Resize mask if needed ===
    if mask_raw.shape != depth_raw.shape:
        mask_raw = cv2.resize(mask_raw, (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

    # === Ensure binary mask ===
    mask_bin = (mask_raw > 128).astype(np.uint8)

    # === Mask RGB and depth ===
    masked_depth = depth_raw.copy()
    masked_depth[mask_bin == 0] = 0

    masked_color = color_resized.copy()
    masked_color[mask_bin == 0] = 0

    # === Convert to Open3D RGBD image ===
    color_o3d = o3d.geometry.Image(cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(masked_depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=30.0
    )

    # === Camera intrinsics
    K = [1066.778, 0.0, 312.9869,
         0.0, 1067.487, 241.3109,
         0.0, 0.0, 1.0]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=depth_raw.shape[1],
        height=depth_raw.shape[0],
        fx=K[0],
        fy=K[4],
        cx=K[2],
        cy=K[5],
    )

    # === Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # === Downsample to 1000 points
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) >= 1000:
        idx = np.random.choice(len(points), 1000, replace=False)
        points_sampled = points[idx]
        colors_sampled = colors[idx]
    else:
        print(f"⚠️ Only {len(points)} points found.")
        points_sampled = points
        colors_sampled = colors

    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(points_sampled)
    pcd_sampled.colors = o3d.utility.Vector3dVector(colors_sampled)

    # === Extract frame ID from path
    frame_id = os.path.splitext(os.path.basename(rgb_path))[0]  # e.g., '000620'

    # === Save point cloud
    save_path = os.path.join(output_dir, f"target_{frame_id}.ply")
    success = o3d.io.write_point_cloud(save_path, pcd_sampled)
    print("✅ Saved:", save_path if success else "❌ Failed to save")

    return save_path  # Optional: return for chaining
