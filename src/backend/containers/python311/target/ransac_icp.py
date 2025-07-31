
import open3d as o3d
import numpy as np
import json
import os
import re

# === Global vectors to collect error ===
angle_errors = []
trans_errors = []

def rotation_error(R_est, R_gt):
    cos_theta = (np.trace(R_est.T @ R_gt) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))

def translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)

def run_ransac_icp_single(
    query_ply,
    query_feat_npy,
    target_ply,
    target_feat_npy,
    scene_gt_path,
    save_dir,
    object_id=3
):
    os.makedirs(save_dir, exist_ok=True)

    # === Extract frame id from target filename ===
    match = re.search(r"target_(\d+)", os.path.basename(target_ply))
    if not match:
        raise ValueError(f"Filename '{target_ply}' does not contain frame id.")
    frame_id = str(int(match.group(1)))  # e.g., "001714" → "1714"

    # === Load point clouds and features ===
    query_pcd = o3d.io.read_point_cloud(query_ply)
    target_pcd = o3d.io.read_point_cloud(target_ply)
    query_features_np = np.load(query_feat_npy).astype(np.float64)
    target_features_np = np.load(target_feat_npy).astype(np.float64)

    # === Format features for Open3D ===
    query_feat = o3d.pipelines.registration.Feature()
    target_feat = o3d.pipelines.registration.Feature()
    query_feat.data = query_features_np.T
    target_feat.data = target_features_np.T

    # === Optional: Convert target to meters if necessary ===
    # Always convert target to meters
    target_pcd.scale(0.1, center=(0, 0, 0))


    # === RANSAC + ICP ===
    distance_threshold = 0.03
    best_fitness = -1
    best_transform = None

    for seed in range(3):
        o3d.utility.random.seed(seed)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=query_pcd,
            target=target_pcd,
            source_feature=query_feat,
            target_feature=target_feat,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000)
        )

        result_icp = o3d.pipelines.registration.registration_icp(
            source=query_pcd,
            target=target_pcd,
            max_correspondence_distance=0.01,
            init=result_ransac.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        if result_icp.fitness > best_fitness:
            best_fitness = result_icp.fitness
            best_transform = result_icp.transformation

    if best_transform is None:
        raise RuntimeError(f"RANSAC+ICP failed for frame {frame_id}")

    # === Save transformation ===
    save_path = os.path.join(save_dir, f"matrix_{frame_id}.npy")
    np.save(save_path, best_transform)
    print(f"✅ Saved transformation matrix frameid_{frame_id}")

    # === Compare with ground truth ===
    with open(scene_gt_path, 'r') as f:
        gt_data = json.load(f)

    if frame_id not in gt_data:
        raise ValueError(f"Frame ID {frame_id} not found in ground truth.")

    gt_entry = gt_data[frame_id]
    gt_pose = next((obj for obj in gt_entry if obj["obj_id"] == object_id), None)
    if gt_pose is None:
        raise ValueError(f"Object ID {object_id} not found in frame {frame_id}.")

    gt_rotation = np.array(gt_pose["cam_R_m2c"]).reshape(3, 3)
    gt_translation = np.array(gt_pose["cam_t_m2c"]) / 1000.0

    gt_transform = np.eye(4)
    gt_transform[:3, :3] = gt_rotation
    gt_transform[:3, 3] = gt_translation

    R_est = best_transform[:3, :3]
    t_est = best_transform[:3, 3]
    R_gt = gt_transform[:3, :3]
    t_gt = gt_transform[:3, 3]

    rot_err = rotation_error(R_est, R_gt)
    trans_err = translation_error(t_est, t_gt)

    angle_errors.append(rot_err)
    trans_errors.append(trans_err)
    print(f"[FRAME {frame_id}] Rotation Error: {rot_err:.2f}°, Translation Error: {trans_err:.4f} m")

