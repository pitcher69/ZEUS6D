
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

# === Load point cloud ===
ply_path = r"./query_5000.ply"
pcd = o3d.io.read_point_cloud(ply_path)
model_pts = np.asarray(pcd.points)/1000

# === Ground truth pose ===

R_gt = np.load("R_gt.npy")
t_gt = np.load("t_gt.npy") / 1000.0  # convert to meters

# === Estimated pose (4x4) ===
trans_est = np.load("matrix.npy")
R_est = trans_est[:3, :3]
t_est = trans_est[:3, 3]

# === ADD ===
def compute_add(R_est, t_est, R_gt, t_gt, model_pts):
    pts_pred = (R_est @ model_pts.T).T + t_est
    pts_gt = (R_gt @ model_pts.T).T + t_gt
    add = np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1))
    return add

# === ADD-S ===
def compute_add_s(R_est, t_est, R_gt, t_gt, model_pts):
    pts_pred = (R_est @ model_pts.T).T + t_est
    pts_gt = (R_gt @ model_pts.T).T + t_gt
    tree = cKDTree(pts_gt)
    dists, _ = tree.query(pts_pred)
    return np.mean(dists)

# === Errors ===
def compute_translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt) * 1000  # mm

def compute_rotation_error(R_est, R_gt):
    R_diff = R_est @ R_gt.T
    trace = np.trace(R_diff)
    trace = np.clip((trace - 1) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(trace)
    return np.degrees(angle_rad)

# === Run Debugging ===
print("\n=== Debugging 6D Pose Estimation ===")

# Model info
print(f"Model point count: {model_pts.shape[0]}")
diameter = np.max(cdist(model_pts, model_pts))  # true max distance
print("Model diameter (approx): {:.6f} m".format(diameter))
print("Recommended 10% threshold: {:.6f} m".format(0.1 * diameter))

# Errors
add = compute_add(R_est, t_est, R_gt, t_gt, model_pts)
add_s = compute_add_s(R_est, t_est, R_gt, t_gt, model_pts)
t_err = compute_translation_error(t_est, t_gt)
r_err = compute_rotation_error(R_est, R_gt)

# Print
print("\nResults:")
print("Translation Error : {:.6f} mm".format(t_err))
print("Rotation Error    : {:.6f} degrees".format(r_err))
print("ADD Score         : {:.6f} meters".format(add))
print("ADD Score         : {:.2f} mm".format(add * 1000))
print("ADD-S Score       : {:.6f} meters".format(add_s))
print("ADD-S Score       : {:.2f} mm".format(add_s * 1000))
