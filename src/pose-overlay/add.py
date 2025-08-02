# import numpy as np
# import open3d as o3d
# from scipy.spatial import cKDTree

# # === Load point cloud from PLY ===
# ply_path = r"C:\Users\ESHWAR\OneDrive\Desktop\cynaptics\iitisoc\ycbv\mustard\query_5000.ply"
# pcd = o3d.io.read_point_cloud(ply_path)
# pts_3d = np.asarray(pcd.points)

# # === Ground truth pose ===
# R_gt = np.array([
#     [-0.9586628303517895, -0.2835287329246605, -0.024014888195919924],
#     [-0.08379337156245308,  0.3619577494350433, -0.9284211691506922],
#     [ 0.27192578395746053, -0.8880300419285506, -0.37075363826953395]
# ])
# T_gt = np.array([115.10576923433375, -41.07723959468405, 755.6857413095329]) / 1000.0  # mm → meters

# # === Estimated pose ===
# trans_est = np.array([
#     [-0.96266065, -0.27077264, -0.00840297,  0.1168339],
#     [-0.0903287,   0.35005618, -0.93236334, -0.04114369],
#     [ 0.25539999, -0.89674012, -0.36142495,  0.75233823],
#     [ 0.,          0.,          0.,          1.]
# ])
# R_est = trans_est[:3, :3]
# T_est = trans_est[:3, 3]

# # === ADD function ===
# def compute_ADD(R, T, R_hat, T_hat, pts_3d):
#     pts_gt = (R @ pts_3d.T).T + T
#     pts_est = (R_hat @ pts_3d.T).T + T_hat
#     return np.mean(np.linalg.norm(pts_gt - pts_est, axis=1))

# # === ADD-S function ===
# def compute_ADD_S(R, T, R_hat, T_hat, pts_3d):
#     pts_gt = (R @ pts_3d.T).T + T
#     pts_est = (R_hat @ pts_3d.T).T + T_hat
#     tree = cKDTree(pts_est)
#     distances, _ = tree.query(pts_gt, k=1)
#     return np.mean(distances)

# import numpy as np
# from scipy.spatial.transform import Rotation as R


# def compute_translation_error(t_est, t_gt):
#     return np.linalg.norm(t_est - t_gt) * 1000  # In mm

# def compute_rotation_error(R_est, R_gt):
#     R_diff = R_est @ R_gt.T
#     trace = np.trace(R_diff)
#     angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
#     return np.degrees(angle)  # In degrees

# # === Compute the metrics ===
# add = compute_ADD(R_gt, T_gt, R_est, T_est, pts_3d)
# add_s = compute_ADD_S(R_gt, T_gt, R_est, T_est, pts_3d)
# t_error = compute_translation_error(T_est,T_gt)
# r_error = compute_rotation_error(R_est,R_gt)
# print(f"Translation Error : {t_error:.6f} mm")
# print(f"Rotation Error : {r_error:.6f} degree")
# print(f"ADD Score   : {add:.6f} meters")
# print(f"ADD-S Score : {add_s:.6f} meters")

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
trans_est = np.array([
    [-0.96266065, -0.27077264, -0.00840297,  0.1168339],
    [-0.0903287,   0.35005618, -0.93236334, -0.04114369],
    [ 0.25539999, -0.89674012, -0.36142495,  0.75233823],
    [ 0.,          0.,          0.,          1.]
])
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
