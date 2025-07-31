import trimesh
import numpy as np
import open3d as o3d
from trimesh.triangles import points_to_barycentric as pob
import os

MODEL = "data/model.ply"
SAVE_PATH = "data/query_5000_scaled.ply"
SAMPLES = 5000
def genpc():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Load mesh
    mesh = trimesh.load(MODEL, process=False)
    
    # Get texture
    texture = np.asarray(mesh.visual.material.image)
    
    # Uniformly sample points and their triangle indices
    points, face_indices = trimesh.sample.sample_surface_even(mesh, count=SAMPLES)
    
    # Get triangles corresponding to sampled faces
    triangles = mesh.triangles[face_indices]
    
    # Get barycentric coords of sampled points
    bc = pob(triangles, points)
    
    # Get UV coords from mesh
    face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
    sample_uvs = np.einsum('ij,ijk->ik', bc, face_uvs)
    
    # Convert UVs for image indexing
    sample_uvs[:, 1] = 1.0 - sample_uvs[:, 1]
    h, w = texture.shape[:2]
    pix = np.clip((sample_uvs * [w - 1, h - 1]).astype(int), 0, [w - 1, h - 1])
    
    # Sample colors from texture
    colors = texture[pix[:, 1], pix[:, 0], :3] / 255.0
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Scale to meters (from mm)
    pcd.scale(0.001, center=(0, 0, 0))
    
    # Save scaled point cloud
    o3d.io.write_point_cloud(SAVE_PATH, pcd)
    print(f"✅ Rescaled and saved query point cloud at {SAVE_PATH}")
