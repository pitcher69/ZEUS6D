import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os

def chamfer_distance(pc1, pc2):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)
    # Using squared distances for consistency with the paper's implication of minimizing L2
    return np.mean(dist1**2) + np.mean(dist2**2)

def sample_rotations(step_degrees=45):
    rotations = []
    angles = np.radians(np.arange(0, 360, step_degrees))
    for x in angles: # Rotation around X
        for y in angles: # Rotation around Y
            for z in angles: # Rotation around Z
                R = o3d.geometry.get_rotation_matrix_from_xyz([x, y, z])
                rotations.append(R)
    return rotations

def is_duplicate_matrix(new_mat, existing_mats, tol=1e-4): # Slightly increased tolerance for matrices
    for mat in existing_mats:
        if np.linalg.norm(new_mat - mat) < tol:
            return True
    return False

# Helper function to clean up matrices for display/storage
def clean_matrix_for_display(matrix, tol=1e-7):
    """Rounds very small values in a matrix to zero for readability."""
    return np.where(np.abs(matrix) < tol, 0, matrix)

def estimate_symmetries(mesh_path, chamfer_threshold=1e-5, num_points=5000, step_degrees=45):
    print(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh at {mesh_path} has no vertices. Check file path or content.")

    mesh.compute_vertex_normals() # Good practice, though not strictly needed for Chamfer on points
    pc = np.asarray(mesh.sample_points_uniformly(number_of_points=num_points).points)
    if pc.shape[0] == 0:
        raise ValueError("Sampled point cloud is empty. Check mesh and num_points.")

    print(f"Original point cloud shape: {pc.shape}")

    symmetries = []
    all_sampled_rotations = sample_rotations(step_degrees)
    print(f"Sampling {len(all_sampled_rotations)} rotations with step {step_degrees} degrees.")

    for i, R in enumerate(all_sampled_rotations):
        # Progress indicator
        if i % 100 == 0:
            print(f"Processing rotation {i}/{len(all_sampled_rotations)}...", end='\r')

        rotated_pc = (R @ pc.T).T # Apply rotation
        dist = chamfer_distance(pc, rotated_pc)

        if dist < chamfer_threshold:
            # Check if this rotation is essentially the identity matrix (no rotation)
            # This logic might need refinement based on your specific definition of "symmetry"
            # If you want to include rotations that are *almost* identity but have higher
            # chamfer distance, you can remove or adjust this check.
            # For strict symmetries, dist should be very close to zero.
            if np.linalg.norm(R - np.eye(3)) < 1e-4 and dist > 1e-8:
                pass # Skip if it's identity-like but not truly symmetric (dist not low enough)

            # Clean the matrix before checking for duplicates and storing
            cleaned_R = clean_matrix_for_display(R)
            if not is_duplicate_matrix(cleaned_R, symmetries): # Use cleaned_R for duplicate check
                print(f"\nSymmetry matrix found (Chamfer Distance = {dist:.8f}, L2_norm(R-I)={np.linalg.norm(R - np.eye(3)):.8f}):")
                print(cleaned_R) # Print the cleaned matrix
                symmetries.append(cleaned_R) # Store the cleaned matrix
    print("\nFinished processing rotations.")
    return np.array(symmetries)