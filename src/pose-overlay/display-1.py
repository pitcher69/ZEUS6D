import cv2
import numpy as np
import trimesh

# --- Inputs (using your provided paths and data) ---
# IMPORTANT: These paths are absolute to your local machine.
# Please ensure these files are accessible when running the script.
mesh_path = r"C:\Users\ESHWAR\OneDrive\Desktop\cynaptics\iitisoc\ycbv\ycbv_models\models\obj_000003.ply"
texture_path = r"C:\Users\ESHWAR\OneDrive\Desktop\cynaptics\iitisoc\ycbv\ycbv_models\models\obj_000003.png"
rgb_path = r"C:\Users\ESHWAR\OneDrive\Desktop\cynaptics\iitisoc\ycbv\ycbv_test_bop19\test\000050\rgb\001874.png"
trans = np.array([
    [-0.96266065, -0.27077264, -0.00840297,  0.1168339],
    [-0.0903287,   0.35005618, -0.93236334, -0.04114369],
    [ 0.25539999, -0.89674012, -0.36142495,  0.75233823],
    [ 0.,          0.,          0.,          1.]
])
# Apply 180° rotation around Y-axis to the object
Ry_180 = np.array([
    [1, 0,  0, 0],
    [ 0, 1,  0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0,  0, 1]
])

# Rotate the object by 180 degrees around Y
transformation = trans @ Ry_180

intrinsics = np.array([
    [1066.778, 0.0, 312.9869],
    [0.0, 1067.487, 241.3109],
    [0.0, 0.0, 1.0]
])

def rgb_to_20_gray(rgb_image):
    """
    Converts an RGB image to grayscale and reduces its intensity to 20%.
    Then converts it back to a 3-channel BGR image for blending.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    gray_20 = (gray * 0.2).astype(np.uint8)
    gray_3ch = cv2.cvtColor(gray_20, cv2.COLOR_GRAY2BGR)
    return gray_3ch

def project_points(points, transformation, intrinsics):
    """
    Projects 3D points from object space to 2D image coordinates.
    Returns projected 2D points, a valid mask, and camera-space Z-depths.
    """
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Transform points from object space to camera space
    transformed = (transformation @ points_hom.T).T[:, :4] # Keep 4th component (W) for perspective division

    # Get Z-depths (Z_c) from camera space coordinates
    # For perspective projection, we use the Z component before division by W
    zs = transformed[:, 2] 
    
    # Determine which points are valid (in front of the camera, Z_c > 0)
    # Also ensure W is not zero for valid division
    valid = (zs > 0) & (transformed[:, 3] != 0)

    # Extract camera intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Initialize projected coordinates with NaN for invalid points
    xs = np.full_like(zs, np.nan)
    ys = np.full_like(zs, np.nan)

    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 0:
        # Perform perspective division (X_c/Z_c, Y_c/Z_c) and then apply intrinsics
        # Note: Open3D uses a standard camera model where the transformation matrix
        # already yields points in camera space (X_c, Y_c, Z_c).
        # We need to explicitly divide by Z_c for perspective projection.
        
        # Clip points that are too close to the camera (Z_c near zero) to avoid division by zero or very large values
        # A small epsilon is used to prevent division by zero for Z.
        epsilon = 1e-6
        clipped_zs = np.maximum(zs[valid_indices], epsilon)

        xs[valid_indices] = (fx * transformed[valid_indices, 0] / clipped_zs) + cx
        ys[valid_indices] = (fy * transformed[valid_indices, 1] / clipped_zs) + cy

    projected = np.stack([xs, ys], axis=1)
    
    # Return projected 2D points, a boolean mask for valid points, and camera Z-depths
    # We return the original Zs (before clipping for projection) for Z-buffering.
    return projected, valid, zs 

def barycentric_coords(p, a, b, c):
    """
    Calculates barycentric coordinates for a point p within a 2D triangle abc.
    Returns (u, v, w) such that p = u*a + v*b + w*c.
    If the triangle is degenerate (area is zero), returns (-1, -1, -1).
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    den = v0[0] * v1[1] - v1[0] * v0[1]
    
    if abs(den) < 1e-9: # Use a small epsilon for floating point comparison for degeneracy
        return -1, -1, -1 
    
    v = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    u = 1.0 - v - w
    
    return u, v, w

def draw_textured_mesh_perspective_correct(image, mesh, projected_points, valid_mask, camera_zs, texture_image):
    """
    Renders a textured 3D mesh onto a 2D image using a custom software rasterizer
    with perspective-correct texture mapping and Z-buffering.
    """
    h, w = image.shape[:2]
    output_image = image.copy()
    
    # Initialize Z-buffer (depth buffer) with a very large value (infinity)
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)

    faces = mesh.faces
    uvs = mesh.visual.uv
    
    # Convert texture image to float32 for interpolation, and normalize to 0-1
    texture_image_float = texture_image.astype(np.float32) / 255.0
    texture_h, texture_w = texture_image.shape[:2]

    drawn_faces = 0

    for face_idx, tri_indices in enumerate(faces):
        # Get 3D vertices of the current triangle in object space
        vtx_3d_obj = mesh.vertices[tri_indices]

        # Get 2D projected points, UV coordinates, and camera Z-depths for the current triangle's vertices
        p_tri = projected_points[tri_indices] # Shape: (3, 2) - 2D (x, y) coordinates
        uv_tri = uvs[tri_indices]             # Shape: (3, 2) - UV (u, v) coordinates
        z_tri = camera_zs[tri_indices]        # Shape: (3,)   - Z-depths in camera space

        # Check if all three vertices are valid AND in front of the camera (Z > 0)
        # Also, check for NaN in projected points.
        if not all(valid_mask[tri_indices]) or np.any(np.isnan(p_tri)) or np.any(z_tri <= 0):
            continue

        # Prepare values for perspective-correct interpolation
        # We interpolate 1/Z, U/Z, V/Z
        # Small epsilon to prevent division by zero if Z is exactly 0 (though we filter Z<=0 above)
        epsilon_z = 1e-6 
        inv_z = 1.0 / np.maximum(z_tri, epsilon_z)
        
        # Multiply UVs by 1/Z
        uv_div_z = uv_tri * inv_z[:, np.newaxis] # np.newaxis makes inv_z (3,1) for broadcasting

        # Convert projected points to integer coordinates for bounding box calculation
        p_tri_int = np.int32(p_tri)

        # Calculate the bounding box for the projected triangle
        min_x = np.min(p_tri_int[:, 0])
        max_x = np.max(p_tri_int[:, 0])
        min_y = np.min(p_tri_int[:, 1])
        max_y = np.max(p_tri_int[:, 1])

        # Clip the bounding box to the actual image dimensions
        min_x = max(0, min_x)
        max_x = min(w - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(h - 1, max_y)

        if min_x > max_x or min_y > max_y:
            continue

        v0, v1, v2 = p_tri[0], p_tri[1], p_tri[2]
        
        # Iterate over each pixel within the clipped bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                pixel_coords = np.array([x, y], dtype=np.float32)
                
                # Calculate barycentric coordinates for the current pixel
                u_bary, v_bary, w_bary = barycentric_coords(pixel_coords, v0, v1, v2)

                # Check if the pixel is inside the triangle (with a small tolerance)
                if u_bary >= -1e-6 and v_bary >= -1e-6 and w_bary >= -1e-6 and (u_bary + v_bary + w_bary <= 1.0 + 1e-6):
                    
                    # Interpolate 1/Z for the current pixel
                    interpolated_inv_z = u_bary * inv_z[0] + v_bary * inv_z[1] + w_bary * inv_z[2]
                    
                    # Calculate the true interpolated Z-depth for Z-buffering
                    # Handle case where interpolated_inv_z is very small (far away or invalid)
                    if interpolated_inv_z < 1e-9: # Prevent division by zero or very large Z
                        continue 
                    interpolated_z = 1.0 / interpolated_inv_z

                    # Z-buffer test: Only draw if this pixel is closer than what's currently in the buffer
                    if interpolated_z < z_buffer[y, x]:
                        z_buffer[y, x] = interpolated_z # Update Z-buffer

                        # Interpolate U/Z and V/Z
                        interpolated_uv_div_z = u_bary * uv_div_z[0] + v_bary * uv_div_z[1] + w_bary * uv_div_z[2]
                        
                        # Perform final perspective division to get true UVs
                        interpolated_u = interpolated_uv_div_z[0] / interpolated_inv_z
                        interpolated_v = interpolated_uv_div_z[1] / interpolated_inv_z
                        
                        # Convert interpolated UVs (0-1 range) to texture pixel coordinates.
                        # Note: trimesh UVs often have (0,0) at bottom-left, while image coordinates
                        # have (0,0) at top-left. So, we invert the V-coordinate (1 - interpolated_v).
                        tex_x = int(interpolated_u * texture_w)
                        tex_y = int((1 - interpolated_v) * texture_h) 

                        # Clamp texture coordinates to ensure they are within the valid range
                        tex_x = max(0, min(texture_w - 1, tex_x))
                        tex_y = max(0, min(texture_h - 1, tex_y))
                        
                        # Get the color from the texture image at the calculated texture coordinates
                        # Convert back to uint8 for OpenCV image
                        color = (texture_image_float[tex_y, tex_x] * 255).astype(np.uint8)
                        
                        # Assign the sampled color to the corresponding pixel in the output image
                        output_image[y, x] = color
        drawn_faces += 1

    print("Total textured triangles drawn:", drawn_faces)
    return output_image


# --- Main execution flow ---

# 1. Load the RGB scene image
rgb_image = cv2.imread(rgb_path)
if rgb_image is None:
    raise FileNotFoundError(f"Error: RGB image not found at '{rgb_path}'. Please check the path and ensure it's valid.")

# Get the dimensions of the RGB image
image_height, image_width = rgb_image.shape[:2]

# 2. Convert the RGB scene to 20% grayscale
gray_image = rgb_to_20_gray(rgb_image)

# 3. Load the texture image for the mesh
texture_image = cv2.imread(texture_path)
if texture_image is None:
    raise FileNotFoundError(f"Error: Texture image not found at '{texture_path}'. Please check the path.")

# 4. Load the PLY mesh using trimesh
mesh = trimesh.load(mesh_path, process=False)

# 5. Validate if the mesh contains UV coordinates, which are essential for texturing
if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
    raise RuntimeError("Error: Mesh does not contain UVs (Texture coordinates). Cannot project texture.")

# 6. Apply the scale factor to the mesh vertices
scale_factor = 0.001
mesh.vertices *= scale_factor

# 7. Project the 3D mesh vertices to 2D image coordinates and get camera Z-depths
vertices = mesh.vertices
projected_points, valid_mask, camera_zs = project_points(vertices, transformation, intrinsics)

# 8. Draw the textured mesh onto the grayscale image using the custom perspective-correct rasterizer
projected_image = draw_textured_mesh_perspective_correct(gray_image.copy(), mesh, projected_points, valid_mask, camera_zs, texture_image)

# 9. Save and display the final projected image
output_filename = 'projected_mesh_textured_rasterizer_v2.png'
cv2.imwrite(output_filename, projected_image)
print(f"Result saved to: {output_filename}")
cv2.imshow('Projected Textured Mesh (Vibrant, Rasterizer V2)', projected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()