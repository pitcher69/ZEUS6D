import cv2
import numpy as np
import open3d as o3d
import json

mesh_path = "model.ply"
rgb_path = "rgb.png"
transformation =np.load("matrix.npy")
intrinsics = np.load("intrinsics.npy")

with open('scene_gt.json', 'r') as f:
    data = json.load(f)

gt_data = np.array(data)

gt_rotation = np.array(gt_data["cam_R_m2c"]).reshape(3, 3)
gt_translation = (np.array(gt_data["cam_t_m2c"]) / 1000.0).reshape(3, 1)

gt_transformation = np.eye(4)
gt_transformation[:3, :3] = gt_rotation
gt_transformation[:3, 3] = gt_translation.flatten()

def rgb_to_20_gray(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    gray_20 = (gray * 0.2).astype(np.uint8)
    gray_3ch = cv2.cvtColor(gray_20, cv2.COLOR_GRAY2BGR)
    return rgb_image

def project_vertices(vertices, transformation, intrinsics):
    vertices_hom = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # Nx4
    camera_space_points = (transformation @ vertices_hom.T).T[:, :3]
    
    zs = camera_space_points[:, 2]
    valid = zs > 0

    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]

    xs = (fx * camera_space_points[:, 0] / zs) + cx
    ys = (fy * camera_space_points[:, 1] / zs) + cy

    projected_2d = np.stack((xs, ys), axis=1)
    return projected_2d, valid

def draw_mesh_edges(image, mesh, projected_points, valid_mask, color):
    h, w = image.shape[:2]
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        pts = []
        for idx in tri:
            if not valid_mask[idx]:
                break
            x, y = projected_points[idx]
            if 0 <= int(x) < w and 0 <= int(y) < h:
                pts.append((int(x), int(y)))
        if len(pts) == 3:
            cv2.line(image, pts[0], pts[1], color, 1)
            cv2.line(image, pts[1], pts[2], color, 1)
            cv2.line(image, pts[2], pts[0], color, 1)
    return image

# Load image and mesh
rgb_image = cv2.imread(rgb_path)
gray_image = rgb_to_20_gray(rgb_image)

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

# Apply scaling
scale_factor = 0.001
mesh.scale(scale_factor, center=(0,0,0))

vertices = np.asarray(mesh.vertices)

# Project using your estimated transformation
projected_points_est, valid_mask_est = project_vertices(vertices, transformation, intrinsics)

# Project using ground truth transformation
projected_points_gt, valid_mask_gt = project_vertices(vertices, gt_transformation, intrinsics)

result_image_red = draw_mesh_edges(gray_image.copy(), mesh, projected_points_gt, valid_mask_gt, color=(0,0,255))
cv2.imwrite("only_red.png", result_image_red)

# Now draw green on top
result_image = draw_mesh_edges(result_image_red.copy(), mesh, projected_points_est, valid_mask_est, color=(0,255,0))
cv2.imwrite("green_on_red.png", result_image)

cv2.imwrite('comparison_projected_mesh_edges.png', result_image)
cv2.imshow('Comparison: Green=Estimated, red=GroundTruth', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
