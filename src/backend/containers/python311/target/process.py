
import cv2
import numpy as np
import open3d as o3d
import os
import imageio


def process_frame(mesh_path, rgb_path, transformation_path, intrinsics, output_folder, output_name):
    rgb_image = cv2.imread(rgb_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    scale_factor = 0.001
    mesh.scale(scale_factor, center=(0,0,0))

    transformation = np.load(transformation_path)
    vertices = np.asarray(mesh.vertices)

    projected_points, valid_mask = project_vertices(vertices, transformation, intrinsics)

    projected_image = draw_mesh_edges(rgb_image.copy(), mesh, projected_points, valid_mask)

    output_path = os.path.join(output_folder, output_name)
    cv2.imwrite(output_path, projected_image)
    return output_path


def project_vertices(vertices, transformation, intrinsics):
    vertices_hom = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    camera_space_points = (transformation @ vertices_hom.T).T[:, :3]

    zs = camera_space_points[:, 2]
    valid = zs > 0

    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]

    xs = (fx * camera_space_points[:, 0] / zs) + cx
    ys = (fy * camera_space_points[:, 1] / zs) + cy

    projected_2d = np.stack((xs, ys), axis=1)
    return projected_2d, valid


def draw_mesh_edges(image, mesh, projected_points, valid_mask):
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
            cv2.line(image, pts[0], pts[1], (0,255,0), 1)
            cv2.line(image, pts[1], pts[2], (0,255,0), 1)
            cv2.line(image, pts[2], pts[0], (0,255,0), 1)
    return image


def create_gif(image_paths, gif_path, duration=0.5):
    frames = [imageio.imread(img_path) for img_path in image_paths]
    imageio.mimsave(gif_path, frames, duration=duration)


def process_folder(mesh_path, rgb_folder, transformation_folder, intrinsics, output_folder, gif_path, duration=0.5):
    os.makedirs(output_folder, exist_ok=True)
    output_images = []

    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])

    for rgb_file in rgb_files:
        frame_id = os.path.splitext(rgb_file)[0]
        rgb_path = os.path.join(rgb_folder, rgb_file)
        transformation_filename = f'frameid_{int(frame_id)}.npy'
        transformation_path = os.path.join(transformation_folder, transformation_filename)

        if not os.path.exists(transformation_path):
            print(f"[WARNING] Transformation for frame {frame_id} not found, skipping.")
            continue

        output_name = f"frame_{frame_id}.png"
        out_img_path = process_frame(mesh_path, rgb_path, transformation_path, intrinsics, output_folder, output_name)
        output_images.append(out_img_path)

    create_gif(output_images, gif_path, duration)


# Example Usage
def process():
    mesh_path = "/content/data/model.ply"
    rgb_folder = "/content/data/rgb"
    transformation_folder = "/content/output/ransac"
    output_folder = "/content/overlay"
    gif_path = "output_animation.gif"

    intrinsics = np.array([
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0]
    ])

    process_folder(mesh_path, rgb_folder, transformation_folder, intrinsics, output_folder, gif_path, duration=0.5)
