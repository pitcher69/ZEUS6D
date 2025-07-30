import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import trimesh
import pyrender
from tqdm import tqdm
import os

cutout_transform = transforms.Compose([
        transforms.Resize((448, 448)), # DINOv2 typically uses 224x224 or 518x518
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
render_transform = transforms.Compose([
    transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def render_with_intrinsics(
    mesh: trimesh.Trimesh,
    mesh_pose: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
):
    #mesh.apply_translation(-mesh.bounding_box.centroid)
    #scale = 1.0 / np.max(mesh.extents)
    #mesh.apply_scale(scale)
    fx,fy,cx,cy=intrinsics[0][0],intrinsics[1][1],intrinsics[0][2],intrinsics[1][2]
    width = int(cx * 2)
    height = int(cy * 2)

    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0], ambient_light=[4, 4, 4])

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    mesh_node = scene.add(render_mesh,pose=mesh_pose)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    scene.add(camera,pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=30.0)
    scene.add(light,pose=camera_pose)  

    renderer = pyrender.OffscreenRenderer(width, height)
    color_image, depth = renderer.render(scene)

    renderer.delete()

    return color_image,depth

def get_dino_features(model, image_tensor):
    """Extracts features from an image using DINOv2."""
    with torch.no_grad():
        outputs = model(image_tensor)
    # We are interested in the patch features, typically the last hidden state
    # The output is (batch_size, num_patches + 1, feature_dim)
    # The first token is the [CLS] token, so we take features from index 1 onwards
    return outputs.last_hidden_state[:, 1:, :]

def cosine_sim(model,reference_tensor,choices_tensor):

    all_cutout_features = get_dino_features(model,choices_tensor)

    all_render_features = get_dino_features(model,reference_tensor)

    all_cutout_features = torch.nn.functional.normalize(all_cutout_features, p=2, dim=-1, eps=1e-12)
    all_render_features = torch.nn.functional.normalize(all_render_features, p=2, dim=-1, eps=1e-12)
    #print(all_cutout_features.shape)

    cosine_sims = torch.einsum("ijk,xjk->ixj",all_cutout_features,all_render_features)
    #print(cosine_sims.shape)
    cosine_sims_avg = torch.mean(cosine_sims,dim=2)
    cosine_sims /= all_render_features.shape[1]

    return torch.argmax(cosine_sims_avg,dim=0),cosine_sims_avg,cosine_sims

def SAR_SingleCutout(cutout,object_pose,sym_matrces,model,output_dir,camera_intrinsic=None):
    cutout = Image.open(cutout)
    rendered_views = []
    rendered_view_image = []

    for sym_mat in sym_matrces:
        flip_mat = np.eye(4)
        flip_mat[1,1] = -1
        flip_mat[2,2] = -1
        final_pose = np.eye(4)
        final_pose[:3,:3] = sym_mat
        final_pose = flip_mat @ object_pose @ final_pose
        intrinsics = camera_intrinsic or np.array([
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0]]) 
        mesh_or_scene = trimesh.load("models/obj_000003.ply")
        if isinstance(mesh_or_scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh_or_scene.geometry.values()]
            )
        else:
            mesh = mesh_or_scene

        mesh.apply_scale(0.005)
        camera_pose = np.eye(4)
        camera_pose[0,0] = 1
        camera_pose[2,2] = 1
        camera_pose[1,3] = 0
        camera_pose[2,3] = 3
        render_image,depth = render_with_intrinsics(mesh,final_pose,camera_pose,intrinsics)
        non_zero_coords = np.argwhere(depth != 0)

        min_row = np.min(non_zero_coords[:, 0])
        max_row = np.max(non_zero_coords[:, 0])
        min_col = np.min(non_zero_coords[:, 1])
        max_col = np.max(non_zero_coords[:, 1])

        render_image = render_image[min_row:max_row,min_col:max_col]
        rendered_view_image.append(Image.fromarray(render_image))
        rendered_views.append(render_transform(Image.fromarray(render_image)))
        renders_tensor = torch.stack(rendered_views).to("cuda")

    cutout_tensor = cutout_transform(cutout).to("cuda").unsqueeze(0)
    cutout_tensor.shape,renders_tensor.shape

    best_match_idicies,cosine_sims_avg,cosine_sims = cosine_sim(model,cutout_tensor,renders_tensor)

    final_pose = np.eye(4)
    final_pose[:3,:3] = sym_matrces[best_match_idicies]

    np.save(output_dir,object_pose @ final_pose)

def SAR(cutouts,object_poses,symmetry_matrices,model,output_dirs,camera_intrinsic):
    for i,(object_pose,cutout,output_dir) in tqdm(enumerate(zip(object_poses,cutouts,output_dirs))):
        SAR_SingleCutout(cutout,object_pose,symmetry_matrices,model,output_dir,camera_intrinsic)