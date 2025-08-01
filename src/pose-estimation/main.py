import sys
import os
import json
import time
import zipfile
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
import torch
import open3d as o3d
import trimesh
import pyrender
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.metrics.pairwise import cosine_similarity
from skimage.exposure import match_histograms
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from flask import Flask, request, jsonify
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MultiFilePoseEstimationPipeline:
    """Complete RANSAC + Symmetry-Aware Refinement Pipeline with Multi-File Support"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        self.setup_models()
        
        # RANSAC parameters with threshold-based early termination
        self.distance_threshold = 0.03
        self.fitness_threshold = 0.8  # Early termination if fitness > 0.8
        self.max_ransac_iterations = 10
        
        logger.info("🎯 Multi-File Pose Estimation Pipeline Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def setup_models(self):
        """Setup DINOv2 model"""
        try:
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
            self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-giant').to(self.device).eval()
            logger.info("✅ DINOv2 Giant model loaded")
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            raise
    
    def extract_files_from_zip_or_single(self, file_path: str, file_type: str) -> list:
        """Extract files from zip or return single file"""
        files = []
        
        if file_path.endswith('.zip'):
            # Extract zip file
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find files based on type
                if file_type == 'features':
                    pattern = '*.npy'
                elif file_type == 'point_clouds':
                    pattern = '*.ply'
                elif file_type == 'meshes':
                    pattern = ['*.ply', '*.obj']
                else:
                    pattern = '*'
                
                # Copy extracted files to persistent location
                extract_dir = f"/app/temp/{int(time.time())}_{file_type}"
                os.makedirs(extract_dir, exist_ok=True)
                
                if isinstance(pattern, list):
                    for p in pattern:
                        for f in Path(temp_dir).rglob(p):
                            shutil.copy2(f, extract_dir)
                            files.append(os.path.join(extract_dir, f.name))
                else:
                    for f in Path(temp_dir).rglob(pattern):
                        shutil.copy2(f, extract_dir)
                        files.append(os.path.join(extract_dir, f.name))
        else:
            # Single file
            files = [file_path]
        
        logger.info(f"✅ Found {len(files)} {file_type} files")
        return sorted(files)
    
    def extract_target_dino_feature(self, rgb_path: str, mask_path: str) -> np.ndarray:
        """Extract DINOv2 feature from target RGB image and mask"""
        try:
            # Load image and mask
            rgb = cv2.imread(rgb_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply mask
            masked_rgb = rgb.copy()
            masked_rgb[mask == 0] = 0
            
            # Crop bounding box of mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                raise ValueError("Mask is empty, no foreground detected!")
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            crop = masked_rgb[y_min:y_max+1, x_min:x_max+1]
            
            # Pad to make it square
            h, w, _ = crop.shape
            size = max(h, w)
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            padded[y_offset:y_offset+h, x_offset:x_offset+w] = crop
            
            # Resize to 224x224
            resized_crop = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Extract DINOv2 feature
            image_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                feature = outputs.last_hidden_state[:, 0, :]  # CLS token
                feature = torch.nn.functional.normalize(feature, dim=1)
            
            feature_np = feature.cpu().numpy()
            logger.info("✅ Target DINOv2 feature extracted")
            return feature_np
            
        except Exception as e:
            logger.error(f"❌ Target feature extraction failed: {e}")
            raise
    
    def chamfer_distance(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Compute chamfer distance between two point clouds"""
        tree1 = cKDTree(pc1)
        tree2 = cKDTree(pc2)
        dist1, _ = tree1.query(pc2)
        dist2, _ = tree2.query(pc1)
        return np.mean(dist1**2) + np.mean(dist2**2)
    
    def estimate_symmetries(self, mesh_path: str, chamfer_threshold: float = 50.0, 
                          num_points: int = 5000, step_degrees: int = 45) -> np.ndarray:
        """Estimate symmetry matrices for a 3D mesh"""
        logger.info(f"🔄 Estimating symmetries for mesh: {os.path.basename(mesh_path)}")
        
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertices():
            raise ValueError(f"Mesh has no vertices: {mesh_path}")
        
        mesh.compute_vertex_normals()
        pc = np.asarray(mesh.sample_points_uniformly(number_of_points=num_points).points)
        
        symmetries = []
        angles = np.radians(np.arange(0, 360, step_degrees))
        total_rotations = len(angles) ** 3
        
        logger.info(f"Testing {total_rotations} rotation combinations")
        
        rotation_count = 0
        for x in angles:
            for y in angles:
                for z in angles:
                    if rotation_count % 100 == 0:
                        logger.info(f"Progress: {rotation_count}/{total_rotations}")
                    
                    R_matrix = o3d.geometry.get_rotation_matrix_from_xyz([x, y, z])
                    rotated_pc = (R_matrix @ pc.T).T
                    dist = self.chamfer_distance(pc, rotated_pc)
                    
                    if dist < chamfer_threshold:
                        cleaned_R = np.where(np.abs(R_matrix) < 1e-7, 0, R_matrix)
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in symmetries:
                            if np.linalg.norm(cleaned_R - existing) < 1e-4:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            symmetries.append(cleaned_R)
                            logger.info(f"✅ Symmetry found (Chamfer: {dist:.4f})")
                    
                    rotation_count += 1
        
        logger.info(f"✅ Found {len(symmetries)} unique symmetries")
        return np.array(symmetries)
    
    def run_ransac_icp_with_threshold(self, query_pcd: o3d.geometry.PointCloud, 
                                     target_pcd: o3d.geometry.PointCloud,
                                     query_features: np.ndarray, 
                                     target_features: np.ndarray) -> tuple:
        """Run RANSAC + ICP with early termination on fitness threshold"""
        logger.info("🔄 Running RANSAC + ICP with early termination...")
        
        # Format features for Open3D
        query_feat = o3d.pipelines.registration.Feature()
        target_feat = o3d.pipelines.registration.Feature()
        query_feat.data = query_features.T
        target_feat.data = target_features.T
        
        best_fitness = -1
        best_transform = None
        
        for seed in range(self.max_ransac_iterations):
            o3d.utility.random.seed(seed)
            
            # RANSAC registration
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source=query_pcd,
                target=target_pcd,
                source_feature=query_feat,
                target_feature=target_feat,
                mutual_filter=True,
                max_correspondence_distance=self.distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000)
            )
            
            # ICP refinement
            result_icp = o3d.pipelines.registration.registration_icp(
                source=query_pcd,
                target=target_pcd,
                max_correspondence_distance=0.01,
                init=result_ransac.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            logger.info(f"Seed {seed}: ICP fitness = {result_icp.fitness:.4f}")
            
            if result_icp.fitness > best_fitness:
                best_fitness = result_icp.fitness
                best_transform = result_icp.transformation
                
                # Early termination if fitness threshold is met
                if best_fitness >= self.fitness_threshold:
                    logger.info(f"🎯 Early termination: fitness {best_fitness:.4f} >= threshold {self.fitness_threshold}")
                    break
        
        logger.info(f"✅ Best fitness: {best_fitness:.4f} (iterations: {seed + 1})")
        return best_transform, best_fitness
    
    def render_mesh_and_prepare_dino_input(self, mesh_path: str, pose: np.ndarray,
                                         img_w: int = 640, img_h: int = 480) -> np.ndarray:
        """Render mesh with given pose for DINOv2 input"""
        mesh = trimesh.load(mesh_path)
        mesh.apply_transform(pose)
        flip = np.diag([1, -1, -1, 1])
        mesh.apply_transform(flip)
        
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.2, 0.2, 0.2])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
        scene.add(light, pose=np.eye(4))
        
        center = mesh.bounding_box.centroid
        size = np.max(mesh.bounding_box.extents)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = center + np.array([0, 0, size * 3.0])
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        scene.add(camera, pose=camera_pose)
        
        renderer = pyrender.OffscreenRenderer(img_w, img_h)
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        
        # Crop to object bounding box
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Rendered object has no visible pixels!")
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        crop = rgb[y_min:y_max+1, x_min:x_max+1]
        
        # Pad to square
        h, w, _ = crop.shape
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = crop
        
        resized = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def extract_dino_feature(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract DINOv2 feature from RGB image"""
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            feature = outputs.last_hidden_state[:, 0, :]
        feature = feature.cpu().numpy()
        return feature / np.linalg.norm(feature)
    
    def match_histogram_to_target(self, source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """Match histogram of source to target image"""
        matched = match_histograms(source_img, target_img, channel_axis=-1)
        return matched.astype(np.uint8)
    
    def symmetry_aware_refinement(self, initial_pose: np.ndarray, symmetries: np.ndarray,
                                mesh_path: str, target_dino_feat: np.ndarray, 
                                target_rgb: np.ndarray, output_dir: str, target_idx: int) -> tuple:
        """Perform symmetry-aware pose refinement"""
        logger.info(f"🔄 Symmetry refinement for target {target_idx}...")
        
        if len(symmetries) == 0:
            logger.warning("No symmetries found, returning initial pose")
            return initial_pose, 0.0, -1
        
        best_similarity = -1
        best_idx = -1
        best_pose = initial_pose
        
        for idx, sym_R in enumerate(symmetries):
            sym_4x4 = np.eye(4)
            sym_4x4[:3, :3] = sym_R
            refined_pose = initial_pose @ sym_4x4
            
            # Render mesh with refined pose
            rendered_resized = self.render_mesh_and_prepare_dino_input(mesh_path, refined_pose)
            
            # Histogram matching
            hist_matched = self.match_histogram_to_target(rendered_resized, target_rgb)
            
            # Save debug image
            debug_path = f"{output_dir}/target_{target_idx}_symmetry_{idx}_matched.png"
            cv2.imwrite(debug_path, cv2.cvtColor(hist_matched, cv2.COLOR_RGB2BGR))
            
            # Extract DINOv2 feature and compute similarity
            query_dino_feat = self.extract_dino_feature(hist_matched)
            similarity = cosine_similarity(target_dino_feat, query_dino_feat)[0, 0]
            
            logger.info(f"Target {target_idx} - Symmetry {idx}: Similarity = {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
                best_pose = refined_pose
        
        logger.info(f"✅ Target {target_idx} - Best symmetry {best_idx}: {best_similarity:.4f}")
        return best_pose, best_similarity, best_idx
    
    def pose_error(self, pred_pose: np.ndarray, gt_pose: np.ndarray) -> tuple:
        """Calculate pose error metrics"""
        pred_R, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
        gt_R, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]
        
        R_diff = pred_R @ gt_R.T
        rot_error_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        rot_error_deg = np.degrees(rot_error_rad)
        trans_error = np.linalg.norm(pred_t - gt_t)
        
        return rot_error_deg, trans_error
    
    def process_multiple_targets(self, data: dict) -> dict:
        """Process multiple target files and find best match"""
        try:
            job_id = data.get('job_id', f'job_{int(time.time())}')
            logger.info(f"🎯 Starting multi-target pose estimation for job {job_id}")
            
            # Create output directory
            output_dir = f"/app/output/{job_id}/pose_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Load query data (single files)
            query_features = np.load(data['query_features_path']).astype(np.float64)
            query_pcd = o3d.io.read_point_cloud(data['query_point_cloud_path'])
            
            # Extract target files (can be zip or single)
            target_feature_files = self.extract_files_from_zip_or_single(
                data['target_features_path'], 'features'
            )
            target_pcd_files = self.extract_files_from_zip_or_single(
                data['target_point_cloud_path'], 'point_clouds'
            )
            
            # Load additional data
            if 'ground_truth_json' in data:
                with open(data['ground_truth_json'], 'r') as f:
                    scene_gt = json.load(f)
            else:
                scene_gt = None
            
            # Extract RGB and mask for target DINOv2 feature extraction
            target_dino_feat = None
            target_rgb = None
            
            if 'rgb_masks_zip' in data and 'binary_masks_zip' in data:
                rgb_files = self.extract_files_from_zip_or_single(data['rgb_masks_zip'], 'images')
                mask_files = self.extract_files_from_zip_or_single(data['binary_masks_zip'], 'images')
                
                if rgb_files and mask_files:
                    target_dino_feat = self.extract_target_dino_feature(rgb_files[0], mask_files[0])
                    target_rgb = cv2.imread(rgb_files[0])
                    target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)
            
            # Process each target combination
            best_overall_score = -1
            best_target_result = None
            all_results = []
            
            # Ensure we have matching numbers of features and point clouds
            min_targets = min(len(target_feature_files), len(target_pcd_files))
            
            for target_idx in range(min_targets):
                logger.info(f"🔄 Processing target {target_idx + 1}/{min_targets}")
                
                try:
                    # Load target data
                    target_features = np.load(target_feature_files[target_idx]).astype(np.float64)
                    target_pcd = o3d.io.read_point_cloud(target_pcd_files[target_idx])
                    
                    logger.info(f"Target {target_idx}: Features {target_features.shape}, Points {len(target_pcd.points)}")
                    
                    # RANSAC + ICP registration
                    initial_pose, ransac_fitness = self.run_ransac_icp_with_threshold(
                        query_pcd, target_pcd, query_features, target_features
                    )
                    
                    # Save initial pose
                    initial_pose_path = f"{output_dir}/target_{target_idx}_initial_transformation.npy"
                    np.save(initial_pose_path, initial_pose)
                    
                    # Estimate symmetries if mesh provided
                    refined_pose = initial_pose
                    best_similarity = 0.0
                    best_sym_idx = -1
                    symmetries = np.array([])
                    
                    if 'target_meshes_zip' in data:
                        mesh_files = self.extract_files_from_zip_or_single(data['target_meshes_zip'], 'meshes')
                        if target_idx < len(mesh_files):
                            mesh_path = mesh_files[target_idx]
                            logger.info(f"Processing mesh: {os.path.basename(mesh_path)}")
                            
                            # Estimate symmetries
                            symmetries = self.estimate_symmetries(mesh_path)
                            
                            # Symmetry-aware refinement
                            if len(symmetries) > 0 and target_dino_feat is not None:
                                refined_pose, best_similarity, best_sym_idx = self.symmetry_aware_refinement(
                                    initial_pose, symmetries, mesh_path, target_dino_feat, target_rgb, 
                                    output_dir, target_idx
                                )
                    
                    # Calculate pose errors if ground truth available
                    pose_metrics = {}
                    if scene_gt and 'frame_id' in data and 'target_obj_id' in data:
                        frame_id = str(data['frame_id'])
                        target_obj_id = data['target_obj_id']
                        
                        gt_pose = None
                        for obj in scene_gt.get(frame_id, []):
                            if obj.get('obj_id') == target_obj_id:
                                R_gt = np.array(obj['cam_R_m2c']).reshape(3, 3)
                                t_gt = np.array(obj['cam_t_m2c']).reshape(3) / 1000.0
                                gt_pose = np.eye(4)
                                gt_pose[:3, :3] = R_gt
                                gt_pose[:3, 3] = t_gt
                                break
                        
                        if gt_pose is not None:
                            rot_error, trans_error = self.pose_error(refined_pose, gt_pose)
                            pose_metrics = {
                                'rotation_error_deg': float(rot_error),
                                'translation_error_m': float(trans_error),
                                'ground_truth_available': True
                            }
                    
                    # Calculate combined score
                    combined_score = (
                        0.5 * ransac_fitness +
                        0.3 * best_similarity +
                        0.2 * (1.0 / (1.0 + pose_metrics.get('rotation_error_deg', 180))) if pose_metrics else 0.0
                    )
                    
                    # Target result
                    target_result = {
                        'target_index': target_idx,
                        'target_feature_file': target_feature_files[target_idx],
                        'target_pointcloud_file': target_pcd_files[target_idx],
                        'initial_transformation': initial_pose.tolist(),
                        'final_transformation': refined_pose.tolist(),
                        'ransac_fitness': float(ransac_fitness),
                        'symmetry_similarity': float(best_similarity),
                        'best_symmetry_index': int(best_sym_idx) if best_sym_idx >= 0 else None,
                        'symmetries_found': len(symmetries),
                        'pose_metrics': pose_metrics,
                        'combined_score': float(combined_score)
                    }
                    
                    all_results.append(target_result)
                    
                    # Save individual result
                    final_pose_path = f"{output_dir}/target_{target_idx}_final_transformation.npy"
                    np.save(final_pose_path, refined_pose)
                    
                    # Check if this is the best result so far
                    if combined_score > best_overall_score:
                        best_overall_score = combined_score
                        best_target_result = target_result.copy()
                        best_target_result['final_pose_path'] = final_pose_path
                    
                    logger.info(f"✅ Target {target_idx} - Score: {combined_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing target {target_idx}: {e}")
                    continue
            
            # Create comprehensive metadata
            result_metadata = {
                'job_id': job_id,
                'processing_summary': {
                    'total_targets_processed': len(all_results),
                    'best_target_index': best_target_result['target_index'] if best_target_result else None,
                    'best_overall_score': float(best_overall_score)
                },
                'best_result': best_target_result,
                'all_results': all_results,
                'timestamp': time.time()
            }
            
            metadata_path = f"{output_dir}/multi_target_pose_estimation_results.json"
            with open(metadata_path, 'w') as f:
                json.dump(result_metadata, f, indent=2)
            
            logger.info(f"✅ Multi-target pose estimation completed for job {job_id}")
            logger.info(f"Best result: Target {best_target_result['target_index']} with score {best_overall_score:.4f}")
            
            return {
                'status': 'success',
                'job_id': job_id,
                'output_directory': output_dir,
                'metadata_path': metadata_path,
                'best_result': best_target_result,
                'summary': result_metadata['processing_summary']
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-target pose estimation failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
pose_pipeline = MultiFilePoseEstimationPipeline()

@app.route('/estimate-pose', methods=['POST'])
def estimate_pose():
    """Multi-target pose estimation pipeline endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = pose_pipeline.process_multiple_targets(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'pose-estimation',
        'pipeline_ready': True,
        'features': [
            'multi_target_processing',
            'zip_file_support',
            'early_ransac_termination',
            'symmetry_aware_refinement',
            'comprehensive_scoring'
        ],
        'ransac_config': {
            'max_iterations': pose_pipeline.max_ransac_iterations,
            'fitness_threshold': pose_pipeline.fitness_threshold,
            'distance_threshold': pose_pipeline.distance_threshold
        },
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'pose-estimation',
        'version': '1.0',
        'description': 'Multi-Target RANSAC + Symmetry-Aware Refinement Pipeline',
        'endpoints': ['/estimate-pose', '/health'],
        'features': [
            'Processes multiple target files from zip archives',
            'Early RANSAC termination on fitness threshold',
            'Symmetry-aware pose refinement',
            'Comprehensive scoring and ranking',
            'Complete pose error analysis'
        ]
    })

if __name__ == '__main__':
    logger.info("🎯 Starting Multi-Target Pose Estimation Pipeline Service")
    app.run(host='0.0.0.0', port=5012, debug=False)
