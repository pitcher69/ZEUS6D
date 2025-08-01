import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import open3d as o3d
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QueryDINOv2Processor:
    """Simplified DINOv2 processor for query visual feature extraction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        self.setup_dinov2_model()
        logger.info("🚀 Query DINOv2 Processor Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def setup_dinov2_model(self):
        """Setup DINOv2 model for feature extraction"""
        try:
            logger.info("🔄 Loading DINOv2 model...")
            
            # Load DINOv2 Giant model
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
            self.model = AutoModel.from_pretrained("facebook/dinov2-giant")
            
            # Move to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Model configuration
            self.patch_size = 14
            self.image_size = 518
            self.num_patches = (self.image_size // self.patch_size) ** 2  # 37x37 = 1369
            self.feature_dim = 1536  # DINOv2 Giant feature dimension
            
            logger.info(f"✅ DINOv2 model loaded on {self.device}")
            logger.info(f"📊 Patch size: {self.patch_size}, Image size: {self.image_size}")
            logger.info(f"📊 Number of patches: {self.num_patches}, Feature dim: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"❌ DINOv2 model loading failed: {e}")
            raise
    
    def load_camera_intrinsics(self, intrinsics_path: str) -> dict:
        """Load camera intrinsics from CNOS output JSON file"""
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics = json.load(f)
            
            logger.info(f"✅ Loaded camera intrinsics from: {intrinsics_path}")
            logger.info(f"📷 fx: {intrinsics['fx']}, fy: {intrinsics['fy']}")
            logger.info(f"📷 cx: {intrinsics['cx']}, cy: {intrinsics['cy']}")
            logger.info(f"📷 Image size: {intrinsics['width']}x{intrinsics['height']}")
            
            return intrinsics
            
        except Exception as e:
            logger.error(f"❌ Failed to load camera intrinsics: {e}")
            raise
    
    def extract_single_image_features(self, image_path: str) -> np.ndarray:
        """Extract DINOv2 features from a single image"""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Process with DINOv2 processor
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size=self.image_size
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get patch features [num_patches, feature_dim]
            patch_features = outputs.last_hidden_state[0, 1:, :].cpu().numpy()
            
            return patch_features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {image_path}: {e}")
            raise
    
    def project_points_to_image(self, points: np.ndarray, intrinsics: dict) -> tuple:
        """Project 3D points to image coordinates using dynamic camera intrinsics"""
        try:
            # Extract intrinsic parameters dynamically
            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx = intrinsics['cx']
            cy = intrinsics['cy']
            img_width = intrinsics['width']
            img_height = intrinsics['height']
            
            # Project 3D points to image plane
            z = points[:, 2]
            valid_depth = z > 0  # Points in front of camera
            
            u = (points[:, 0] * fx / z) + cx
            v = (points[:, 1] * fy / z) + cy
            
            # Check bounds
            in_bounds = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
            valid_mask = valid_depth & in_bounds
            
            uv_coords = np.column_stack([u, v])
            
            logger.info(f"📍 Projected {np.sum(valid_mask)}/{len(points)} points to image")
            
            return uv_coords, valid_mask
            
        except Exception as e:
            logger.error(f"❌ Point projection failed: {e}")
            raise
    
    def get_patch_indices(self, uv_coords: np.ndarray, intrinsics: dict) -> np.ndarray:
        """Get patch indices for projected points"""
        try:
            img_width = intrinsics['width']
            img_height = intrinsics['height']
            
            # Calculate patch grid size (37x37 for 518x518 image with 14x14 patches)
            patches_per_row = img_width // self.patch_size
            patches_per_col = img_height // self.patch_size
            
            # Convert image coordinates to patch indices
            u_idx = np.clip((uv_coords[:, 0] / self.patch_size).astype(int), 0, patches_per_row - 1)
            v_idx = np.clip((uv_coords[:, 1] / self.patch_size).astype(int), 0, patches_per_col - 1)
            
            # Convert to linear patch indices
            patch_indices = v_idx * patches_per_row + u_idx
            patch_indices = np.clip(patch_indices, 0, self.num_patches - 1)
            
            return patch_indices
            
        except Exception as e:
            logger.error(f"❌ Patch index calculation failed: {e}")
            raise
    
    def process_visual_features(self, data: dict) -> dict:
        """Main processing function for visual feature extraction"""
        try:
            # Extract input parameters
            rendered_images = data.get('rendered_images', [])
            camera_intrinsics_path = data.get('camera_intrinsics_path')
            point_cloud_path = data.get('point_cloud_path')
            job_id = data.get('job_id', f'job_{int(time.time())}')
            
            if not all([rendered_images, camera_intrinsics_path, point_cloud_path]):
                raise ValueError("Missing required inputs: rendered_images, camera_intrinsics_path, point_cloud_path")
            
            logger.info(f"🔷 Processing visual features for job {job_id}")
            logger.info(f"📸 Rendered images: {len(rendered_images)}")
            logger.info(f"📷 Camera intrinsics: {camera_intrinsics_path}")
            logger.info(f"☁️ Point cloud: {point_cloud_path}")
            
            # Load camera intrinsics from CNOS output
            intrinsics = self.load_camera_intrinsics(camera_intrinsics_path)
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
            num_points = points.shape[0]
            
            logger.info(f"☁️ Loaded point cloud with {num_points} points")
            
            # Initialize feature accumulator for each point
            point_features = [[] for _ in range(num_points)]
            
            # Process each rendered image
            start_time = time.time()
            
            for i, image_path in enumerate(rendered_images):
                try:
                    logger.info(f"🔄 Processing image {i+1}/{len(rendered_images)}: {Path(image_path).name}")
                    
                    # Extract DINOv2 features from image
                    patch_features = self.extract_single_image_features(image_path)
                    
                    # Project point cloud to image coordinates
                    uv_coords, valid_mask = self.project_points_to_image(points, intrinsics)
                    
                    if np.sum(valid_mask) == 0:
                        logger.warning(f"⚠️ No valid points projected for image {i+1}")
                        continue
                    
                    # Get patch indices for valid points
                    valid_uv = uv_coords[valid_mask]
                    valid_point_indices = np.where(valid_mask)[0]
                    patch_indices = self.get_patch_indices(valid_uv, intrinsics)
                    
                    # Assign features to corresponding points
                    for point_idx, patch_idx in zip(valid_point_indices, patch_indices):
                        feature_vector = patch_features[patch_idx]
                        point_features[point_idx].append(feature_vector)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process image {i+1}: {e}")
                    continue
            
            # Aggregate features for each point
            logger.info("🔄 Aggregating features across all images...")
            
            final_features = np.zeros((num_points, self.feature_dim), dtype=np.float32)
            
            for i, features_list in enumerate(point_features):
                if len(features_list) > 0:
                    # Average features across all views where point is visible
                    final_features[i] = np.mean(features_list, axis=0)
                else:
                    # Zero features for points not visible in any image
                    final_features[i] = np.zeros(self.feature_dim)
            
            # Apply PCA reduction to 64 dimensions
            logger.info("🔄 Applying PCA reduction to 64 dimensions...")
            pca = PCA(n_components=64)
            reduced_features = pca.fit_transform(final_features)
            
            # Apply L2 normalization
            logger.info("🔄 Applying L2 normalization...")
            normalized_features = normalize(reduced_features, norm='l2', axis=1)
            
            # Create output directory and save features
            output_base = Path(f"/app/output/{job_id}")
            output_dir = output_base / "dino_features"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "query_visual_features.npy"
            np.save(str(output_path), normalized_features)
            
            # Calculate processing metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                'status': 'success',
                'job_id': job_id,
                'output_path': str(output_path),
                'feature_dimensions': normalized_features.shape,
                'pca_components': 64,
                'l2_normalized': True,
                'processing_metrics': {
                    'processing_time': processing_time,
                    'images_processed': len(rendered_images),
                    'points_processed': num_points,
                    'images_per_second': len(rendered_images) / processing_time if processing_time > 0 else 0
                }
            }
            
            logger.info(f"✅ Visual feature extraction completed in {processing_time:.2f}s")
            logger.info(f"📊 Final features shape: {normalized_features.shape}")
            logger.info(f"💾 Saved to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Visual feature processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
processor = QueryDINOv2Processor()

@app.route('/extract-features', methods=['POST'])
def extract_visual_features():
    """Extract visual features from rendered images and point cloud"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = processor.process_visual_features(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'query-dinov2',
        'device': str(processor.device),
        'dinov2_available': True,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'query-dinov2',
        'version': '1.0',
        'description': 'DINOv2 visual feature extraction for query objects',
        'endpoints': ['/extract-features', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Query DINOv2 Processor Service")
    app.run(host='0.0.0.0', port=5009, debug=False)
