import sys
import os
import json
import time
import zipfile
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import open3d as o3d
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from flask import Flask, request, jsonify
import logging
import re

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TargetDINOv2Processor:
    """Simplified DINOv2 processor for target visual feature extraction"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        self.setup_dinov2()
        logger.info("🎨 Target DINOv2 Processor Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def setup_dinov2(self):
        """Setup DINOv2 model"""
        logger.info("🔄 Loading DINOv2 model...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load model and processor
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
            
            logger.info(f"✅ DINOv2 loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load DINOv2: {e}")
            raise
    
    def load_camera_intrinsics(self, intrinsics_path: str) -> np.ndarray:
        """Load camera intrinsics from JSON file"""
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = json.load(f)
            
            # Handle different intrinsics formats
            if isinstance(intrinsics_data, dict):
                if 'cam_K' in intrinsics_data:
                    return np.array(intrinsics_data['cam_K']).reshape(3, 3)
                else:
                    fx = float(intrinsics_data.get('fx', 525.0))
                    fy = float(intrinsics_data.get('fy', 525.0))
                    cx = float(intrinsics_data.get('cx', 320.0))
                    cy = float(intrinsics_data.get('cy', 240.0))
                    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            elif isinstance(intrinsics_data, list) and len(intrinsics_data) >= 9:
                return np.array(intrinsics_data).reshape(3, 3)
            
            # Default fallback
            return np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
            
        except Exception as e:
            logger.error(f"❌ Failed to load intrinsics: {e}")
            return np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from filename"""
        name = Path(filename).stem
        patterns = [r'(\d{6})', r'(\d{5})', r'(\d{4})', r'(\d{3})', r'(\d+)']
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def extract_frames_from_video(self, video_path: str, output_dir: Path, fps: float = 1.0) -> list:
        """Extract frames from video at specified FPS"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"📹 Extracting frames from video at {fps} FPS (interval: {frame_interval})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{extracted_count:06d}.png"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frames.append(str(frame_path))
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"✅ Extracted {len(frames)} frames from video")
        return frames
    
    def extract_zip_contents(self, zip_path: str, output_dir: Path) -> list:
        """Extract images from ZIP file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
            image_files = []
            
            for file_path in output_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(str(file_path))
            
            # Sort files naturally
            image_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', Path(x).name)])
            
            logger.info(f"✅ Extracted {len(image_files)} images from ZIP")
            return image_files
            
        except Exception as e:
            logger.error(f"❌ Failed to extract ZIP file {zip_path}: {e}")
            return []
    
    def process_input_source(self, input_path: str, output_dir: Path, source_type: str) -> list:
        """Process input source based on type"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        suffix = input_path.suffix.lower()
        
        if suffix == '.zip':
            return self.extract_zip_contents(str(input_path), output_dir)
        elif suffix in {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}:
            return self.extract_frames_from_video(str(input_path), output_dir, fps=1.0)
        elif suffix in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}:
            dest_path = output_dir / input_path.name
            shutil.copy2(input_path, dest_path)
            return [str(dest_path)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def project_points_to_image(self, points: np.ndarray, intrinsic: np.ndarray) -> tuple:
        """Project 3D points to image coordinates"""
        z = points[:, 2]
        valid = z > 0.01  # Filter out points too close to camera
        
        u = (points[:, 0] * intrinsic[0, 0]) / z + intrinsic[0, 2]
        v = (points[:, 1] * intrinsic[1, 1]) / z + intrinsic[1, 2]
        uv = np.stack([u, v], axis=1)
        
        return uv, valid
    
    def get_patch_indices(self, uv: np.ndarray, image_shape: tuple, patch_size: int = 14) -> np.ndarray:
        """Get patch indices for DINOv2 feature mapping"""
        h_patches = image_shape[0] // patch_size
        w_patches = image_shape[1] // patch_size
        
        u_idx = np.clip((uv[:, 0] / patch_size).astype(int), 0, w_patches - 1)
        v_idx = np.clip((uv[:, 1] / patch_size).astype(int), 0, h_patches - 1)
        
        return v_idx * w_patches + u_idx
    
    def extract_dinov2_features(self, rgb_image: Image.Image, point_cloud: o3d.geometry.PointCloud, 
                              intrinsic: np.ndarray, frame_id: str) -> np.ndarray:
        """Extract DINOv2 features for point cloud"""
        try:
            # Get image dimensions
            image_size = rgb_image.size[::-1]  # (height, width)
            patch_size = 14
            
            # Get point cloud points
            points = np.asarray(point_cloud.points)
            logger.info(f"📸 Processing {len(points)} points for frame {frame_id}")
            
            # Process image through DINOv2
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model(**inputs)
            
            # Get feature tokens (exclude CLS token)
            dino_features = output.last_hidden_state[0, 1:, :].cpu().numpy()  # [tokens, feature_dim]
            
            logger.info(f"   🎯 DINOv2 features shape: {dino_features.shape}")
            
            # Project points to image
            uv, valid = self.project_points_to_image(points, intrinsic)
            
            # Get patch indices for valid points
            patch_indices = self.get_patch_indices(uv[valid], image_size, patch_size)
            patch_indices = np.clip(patch_indices, 0, dino_features.shape[0] - 1)
            
            # Initialize feature array for all points
            point_features = np.zeros((points.shape[0], dino_features.shape[1]), dtype=np.float32)
            
            # Assign features to valid points
            if valid.sum() > 0:
                point_features[valid] = dino_features[patch_indices]
            
            logger.info(f"   📊 Valid points: {valid.sum()}/{len(valid)}")
            
            # Apply PCA to reduce dimensions (if needed)
            if dino_features.shape[1] > 256:
                logger.info("🎛️ Reducing dimensions with PCA")
                pca = PCA(n_components=256)
                point_features = pca.fit_transform(point_features)
            
            # Apply L2 normalization
            logger.info("🔄 Applying L2 normalization")
            normalized_features = normalize(point_features, norm='l2', axis=1)
            
            logger.info(f"   🎯 Final features shape: {normalized_features.shape}")
            
            return normalized_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for frame {frame_id}: {e}")
            # Return zero features as fallback
            return np.zeros((len(points), 256), dtype=np.float32)
    
    def process_visual_features(self, data: dict) -> dict:
        """Main processing function for visual feature extraction"""
        try:
            # Extract input parameters
            pointcloud_input = data.get('pointcloud_input')
            image_input = data.get('image_input')
            intrinsics_path = data.get('intrinsics_path')
            
            if not all([pointcloud_input, image_input, intrinsics_path]):
                raise ValueError("Missing required inputs: pointcloud_input, image_input, intrinsics_path")
            
            # Create output directory
            output_base = Path("/app/output")
            output_base.mkdir(exist_ok=True)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create subdirectories
                pc_dir = temp_path / 'pointclouds'
                img_dir = temp_path / 'images'
                features_dir = temp_path / 'features'
                
                for dir_path in [pc_dir, img_dir, features_dir]:
                    dir_path.mkdir()
                
                logger.info("🔷 Processing input files...")
                
                # Process inputs
                pc_files = self.process_input_source(pointcloud_input, pc_dir, 'pointcloud')
                img_files = self.process_input_source(image_input, img_dir, 'image')
                
                # Load camera intrinsics
                intrinsic = self.load_camera_intrinsics(intrinsics_path)
                
                # Ensure we have matching files
                min_files = min(len(pc_files), len(img_files))
                if min_files == 0:
                    raise ValueError("No valid files found")
                
                pc_files = pc_files[:min_files]
                img_files = img_files[:min_files]
                
                # Process each frame
                feature_files = []
                
                for i, (pc_file, img_file) in enumerate(zip(pc_files, img_files)):
                    frame_id = self.extract_frame_id(Path(img_file).name)
                    
                    try:
                        # Load point cloud and image
                        point_cloud = o3d.io.read_point_cloud(pc_file)
                        rgb_image = Image.open(img_file).convert("RGB")
                        
                        if len(point_cloud.points) == 0:
                            logger.warning(f"Empty point cloud for frame {frame_id}")
                            continue
                        
                        # Extract visual features
                        features = self.extract_dinov2_features(
                            rgb_image, point_cloud, intrinsic, frame_id
                        )
                        
                        # Save features
                        feature_filename = f"dinov2_features_{frame_id}.npy"
                        feature_path = features_dir / feature_filename
                        np.save(str(feature_path), features)
                        feature_files.append(str(feature_path))
                        
                        logger.info(f"✅ Processed frame {i+1}/{min_files}: {features.shape}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process frame {frame_id}: {e}")
                        continue
                
                # Generate output
                timestamp = int(time.time())
                
                if len(feature_files) > 1:
                    # Create ZIP for multiple features
                    zip_filename = f"visual_features_{timestamp}.zip"
                    zip_path = output_base / zip_filename
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                        for feature_file in feature_files:
                            zip_ref.write(feature_file, Path(feature_file).name)
                    
                    return {
                        'status': 'success',
                        'output_type': 'zip',
                        'output_path': str(zip_path),
                        'feature_count': len(feature_files)
                    }
                
                elif len(feature_files) == 1:
                    # Copy single feature file
                    feature_filename = f"visual_features_{timestamp}.npy"
                    output_path = output_base / feature_filename
                    shutil.copy2(feature_files[0], output_path)
                    
                    return {
                        'status': 'success',
                        'output_type': 'single',
                        'output_path': str(output_path),
                        'feature_count': 1
                    }
                
                else:
                    return {'status': 'error', 'message': 'No valid features generated'}
                    
        except Exception as e:
            logger.error(f"❌ Visual feature processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
processor = TargetDINOv2Processor()

@app.route('/extract-features', methods=['POST'])
def extract_visual_features():
    """Extract visual features from point clouds and images"""
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
        'service': 'target-dinov2',
        'device': str(processor.device),
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'target-dinov2',
        'version': '1.0',
        'endpoints': ['/extract-features', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Target DINOv2 Processor Service")
    app.run(host='0.0.0.0', port=5004, debug=False)
