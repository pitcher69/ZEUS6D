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
import open3d as o3d
from flask import Flask, request, jsonify
import logging
import re

# Add GeDi and PointNet2 to path
sys.path.insert(0, '/app/gedi-repo')
sys.path.insert(0, '/app/pointnet2')

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    # Import GeDi after path setup
    from gedi import GeDi
    logger.info("✅ GeDi imported successfully")
    GEDI_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import GeDi: {e}")
    GEDI_AVAILABLE = False

class TargetGeDiProcessor:
    """Simplified GeDi processor for target geometric feature extraction"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        self.setup_gedi()
        logger.info("🔷 Target GeDi Processor Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def setup_gedi(self):
        """Setup GeDi models"""
        try:
            if GEDI_AVAILABLE:
                # Base configuration for GeDi models
                self.base_config = {
                    'dim': 32,
                    'samples_per_batch': 500,
                    'samples_per_patch_lrf': 4000,
                    'samples_per_patch_out': 512,
                    # Note: checkpoint path would need to be available
                    # 'fchkpt_gedi_net': '/app/gedi-repo/data/chkpts/3dmatch/chkpt.tar'
                }
                logger.info("✅ GeDi configuration ready")
            else:
                self.base_config = None
                logger.warning("⚠️ GeDi not available, using fallback geometric features")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup GeDi: {e}")
            self.base_config = None
    
    def normalize_pcd_max_extent(self, pcd):
        """Normalize point cloud to unit scale based on maximum extent"""
        try:
            aabb = pcd.get_axis_aligned_bounding_box()
            extents = np.asarray(aabb.get_extent())
            max_extent = extents.max()
            
            if max_extent == 0:
                logger.warning("Point cloud has zero size, skipping normalization")
                return pcd
            
            pcd.scale(1.0 / max_extent, center=(0, 0, 0))
            return pcd
            
        except Exception as e:
            logger.error(f"Failed to normalize point cloud: {e}")
            return pcd
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from PLY filename"""
        name = Path(filename).stem
        
        patterns = [
            r'target_(\d+)',
            r'pointcloud_(\d+)',
            r'(\d{6})',
            r'(\d{5})',
            r'(\d{4})',
            r'(\d{3})',
            r'(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def extract_frames_from_video(self, video_path: str, output_dir: Path, fps: float = 1.0) -> list:
        """Extract frames from video at specified FPS (if needed for point cloud videos)"""
        # This would be implemented if point cloud videos are supported
        # For now, we assume point cloud inputs are ZIP archives or single files
        return []
    
    def extract_zip_contents(self, zip_path: str, output_dir: Path) -> list:
        """Extract point cloud files from ZIP archive"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Find all PLY files
            ply_files = []
            for file_path in output_dir.rglob('*.ply'):
                ply_files.append(str(file_path))
            
            # Sort files naturally
            ply_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', Path(x).name)])
            
            logger.info(f"✅ Extracted {len(ply_files)} point cloud files from ZIP")
            return ply_files
            
        except Exception as e:
            logger.error(f"❌ Failed to extract ZIP file {zip_path}: {e}")
            return []
    
    def process_input_source(self, input_path: str, output_dir: Path) -> list:
        """Process input source based on type"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        suffix = input_path.suffix.lower()
        
        if suffix == '.zip':
            return self.extract_zip_contents(str(input_path), output_dir)
        elif suffix == '.ply':
            dest_path = output_dir / input_path.name
            shutil.copy2(input_path, dest_path)
            return [str(dest_path)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def extract_gedi_features(self, point_cloud_path: str, frame_id: str) -> np.ndarray:
        """Extract geometric features using GeDi or fallback method"""
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            
            if len(pcd.points) == 0:
                raise ValueError(f"Empty point cloud: {point_cloud_path}")
            
            # Normalize point cloud
            pcd = self.normalize_pcd_max_extent(pcd)
            
            if GEDI_AVAILABLE and self.base_config is not None:
                # Use GeDi for feature extraction
                return self._extract_gedi_features_advanced(pcd, frame_id)
            else:
                # Use fallback basic geometric features
                return self._extract_basic_geometric_features(pcd, frame_id)
                
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {point_cloud_path}: {e}")
            # Return zero features as fallback
            return np.zeros((1000, 64), dtype=np.float32)
    
    def _extract_gedi_features_advanced(self, pcd, frame_id: str) -> np.ndarray:
        """Advanced GeDi feature extraction"""
        try:
            # Estimate diameter for radius calculation
            aabb = pcd.get_axis_aligned_bounding_box()
            min_b, max_b = np.asarray(aabb.min_bound), np.asarray(aabb.max_bound)
            diameter = np.linalg.norm(max_b - min_b)
            
            # Convert to tensor
            pts = torch.tensor(np.asarray(pcd.points)).float()
            
            # Initialize GeDi models with different radii
            config_03 = self.base_config.copy()
            config_03['r_lrf'] = 0.3 * diameter
            
            config_04 = self.base_config.copy()
            config_04['r_lrf'] = 0.4 * diameter
            
            gedi_03 = GeDi(config=config_03)
            gedi_04 = GeDi(config=config_04)
            
            # Compute descriptors for both radii
            desc_03 = gedi_03.compute(pts=pts, pcd=pts)
            desc_04 = gedi_04.compute(pts=pts, pcd=pts)
            
            # Ensure outputs are numpy arrays
            if isinstance(desc_03, torch.Tensor):
                desc_03 = desc_03.detach().cpu().numpy()
            if isinstance(desc_04, torch.Tensor):
                desc_04 = desc_04.detach().cpu().numpy()
            
            # Concatenate features → N x 64
            desc_combined = np.concatenate([desc_03, desc_04], axis=1)
            
            logger.info(f"✅ GeDi features extracted for {frame_id}: {desc_combined.shape}")
            
            # Clean up GPU memory
            del gedi_03, gedi_04, desc_03, desc_04, pts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return desc_combined.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Advanced GeDi extraction failed: {e}")
            return self._extract_basic_geometric_features(pcd, frame_id)
    
    def _extract_basic_geometric_features(self, pcd, frame_id: str) -> np.ndarray:
        """Fallback basic geometric feature extraction"""
        try:
            points = np.asarray(pcd.points)
            
            # Ensure we have points to work with
            if len(points) == 0:
                return np.zeros((1000, 64), dtype=np.float32)
            
            # Resample to 1000 points if needed
            if len(points) != 1000:
                if len(points) > 1000:
                    indices = np.random.choice(len(points), 1000, replace=False)
                    points = points[indices]
                else:
                    # Upsample by repeating points
                    while len(points) < 1000:
                        needed = min(len(points), 1000 - len(points))
                        points = np.vstack([points, points[:needed]])
            
            # Extract basic geometric features for each point
            features = []
            
            for i, point in enumerate(points):
                # Distance-based features
                distances = np.linalg.norm(points - point, axis=1)
                
                # Statistical features
                point_features = [
                    point[0], point[1], point[2],  # Coordinates (3)
                    np.mean(distances),             # Mean distance to other points (1)
                    np.std(distances),              # Std distance (1)
                    np.min(distances[distances > 0]) if np.any(distances > 0) else 0,  # Min non-zero distance (1)
                    np.max(distances),              # Max distance (1)
                ]
                
                # Add more geometric features
                centroid = np.mean(points, axis=0)
                point_features.extend([
                    np.linalg.norm(point - centroid),  # Distance to centroid (1)
                ])
                
                # Pad or truncate to 64 dimensions
                while len(point_features) < 64:
                    point_features.append(0.0)
                
                point_features = point_features[:64]
                features.append(point_features)
            
            features_array = np.array(features, dtype=np.float32)
            
            # L2 normalize each feature vector
            norms = np.linalg.norm(features_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            features_array = features_array / norms
            
            logger.info(f"✅ Basic geometric features extracted for {frame_id}: {features_array.shape}")
            return features_array
            
        except Exception as e:
            logger.error(f"❌ Basic geometric feature extraction failed: {e}")
            return np.zeros((1000, 64), dtype=np.float32)
    
    def process_geometric_features(self, data: dict) -> dict:
        """Main processing function for geometric feature extraction"""
        try:
            # Extract input parameters
            pointcloud_input = data.get('pointcloud_input')
            
            if not pointcloud_input:
                raise ValueError("Missing required input: pointcloud_input")
            
            # Create output directory
            output_base = Path("/app/output")
            output_base.mkdir(exist_ok=True)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pc_dir = temp_path / 'pointclouds'
                features_dir = temp_path / 'features'
                
                for dir_path in [pc_dir, features_dir]:
                    dir_path.mkdir()
                
                logger.info("🔷 Processing input point cloud files...")
                
                # Process inputs
                pc_files = self.process_input_source(pointcloud_input, pc_dir)
                
                if not pc_files:
                    raise ValueError("No valid point cloud files found")
                
                # Process each point cloud
                feature_files = []
                
                for i, pc_file in enumerate(pc_files):
                    frame_id = self.extract_frame_id(Path(pc_file).name)
                    
                    try:
                        # Extract geometric features
                        features = self.extract_gedi_features(pc_file, frame_id)
                        
                        # Save features
                        feature_filename = f"gedi_features_{frame_id}.npy"
                        feature_path = features_dir / feature_filename
                        np.save(str(feature_path), features)
                        feature_files.append(str(feature_path))
                        
                        logger.info(f"✅ Processed frame {i+1}/{len(pc_files)}: {features.shape}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process frame {frame_id}: {e}")
                        continue
                
                # Generate output
                timestamp = int(time.time())
                
                if len(feature_files) > 1:
                    # Create ZIP for multiple features
                    zip_filename = f"geometric_features_{timestamp}.zip"
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
                    feature_filename = f"geometric_features_{timestamp}.npy"
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
            logger.error(f"❌ Geometric feature processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
processor = TargetGeDiProcessor()

@app.route('/extract-features', methods=['POST'])
def extract_geometric_features():
    """Extract geometric features from point clouds"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = processor.process_geometric_features(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'target-gedi',
        'device': str(processor.device),
        'gedi_available': GEDI_AVAILABLE,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'target-gedi',
        'version': '1.0',
        'endpoints': ['/extract-features', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Target GeDi Processor Service")
    app.run(host='0.0.0.0', port=5005, debug=False)
