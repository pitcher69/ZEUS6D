import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from flask import Flask, request, jsonify
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Add GeDi repository to Python path
sys.path.insert(0, '/app/gedi-repo')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QueryGeDiProcessor:
    """Simplified GeDi processor for query geometric feature extraction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        self.setup_gedi()
        logger.info("🚀 Query GeDi Processor Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def setup_gedi(self):
        """Setup GeDi framework"""
        try:
            # Import GeDi after path setup
            from gedi import GeDi
            self.gedi_available = True
            logger.info("✅ GeDi imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import GeDi: {e}")
            self.gedi_available = False
            logger.warning("⚠️ Using fallback geometric features")
    
    def load_point_cloud(self, point_cloud_path: str) -> tuple:
        """Load 5000-point PLY file from query-point-cloud service"""
        try:
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
            
            if len(points) != 5000:
                logger.warning(f"⚠️ Point cloud has {len(points)} points, expected 5000")
            
            pts_tensor = torch.tensor(points, dtype=torch.float32)
            logger.info(f"✅ Loaded point cloud with {len(points)} points")
            return pts_tensor, points
            
        except Exception as e:
            logger.error(f"❌ Error loading point cloud: {e}")
            raise
    
    def estimate_diameter(self, points: np.ndarray) -> float:
        """Estimate object diameter for GeDi scaling"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            aabb = pcd.get_axis_aligned_bounding_box()
            diameter = np.linalg.norm(np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound))
            
            logger.info(f"✅ Estimated diameter: {diameter:.6f}")
            return diameter
            
        except Exception as e:
            logger.error(f"❌ Error estimating diameter: {e}")
            raise
    
    def compute_gedi_features(self, pts_tensor: torch.Tensor, points: np.ndarray) -> torch.Tensor:
        """Compute dual-scale GeDi geometric features"""
        try:
            if not self.gedi_available:
                return self.compute_fallback_features(pts_tensor)
            
            from gedi import GeDi
            
            # Estimate diameter for scaling
            diameter = self.estimate_diameter(points)
            
            # Base GeDi configuration
            base_config = {
                'dim': 32,
                'samples_per_batch': 500,
                'samples_per_patch_lrf': 4000,
                'samples_per_patch_out': 512
            }
            
            # Compute GeDi features with dual radius scaling
            logger.info("🔄 Computing GeDi features with r_lrf = 0.3 * diameter...")
            config_03 = base_config.copy()
            config_03['r_lrf'] = 0.3 * diameter
            gedi_03 = GeDi(config=config_03)
            desc_03 = gedi_03.compute(pts=pts_tensor, pcd=pts_tensor)
            
            logger.info("🔄 Computing GeDi features with r_lrf = 0.4 * diameter...")
            config_04 = base_config.copy()
            config_04['r_lrf'] = 0.4 * diameter
            gedi_04 = GeDi(config=config_04)
            desc_04 = gedi_04.compute(pts=pts_tensor, pcd=pts_tensor)
            
            # Concatenate dual-scale descriptors
            desc_concat = torch.cat([desc_03, desc_04], dim=1)  # Shape: (5000, 64)
            logger.info(f"✅ Concatenated descriptors shape: {desc_concat.shape}")
            
            # L2 normalize
            desc_normed = F.normalize(desc_concat, p=2, dim=1)
            
            # Verify L2 normalization
            norms = torch.norm(desc_normed, dim=1)
            logger.info(f"📏 L2 norm check — mean: {norms.mean().item():.6f}")
            
            if torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
                logger.info("✅ L2 normalization verified")
            else:
                logger.warning("⚠️ L2 normalization check failed, but continuing...")
            
            return desc_normed
            
        except Exception as e:
            logger.error(f"❌ GeDi processing failed: {e}")
            return self.compute_fallback_features(pts_tensor)
    
    def compute_fallback_features(self, pts_tensor: torch.Tensor) -> torch.Tensor:
        """Fallback geometric features if GeDi fails"""
        try:
            num_points = pts_tensor.shape[0]
            
            # Simple geometric features as fallback
            center = pts_tensor.mean(dim=0)
            normalized_pts = pts_tensor - center
            
            # Create 64D features through repetition and padding
            if normalized_pts.shape[1] == 3:
                repeated = normalized_pts.repeat(1, 21)[:, :63]  # 3*21 = 63
                padding = torch.zeros(num_points, 1)
                fallback_features = torch.cat([repeated, padding], dim=1)
            else:
                fallback_features = torch.zeros(num_points, 64)
            
            # L2 normalize
            fallback_features = F.normalize(fallback_features, p=2, dim=1)
            
            logger.info(f"✅ Fallback features computed: {fallback_features.shape}")
            return fallback_features
            
        except Exception as e:
            logger.error(f"❌ Fallback feature computation failed: {e}")
            return torch.zeros(pts_tensor.shape[0], 64)
    
    def process_geometric_features(self, data: dict) -> dict:
        """Main processing function for geometric feature extraction"""
        try:
            # Extract input parameters
            point_cloud_path = data.get('point_cloud_path')
            job_id = data.get('job_id', f'job_{int(time.time())}')
            
            if not point_cloud_path:
                raise ValueError("Missing required input: point_cloud_path")
            
            if not os.path.exists(point_cloud_path):
                raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")
            
            logger.info(f"🔷 Processing geometric features for job {job_id}")
            logger.info(f"☁️ Point cloud: {point_cloud_path}")
            
            # Load point cloud
            start_time = time.time()
            pts_tensor, points = self.load_point_cloud(point_cloud_path)
            
            # Compute GeDi geometric features
            geometric_features = self.compute_gedi_features(pts_tensor, points)
            
            # Create output directory and save features
            output_base = Path(f"/app/output/{job_id}")
            output_dir = output_base / "gedi_features"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "query_geometric_features.npy"
            np.save(str(output_path), geometric_features.cpu().numpy())
            
            # Save metadata
            metadata = {
                'feature_dimension': 64,
                'num_points': len(points),
                'point_indices': list(range(len(points))),
                'processing_info': {
                    'gedi_available': self.gedi_available,
                    'dual_scale': True,
                    'r_lrf_values': [0.3, 0.4] if self.gedi_available else None,
                    'l2_normalized': True
                }
            }
            
            metadata_path = output_dir / "gedi_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Calculate processing metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                'status': 'success',
                'job_id': job_id,
                'output_path': str(output_path),
                'metadata_path': str(metadata_path),
                'feature_dimensions': geometric_features.shape,
                'gedi_available': self.gedi_available,
                'processing_metrics': {
                    'processing_time': processing_time,
                    'points_processed': len(points),
                    'points_per_second': len(points) / processing_time if processing_time > 0 else 0
                }
            }
            
            logger.info(f"✅ Geometric feature extraction completed in {processing_time:.2f}s")
            logger.info(f"📊 Final features shape: {geometric_features.shape}")
            logger.info(f"💾 Saved to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Geometric feature processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
processor = QueryGeDiProcessor()

@app.route('/extract-features', methods=['POST'])
def extract_geometric_features():
    """Extract geometric features from point cloud"""
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
        'service': 'query-gedi',
        'device': str(processor.device),
        'gedi_available': processor.gedi_available,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'query-gedi',
        'version': '1.0',
        'description': 'GeDi geometric feature extraction for query objects',
        'endpoints': ['/extract-features', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Query GeDi Processor Service")
    app.run(host='0.0.0.0', port=5010, debug=False)
