import sys
import os
import json
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d
from PIL import Image
from flask import Flask, request, jsonify
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QueryPointCloudGenerator:
    """Simplified point cloud generator for query CAD models"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        logger.info("🚀 Query Point Cloud Generator Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def load_mesh_with_texture(self, ply_path: str, texture_path: str = None) -> tuple:
        """Load mesh and texture with fallback options"""
        try:
            # Load mesh
            mesh = trimesh.load(ply_path, process=False)
            logger.info(f"✅ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Try to get texture from mesh first
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
                texture = np.asarray(mesh.visual.material.image)
                logger.info(f"✅ Loaded texture from PLY file: {texture.shape}")
                return texture, mesh
            
            # If no texture in mesh, try external texture file
            if texture_path and os.path.exists(texture_path):
                texture = np.asarray(Image.open(texture_path))
                logger.info(f"✅ Loaded external texture: {texture.shape}")
                return texture, mesh
            
            # If no texture available, create default texture
            logger.warning("⚠️ No texture found, creating default white texture")
            texture = np.ones((256, 256, 3), dtype=np.uint8) * 255
            return texture, mesh
            
        except Exception as e:
            logger.error(f"❌ Error loading mesh/texture: {e}")
            raise
    
    def sample_points_from_mesh(self, mesh, samples: int = 5000) -> tuple:
        """Sample points uniformly from mesh surface"""
        try:
            # Use trimesh uniform surface sampling
            points, face_indices = trimesh.sample.sample_surface_even(mesh, count=samples)
            
            logger.info(f"✅ Sampled {len(points)} points from mesh surface")
            return points, face_indices
            
        except Exception as e:
            logger.error(f"❌ Point sampling failed: {e}")
            raise
    
    def sample_colors_from_texture(self, mesh, points: np.ndarray, face_indices: np.ndarray, 
                                 texture: np.ndarray) -> np.ndarray:
        """Sample colors from texture using barycentric coordinates"""
        try:
            # Get triangles and calculate barycentric coordinates
            triangles = mesh.triangles[face_indices]
            bc = trimesh.triangles.points_to_barycentric(triangles, points)
            
            # Get UV coordinates
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
                sample_uvs = np.einsum('ij,ijk->ik', bc, face_uvs)
            else:
                # Create default UV coordinates if not available
                logger.warning("⚠️ No UV coordinates found, using default mapping")
                sample_uvs = np.random.rand(len(points), 2)
            
            # Sample colors from texture
            h, w = texture.shape[:2]
            colors = np.zeros((len(sample_uvs), 3), dtype=np.float32)
            
            for i, (u, v) in enumerate(sample_uvs):
                # Convert UV to pixel coordinates
                x = int(np.clip(u * (w - 1), 0, w - 1))
                y = int(np.clip((1.0 - v) * (h - 1), 0, h - 1))
                colors[i] = texture[y, x, :3] / 255.0
            
            logger.info("✅ Color sampling completed")
            return colors
            
        except Exception as e:
            logger.error(f"❌ Color sampling failed: {e}")
            # Return default gray colors
            return np.ones((len(points), 3), dtype=np.float32) * 0.7
    
    def create_point_cloud(self, points: np.ndarray, colors: np.ndarray, 
                          scale: float = 0.001) -> o3d.geometry.PointCloud:
        """Create Open3D point cloud with colors"""
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Apply scaling (from mm to meters)
            pcd.scale(scale, center=(0, 0, 0))
            
            # Remove statistical outliers
            pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            logger.info(f"✅ Created point cloud with {len(pcd.points)} points")
            return pcd, inlier_indices
            
        except Exception as e:
            logger.error(f"❌ Point cloud creation failed: {e}")
            raise
    
    def ensure_exact_point_count(self, points: np.ndarray, colors: np.ndarray, 
                                face_indices: np.ndarray, target_count: int = 5000) -> tuple:
        """Ensure exactly target_count points"""
        current_count = len(points)
        
        if current_count == target_count:
            return points, colors, face_indices
        
        logger.info(f"⚠️ Adjusting point count: {current_count} → {target_count}")
        
        if current_count > target_count:
            # Randomly select subset
            indices = np.random.choice(current_count, target_count, replace=False)
            return points[indices], colors[indices], face_indices[indices]
        else:
            # Duplicate points to reach target
            repeat_factor = target_count // current_count + 1
            points_repeated = np.tile(points, (repeat_factor, 1))[:target_count]
            colors_repeated = np.tile(colors, (repeat_factor, 1))[:target_count]
            face_indices_repeated = np.tile(face_indices, repeat_factor)[:target_count]
            return points_repeated, colors_repeated, face_indices_repeated
    
    def process_point_cloud_generation(self, data: dict) -> dict:
        """Main processing function for point cloud generation"""
        try:
            # Extract input parameters
            ply_path = data.get('ply_path')
            texture_path = data.get('texture_path')
            job_id = data.get('job_id', f'job_{int(time.time())}')
            samples = data.get('samples', 5000)
            
            if not ply_path:
                raise ValueError("Missing required input: ply_path")
            
            if not os.path.exists(ply_path):
                raise FileNotFoundError(f"PLY file not found: {ply_path}")
            
            # Create output directory
            output_base = Path(f"/app/output/{job_id}")
            output_dir = output_base / "point_cloud"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🚀 Starting point cloud generation for job {job_id}")
            logger.info(f"📁 PLY: {ply_path}")
            logger.info(f"📁 Texture: {texture_path}")
            logger.info(f"🎯 Target samples: {samples}")
            
            # Load mesh and texture
            start_time = time.time()
            texture, mesh = self.load_mesh_with_texture(ply_path, texture_path)
            
            # Sample points from mesh surface
            points, face_indices = self.sample_points_from_mesh(mesh, samples)
            
            # Sample colors from texture
            colors = self.sample_colors_from_texture(mesh, points, face_indices, texture)
            
            # Ensure exact point count
            points, colors, face_indices = self.ensure_exact_point_count(
                points, colors, face_indices, samples
            )
            
            # Create point cloud
            pcd, inlier_indices = self.create_point_cloud(points, colors)
            
            # Save point cloud
            output_path = output_dir / f"query_{len(pcd.points)}_scaled.ply"
            o3d.io.write_point_cloud(str(output_path), pcd)
            
            # Save index mapping for feature pairing
            if len(inlier_indices) > 0:
                point_indices = np.arange(len(points))[inlier_indices]
                final_face_indices = face_indices[inlier_indices] if len(inlier_indices) < len(face_indices) else face_indices
            else:
                point_indices = np.arange(len(pcd.points))
                final_face_indices = face_indices[:len(pcd.points)]
            
            index_mapping = {
                'point_indices': point_indices.tolist(),
                'face_indices': final_face_indices.tolist(),
                'total_points': len(pcd.points)
            }
            
            index_path = output_dir / "point_indices.json"
            with open(index_path, 'w') as f:
                json.dump(index_mapping, f, indent=2)
            
            # Performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                'status': 'success',
                'job_id': job_id,
                'point_cloud_path': str(output_path),
                'index_mapping_path': str(index_path),
                'num_points': len(pcd.points),
                'indexed': True,
                'performance_metrics': {
                    'processing_time': processing_time,
                    'points_per_second': len(pcd.points) / processing_time if processing_time > 0 else 0,
                    'original_samples': samples,
                    'final_points': len(pcd.points)
                }
            }
            
            logger.info(f"✅ Point cloud generation completed in {processing_time:.2f}s")
            logger.info(f"📊 Processing rate: {result['performance_metrics']['points_per_second']:.2f} points/second")
            logger.info(f"💾 Saved to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Point cloud generation failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
generator = QueryPointCloudGenerator()

@app.route('/generate-pointcloud', methods=['POST'])
def generate_point_cloud():
    """Generate point cloud from PLY mesh and texture"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = generator.process_point_cloud_generation(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'query-point-cloud',
        'trimesh_available': True,
        'open3d_available': True,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'query-point-cloud',
        'version': '1.0',
        'description': 'Point cloud generation from PLY mesh and texture',
        'endpoints': ['/generate-pointcloud', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Query Point Cloud Generator Service")
    app.run(host='0.0.0.0', port=5008, debug=False)
