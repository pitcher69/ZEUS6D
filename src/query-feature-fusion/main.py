import sys
import os
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize
from flask import Flask, request, jsonify
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QueryFeatureFusion:
    """Simplified feature fusion processor for query features"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        
        # Job tracking for coordinating DINOv2 and GeDi features
        self.pending_jobs = {}
        
        logger.info("🚀 Query Feature Fusion Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def load_features(self, dinov2_path: str, gedi_path: str) -> tuple:
        """Load DINOv2 and GeDi features from numpy files"""
        try:
            dinov2_features = np.load(dinov2_path)
            gedi_features = np.load(gedi_path)
            
            logger.info(f"✅ Features loaded:")
            logger.info(f"   📊 DINOv2: {dinov2_features.shape}")
            logger.info(f"   📊 GeDi: {gedi_features.shape}")
            
            return dinov2_features, gedi_features
            
        except Exception as e:
            logger.error(f"❌ Error loading features: {e}")
            raise
    
    def validate_feature_dimensions(self, dinov2_features: np.ndarray, gedi_features: np.ndarray) -> bool:
        """Validate feature dimensions for fusion"""
        try:
            expected_points = 5000
            expected_dim = 64
            
            if dinov2_features.shape != (expected_points, expected_dim):
                logger.error(f"❌ DINOv2 shape mismatch: {dinov2_features.shape}, expected: ({expected_points}, {expected_dim})")
                return False
            
            if gedi_features.shape != (expected_points, expected_dim):
                logger.error(f"❌ GeDi shape mismatch: {gedi_features.shape}, expected: ({expected_points}, {expected_dim})")
                return False
            
            logger.info(f"✅ Feature validation passed:")
            logger.info(f"   📊 DINOv2: {dinov2_features.shape}")
            logger.info(f"   📊 GeDi: {gedi_features.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature validation failed: {e}")
            return False
    
    def verify_l2_normalization(self, features: np.ndarray, feature_type: str) -> bool:
        """Verify L2 normalization of features"""
        try:
            norms = np.linalg.norm(features, axis=1)
            mean_norm = np.mean(norms)
            min_norm = np.min(norms)
            max_norm = np.max(norms)
            
            logger.info(f"📊 {feature_type} L2 norms - mean: {mean_norm:.6f}, min: {min_norm:.6f}, max: {max_norm:.6f}")
            
            # Check if features are properly L2 normalized (norms should be ~1)
            is_normalized = np.allclose(norms, 1.0, atol=1e-4)
            
            if is_normalized:
                logger.info(f"✅ {feature_type} features are properly L2 normalized")
                return True
            else:
                logger.warning(f"⚠️ {feature_type} features need L2 normalization")
                return False
            
        except Exception as e:
            logger.error(f"❌ L2 normalization check failed for {feature_type}: {e}")
            return False
    
    def fuse_features(self, dinov2_features: np.ndarray, gedi_features: np.ndarray) -> np.ndarray:
        """Fuse DINOv2 and GeDi features by concatenation"""
        try:
            # Validate dimensions
            if not self.validate_feature_dimensions(dinov2_features, gedi_features):
                raise ValueError("Feature dimension validation failed")
            
            # Verify L2 normalization
            dinov2_normalized = self.verify_l2_normalization(dinov2_features, "DINOv2")
            gedi_normalized = self.verify_l2_normalization(gedi_features, "GeDi")
            
            # Apply L2 normalization if needed
            if not dinov2_normalized:
                dinov2_features = normalize(dinov2_features, norm='l2', axis=1)
                logger.info("✅ Applied L2 normalization to DINOv2 features")
            
            if not gedi_normalized:
                gedi_features = normalize(gedi_features, norm='l2', axis=1)
                logger.info("✅ Applied L2 normalization to GeDi features")
            
            # Concatenate features: GeDi (64D) + DINOv2 (64D) = 128D
            logger.info("🔄 Concatenating features...")
            fused_features = np.concatenate([gedi_features, dinov2_features], axis=1)
            
            # Verify output dimensions
            expected_shape = (5000, 128)
            if fused_features.shape != expected_shape:
                raise ValueError(f"Output shape mismatch: {fused_features.shape}, expected: {expected_shape}")
            
            logger.info(f"✅ Feature fusion completed: {fused_features.shape}")
            return fused_features
            
        except Exception as e:
            logger.error(f"❌ Feature fusion failed: {e}")
            raise
    
    def process_feature_fusion(self, data: dict) -> dict:
        """Main processing function for feature fusion"""
        try:
            # Extract input parameters
            dinov2_path = data.get('dinov2_features_path')
            gedi_path = data.get('gedi_features_path')
            job_id = data.get('job_id', f'job_{int(time.time())}')
            
            if not all([dinov2_path, gedi_path]):
                raise ValueError("Missing required inputs: dinov2_features_path, gedi_features_path")
            
            if not os.path.exists(dinov2_path):
                raise FileNotFoundError(f"DINOv2 features file not found: {dinov2_path}")
            
            if not os.path.exists(gedi_path):
                raise FileNotFoundError(f"GeDi features file not found: {gedi_path}")
            
            logger.info(f"🔷 Processing feature fusion for job {job_id}")
            logger.info(f"📊 DINOv2 features: {dinov2_path}")
            logger.info(f"📊 GeDi features: {gedi_path}")
            
            # Load features
            start_time = time.time()
            dinov2_features, gedi_features = self.load_features(dinov2_path, gedi_path)
            
            # Fuse features
            fused_features = self.fuse_features(dinov2_features, gedi_features)
            
            # Create output directory and save fused features
            output_base = Path(f"/app/output/{job_id}")
            output_dir = output_base / "fused_features"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "query_fused_features_128d.npy"
            np.save(str(output_path), fused_features)
            
            # Save fusion metadata
            fusion_metadata = {
                'feature_dimension': 128,
                'num_points': 5000,
                'fusion_components': {
                    'gedi_geometric': 64,
                    'dinov2_visual': 64,
                    'total_dimension': 128
                },
                'normalization': {
                    'dinov2_l2_normalized': True,
                    'gedi_l2_normalized': True,
                    'fusion_method': 'concatenation'
                },
                'input_files': {
                    'dinov2_path': dinov2_path,
                    'gedi_path': gedi_path
                }
            }
            
            metadata_path = output_dir / "fusion_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(fusion_metadata, f, indent=2)
            
            # Calculate processing metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                'status': 'success',
                'job_id': job_id,
                'output_path': str(output_path),
                'metadata_path': str(metadata_path),
                'feature_dimensions': fused_features.shape,
                'fusion_components': fusion_metadata['fusion_components'],
                'processing_metrics': {
                    'processing_time': processing_time,
                    'points_processed': 5000,
                    'points_per_second': 5000 / processing_time if processing_time > 0 else 0
                }
            }
            
            logger.info(f"✅ Feature fusion completed in {processing_time:.2f}s")
            logger.info(f"📊 Final features shape: {fused_features.shape}")
            logger.info(f"💾 Saved to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Feature fusion processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
fusion_processor = QueryFeatureFusion()

@app.route('/fuse-features', methods=['POST'])
def fuse_features():
    """Fuse DINOv2 and GeDi features"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = fusion_processor.process_feature_fusion(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'query-feature-fusion',
        'fusion_ready': True,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'query-feature-fusion',
        'version': '1.0',
        'description': 'Multi-modal feature fusion for query objects',
        'endpoints': ['/fuse-features', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Query Feature Fusion Service")
    app.run(host='0.0.0.0', port=5011, debug=False)
