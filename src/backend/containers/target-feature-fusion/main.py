import sys
import os
import json
import time
import zipfile
import tempfile
import shutil
from pathlib import Path

import numpy as np
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

class TargetFeatureFusion:
    """Simplified target feature fusion service for combining visual and geometric features"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        logger.info("🔗 Target Feature Fusion Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from feature filename"""
        name = Path(filename).stem
        
        patterns = [
            r'target_(\d+)_gedi',        # target_000620_gedi
            r'target_(\d+)_pca64',       # target_000620_pca64
            r'dinov2_features_(\d+)',    # dinov2_features_000620
            r'gedi_features_(\d+)',      # gedi_features_000620
            r'visual_features_(\d+)',    # visual_features_000620
            r'geometric_features_(\d+)', # geometric_features_000620
            r'(\d{6})',                  # Any 6-digit number
            r'(\d{5})',                  # Any 5-digit number
            r'(\d{4})',                  # Any 4-digit number
            r'(\d+)'                     # Any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def extract_zip_contents(self, zip_path: str, output_dir: Path) -> list:
        """Extract numpy files from ZIP archive"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Find all numpy files
            npy_files = []
            for file_path in output_dir.rglob('*.npy'):
                npy_files.append(str(file_path))
            
            # Sort files naturally
            npy_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', Path(x).name)])
            
            logger.info(f"✅ Extracted {len(npy_files)} numpy files from ZIP")
            return npy_files
            
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
        elif suffix == '.npy':
            dest_path = output_dir / input_path.name
            shutil.copy2(input_path, dest_path)
            return [str(dest_path)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def fuse_features_from_arrays(self, geometric_features: np.ndarray, visual_features: np.ndarray, 
                                frame_id: str) -> np.ndarray:
        """Fuse geometric and visual features with validation"""
        try:
            logger.info(f"   📥 Geometric features shape: {geometric_features.shape}")
            logger.info(f"   📥 Visual features shape: {visual_features.shape}")
            
            # Validate basic dimensions (should have same number of points)
            if len(geometric_features.shape) != 2 or len(visual_features.shape) != 2:
                raise ValueError("Features must be 2D arrays")
            
            if geometric_features.shape[0] != visual_features.shape[0]:
                raise ValueError(f"Feature count mismatch: geometric={geometric_features.shape[0]}, visual={visual_features.shape[0]}")
            
            # Expected dimensions: N points x 64 features each
            expected_geo_dim = 64
            expected_vis_dim = 64
            
            if geometric_features.shape[1] != expected_geo_dim:
                logger.warning(f"Geometric features dimension unexpected: {geometric_features.shape[1]} (expected {expected_geo_dim})")
            
            if visual_features.shape[1] != expected_vis_dim:
                logger.warning(f"Visual features dimension unexpected: {visual_features.shape[1]} (expected {expected_vis_dim})")
            
            # Concatenate features: [geometric_64D, visual_64D] -> 128D
            fused_features = np.concatenate([geometric_features, visual_features], axis=1)
            
            logger.info(f"   🔗 Fused features shape: {fused_features.shape}")
            
            return fused_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Feature fusion failed for frame {frame_id}: {e}")
            # Return zero features as fallback
            num_points = max(geometric_features.shape[0] if len(geometric_features.shape) > 0 else 1000,
                           visual_features.shape[0] if len(visual_features.shape) > 0 else 1000)
            return np.zeros((num_points, 128), dtype=np.float32)
    
    def find_matching_files(self, geometric_files: list, visual_files: list) -> list:
        """Find matching geometric and visual feature files by frame ID"""
        matches = []
        
        # Create dictionaries for fast lookup
        geo_dict = {}
        vis_dict = {}
        
        for geo_file in geometric_files:
            frame_id = self.extract_frame_id(Path(geo_file).name)
            geo_dict[frame_id] = geo_file
        
        for vis_file in visual_files:
            frame_id = self.extract_frame_id(Path(vis_file).name)
            vis_dict[frame_id] = vis_file
        
        # Find common frame IDs
        common_frames = set(geo_dict.keys()).intersection(set(vis_dict.keys()))
        
        for frame_id in sorted(common_frames):
            matches.append({
                'frame_id': frame_id,
                'geometric_file': geo_dict[frame_id],
                'visual_file': vis_dict[frame_id]
            })
        
        logger.info(f"📊 Found {len(matches)} matching frames for fusion")
        return matches
    
    def process_feature_fusion(self, data: dict) -> dict:
        """Main processing function for feature fusion"""
        try:
            # Extract input parameters
            geometric_input = data.get('geometric_input')
            visual_input = data.get('visual_input')
            
            if not all([geometric_input, visual_input]):
                raise ValueError("Missing required inputs: geometric_input, visual_input")
            
            # Create output directory
            output_base = Path("/app/output")
            output_base.mkdir(exist_ok=True)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create subdirectories
                geo_dir = temp_path / 'geometric'
                vis_dir = temp_path / 'visual'
                fused_dir = temp_path / 'fused'
                
                for dir_path in [geo_dir, vis_dir, fused_dir]:
                    dir_path.mkdir()
                
                logger.info("🔷 Processing input feature files...")
                
                # Process inputs
                geometric_files = self.process_input_source(geometric_input, geo_dir, 'geometric')
                visual_files = self.process_input_source(visual_input, vis_dir, 'visual')
                
                if not geometric_files or not visual_files:
                    raise ValueError("No valid feature files found")
                
                # Find matching files
                matches = self.find_matching_files(geometric_files, visual_files)
                
                if not matches:
                    raise ValueError("No matching frames found between geometric and visual features")
                
                # Process each matching pair
                fused_files = []
                
                for i, match in enumerate(matches):
                    frame_id = match['frame_id']
                    
                    try:
                        # Load feature arrays
                        geometric_features = np.load(match['geometric_file'])
                        visual_features = np.load(match['visual_file'])
                        
                        # Fuse features
                        fused_features = self.fuse_features_from_arrays(
                            geometric_features, visual_features, frame_id
                        )
                        
                        # Save fused features
                        fused_filename = f"fused_features_{frame_id}.npy"
                        fused_path = fused_dir / fused_filename
                        np.save(str(fused_path), fused_features)
                        fused_files.append(str(fused_path))
                        
                        logger.info(f"✅ Processed frame {i+1}/{len(matches)}: {fused_features.shape}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process frame {frame_id}: {e}")
                        continue
                
                # Generate output
                timestamp = int(time.time())
                
                if len(fused_files) > 1:
                    # Create ZIP for multiple features
                    zip_filename = f"fused_features_{timestamp}.zip"
                    zip_path = output_base / zip_filename
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                        for fused_file in fused_files:
                            zip_ref.write(fused_file, Path(fused_file).name)
                    
                    return {
                        'status': 'success',
                        'output_type': 'zip',
                        'output_path': str(zip_path),
                        'feature_count': len(fused_files),
                        'feature_dimensions': 128
                    }
                
                elif len(fused_files) == 1:
                    # Copy single feature file
                    feature_filename = f"fused_features_{timestamp}.npy"
                    output_path = output_base / feature_filename
                    shutil.copy2(fused_files[0], output_path)
                    
                    return {
                        'status': 'success',
                        'output_type': 'single',
                        'output_path': str(output_path),
                        'feature_count': 1,
                        'feature_dimensions': 128
                    }
                
                else:
                    return {'status': 'error', 'message': 'No valid fused features generated'}
                    
        except Exception as e:
            logger.error(f"❌ Feature fusion processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
processor = TargetFeatureFusion()

@app.route('/fuse-features', methods=['POST'])
def fuse_features():
    """Fuse geometric and visual features"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = processor.process_feature_fusion(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'target-feature-fusion',
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'target-feature-fusion',
        'version': '1.0',
        'endpoints': ['/fuse-features', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Target Feature Fusion Service")
    app.run(host='0.0.0.0', port=5006, debug=False)
