import sys
import os
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PoseOverlapService:
    """Single instance service that collects all pose estimation results"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        logger.info("🔄 Pose Overlap Service Ready - Waiting for pose estimation results")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def wait_for_all_pose_files(self, object_count, timeout=600, check_interval=5):
        """Wait for all pose estimation JSON files to be ready"""
        expected_files = []
        
        for obj_id in range(object_count):
            json_path = f"/shared/outputs/pose_estimation_{obj_id}_results.json"
            expected_files.append(json_path)
        
        logger.info(f"⏳ Waiting for {object_count} pose estimation files...")
        
        start_time = time.time()
        while True:
            # Check if all files exist
            all_exist = all(os.path.exists(f) for f in expected_files)
            
            if all_exist:
                logger.info("✅ All pose estimation files found!")
                return expected_files
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for pose estimation files after {timeout}s")
            
            # Wait before next check
            time.sleep(check_interval)
    
    def load_all_poses(self, pose_files):
        """Load all pose estimation JSON files"""
        all_poses = []
        
        for i, file_path in enumerate(pose_files):
            try:
                with open(file_path, 'r') as f:
                    pose_data = json.load(f)
                    pose_data['object_id'] = i
                    all_poses.append(pose_data)
                    logger.info(f"✅ Loaded pose data for object {i}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
                raise
        
        return all_poses
    
    def analyze_pose_overlaps(self, all_poses):
        """Analyze overlaps between all poses - PLACEHOLDER for your friend's code"""
        logger.info(f"🔄 Analyzing pose overlaps for {len(all_poses)} objects...")
        
        # PLACEHOLDER: This is where your friend's overlap analysis code will go
        overlap_results = {
            'total_objects': len(all_poses),
            'overlap_detected': False,
            'conflicts': [],
            'final_poses': all_poses,
            'analysis_complete': True,
            'message': 'Pose overlap analysis placeholder - awaiting implementation'
        }
        
        logger.info("✅ Pose overlap analysis completed")
        return overlap_results
    
    def process_overlap_analysis(self, data):
        """Main processing function"""
        try:
            object_count = data.get('object_count')
            if not object_count:
                raise ValueError("Missing object_count parameter")
            
            job_id = data.get('job_id', f'overlap_{int(time.time())}')
            logger.info(f"🎯 Starting pose overlap analysis for {object_count} objects (Job: {job_id})")
            
            # Wait for all pose estimation files
            pose_files = self.wait_for_all_pose_files(object_count)
            
            # Load all pose data
            all_poses = self.load_all_poses(pose_files)
            
            # Analyze overlaps (placeholder for your friend's code)
            overlap_results = self.analyze_pose_overlaps(all_poses)
            
            # Save results
            output_dir = f"/shared/outputs/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            result_file = f"{output_dir}/pose_overlap_final_results.json"
            with open(result_file, 'w') as f:
                json.dump(overlap_results, f, indent=2)
            
            logger.info(f"💾 Final results saved to: {result_file}")
            
            return {
                'status': 'success',
                'job_id': job_id,
                'objects_processed': object_count,
                'output_file': result_file,
                'overlap_results': overlap_results
            }
            
        except Exception as e:
            logger.error(f"❌ Pose overlap analysis failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
overlap_service = PoseOverlapService()

@app.route('/analyze-overlaps', methods=['POST'])
def analyze_overlaps():
    """Analyze pose overlaps for multiple objects"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = overlap_service.process_overlap_analysis(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'pose-overlap',
        'ready_for_analysis': True,
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'pose-overlap',
        'version': '1.0',
        'description': 'Single instance pose overlap analysis service',
        'endpoints': ['/analyze-overlaps', '/health'],
        'note': 'Collects all pose estimation results and analyzes overlaps once'
    })

if __name__ == '__main__':
    logger.info("🔄 Starting Pose Overlap Analysis Service")
    app.run(host='0.0.0.0', port=5013, debug=False)
