import sys
import os
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
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

class QueryCNOSRenderer:
    """Simplified CNOS renderer for query object rendering"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        self.setup_rendering_environment()
        logger.info("🚀 Query CNOS Renderer Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def setup_rendering_environment(self):
        """Setup rendering environment"""
        # Ensure CNOS is properly configured
        self.cnos_path = "/app/cnos"
        self.poses_file = "/app/cnos/src/poses/predefined_poses/obj_poses_level0.npy"
        
        # Check if CNOS is available
        if not os.path.exists(self.cnos_path):
            logger.error("❌ CNOS repository not found")
            raise FileNotFoundError("CNOS repository not found")
        
        logger.info("✅ CNOS rendering environment configured")
    
    def generate_camera_intrinsics(self) -> dict:
        """Generate camera intrinsics for CNOS rendering"""
        return {
            "fx": 577.5,
            "fy": 577.5,
            "cx": 319.5,
            "cy": 239.5,
            "width": 640,
            "height": 480,
            "camera_matrix": [
                [577.5, 0, 319.5],
                [0, 577.5, 239.5],
                [0, 0, 1]
            ],
            "distortion_coefficients": [0, 0, 0, 0, 0]
        }
    
    def render_single_view(self, ply_path: str, pose_idx: int, output_dir: str, 
                          texture_path: str = None) -> str:
        """Render a single view using CNOS"""
        try:
            output_path = f"{output_dir}/view_{pose_idx:03d}.png"
            
            # CNOS rendering command
            cmd = [
                "python", f"{self.cnos_path}/src/poses/generate_views.py",
                ply_path,
                self.poses_file,
                output_dir,
                str(pose_idx),
                "True",  # Save image
                "1",     # Single view
                "0.3"    # Distance factor
            ]
            
            # Add texture if provided
            if texture_path and os.path.exists(texture_path):
                cmd.extend(["--texture", texture_path])
            
            # Set up environment for headless rendering
            env = os.environ.copy()
            env['DISPLAY'] = ':99'
            env['PYOPENGL_PLATFORM'] = 'egl'
            
            # Execute rendering
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.cnos_path,
                env=env
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"✅ Rendered view {pose_idx}")
                return output_path
            else:
                logger.error(f"❌ Rendering failed for view {pose_idx}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Single view rendering failed: {e}")
            return None
    
    def render_42_views(self, ply_path: str, output_dir: str, texture_path: str = None) -> list:
        """Render 42 views of the object"""
        logger.info("🔷 Starting 42-view rendering")
        
        rendered_images = []
        
        # Render each of the 42 predefined poses
        for pose_idx in range(42):
            try:
                image_path = self.render_single_view(ply_path, pose_idx, output_dir, texture_path)
                if image_path:
                    rendered_images.append(image_path)
                else:
                    logger.warning(f"⚠️ Failed to render view {pose_idx}")
            except Exception as e:
                logger.error(f"❌ Error rendering view {pose_idx}: {e}")
                continue
        
        logger.info(f"✅ Rendered {len(rendered_images)}/42 views successfully")
        return rendered_images
    
    def create_gif_from_images(self, image_paths: list, gif_path: str, duration: int = 100) -> str:
        """Create an optimized GIF from rendered images"""
        try:
            if not image_paths:
                raise ValueError("No images provided for GIF creation")
            
            # Load and process images
            frames = []
            for img_path in sorted(image_paths):
                try:
                    img = Image.open(img_path).convert("RGBA")
                    # Create white background for transparency
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                    frames.append(background)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process image {img_path}: {e}")
                    continue
            
            if not frames:
                raise ValueError("No valid frames for GIF creation")
            
            # Create optimized GIF
            frames[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            logger.info(f"🎬 GIF created successfully: {gif_path}")
            return gif_path
            
        except Exception as e:
            logger.error(f"❌ GIF creation failed: {e}")
            return None
    
    def process_object_rendering(self, data: dict) -> dict:
        """Main processing function for object rendering"""
        try:
            # Extract input parameters
            ply_path = data.get('ply_path')
            texture_path = data.get('texture_path')
            job_id = data.get('job_id', f'job_{int(time.time())}')
            
            if not ply_path:
                raise ValueError("Missing required input: ply_path")
            
            if not os.path.exists(ply_path):
                raise FileNotFoundError(f"PLY file not found: {ply_path}")
            
            # Create output directories
            output_base = Path(f"/app/output/{job_id}")
            renders_dir = output_base / "renders"
            gifs_dir = output_base / "gifs"
            
            for dir_path in [renders_dir, gifs_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🔷 Processing object rendering for job {job_id}")
            logger.info(f"📁 PLY: {ply_path}")
            logger.info(f"📁 Texture: {texture_path}")
            
            # Render 42 views
            start_time = time.time()
            rendered_images = self.render_42_views(str(ply_path), str(renders_dir), texture_path)
            
            if not rendered_images:
                raise ValueError("No images were successfully rendered")
            
            # Generate camera intrinsics
            camera_intrinsics = self.generate_camera_intrinsics()
            intrinsics_path = output_base / "camera_intrinsics.json"
            
            with open(intrinsics_path, 'w') as f:
                json.dump(camera_intrinsics, f, indent=2)
            
            # Create GIF for frontend display
            gif_path = gifs_dir / "rendered_views.gif"
            created_gif = self.create_gif_from_images(rendered_images, str(gif_path))
            
            # Calculate performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                'status': 'success',
                'job_id': job_id,
                'rendered_images': rendered_images,
                'camera_intrinsics': camera_intrinsics,
                'intrinsics_path': str(intrinsics_path),
                'gif_path': created_gif,
                'num_views': len(rendered_images),
                'performance_metrics': {
                    'processing_time': processing_time,
                    'fps': len(rendered_images) / processing_time if processing_time > 0 else 0,
                    'views_rendered': len(rendered_images),
                    'success_rate': len(rendered_images) / 42 * 100
                }
            }
            
            logger.info(f"✅ Object rendering completed in {processing_time:.2f}s")
            logger.info(f"📊 Success rate: {result['performance_metrics']['success_rate']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Object rendering processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
renderer = QueryCNOSRenderer()

@app.route('/render-object', methods=['POST'])
def render_object():
    """Render 42 views of an object from PLY file"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = renderer.process_object_rendering(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'query-cnos',
        'cnos_available': os.path.exists(renderer.cnos_path),
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'query-cnos',
        'version': '1.0',
        'description': 'CNOS 3D object rendering service',
        'endpoints': ['/render-object', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Query CNOS Renderer Service")
    app.run(host='0.0.0.0', port=5007, debug=False)
