import sys
import os
import json
import time
import zipfile
import shutil
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import re
from flask import Flask, request, jsonify
import logging

sys.path.append('/app/shared-libs')
from kafka_handler import KafkaMessageHandler, TOPICS
from message_schemas import PipelineMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TargetPointCloudGenerator:
    """Simplified point cloud generator with dynamic camera intrinsics"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        logger.info("🔷 Point Cloud Generator Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka communication"""
        try:
            self.kafka_handler = KafkaMessageHandler()
            logger.info("✅ Kafka connected")
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            self.kafka_handler = None
    
    def load_camera_intrinsics_dynamic(self, intrinsics_path: str, frame_id: Optional[str] = None) -> Dict[str, float]:
        """Dynamically load camera intrinsics from JSON file with improved error handling"""
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = json.load(f)
            
            # Try multiple possible keys for frame-specific intrinsics
            frame_intrinsics = None
            if frame_id:
                possible_keys = [
                    frame_id,
                    f"frame_{frame_id}",
                    f"{frame_id}.jpg",
                    f"{frame_id}.png",
                    "default"
                ]
                
                for key in possible_keys:
                    if key in intrinsics_data:
                        frame_intrinsics = intrinsics_data[key]
                        break
            
            # Fallback to first available intrinsics
            if frame_intrinsics is None:
                if isinstance(intrinsics_data, dict) and intrinsics_data:
                    frame_intrinsics = list(intrinsics_data.values())[0]
                else:
                    frame_intrinsics = intrinsics_data
            
            # Handle different intrinsics formats with better validation
            if isinstance(frame_intrinsics, dict):
                return {
                    'fx': float(frame_intrinsics.get('fx', 525.0)),
                    'fy': float(frame_intrinsics.get('fy', 525.0)),
                    'cx': float(frame_intrinsics.get('cx', 320.0)),
                    'cy': float(frame_intrinsics.get('cy', 240.0))
                }
            elif isinstance(frame_intrinsics, list):
                if len(frame_intrinsics) >= 9:  # 3x3 matrix format
                    return {
                        'fx': float(frame_intrinsics[0]),
                        'fy': float(frame_intrinsics[4]),
                        'cx': float(frame_intrinsics[2]),
                        'cy': float(frame_intrinsics[5])
                    }
                elif len(frame_intrinsics) >= 4:  # fx, fy, cx, cy format
                    return {
                        'fx': float(frame_intrinsics[0]),
                        'fy': float(frame_intrinsics[1]),
                        'cx': float(frame_intrinsics[2]),
                        'cy': float(frame_intrinsics[3])
                    }
            
            # Default fallback
            logger.warning(f"Using default intrinsics for frame {frame_id}")
            return {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
            
        except Exception as e:
            logger.error(f"❌ Failed to load intrinsics: {e}")
            return {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from filename with improved pattern matching"""
        name = Path(filename).stem
        
        # Try various numeric patterns
        patterns = [
            r'frame_(\d+)',      # frame_000001
            r'(\d{6})',          # 000001
            r'(\d{5})',          # 00001
            r'(\d{4})',          # 0001
            r'(\d{3})',          # 001
            r'(\d+)'             # any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def extract_video_frames(self, video_path: str, output_dir: Path, fps: Optional[float] = None) -> List[str]:
        """Extract frames from video with optional FPS control"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame extraction interval
        if fps and fps < video_fps:
            frame_interval = int(video_fps / fps)
        else:
            frame_interval = 1
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"📹 Extracting frames from video (FPS: {video_fps}, Target FPS: {fps or 'all'})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{extracted_count:06d}.png"
                frame_path = output_dir / frame_filename
                
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    frames.append(str(frame_path))
                    extracted_count += 1
                else:
                    logger.warning(f"Failed to save frame {extracted_count}")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"✅ Extracted {len(frames)} frames from {total_frames} total frames")
        return frames
    
    def extract_zip_contents(self, zip_path: str, output_dir: Path) -> List[str]:
        """Extract images from ZIP file with improved sorting"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Get sorted list of image files
            image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
            image_files = []
            
            for file_path in output_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(str(file_path))
            
            # Sort files naturally (handling numeric sequences)
            image_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', Path(x).name)])
            
            logger.info(f"✅ Extracted {len(image_files)} images from ZIP")
            return image_files
            
        except Exception as e:
            logger.error(f"❌ Failed to extract ZIP file {zip_path}: {e}")
            return []
    
    def process_input_source(self, input_path: str, output_dir: Path, input_type: str) -> List[str]:
        """Process input source (single image, video, or ZIP) with improved handling"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        suffix = input_path.suffix.lower()
        
        # Handle ZIP files
        if suffix == '.zip':
            return self.extract_zip_contents(str(input_path), output_dir)
        
        # Handle video files
        elif suffix in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}:
            return self.extract_video_frames(str(input_path), output_dir, fps=1.0)  # 1 FPS extraction
        
        # Handle single image files
        elif suffix in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}:
            dest_path = output_dir / input_path.name
            shutil.copy2(input_path, dest_path)
            return [str(dest_path)]
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def generate_point_cloud_from_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                                     mask: np.ndarray, intrinsics: Dict[str, float], 
                                     frame_id: str) -> o3d.geometry.PointCloud:
        """Generate point cloud from RGB-D data with mask - fixed version"""
        try:
            # Ensure all images have the same dimensions
            h, w = depth_image.shape[:2]
            
            if rgb_image.shape[:2] != (h, w):
                rgb_image = cv2.resize(rgb_image, (w, h))
            
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Apply mask with proper thresholding
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            mask_binary = (mask > 127).astype(np.uint8)
            
            # Apply mask to depth and RGB
            masked_depth = depth_image.copy().astype(np.float32)
            masked_depth[mask_binary == 0] = 0
            
            masked_rgb = rgb_image.copy()
            masked_rgb[mask_binary == 0] = [0, 0, 0]
            
            # Create Open3D images with proper data types
            color_o3d = o3d.geometry.Image(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGB).astype(np.uint8))
            depth_o3d = o3d.geometry.Image(masked_depth.astype(np.float32))
            
            # Create RGBD image with appropriate parameters
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=color_o3d,
                depth=depth_o3d,
                depth_scale=1000.0 if masked_depth.max() > 100 else 1.0,
                depth_trunc=10.0,
                convert_rgb_to_intensity=False
            )
            
            # Create camera intrinsic matrix
            intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
                width=w,
                height=h,
                fx=intrinsics['fx'],
                fy=intrinsics['fy'],
                cx=intrinsics['cx'],
                cy=intrinsics['cy']
            )
            
            # Generate point cloud
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_matrix)
            
            # Filter out invalid points
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)
            
            if len(points) == 0:
                logger.warning(f"No valid points generated for frame {frame_id}")
                return o3d.geometry.PointCloud()
            
            # Remove points at origin (often invalid)
            valid_mask = np.linalg.norm(points, axis=1) > 0.01
            points = points[valid_mask]
            colors = colors[valid_mask]
            
            # Statistical outlier removal if enough points
            if len(points) > 100:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points)
                temp_pcd.colors = o3d.utility.Vector3dVector(colors)
                
                temp_pcd, _ = temp_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                points = np.asarray(temp_pcd.points)
                colors = np.asarray(temp_pcd.colors)
            
            # Downsample to target number of points
            target_points = 1000
            if len(points) > target_points:
                indices = np.random.choice(len(points), target_points, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Create final point cloud
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(points)
            final_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            logger.info(f"✅ Generated point cloud for frame {frame_id}: {len(points)} points")
            return final_pcd
            
        except Exception as e:
            logger.error(f"❌ Point cloud generation failed for frame {frame_id}: {e}")
            return o3d.geometry.PointCloud()
    
    def process_target_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing function with improved error handling and output management"""
        try:
            # Extract input parameters
            rgb_input = data.get('rgb_input')
            depth_input = data.get('depth_input') 
            mask_input = data.get('mask_input')
            intrinsics_path = data.get('intrinsics_path')
            
            if not all([rgb_input, depth_input, mask_input, intrinsics_path]):
                raise ValueError("Missing required input parameters: rgb_input, depth_input, mask_input, intrinsics_path")
            
            # Create output directory
            output_base = Path("/app/output")
            output_base.mkdir(exist_ok=True)
            
            # Use temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create subdirectories
                rgb_dir = temp_path / 'rgb'
                depth_dir = temp_path / 'depth'  
                mask_dir = temp_path / 'mask'
                pc_dir = temp_path / 'point_clouds'
                
                for dir_path in [rgb_dir, depth_dir, mask_dir, pc_dir]:
                    dir_path.mkdir()
                
                logger.info("🔷 Processing input files...")
                
                # Process all input sources
                rgb_files = self.process_input_source(rgb_input, rgb_dir, 'rgb')
                depth_files = self.process_input_source(depth_input, depth_dir, 'depth')
                mask_files = self.process_input_source(mask_input, mask_dir, 'mask')
                
                # Validate file counts match
                min_files = min(len(rgb_files), len(depth_files), len(mask_files))
                if min_files == 0:
                    raise ValueError("No valid files found in one or more input sources")
                
                if len(rgb_files) != len(depth_files) or len(rgb_files) != len(mask_files):
                    logger.warning(f"File count mismatch: RGB({len(rgb_files)}), Depth({len(depth_files)}), Mask({len(mask_files)}). Using first {min_files} files.")
                    rgb_files = rgb_files[:min_files]
                    depth_files = depth_files[:min_files]
                    mask_files = mask_files[:min_files]
                
                # Load camera intrinsics
                intrinsics = self.load_camera_intrinsics_dynamic(intrinsics_path)
                
                # Process each frame sequentially
                point_cloud_files = []
                
                for i, (rgb_file, depth_file, mask_file) in enumerate(zip(rgb_files, depth_files, mask_files)):
                    frame_id = self.extract_frame_id(Path(rgb_file).name)
                    
                    try:
                        # Load images with proper error handling
                        rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
                        depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
                        mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        
                        if rgb_image is None:
                            raise ValueError(f"Cannot load RGB image: {rgb_file}")
                        if depth_image is None:
                            raise ValueError(f"Cannot load depth image: {depth_file}")
                        if mask_image is None:
                            raise ValueError(f"Cannot load mask image: {mask_file}")
                        
                        # Generate point cloud
                        point_cloud = self.generate_point_cloud_from_rgbd(
                            rgb_image, depth_image, mask_image, intrinsics, frame_id
                        )
                        
                        # Save point cloud if valid
                        if len(point_cloud.points) > 0:
                            pc_filename = f"pointcloud_{frame_id}.ply"
                            pc_path = pc_dir / pc_filename
                            
                            success = o3d.io.write_point_cloud(str(pc_path), point_cloud)
                            if success:
                                point_cloud_files.append(str(pc_path))
                                logger.info(f"✅ Processed frame {i+1}/{min_files}: {len(point_cloud.points)} points")
                            else:
                                logger.error(f"Failed to save point cloud for frame {frame_id}")
                        else:
                            logger.warning(f"Empty point cloud generated for frame {frame_id}")
                    
                    except Exception as e:
                        logger.error(f"❌ Failed to process frame {frame_id}: {e}")
                        continue
                
                # Generate output based on number of point clouds
                timestamp = int(time.time())
                
                if len(point_cloud_files) > 1:
                    # Create ZIP file for multiple point clouds
                    zip_filename = f"point_clouds_{timestamp}.zip"
                    zip_path = output_base / zip_filename
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                        for pc_file in point_cloud_files:
                            zip_ref.write(pc_file, Path(pc_file).name)
                    
                    return {
                        'status': 'success',
                        'output_type': 'zip',
                        'output_path': str(zip_path),
                        'point_cloud_count': len(point_cloud_files),
                        'processing_time': time.time() - timestamp
                    }
                
                elif len(point_cloud_files) == 1:
                    # Copy single point cloud to output directory
                    ply_filename = f"pointcloud_{timestamp}.ply"
                    output_path = output_base / ply_filename
                    shutil.copy2(point_cloud_files[0], output_path)
                    
                    return {
                        'status': 'success',
                        'output_type': 'single',
                        'output_path': str(output_path),
                        'point_cloud_count': 1,
                        'processing_time': time.time() - timestamp
                    }
                
                else:
                    return {
                        'status': 'error',
                        'message': 'No valid point clouds generated',
                        'processed_frames': min_files
                    }
                    
        except Exception as e:
            logger.error(f"❌ Target data processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global instance
generator = TargetPointCloudGenerator()

@app.route('/generate', methods=['POST'])
def generate_point_clouds():
    """Generate point clouds from RGB-D input data"""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        result = generator.process_target_data(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'target-point-cloud',
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'target-point-cloud',
        'version': '1.0',
        'endpoints': ['/generate', '/health']
    })

if __name__ == '__main__':
    logger.info("🔷 Starting Target Point Cloud Generator Service")
    app.run(host='0.0.0.0', port=5003, debug=False)
