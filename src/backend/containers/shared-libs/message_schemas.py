import time
from typing import Dict, Any, List, Optional
from kafka_handler import KafkaMessageHandler, TOPICS

class PipelineMessage:
    """Enhanced message format for complex multi-pipeline processing"""
    
    def __init__(self, object_id: int, pipeline_type: str, stage: str, 
                 data: Dict[Any, Any], status: str = "processing", 
                 error: Optional[str] = None):
        self.object_id = object_id
        self.pipeline_type = pipeline_type  # 'target', 'query', 'pose_estimation'
        self.stage = stage
        self.data = data
        self.status = status
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'object_id': self.object_id,
            'pipeline_type': self.pipeline_type,
            'stage': self.stage,
            'data': self.data,
            'status': self.status,
            'error': self.error,
            'timestamp': self.timestamp
        }

# Enhanced processing stages for complex pipeline
PIPELINE_STAGES = {
    # Target pipeline stages
    'TARGET_GEDI': 'target_gedi_processor',
    'TARGET_DINO': 'target_dino_processor',
    'TARGET_POINT_CLOUD': 'target_point_cloud_generator',
    'TARGET_FEATURE_FUSION': 'target_feature_fusion',
    
    # Query pipeline stages
    'QUERY_CNOS': 'query_cnos_renderer',
    'QUERY_DINO': 'query_dino_processor',
    'QUERY_FEATURE_FUSION': 'query_feature_fusion',
    'QUERY_GEDI': 'query_gedi_processor',
    'QUERY_POINT_CLOUD': 'query_point_cloud_generator',
    
    # Pose estimation pipeline stages
    'POSE_DINO': 'pose_estimation_dino',
    'POSE_EVAL_SYMMETRY': 'pose_estimation_eval_symmetry',
    'POSE_GET_SYMMETRY': 'pose_estimation_get_symmetry',
    'POSE_RANSAC': 'pose_estimation_ransac',
    'POSE_SEE_SYMMETRY': 'pose_estimation_see_symmetry',
    'POSE_SIDE_BY_SIDE': 'pose_estimation_side_by_side',
    'POSE_EXTRA_1': 'pose_estimation_extra_1',
    'POSE_EXTRA_2': 'pose_estimation_extra_2',
    
    # Final processing stages
    'RANSAC': 'ransac',
    'SYMMETRY_AWARE_REFINEMENT': 'symmetry_aware_refinement',
    'COMPLETE': 'complete'
}

class MultiPipelineController:
    """Enhanced controller for complex multi-pipeline processing"""
    
    def __init__(self, total_objects: int):
        self.total_objects = total_objects
        self.current_object = 1
        self.completed_objects = []
        self.pipeline_status = {}  # Track status of each pipeline for each object
        self.kafka_handler = KafkaMessageHandler()
        
        # Initialize pipeline tracking
        for obj_id in range(1, total_objects + 1):
            self.pipeline_status[obj_id] = {
                'target_pipeline': False,
                'query_pipeline': False,
                'pose_estimation_pipeline': False,
                'final_processing': False
            }
    
    def start_object_processing(self, object_id: int, custom_flow: List[str] = None):
        """Start processing for object with custom flow definition"""
        if custom_flow:
            # Use custom flow order as specified by user
            self.execute_custom_flow(object_id, custom_flow)
        else:
            # Default flow if no custom flow specified
            self.start_default_flow(object_id)
    
    def execute_custom_flow(self, object_id: int, flow_steps: List[str]):
        """Execute custom processing flow as specified by user"""
        message = PipelineMessage(
            object_id=object_id,
            pipeline_type='custom',
            stage=flow_steps[0],
            data={
                'action': 'start_custom_flow',
                'flow_sequence': flow_steps,
                'current_step': 0
            },
            status='ready'
        )
        
        # Send to appropriate topic based on first step
        topic = self.get_topic_for_stage(flow_steps[0])
        self.kafka_handler.send_message(topic, message.to_dict())
    
    def get_topic_for_stage(self, stage: str) -> str:
        """Get appropriate Kafka topic for processing stage"""
        stage_topic_mapping = {
            'target_gedi_processor': TOPICS['TARGET_GEDI_READY'],
            'target_dino_processor': TOPICS['TARGET_DINO_READY'],
            'target_point_cloud_generator': TOPICS['TARGET_POINT_CLOUD_READY'],
            'target_feature_fusion': TOPICS['TARGET_FEATURE_FUSION_READY'],
            'query_cnos_renderer': TOPICS['QUERY_CNOS_READY'],
            'query_dino_processor': TOPICS['QUERY_DINO_READY'],
            'query_feature_fusion': TOPICS['QUERY_FEATURE_FUSION_READY'],
            'query_gedi_processor': TOPICS['QUERY_GEDI_READY'],
            'query_point_cloud_generator': TOPICS['QUERY_POINT_CLOUD_READY'],
            'pose_estimation_dino': TOPICS['POSE_DINO_READY'],
            'pose_estimation_eval_symmetry': TOPICS['POSE_EVAL_SYMMETRY_READY'],
            'pose_estimation_get_symmetry': TOPICS['POSE_GET_SYMMETRY_READY'],
            'pose_estimation_ransac': TOPICS['POSE_RANSAC_READY'],
            'pose_estimation_see_symmetry': TOPICS['POSE_SEE_SYMMETRY_READY'],
            'pose_estimation_side_by_side': TOPICS['POSE_SIDE_BY_SIDE_READY'],
            'pose_estimation_extra_1': TOPICS['POSE_EXTRA_1_READY'],
            'pose_estimation_extra_2': TOPICS['POSE_EXTRA_2_READY'],
            'ransac': TOPICS['RANSAC_READY'],
            'symmetry_aware_refinement': TOPICS['SYMMETRY_AWARE_REFINEMENT_READY']
        }
        
        return stage_topic_mapping.get(stage, TOPICS['PIPELINE_STATUS'])
    
    def stage_completed(self, object_id: int, completed_stage: str, next_stage: str = None):
        """Handle completion of processing stage and trigger next stage"""
        if next_stage:
            # Continue with next stage in custom flow
            message = PipelineMessage(
                object_id=object_id,
                pipeline_type='custom',
                stage=next_stage,
                data={
                    'action': 'continue_flow',
                    'previous_stage': completed_stage
                },
                status='ready'
            )
            
            topic = self.get_topic_for_stage(next_stage)
            self.kafka_handler.send_message(topic, message.to_dict())
        else:
            # Object processing complete
            self.object_completed(object_id)
    
    def object_completed(self, object_id: int):
        """Mark object as completed and start next object"""
        self.completed_objects.append(object_id)
        
        if len(self.completed_objects) == self.total_objects:
            # All objects complete, trigger final overlay
            self.trigger_final_overlay()
        else:
            # Start next object (will use custom flow when specified)
            self.current_object += 1
            if self.current_object <= self.total_objects:
                # Wait for user to specify custom flow for next object
                pass
    
    def trigger_final_overlay(self):
        """Trigger final overlay combination"""
        message = {
            'total_objects': self.total_objects,
            'completed_objects': self.completed_objects,
            'action': 'create_final_overlay'
        }
        
        self.kafka_handler.send_message(TOPICS['OVERLAY_READY'], message)
