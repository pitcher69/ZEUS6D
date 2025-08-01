from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
import json
import logging
import time

class KafkaMessageHandler:
    def __init__(self, bootstrap_servers=['kafka:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
        self.logger = logging.getLogger(__name__)
        
        # Wait for Kafka to be ready
        self._wait_for_kafka()
    
    def _wait_for_kafka(self, max_retries=30):
        """Wait for Kafka to be available"""
        for attempt in range(max_retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
                self.logger.info("Successfully connected to Kafka")
                break
            except NoBrokersAvailable:
                self.logger.warning(f"Kafka not ready, attempt {attempt + 1}/{max_retries}")
                time.sleep(2)
        else:
            raise Exception("Could not connect to Kafka after maximum retries")
    
    def send_message(self, topic, message):
        """Send message to Kafka topic"""
        try:
            future = self.producer.send(topic, message)
            result = future.get(timeout=10)
            self.logger.info(f"Message sent to {topic}: {message.get('object_id', 'unknown')}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to send message to {topic}: {e}")
            raise
    
    def create_consumer(self, topic, group_id):
        """Create Kafka consumer for specific topic"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
        return consumer
    
    def close(self):
        """Close Kafka connections"""
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()

# Updated topic definitions for complex multi-pipeline architecture
TOPICS = {
    # Input handling
    'INPUT_READY': 'object-input-ready',
    
    # Target pipeline topics
    'TARGET_GEDI_READY': 'target-gedi-ready',
    'TARGET_DINO_READY': 'target-dino-ready',
    'TARGET_POINT_CLOUD_READY': 'target-point-cloud-ready',
    'TARGET_FEATURE_FUSION_READY': 'target-feature-fusion-ready',
    'TARGET_GEDI_COMPLETE': 'target-gedi-complete',
    
    # Query pipeline topics
    'QUERY_CNOS_READY': 'query-cnos-ready',
    'QUERY_DINO_READY': 'query-dino-ready',
    'QUERY_FEATURE_FUSION_READY': 'query-feature-fusion-ready',
    'QUERY_GEDI_READY': 'query-gedi-ready',
    'QUERY_POINT_CLOUD_READY': 'query-point-cloud-ready',
    
    # Pose estimation pipeline topics
    'POSE_DINO_READY': 'pose-dino-ready',
    'POSE_EVAL_SYMMETRY_READY': 'pose-eval-symmetry-ready',
    'POSE_GET_SYMMETRY_READY': 'pose-get-symmetry-ready',
    'POSE_RANSAC_READY': 'pose-ransac-ready',
    'POSE_SEE_SYMMETRY_READY': 'pose-see-symmetry-ready',
    'POSE_SIDE_BY_SIDE_READY': 'pose-side-by-side-ready',
    'POSE_EXTRA_1_READY': 'pose-extra-1-ready',
    'POSE_EXTRA_2_READY': 'pose-extra-2-ready',
    
    # Final processing topics
    'RANSAC_READY': 'ransac-ready',
    'SYMMETRY_AWARE_REFINEMENT_READY': 'symmetry-aware-refinement-ready',
    'OVERLAY_READY': 'overlay-ready',
    'PIPELINE_STATUS': 'pipeline-status'
}
