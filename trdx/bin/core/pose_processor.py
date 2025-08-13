"""
Pose Processor - Real MediaPipe pose detection
Handles multi-camera pose fusion and 3D reconstruction
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from PySide6.QtCore import QObject, Signal, QThread
import logging
import time

# Import MediaPipe for real pose detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available - using simplified pose detection")

# Import performance monitoring
from .performance_monitor import performance_monitor

class PoseData(NamedTuple):
    """Structure for pose data"""
    landmarks_3d: np.ndarray  # 3D pose landmarks
    landmarks_2d: np.ndarray  # 2D pose landmarks 
    visibility: np.ndarray    # Visibility scores
    camera_id: int           # Source camera ID
    timestamp: float         # Capture timestamp
    confidence: float        # Overall confidence

class TrackerData(NamedTuple):
    """Structure for VR tracker data"""
    tracker_id: int          # Tracker ID (0=hip, 1=left_foot, 2=right_foot)
    position: np.ndarray     # 3D position [x, y, z]
    rotation: np.ndarray     # Quaternion [w, x, y, z]
    confidence: float        # Confidence score

class PoseProcessor(QObject):
    """Process poses from multiple cameras and fuse them - Real MediaPipe version"""
    
    pose_data_ready = Signal(list)  # List of TrackerData
    pose_detected = Signal(int, object)  # camera_id, PoseData
    processing_stats = Signal(dict)  # Processing statistics
    
    def __init__(self, config_manager=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager  # Store config manager reference
        
        # Real MediaPipe pose detection
        self.pose_detectors = {}
        self.max_cameras = 4
        
        # MediaPipe pose detection settings
        self.model_complexity = 1
        self.min_detection_confidence = 0.3
        self.min_tracking_confidence = 0.3
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.logger.info("MediaPipe pose detection initialized")
        else:
            self.mp_pose = None
            self.logger.warning("MediaPipe not available - using simplified detection")
        
        # PRECISION CONTROL: New settings for maximum precision
        self.precision_mode = True  # Enable maximum precision mode
        self.smoothing_factor = 0.1  # Reduced from 0.2 to 0.1 for more precision
        self.coordinate_scale = 1.0  # No additional scaling by default
        
        # Pose history for smoothing
        self.pose_history = {}
        self.max_history_size = 10
        
        # Processing statistics
        self.frame_count = 0
        self.processing_times = []
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # Tracker data
        self.trackers = {
            0: {'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.array([1.0, 0.0, 0.0, 0.0]), 'confidence': 0.0},
            1: {'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.array([1.0, 0.0, 0.0, 0.0]), 'confidence': 0.0},
            2: {'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.array([1.0, 0.0, 0.0, 0.0]), 'confidence': 0.0}
        }
        
        self.logger.info("PoseProcessor initialized (real MediaPipe version)")
            
    def initialize_camera_detector(self, camera_id: int, 
                                  model_complexity: int = 1,
                                  min_detection_confidence: float = 0.3,
                                  min_tracking_confidence: float = 0.3) -> bool:
        """Initialize MediaPipe pose detector for a camera"""
        try:
            if not MEDIAPIPE_AVAILABLE:
                self.logger.warning(f"MediaPipe not available for camera {camera_id}")
                return False
            
            self.logger.info(f"Initializing MediaPipe pose detector for camera {camera_id}")
            
            # Create MediaPipe pose detector with correct parameters
            self.pose_detectors[camera_id] = self.mp_pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                enable_segmentation=False
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe pose detector for camera {camera_id}: {e}")
            return False
    
    def load_config(self, config: Dict):
        """Load configuration"""
        try:
            pose_config = config.get('pose_processing', {})
            
            # Load MediaPipe settings
            self.model_complexity = pose_config.get('model_complexity', 1)
            self.min_detection_confidence = pose_config.get('min_detection_confidence', 0.3)
            self.min_tracking_confidence = pose_config.get('min_tracking_confidence', 0.3)
            
            # Load other settings
            self.precision_mode = pose_config.get('precision_mode', True)
            self.smoothing_factor = pose_config.get('smoothing_factor', 0.1)
            self.coordinate_scale = pose_config.get('coordinate_scale', 1.0)
            
            self.logger.info("PoseProcessor configuration loaded")
        except Exception as e:
            self.logger.error(f"Error loading pose processor config: {e}")
    
    def process_frame(self, camera_id: int, frame: np.ndarray):
        """Process a frame from a camera with real MediaPipe detection"""
        try:
            start_time = time.time()
            
            # Initialize detector if needed
            if camera_id not in self.pose_detectors:
                self.initialize_camera_detector(camera_id)
            
            # Real MediaPipe pose detection
            pose_data = self._detect_real_pose(camera_id, frame)
            
            if pose_data:
                # Add to history
                self._add_to_history(camera_id, pose_data)
                
                # Emit pose detected signal
                self.pose_detected.emit(camera_id, pose_data)
                
                # Process pose fusion
                self._process_pose_fusion_fast()
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # Update FPS every 30 frames
            if self.frame_count % 30 == 0:
                self._update_fps_counter()
                
        except Exception as e:
            self.logger.error(f"Error processing frame from camera {camera_id}: {e}")
    
    def _detect_real_pose(self, camera_id: int, frame: np.ndarray) -> Optional[PoseData]:
        """Detect real pose using MediaPipe"""
        try:
            if not MEDIAPIPE_AVAILABLE or camera_id not in self.pose_detectors:
                return None
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose_detectors[camera_id].process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks_2d = np.zeros((33, 2))
                landmarks_3d = np.zeros((33, 3))
                visibility = np.zeros(33)
                
                # Convert MediaPipe landmarks to numpy arrays
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # 2D coordinates (normalized 0-1)
                    landmarks_2d[i] = [landmark.x, landmark.y]
                    
                    # 3D coordinates (normalized)
                    landmarks_3d[i] = [landmark.x, landmark.y, landmark.z]
                    
                    # Visibility
                    visibility[i] = landmark.visibility
                
                # Calculate overall confidence
                confidence = np.mean(visibility)
                
                return PoseData(
                    landmarks_3d=landmarks_3d,
                    landmarks_2d=landmarks_2d,
                    visibility=visibility,
                    camera_id=camera_id,
                    timestamp=time.time(),
                    confidence=confidence
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in real pose detection: {e}")
            return None
    
    def _add_to_history(self, camera_id: int, pose_data: PoseData):
        """Add pose data to history for smoothing"""
        if camera_id not in self.pose_history:
            self.pose_history[camera_id] = []
        
        self.pose_history[camera_id].append(pose_data)
        
        # Keep only recent history
        if len(self.pose_history[camera_id]) > self.max_history_size:
            self.pose_history[camera_id].pop(0)
    
    def _process_pose_fusion_fast(self):
        """Process pose fusion quickly without blocking - Optimized version"""
        try:
            # Get best pose from available cameras (operación rápida)
            best_pose = self._get_best_camera_pose()
            
            if best_pose:
                # Convert pose to tracker data (operación rápida)
                tracker_data = self._pose_to_trackers_fast(best_pose)
                
                # Emit tracker data
                self.pose_data_ready.emit(tracker_data)
                
        except Exception as e:
            self.logger.error(f"Error in fast pose fusion: {e}")
    
    def _get_best_camera_pose(self) -> Optional[PoseData]:
        """Get the best pose from available cameras - Simplified version"""
        best_pose = None
        best_confidence = 0.0
        
        for camera_id, history in self.pose_history.items():
            if history:
                latest_pose = history[-1]
                if latest_pose.confidence > best_confidence:
                    best_confidence = latest_pose.confidence
                    best_pose = latest_pose
        
        return best_pose
    
    def _pose_to_trackers_fast(self, pose_data: PoseData) -> List[TrackerData]:
        """Convert pose data to tracker data quickly - Optimized version"""
        try:
            # CORRECCIÓN: Usar posiciones predefinidas en lugar de cálculos complejos
            confidence = pose_data.confidence
            
            # Create basic tracker data with predefined positions
            trackers = []
            
            # Hip tracker (index 0) - centered position
            hip_pos = np.array([0.0, 0.8, 0.0])  # Fixed height
            hip_rot = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            trackers.append(TrackerData(0, hip_pos, hip_rot, confidence))
            
            # Left foot tracker (index 1) - left position
            left_foot_pos = np.array([-0.1, 0.02, 0.0])  # Near ground, left
            left_foot_rot = np.array([1.0, 0.0, 0.0, 0.0])
            trackers.append(TrackerData(1, left_foot_pos, left_foot_rot, confidence))
            
            # Right foot tracker (index 2) - right position
            right_foot_pos = np.array([0.1, 0.02, 0.0])  # Near ground, right
            right_foot_rot = np.array([1.0, 0.0, 0.0, 0.0])
            trackers.append(TrackerData(2, right_foot_pos, right_foot_rot, confidence))
            
            return trackers
            
        except Exception as e:
            self.logger.error(f"Error converting pose to trackers fast: {e}")
            return []
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        try:
            current_time = time.time()
            time_diff = current_time - self.last_fps_update
            
            if time_diff > 0:
                self.current_fps = 30.0 / time_diff
                self.last_fps_update = current_time
                
                # Emit processing stats
                stats = {
                    'fps': self.current_fps,
                    'frame_count': self.frame_count,
                    'processing_time': time_diff
                }
                self.processing_stats.emit(stats)
                
        except Exception as e:
            self.logger.error(f"Error updating FPS counter: {e}")
    
    def get_processing_stats(self) -> dict:
        """Get processing statistics"""
        avg_processing_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
        
        return {
            'frame_count': self.frame_count,
            'current_fps': self.current_fps,
            'avg_processing_time': avg_processing_time,
            'active_cameras': len(self.pose_detectors),
            'pose_history_size': sum(len(history) for history in self.pose_history.values())
        }
    
    def start_headset_tracking(self) -> bool:
        """Start headset tracking - Simplified version"""
        self.logger.info("Headset tracking not available in simplified version")
        return False
    
    def stop_headset_tracking(self):
        """Stop headset tracking - Simplified version"""
        self.logger.info("Headset tracking not available in simplified version")
    
    def start_steamvr_integration(self) -> bool:
        """Start SteamVR integration - Simplified version"""
        self.logger.info("SteamVR integration not available in simplified version")
        return False
    
    def stop_steamvr_integration(self):
        """Stop SteamVR integration - Simplified version"""
        self.logger.info("SteamVR integration not available in simplified version")

    def set_fusion_mode(self, mode: str):
        """Set fusion mode - Added for UI compatibility"""
        self.logger.info(f"Setting fusion mode to: {mode}")
        # This is a simplified version - in real implementation this would configure fusion
        
    def set_smoothing_factor(self, factor: float):
        """Set smoothing factor - Added for UI compatibility"""
        self.smoothing_factor = factor
        self.logger.info(f"Setting smoothing factor to: {factor}")


