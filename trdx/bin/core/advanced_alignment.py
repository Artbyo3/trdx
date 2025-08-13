"""
Enhanced Tracker Alignment System
Advanced calibration system to ensure trackers feel perfectly attached to the user's body
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
import json

@dataclass
class BodyMeasurements:
    """Physical body measurements for precise tracker placement"""
    total_height: float = 1.75  # Total height in meters
    torso_length: float = 0.60  # Shoulder to hip distance
    leg_length: float = 0.90    # Hip to ankle distance
    foot_length: float = 0.25   # Foot length
    shoulder_width: float = 0.45 # Shoulder to shoulder distance
    hip_width: float = 0.35     # Hip to hip distance
    
    @classmethod
    def from_vr_detection(cls, headset_height: float, controller_positions: List[np.ndarray]) -> 'BodyMeasurements':
        """Create body measurements from VR system detection"""
        measurements = cls()
        
        # Estimate total height from headset position (headset is ~90% of total height)
        if headset_height > 0:
            measurements.total_height = headset_height / 0.9
            measurements.leg_length = measurements.total_height * 0.52  # Legs are ~52% of height
            measurements.torso_length = measurements.total_height * 0.35  # Torso is ~35% of height
        
        # Estimate shoulder/hip width from controller spread if available
        if len(controller_positions) >= 2:
            controller_distance = np.linalg.norm(controller_positions[0] - controller_positions[1])
            if 0.3 < controller_distance < 1.0:  # Reasonable range
                measurements.shoulder_width = controller_distance * 0.8  # Controllers wider than shoulders
                measurements.hip_width = measurements.shoulder_width * 0.8
        
        return measurements

class AdvancedTrackerAlignment:
    """Advanced tracker alignment system for perfect body attachment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Body measurements and calibration
        self.body_measurements = BodyMeasurements()
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_start_time = None
        
        # VR reference system
        self.vr_floor_level = 0.0
        self.vr_origin_position = np.array([0.0, 0.0, 0.0])
        self.headset_to_body_offset = np.array([0.0, -0.15, 0.0])  # Headset is ~15cm above shoulder center
        
        # Tracker attachment points (relative to body center)
        self.tracker_attachment_points = {
            'hip': np.array([0.0, 0.0, 0.0]),      # Hip center (body origin)
            'left_foot': np.array([-0.15, -0.90, 0.05]),   # Left foot position
            'right_foot': np.array([0.15, -0.90, 0.05])    # Right foot position
        }
        
        # Smoothing and stability
        self.position_history = {'hip': [], 'left_foot': [], 'right_foot': []}
        self.max_history_length = 10
        self.smoothing_factor = 0.85
        
        # Movement prediction
        self.last_positions = {}
        self.velocity_estimates = {}
        
    def start_calibration(self, vr_headset_pose: Optional[np.ndarray] = None, 
                         vr_controller_poses: Optional[List[np.ndarray]] = None):
        """Start calibration process using VR reference data"""
        self.calibration_start_time = time.time()
        self.calibration_samples = []
        
        # Use VR data to estimate body measurements
        if vr_headset_pose is not None:
            headset_height = vr_headset_pose[1]
            controller_positions = vr_controller_poses or []
            
            self.body_measurements = BodyMeasurements.from_vr_detection(
                headset_height, controller_positions
            )
            
            # Set VR reference points
            self.vr_origin_position = vr_headset_pose.copy()
            self.vr_floor_level = headset_height - self.body_measurements.total_height
            
            self.logger.info(f"VR-based calibration started:")
            self.logger.info(f"  Estimated height: {self.body_measurements.total_height:.2f}m")
            self.logger.info(f"  Floor level: {self.vr_floor_level:.3f}m")
            self.logger.info(f"  Headset reference: {vr_headset_pose}")
        
        self.logger.info("Advanced tracker calibration started")
    
    def add_calibration_sample(self, mediapose_landmarks: np.ndarray, 
                              vr_headset_pose: Optional[np.ndarray] = None):
        """Add a calibration sample"""
        if self.calibration_start_time is None:
            return
        
        sample = {
            'timestamp': time.time(),
            'mediapose_landmarks': mediapose_landmarks.copy(),
            'vr_headset_pose': vr_headset_pose.copy() if vr_headset_pose is not None else None
        }
        
        self.calibration_samples.append(sample)
        
        # Auto-complete calibration after collecting enough samples
        if len(self.calibration_samples) >= 30:  # 1 second at 30fps
            self.complete_calibration()
    
    def complete_calibration(self):
        """Complete calibration and compute optimal tracker alignment"""
        if len(self.calibration_samples) < 10:
            self.logger.warning("Not enough calibration samples")
            return
        
        try:
            # Analyze calibration samples to refine body measurements
            self._analyze_calibration_samples()
            
            # Compute optimal tracker attachment points
            self._compute_tracker_attachment_points()
            
            # Mark calibration as complete
            self.is_calibrated = True
            self.calibration_start_time = None
            
            self.logger.info("Advanced tracker calibration completed")
            self.logger.info(f"Final body measurements:")
            self.logger.info(f"  Height: {self.body_measurements.total_height:.2f}m")
            self.logger.info(f"  Torso: {self.body_measurements.torso_length:.2f}m")
            self.logger.info(f"  Legs: {self.body_measurements.leg_length:.2f}m")
            
        except Exception as e:
            self.logger.error(f"Error completing calibration: {e}")
    
    def _analyze_calibration_samples(self):
        """Analyze calibration samples to refine measurements"""
        if not self.calibration_samples:
            return
        
        # Extract key landmarks from samples
        hip_positions = []
        shoulder_positions = []
        ankle_positions = []
        
        for sample in self.calibration_samples:
            landmarks = sample['mediapose_landmarks']
            if len(landmarks) >= 33:  # MediaPipe pose has 33 landmarks
                # Hip center (average of left and right hip)
                left_hip = landmarks[23]  # Left hip
                right_hip = landmarks[24]  # Right hip
                hip_center = (left_hip + right_hip) / 2
                hip_positions.append(hip_center)
                
                # Shoulder center
                left_shoulder = landmarks[11]  # Left shoulder
                right_shoulder = landmarks[12]  # Right shoulder
                shoulder_center = (left_shoulder + right_shoulder) / 2
                shoulder_positions.append(shoulder_center)
                
                # Ankle positions
                left_ankle = landmarks[27]  # Left ankle
                right_ankle = landmarks[28]  # Right ankle
                ankle_positions.append([left_ankle, right_ankle])
        
        if hip_positions and shoulder_positions:
            # Calculate average body proportions
            hip_positions = np.array(hip_positions)
            shoulder_positions = np.array(shoulder_positions)
            
            # Update torso length (shoulder to hip distance)
            torso_distances = []
            for hip, shoulder in zip(hip_positions, shoulder_positions):
                distance = np.linalg.norm(hip - shoulder)
                torso_distances.append(distance)
            
            if torso_distances:
                avg_torso = np.mean(torso_distances)
                # Convert MediaPipe normalized distance to real distance
                self.body_measurements.torso_length = avg_torso * self.body_measurements.total_height
        
        self.logger.info("Calibration analysis completed")
    
    def _compute_tracker_attachment_points(self):
        """Compute optimal tracker attachment points based on body measurements"""
        # Hip tracker at body center (torso center)
        self.tracker_attachment_points['hip'] = np.array([0.0, 0.0, 0.0])
        
        # Foot trackers at ankle level, offset from center
        foot_separation = self.body_measurements.hip_width * 0.8  # Feet closer than hips
        foot_height = -self.body_measurements.leg_length * 0.95  # Slightly above floor
        
        self.tracker_attachment_points['left_foot'] = np.array([
            -foot_separation/2, foot_height, self.body_measurements.foot_length * 0.3
        ])
        
        self.tracker_attachment_points['right_foot'] = np.array([
            foot_separation/2, foot_height, self.body_measurements.foot_length * 0.3
        ])
        
        self.logger.info("Tracker attachment points computed")
    
    def align_trackers_to_body(self, mediapose_landmarks: np.ndarray, 
                              vr_headset_pose: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """Align trackers to body with aggressive direct positioning - ZERO tolerance for drift"""
        try:
            # AGGRESSIVE DIRECT POSITIONING - No smoothing, no prediction, pure landmark tracking
            if len(mediapose_landmarks) < 33:
                return {}
            
            # Direct landmark extraction with scaling factor for real-world positions
            landmarks = mediapose_landmarks
            scale_factor = 1.75  # Average human height scaling
            
            # DIRECT HIP TRACKING - Exact position from landmarks
            left_hip = landmarks[23] * scale_factor  # Left hip landmark
            right_hip = landmarks[24] * scale_factor  # Right hip landmark
            hip_center = (left_hip + right_hip) / 2.0
            
            # DIRECT FOOT TRACKING - Exact position from landmarks  
            left_ankle = landmarks[27] * scale_factor  # Left ankle landmark
            right_ankle = landmarks[28] * scale_factor  # Right ankle landmark
            
            # Apply VR headset reference for world positioning
            if vr_headset_pose is not None:
                # Direct application of headset offset - no dampening
                headset_offset = vr_headset_pose[:3] if len(vr_headset_pose) >= 3 else np.array([0, 0, 0])
                hip_center += headset_offset
                left_ankle += headset_offset
                right_ankle += headset_offset
            
            # ZERO SMOOTHING - Direct position assignment for instant response
            tracker_positions = {
                'hip': {
                    'position': hip_center + np.array([0.0, 0.05, 0.0]),  # Slight offset for tracker placement
                    'rotation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
                    'confidence': 1.0  # Maximum confidence for direct tracking
                },
                'left_foot': {
                    'position': left_ankle + np.array([0.0, 0.02, 0.0]),  # Slight offset above ankle
                    'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                    'confidence': 1.0
                },
                'right_foot': {
                    'position': right_ankle + np.array([0.0, 0.02, 0.0]),  # Slight offset above ankle
                    'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                    'confidence': 1.0
                }
            }
            
            # Store current positions for next frame (but don't use for smoothing)
            for tracker_name, data in tracker_positions.items():
                self.last_positions[tracker_name] = data['position'].copy()
            
            self.logger.debug("Applied AGGRESSIVE direct tracker positioning - zero smoothing")
            return tracker_positions
            
        except Exception as e:
            self.logger.error(f"Error in aggressive tracker alignment: {e}")
            # Fallback to basic direct positioning
            return self._basic_direct_alignment(mediapose_landmarks, vr_headset_pose)
    
    def _get_body_center_from_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Get body center position from MediaPipe landmarks"""
        if len(landmarks) < 25:
            return np.array([0.0, 0.0, 0.0])
        
        # Body center is average of hip landmarks
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        return (left_hip + right_hip) / 2
    
    def _apply_vr_reference_alignment(self, body_center: np.ndarray, 
                                    vr_headset_pose: np.ndarray) -> np.ndarray:
        """Apply VR reference for perfect world alignment"""
        # Calculate headset movement relative to initial position
        headset_offset = vr_headset_pose - self.vr_origin_position
        
        # Apply headset movement to body center (with some dampening)
        dampening = 0.8
        adjusted_body_center = body_center + (headset_offset * dampening)
        
        # Ensure floor alignment
        adjusted_body_center[1] = max(adjusted_body_center[1], self.vr_floor_level)
        
        return adjusted_body_center
    
    def _apply_position_smoothing(self, tracker_name: str, position: np.ndarray) -> np.ndarray:
        """Apply position smoothing to reduce jitter"""
        if tracker_name not in self.position_history:
            self.position_history[tracker_name] = []
        
        # Add to history
        self.position_history[tracker_name].append(position.copy())
        
        # Keep only recent history
        if len(self.position_history[tracker_name]) > self.max_history_length:
            self.position_history[tracker_name].pop(0)
        
        # Apply exponential smoothing
        if len(self.position_history[tracker_name]) > 1:
            prev_position = self.position_history[tracker_name][-2]
            smoothed = prev_position * self.smoothing_factor + position * (1 - self.smoothing_factor)
            return smoothed
        
        return position
    
    def _apply_movement_prediction(self, tracker_name: str, position: np.ndarray) -> np.ndarray:
        """Apply movement prediction to reduce perceived latency"""
        if tracker_name in self.last_positions:
            # Calculate velocity
            velocity = position - self.last_positions[tracker_name]
            self.velocity_estimates[tracker_name] = velocity
            
            # Predict future position (small lookahead)
            prediction_time = 0.033  # ~30ms prediction
            predicted_position = position + velocity * prediction_time
            
            # Limit prediction to reasonable bounds
            max_prediction = 0.05  # 5cm max prediction
            prediction_magnitude = np.linalg.norm(velocity * prediction_time)
            if prediction_magnitude > max_prediction:
                scale = max_prediction / prediction_magnitude
                predicted_position = position + velocity * prediction_time * scale
            
            self.last_positions[tracker_name] = position
            return predicted_position
        
        self.last_positions[tracker_name] = position
        return position
    
    def _calculate_tracker_rotation(self, tracker_name: str, landmarks: np.ndarray) -> np.ndarray:
        """Calculate tracker rotation based on body orientation"""
        # For now, return identity rotation
        # TODO: Implement proper body orientation tracking
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    def _calculate_confidence(self, tracker_name: str, landmarks: np.ndarray) -> float:
        """Calculate tracker confidence based on landmark visibility"""
        if len(landmarks) < 25:
            return 0.0
        
        # Check relevant landmarks for each tracker
        relevant_indices = {
            'hip': [23, 24],  # Hip landmarks
            'left_foot': [27, 29, 31],  # Left ankle, heel, foot index
            'right_foot': [28, 30, 32]  # Right ankle, heel, foot index
        }
        
        indices = relevant_indices.get(tracker_name, [])
        if not indices:
            return 0.5
        
        # Calculate average visibility of relevant landmarks
        # Note: This assumes landmarks have visibility info in 4th dimension
        if landmarks.shape[1] >= 4:
            visibilities = [landmarks[i][3] for i in indices if i < len(landmarks)]
            return np.mean(visibilities) if visibilities else 0.0
        
        return 0.8  # Default confidence if no visibility info
    
    def _basic_alignment(self, landmarks: np.ndarray, vr_headset_pose: Optional[np.ndarray]) -> Dict[str, Dict]:
        """Basic alignment for when calibration is not complete"""
        # Simple alignment using default proportions
        if len(landmarks) < 25:
            return {}
        
        hip_center = (landmarks[23] + landmarks[24]) / 2
        
        # Apply simple scaling
        scale = 1.8  # Default scale
        
        trackers = {
            'hip': {
                'position': hip_center * scale + np.array([0, 1.0, 0]),
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.7
            },
            'left_foot': {
                'position': landmarks[27] * scale + np.array([0, 0.1, 0]),
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.7
            },
            'right_foot': {
                'position': landmarks[28] * scale + np.array([0, 0.1, 0]),
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.7
            }
        }
        
        return trackers
    
    def _basic_direct_alignment(self, landmarks: np.ndarray, vr_headset_pose: Optional[np.ndarray]) -> Dict[str, Dict]:
        """Direct alignment fallback with zero smoothing - emergency positioning"""
        if len(landmarks) < 25:
            return {}
        
        # Direct landmark access with scaling
        scale = 1.75  # Default human scale
        
        # Calculate positions directly from landmarks
        hip_center = (landmarks[23] + landmarks[24]) / 2 * scale
        left_ankle = landmarks[27] * scale if len(landmarks) > 27 else hip_center + np.array([-0.2, -0.9, 0])
        right_ankle = landmarks[28] * scale if len(landmarks) > 28 else hip_center + np.array([0.2, -0.9, 0])
        
        # Apply headset offset if available
        if vr_headset_pose is not None:
            offset = vr_headset_pose[:3] if len(vr_headset_pose) >= 3 else np.array([0, 0, 0])
            hip_center += offset
            left_ankle += offset
            right_ankle += offset
        
        # Return direct positions - NO SMOOTHING
        return {
            'hip': {
                'position': hip_center + np.array([0, 0.05, 0]),
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 1.0
            },
            'left_foot': {
                'position': left_ankle + np.array([0, 0.02, 0]),
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 1.0
            },
            'right_foot': {
                'position': right_ankle + np.array([0, 0.02, 0]),
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 1.0
            }
        }
    
    def export_calibration_data(self) -> Dict:
        """Export calibration data for saving"""
        return {
            'is_calibrated': self.is_calibrated,
            'body_measurements': {
                'total_height': self.body_measurements.total_height,
                'torso_length': self.body_measurements.torso_length,
                'leg_length': self.body_measurements.leg_length,
                'foot_length': self.body_measurements.foot_length,
                'shoulder_width': self.body_measurements.shoulder_width,
                'hip_width': self.body_measurements.hip_width
            },
            'vr_reference': {
                'floor_level': self.vr_floor_level,
                'origin_position': self.vr_origin_position.tolist()
            },
            'tracker_attachment_points': {
                name: point.tolist() for name, point in self.tracker_attachment_points.items()
            }
        }
    
    def import_calibration_data(self, data: Dict):
        """Import calibration data from saved state"""
        try:
            self.is_calibrated = data.get('is_calibrated', False)
            
            # Import body measurements
            body_data = data.get('body_measurements', {})
            self.body_measurements.total_height = body_data.get('total_height', 1.75)
            self.body_measurements.torso_length = body_data.get('torso_length', 0.60)
            self.body_measurements.leg_length = body_data.get('leg_length', 0.90)
            self.body_measurements.foot_length = body_data.get('foot_length', 0.25)
            self.body_measurements.shoulder_width = body_data.get('shoulder_width', 0.45)
            self.body_measurements.hip_width = body_data.get('hip_width', 0.35)
            
            # Import VR reference
            vr_data = data.get('vr_reference', {})
            self.vr_floor_level = vr_data.get('floor_level', 0.0)
            if 'origin_position' in vr_data:
                self.vr_origin_position = np.array(vr_data['origin_position'])
            
            # Import tracker attachment points
            attachment_data = data.get('tracker_attachment_points', {})
            for name, point in attachment_data.items():
                self.tracker_attachment_points[name] = np.array(point)
            
            self.logger.info("Calibration data imported successfully")
            
        except Exception as e:
            self.logger.error(f"Error importing calibration data: {e}")
