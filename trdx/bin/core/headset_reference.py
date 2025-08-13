"""
Headset Reference System - Enhanced tracking using SteamVR headset data
Integrates headset position as primary reference for body tracking alignment
"""

import openvr
import time
import math
import numpy as np
import logging
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import threading
from PySide6.QtCore import QObject, Signal
from scipy.spatial.transform import Rotation as R


class HeadsetPose(NamedTuple):
    """Structure for headset pose data"""
    position: np.ndarray      # 3D position [x, y, z] in meters
    rotation: np.ndarray      # Quaternion [w, x, y, z]
    euler_angles: np.ndarray  # Euler angles [pitch, yaw, roll] in degrees
    velocity: np.ndarray      # Linear velocity [x, y, z]
    angular_velocity: np.ndarray  # Angular velocity [x, y, z]
    is_valid: bool           # Tracking validity
    timestamp: float         # Pose timestamp
    confidence: float        # Tracking confidence (0-1)


@dataclass
class UserProfile:
    """User physical measurements for better tracking alignment"""
    total_height: float = 1.75  # meters
    eye_to_top_head: float = 0.12  # meters
    shoulder_width: float = 0.45  # meters
    torso_length: float = 0.60  # meters
    arm_length: float = 0.65  # meters
    leg_length: float = 0.90  # meters
    neck_length: float = 0.15  # meters


class HeadsetReferenceSystem(QObject):
    """Enhanced tracking system using SteamVR headset as primary reference"""
    
    # Signals
    headset_pose_updated = Signal(object)  # HeadsetPose
    tracking_lost = Signal()
    tracking_recovered = Signal()
    user_calibration_ready = Signal(object)  # UserProfile
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # VR System
        self.vr_system = None
        self.is_initialized = False
        self.is_running = False
        
        # Tracking thread
        self.tracking_thread = None
        
        # Headset data
        self.current_pose: Optional[HeadsetPose] = None
        self.pose_history: list = []  # For smoothing and analysis
        self.max_history_size = 30  # Keep last 30 poses for smoothing
        
        # User profile and calibration
        self.user_profile = UserProfile()
        self.is_calibrated = False
        self.calibration_data = {}
        
        # Tracking parameters
        self.config = config or {}
        self.update_frequency = self.config.get('headset_update_frequency', 90)  # Hz
        self.smoothing_factor = self.config.get('pose_smoothing_factor', 0.7)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        
        # Origin and reference frame
        self.world_origin = np.array([0.0, 0.0, 0.0])
        self.floor_level = 0.0
        self.standing_height = None
        
        # Tracking state
        self.tracking_lost_count = 0
        self.max_lost_frames = 30  # Frames before considering tracking lost
        
        self.logger.info("Headset Reference System initialized")
    
    def initialize_steamvr(self) -> bool:
        """Initialize SteamVR connection"""
        try:
            # Check if already initialized
            if self.is_initialized:
                return True
            
            # Initialize OpenVR
            openvr.init(openvr.VRApplication_Background)
            self.vr_system = openvr.VRSystem()
            
            if self.vr_system is None:
                raise Exception("Could not initialize VR system")
            
            # Verify headset is connected
            if not self.vr_system.isTrackedDeviceConnected(openvr.k_unTrackedDeviceIndex_Hmd):
                raise Exception("Headset not detected or not connected")
            
            self.is_initialized = True
            
            # Get headset info
            model = self.vr_system.getStringTrackedDeviceProperty(
                openvr.k_unTrackedDeviceIndex_Hmd, 
                openvr.Prop_ModelNumber_String
            )
            
            self.logger.info(f"✓ SteamVR initialized successfully")
            self.logger.info(f"✓ Headset detected: {model}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SteamVR: {e}")
            self.logger.error("Make sure:")
            self.logger.error("- SteamVR is running")
            self.logger.error("- Headset is connected and working")
            return False
    
    def shutdown_steamvr(self):
        """Shutdown SteamVR connection"""
        if self.is_initialized:
            try:
                openvr.shutdown()
                self.is_initialized = False
                self.logger.info("✓ SteamVR disconnected")
            except:
                pass
    
    def start_tracking(self) -> bool:
        """Start headset tracking in background thread"""
        if not self.is_initialized:
            if not self.initialize_steamvr():
                return False
        
        if self.is_running:
            return True
        
        self.is_running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        self.logger.info("Headset tracking started")
        return True
    
    def stop_tracking(self):
        """Stop headset tracking"""
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
        
        self.shutdown_steamvr()
        self.logger.info("Headset tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop running in background thread"""
        interval = 1.0 / self.update_frequency
        
        while self.is_running:
            start_time = time.time()
            
            # Get and process headset pose
            pose = self._get_headset_pose()
            if pose:
                self._process_pose(pose)
                self.headset_pose_updated.emit(pose)
            
            # Sleep to maintain update frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def _get_headset_pose(self) -> Optional[HeadsetPose]:
        """Get current headset pose from SteamVR"""
        if not self.is_initialized:
            return None
        
        try:
            # Get device poses
            poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
            self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 
                0,
                poses
            )
            
            # Get headset pose (always at index 0)
            hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
            
            if hmd_pose.bPoseIsValid:
                # Convert matrix to position and rotation
                matrix = hmd_pose.mDeviceToAbsoluteTracking
                position, rotation_matrix = self._matrix_to_pose(matrix)
                
                # Convert to quaternion
                r = R.from_matrix(rotation_matrix)
                quaternion = r.as_quat()  # [x, y, z, w] format
                quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])  # [w, x, y, z]
                
                # Convert to Euler angles for easier understanding
                euler = r.as_euler('xyz', degrees=True)
                
                # Extract velocities
                velocity = np.array([
                    hmd_pose.vVelocity.v[0],
                    hmd_pose.vVelocity.v[1], 
                    hmd_pose.vVelocity.v[2]
                ])
                
                angular_velocity = np.array([
                    hmd_pose.vAngularVelocity.v[0],
                    hmd_pose.vAngularVelocity.v[1],
                    hmd_pose.vAngularVelocity.v[2]
                ])
                
                # Calculate confidence based on tracking quality
                confidence = self._calculate_confidence(hmd_pose, velocity, angular_velocity)
                
                pose = HeadsetPose(
                    position=position,
                    rotation=quaternion,
                    euler_angles=euler,
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    is_valid=True,
                    timestamp=time.time(),
                    confidence=confidence
                )
                
                # Reset lost tracking counter
                self.tracking_lost_count = 0
                
                return pose
            
            else:
                # Invalid pose
                self.tracking_lost_count += 1
                if self.tracking_lost_count == self.max_lost_frames:
                    self.tracking_lost.emit()
                elif self.tracking_lost_count == 1 and self.current_pose:
                    # Was tracking before, emit recovery when it comes back
                    pass
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting headset pose: {e}")
            return None
    
    def _matrix_to_pose(self, matrix) -> Tuple[np.ndarray, np.ndarray]:
        """Convert OpenVR transformation matrix to position and rotation matrix"""
        # Extract position (last column)
        position = np.array([matrix[0][3], matrix[1][3], matrix[2][3]])
        
        # Extract rotation matrix (3x3)
        rotation_matrix = np.array([
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]]
        ])
        
        return position, rotation_matrix
    
    def _calculate_confidence(self, pose, velocity, angular_velocity) -> float:
        """Calculate tracking confidence based on various factors"""
        confidence = 1.0
        
        # Reduce confidence for high velocities (might indicate tracking issues)
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > 5.0:  # 5 m/s is very fast for head movement
            confidence *= 0.5
        elif velocity_magnitude > 2.0:  # 2 m/s is still quite fast
            confidence *= 0.8
        
        # Reduce confidence for high angular velocities
        angular_magnitude = np.linalg.norm(angular_velocity)
        if angular_magnitude > 10.0:  # High angular velocity
            confidence *= 0.6
        elif angular_magnitude > 5.0:
            confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))
    
    def _process_pose(self, pose: HeadsetPose):
        """Process and smooth the pose data"""
        # Add to history
        self.pose_history.append(pose)
        if len(self.pose_history) > self.max_history_size:
            self.pose_history.pop(0)
        
        # Apply smoothing if we have previous poses
        if self.current_pose and self.smoothing_factor > 0:
            pose = self._smooth_pose(pose, self.current_pose, self.smoothing_factor)
        
        self.current_pose = pose
        
        # Update calibration if needed
        if not self.is_calibrated and len(self.pose_history) > 10:
            self._update_calibration()
    
    def _smooth_pose(self, new_pose: HeadsetPose, prev_pose: HeadsetPose, factor: float) -> HeadsetPose:
        """Apply smoothing between poses"""
        # Smooth position
        smooth_position = prev_pose.position * factor + new_pose.position * (1 - factor)
        
        # Smooth quaternion rotation using SLERP
        from scipy.spatial.transform import Slerp
        
        key_times = [0, 1]
        key_rots = R.from_quat([
            [prev_pose.rotation[1], prev_pose.rotation[2], prev_pose.rotation[3], prev_pose.rotation[0]],  # [x,y,z,w]
            [new_pose.rotation[1], new_pose.rotation[2], new_pose.rotation[3], new_pose.rotation[0]]
        ])
        
        slerp = Slerp(key_times, key_rots)
        smooth_rot = slerp(1 - factor)
        smooth_quat = smooth_rot.as_quat()
        smooth_quaternion = np.array([smooth_quat[3], smooth_quat[0], smooth_quat[1], smooth_quat[2]])  # [w,x,y,z]
        
        # Smooth velocities
        smooth_velocity = prev_pose.velocity * factor + new_pose.velocity * (1 - factor)
        smooth_angular_velocity = prev_pose.angular_velocity * factor + new_pose.angular_velocity * (1 - factor)
        
        # Update euler angles
        smooth_euler = smooth_rot.as_euler('xyz', degrees=True)
        
        return HeadsetPose(
            position=smooth_position,
            rotation=smooth_quaternion,
            euler_angles=smooth_euler,
            velocity=smooth_velocity,
            angular_velocity=smooth_angular_velocity,
            is_valid=new_pose.is_valid,
            timestamp=new_pose.timestamp,
            confidence=new_pose.confidence
        )
    
    def _update_calibration(self):
        """Update user calibration based on collected pose data"""
        if len(self.pose_history) < 10:
            return
        
        # Calculate average floor level from headset positions
        positions = [pose.position for pose in self.pose_history[-10:]]
        avg_position = np.mean(positions, axis=0)
        
        # Estimate standing height (headset Y position relative to floor)
        self.standing_height = avg_position[1]
        
        # Estimate user height based on headset position
        # Typical eye height is about 0.12m below top of head
        estimated_height = self.standing_height + self.user_profile.eye_to_top_head
        self.user_profile.total_height = estimated_height
        
        # Set world origin at user's average position
        self.world_origin = np.array([avg_position[0], 0, avg_position[2]])
        self.floor_level = avg_position[1] - self.standing_height
        
        self.is_calibrated = True
        self.calibration_data = {
            'standing_height': self.standing_height,
            'floor_level': self.floor_level,
            'world_origin': self.world_origin.tolist(),
            'estimated_user_height': estimated_height
        }
        
        self.user_calibration_ready.emit(self.user_profile)
        self.logger.info(f"User calibration complete - Height: {estimated_height:.2f}m, Floor: {self.floor_level:.2f}m")
    
    def get_body_reference_points(self) -> Dict[str, np.ndarray]:
        """Calculate estimated body reference points based on headset position"""
        if not self.current_pose or not self.is_calibrated:
            return {}
        
        # Head position (headset position adjusted for headset offset)
        head_pos = self.current_pose.position.copy()
        
        # Calculate body orientation from headset yaw
        yaw_rad = math.radians(self.current_pose.euler_angles[1])
        
        # Forward and right vectors based on head orientation
        forward = np.array([math.sin(yaw_rad), 0, -math.cos(yaw_rad)])
        right = np.array([math.cos(yaw_rad), 0, math.sin(yaw_rad)])
        up = np.array([0, 1, 0])
        
        # Estimate body positions relative to head
        reference_points = {
            'head': head_pos,
            'neck': head_pos + up * (-self.user_profile.neck_length),
            'chest': head_pos + up * (-self.user_profile.neck_length - self.user_profile.torso_length * 0.3),
            'waist': head_pos + up * (-self.user_profile.neck_length - self.user_profile.torso_length * 0.7),
            'hips': head_pos + up * (-self.user_profile.neck_length - self.user_profile.torso_length),
            'left_shoulder': head_pos + up * (-self.user_profile.neck_length * 0.5) + right * (-self.user_profile.shoulder_width * 0.5),
            'right_shoulder': head_pos + up * (-self.user_profile.neck_length * 0.5) + right * (self.user_profile.shoulder_width * 0.5),
            'left_foot': head_pos + up * (-self.user_profile.total_height + 0.1),
            'right_foot': head_pos + up * (-self.user_profile.total_height + 0.1)
        }
        
        return reference_points
    
    def transform_pose_to_headset_space(self, world_pose: np.ndarray) -> np.ndarray:
        """Transform a world space pose to be relative to headset"""
        if not self.current_pose:
            return world_pose
        
        # Translate relative to headset position
        relative_pose = world_pose - self.current_pose.position
        
        # Rotate based on headset orientation
        # Create rotation matrix from headset quaternion
        r = R.from_quat([
            self.current_pose.rotation[1], 
            self.current_pose.rotation[2], 
            self.current_pose.rotation[3], 
            self.current_pose.rotation[0]
        ])
        
        # Apply inverse rotation to get pose in headset space
        headset_space_pose = r.inv().apply(relative_pose)
        
        return headset_space_pose
    
    def get_tracking_quality(self) -> Dict[str, float]:
        """Get current tracking quality metrics"""
        if not self.current_pose:
            return {'overall': 0.0, 'confidence': 0.0, 'stability': 0.0}
        
        # Calculate stability based on recent pose variations
        stability = 1.0
        if len(self.pose_history) >= 5:
            recent_positions = [pose.position for pose in self.pose_history[-5:]]
            position_variance = np.var(recent_positions, axis=0)
            avg_variance = np.mean(position_variance)
            stability = max(0.0, 1.0 - avg_variance * 100)  # Scale variance to 0-1
        
        overall_quality = (self.current_pose.confidence + stability) / 2.0
        
        return {
            'overall': overall_quality,
            'confidence': self.current_pose.confidence,
            'stability': stability,
            'is_valid': self.current_pose.is_valid
        }
    
    def get_steamvr_floor_level(self) -> Optional[float]:
        """Get the floor level from SteamVR tracking system"""
        if not self.is_initialized or not self.vr_system:
            return None
            
        try:
            # Get the tracking space bounds from SteamVR
            # This gives us the floor level in SteamVR space
            chaperone = openvr.VRChaperone()
            if chaperone:
                # Get the play area bounds
                bounds = chaperone.getPlayAreaSize()
                if bounds[0]:  # Success
                    # The play area Y=0 is typically the floor level
                    floor_y = 0.0
                    self.logger.debug(f"SteamVR floor level detected: {floor_y}")
                    return floor_y
            
            # Alternative method: Get tracking space to standing absolute transform
            transform = []
            result = self.vr_system.getSeatedZeroPoseToStandingAbsoluteTrackingPose()
            if result:
                # Extract Y translation - this is the floor offset
                floor_offset = result[1][3]  # Y component of translation
                self.logger.debug(f"SteamVR floor offset: {floor_offset}")
                return -floor_offset  # Negate to get actual floor level
                
        except Exception as e:
            self.logger.warning(f"Could not get SteamVR floor level: {e}")
            
        return None
    
    def get_steamvr_tracking_origin(self) -> Optional[np.ndarray]:
        """Get the SteamVR tracking space origin"""
        if not self.is_initialized or not self.vr_system:
            return None
            
        try:
            # Get the tracking origin transform
            transform = self.vr_system.getSeatedZeroPoseToStandingAbsoluteTrackingPose()
            if transform:
                # Extract translation vector (origin offset)
                origin = np.array([transform[0][3], transform[1][3], transform[2][3]])
                return origin
        except Exception as e:
            self.logger.warning(f"Could not get SteamVR tracking origin: {e}")
            
        return None
