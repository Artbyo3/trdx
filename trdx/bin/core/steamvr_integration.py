"""
SteamVR Integration - Access headset and controller positions for improved tracking
Provides real-time access to VR headset and controller poses for reference
"""

import numpy as np
import logging
import time
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import json
import socket
import struct
import threading

class VRDevicePose(NamedTuple):
    """Structure for VR device pose data"""
    device_id: int
    device_type: str  # 'headset', 'controller_left', 'controller_right', 'tracker'
    position: np.ndarray  # 3D position [x, y, z]
    rotation: np.ndarray  # Quaternion [w, x, y, z]
    velocity: np.ndarray  # Linear velocity [x, y, z]
    angular_velocity: np.ndarray  # Angular velocity [x, y, z]
    is_connected: bool
    is_tracking: bool
    timestamp: float

@dataclass
class VRReferenceFrame:
    """Reference frame data from VR system"""
    headset_pose: Optional[VRDevicePose] = None
    left_controller_pose: Optional[VRDevicePose] = None
    right_controller_pose: Optional[VRDevicePose] = None
    play_area_bounds: Optional[np.ndarray] = None
    standing_height: Optional[float] = None
    eye_height: Optional[float] = None

class SteamVRIntegration:
    """Integration with SteamVR system to access headset and controller data"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # VR system state
        self.vr_system_connected = False
        self.device_poses: Dict[int, VRDevicePose] = {}
        self.reference_frame = VRReferenceFrame()
        
        # Communication setup
        self.vr_data_port = config.get('vr_data_port', 6970)  # Different from tracker port
        self.socket = None
        self.running = False
        self.update_thread = None
        
        # Reference tracking
        self.headset_origin = None
        self.floor_level = None
        self.user_height = None
        self.calibration_complete = False
        
        # Integration settings
        self.use_headset_as_origin = config.get('use_headset_as_origin', True)
        self.use_controller_height_reference = config.get('use_controller_height_reference', True)
        self.auto_detect_floor_from_controllers = config.get('auto_detect_floor_from_controllers', True)
        self.stabilization_factor = config.get('stabilization_factor', 0.8)
        
        self.logger.info("SteamVR Integration initialized")
    
    def start(self) -> bool:
        """Start VR system integration"""
        try:
            # Try to connect to SteamVR data source
            success = self._initialize_vr_connection()
            if success:
                self.running = True
                self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
                self.update_thread.start()
                self.logger.info("SteamVR integration started successfully")
                return True
            else:
                self.logger.warning("Could not connect to SteamVR system")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start SteamVR integration: {e}")
            return False
    
    def stop(self):
        """Stop VR system integration"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
        self.logger.info("SteamVR integration stopped")
    
    def _initialize_vr_connection(self) -> bool:
        """Initialize connection to VR system"""
        try:
            # Try to connect via UDP socket to a VR data bridge
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('127.0.0.1', self.vr_data_port))
            self.socket.settimeout(0.1)  # Non-blocking with timeout
            
            self.vr_system_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VR connection: {e}")
            return False
    
    def _update_loop(self):
        """Main update loop for VR data"""
        while self.running:
            try:
                self._update_vr_poses()
                self._update_reference_frame()
                time.sleep(1/90)  # 90 Hz update rate
                
            except Exception as e:
                self.logger.debug(f"VR update error: {e}")
                time.sleep(0.1)
    
    def _update_vr_poses(self):
        """Update VR device poses"""
        if not self.socket:
            return
            
        try:
            # Try to receive VR pose data
            data, addr = self.socket.recvfrom(1024)
            pose_data = self._parse_vr_pose_data(data)
            
            if pose_data:
                device_id = pose_data['device_id']
                self.device_poses[device_id] = VRDevicePose(
                    device_id=device_id,
                    device_type=pose_data['device_type'],
                    position=np.array(pose_data['position']),
                    rotation=np.array(pose_data['rotation']),
                    velocity=np.array(pose_data.get('velocity', [0, 0, 0])),
                    angular_velocity=np.array(pose_data.get('angular_velocity', [0, 0, 0])),
                    is_connected=pose_data.get('is_connected', True),
                    is_tracking=pose_data.get('is_tracking', True),
                    timestamp=time.time()
                )
                
        except socket.timeout:
            pass  # Normal timeout, continue
        except Exception as e:
            self.logger.debug(f"Error updating VR poses: {e}")
    
    def _parse_vr_pose_data(self, data: bytes) -> Optional[Dict]:
        """Parse received VR pose data"""
        try:
            # Expected format: JSON string with pose data
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            self.logger.debug(f"Error parsing VR pose data: {e}")
            return None
    
    def _update_reference_frame(self):
        """Update the VR reference frame data"""
        # Find headset pose
        headset_pose = None
        for pose in self.device_poses.values():
            if pose.device_type == 'headset' and pose.is_tracking:
                headset_pose = pose
                break
        
        # Find controller poses
        left_controller = None
        right_controller = None
        for pose in self.device_poses.values():
            if pose.device_type == 'controller_left' and pose.is_tracking:
                left_controller = pose
            elif pose.device_type == 'controller_right' and pose.is_tracking:
                right_controller = pose
        
        # Update reference frame
        self.reference_frame.headset_pose = headset_pose
        self.reference_frame.left_controller_pose = left_controller
        self.reference_frame.right_controller_pose = right_controller
        
        # Auto-calibrate based on VR data
        if headset_pose and not self.calibration_complete:
            self._auto_calibrate_from_vr_data()
    
    def _auto_calibrate_from_vr_data(self):
        """Auto-calibrate tracking system using VR reference data"""
        headset_pose = self.reference_frame.headset_pose
        if not headset_pose:
            return
        
        # Estimate user height from headset position
        if self.user_height is None:
            # Assume headset is roughly at eye level (90% of total height)
            estimated_height = headset_pose.position[1] / 0.9
            if 1.4 < estimated_height < 2.2:  # Reasonable height range
                self.user_height = estimated_height
                self.reference_frame.standing_height = estimated_height
                self.reference_frame.eye_height = headset_pose.position[1]
                self.logger.info(f"Estimated user height: {estimated_height:.2f}m from headset position")
        
        # Set floor level from controllers if available
        if (self.floor_level is None and 
            self.reference_frame.left_controller_pose and 
            self.reference_frame.right_controller_pose):
            
            # When controllers are on the floor, use their Y position as floor reference
            left_y = self.reference_frame.left_controller_pose.position[1]
            right_y = self.reference_frame.right_controller_pose.position[1]
            
            # If both controllers are roughly at the same low height
            if abs(left_y - right_y) < 0.1 and min(left_y, right_y) < 0.3:
                self.floor_level = min(left_y, right_y)
                self.logger.info(f"Floor level detected at Y={self.floor_level:.3f}m from controllers")
        
        # Set headset origin for relative tracking
        if self.headset_origin is None:
            self.headset_origin = headset_pose.position.copy()
            self.logger.info(f"Headset origin set at: {self.headset_origin}")
        
        # Mark calibration as complete when we have basic references
        if (self.user_height is not None and 
            self.headset_origin is not None):
            self.calibration_complete = True
            self.logger.info("VR-based calibration completed")
    
    def get_headset_pose(self) -> Optional[VRDevicePose]:
        """Get current headset pose"""
        return self.reference_frame.headset_pose
    
    def get_controller_poses(self) -> Tuple[Optional[VRDevicePose], Optional[VRDevicePose]]:
        """Get current controller poses (left, right)"""
        return (self.reference_frame.left_controller_pose, 
                self.reference_frame.right_controller_pose)
    
    def get_relative_headset_position(self) -> Optional[np.ndarray]:
        """Get headset position relative to initial origin"""
        if not self.reference_frame.headset_pose or self.headset_origin is None:
            return None
        
        return self.reference_frame.headset_pose.position - self.headset_origin
    
    def get_user_height_estimate(self) -> Optional[float]:
        """Get estimated user height from VR data"""
        return self.user_height
    
    def get_floor_level(self) -> Optional[float]:
        """Get detected floor level"""
        return self.floor_level
    
    def is_vr_system_ready(self) -> bool:
        """Check if VR system is connected and tracking"""
        return (self.vr_system_connected and 
                self.reference_frame.headset_pose is not None and
                self.reference_frame.headset_pose.is_tracking)
    
    def apply_vr_reference_to_pose(self, pose_3d: np.ndarray) -> np.ndarray:
        """Apply VR reference frame to adjust MediaPipe pose coordinates"""
        if not self.calibration_complete:
            return pose_3d
        
        adjusted_pose = pose_3d.copy()
        
        # Apply headset-relative positioning if enabled
        if self.use_headset_as_origin and self.headset_origin is not None:
            headset_offset = self.get_relative_headset_position()
            if headset_offset is not None:
                # Adjust pose relative to headset movement
                adjusted_pose[:, 0] += headset_offset[0]  # X offset
                adjusted_pose[:, 2] += headset_offset[2]  # Z offset
        
        # Apply floor level correction
        if self.floor_level is not None:
            adjusted_pose[:, 1] -= self.floor_level  # Adjust Y to floor level
        
        # Apply height scaling based on VR-detected user height
        if self.user_height is not None:
            # Scale pose based on detected vs estimated height
            height_scale = self.user_height / 1.75  # Normalize to 1.75m reference
            adjusted_pose *= height_scale
        
        return adjusted_pose
    
    def get_vr_stabilized_tracking_scale(self) -> float:
        """Get tracking scale factor stabilized by VR reference"""
        if not self.calibration_complete or self.user_height is None:
            return 1.0
        
        # Return scale factor based on VR-detected user height
        return self.user_height / 1.75  # Normalize to reference height
    
    def export_vr_calibration_data(self) -> Dict:
        """Export VR calibration data for saving"""
        return {
            'user_height': self.user_height,
            'floor_level': self.floor_level,
            'headset_origin': self.headset_origin.tolist() if self.headset_origin is not None else None,
            'calibration_complete': self.calibration_complete,
            'eye_height': self.reference_frame.eye_height,
            'standing_height': self.reference_frame.standing_height
        }
    
    def import_vr_calibration_data(self, data: Dict):
        """Import VR calibration data from saved state"""
        self.user_height = data.get('user_height')
        self.floor_level = data.get('floor_level')
        if data.get('headset_origin'):
            self.headset_origin = np.array(data['headset_origin'])
        self.calibration_complete = data.get('calibration_complete', False)
        self.reference_frame.eye_height = data.get('eye_height')
        self.reference_frame.standing_height = data.get('standing_height')
        
        self.logger.info("VR calibration data imported")
