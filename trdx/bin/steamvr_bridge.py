"""
SteamVR Data Bridge - Extracts headset and controller data from SteamVR
Sends data to TRDX system for enhanced tracking reference
"""

import openvr
import time
import json
import socket
import numpy as np
import logging
from typing import Dict, List, Optional
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SteamVRDataBridge:
    """Bridge to extract data from SteamVR system"""
    
    def __init__(self, target_port: int = 6970):
        self.logger = logging.getLogger(__name__)
        self.target_port = target_port
        self.socket = None
        self.vr_system = None
        self.running = False
        
        # Device tracking
        self.tracked_devices = {}
        self.device_classes = {
            openvr.TrackedDeviceClass_HMD: 'headset',
            openvr.TrackedDeviceClass_Controller: 'controller',
            openvr.TrackedDeviceClass_GenericTracker: 'tracker',
            openvr.TrackedDeviceClass_TrackingReference: 'base_station'
        }
        
    def initialize(self) -> bool:
        """Initialize OpenVR connection"""
        try:
            # Initialize OpenVR
            openvr.init(openvr.VRApplication_Background)
            self.vr_system = openvr.VRSystem()
            
            if not self.vr_system:
                self.logger.error("Failed to initialize VR system")
                return False
            
            # Initialize UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            self.logger.info("SteamVR Data Bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenVR: {e}")
            return False
    
    def start(self):
        """Start the data bridge"""
        if not self.initialize():
            return False
        
        self.running = True
        self.logger.info("Starting SteamVR data extraction...")
        
        try:
            while self.running:
                self._update_and_send_data()
                time.sleep(1/90)  # 90 Hz update rate
                
        except KeyboardInterrupt:
            self.logger.info("Received stop signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop the data bridge"""
        self.running = False
    
    def cleanup(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        
        try:
            openvr.shutdown()
        except:
            pass
        
        self.logger.info("SteamVR Data Bridge stopped")
    
    def _update_and_send_data(self):
        """Update device poses and send to TRDX"""
        try:
            # Create pose arrays
            poses = []
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                poses.append(openvr.TrackedDevicePose_t())

            # Get device poses
            self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, poses)

            # Process each tracked device
            for device_id in range(openvr.k_unMaxTrackedDeviceCount):
                device_class = self.vr_system.getTrackedDeviceClass(device_id)
                device_type = self.device_classes.get(device_class, 'unknown')
                pose = poses[device_id]

                if device_type == 'headset':
                    self.logger.info(f"[DEBUG] HMD encontrado: device_id={device_id}, conectado={self.vr_system.isTrackedDeviceConnected(device_id)}, poseValida={pose.bPoseIsValid}")

                if not self.vr_system.isTrackedDeviceConnected(device_id):
                    if device_type == 'headset':
                        self.logger.warning(f"[DEBUG] HMD device_id={device_id} NO conectado")
                    continue

                if not pose.bPoseIsValid:
                    if device_type == 'headset':
                        self.logger.warning(f"[DEBUG] HMD device_id={device_id} conectado pero pose NO valida")
                    continue

                # Extract pose data
                pose_data = self._extract_pose_data(device_id, device_class, pose)
                if pose_data:
                    if device_type == 'headset':
                        self.logger.info(f"[DEBUG] HMD pose enviada: {pose_data}")
                    self._send_pose_data(pose_data)
                    
        except Exception as e:
            self.logger.debug(f"Error updating poses: {e}")
    
    def _extract_pose_data(self, device_id: int, device_class: int, pose) -> Optional[Dict]:
        """Extract pose data from OpenVR pose"""
        try:
            # Get device type
            device_type = self.device_classes.get(device_class, 'unknown')
            
            # Skip unknown devices
            if device_type == 'unknown':
                return None
            
            # For controllers, determine left/right
            if device_type == 'controller':
                controller_role = self.vr_system.getControllerRoleForTrackedDeviceIndex(device_id)
                if controller_role == openvr.TrackedControllerRole_LeftHand:
                    device_type = 'controller_left'
                elif controller_role == openvr.TrackedControllerRole_RightHand:
                    device_type = 'controller_right'
            
            # Extract transformation matrix
            matrix = pose.mDeviceToAbsoluteTracking
            
            # Extract position (translation)
            position = [matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]]
            
            # Extract rotation matrix and convert to quaternion
            rotation_matrix = np.array([
                [matrix.m[0][0], matrix.m[0][1], matrix.m[0][2]],
                [matrix.m[1][0], matrix.m[1][1], matrix.m[1][2]],
                [matrix.m[2][0], matrix.m[2][1], matrix.m[2][2]]
            ])
            
            # Convert rotation matrix to quaternion
            quaternion = self._matrix_to_quaternion(rotation_matrix)
            
            # Extract velocities
            velocity = [pose.vVelocity.v[0], pose.vVelocity.v[1], pose.vVelocity.v[2]]
            angular_velocity = [pose.vAngularVelocity.v[0], pose.vAngularVelocity.v[1], pose.vAngularVelocity.v[2]]
            
            return {
                'device_id': device_id,
                'device_type': device_type,
                'position': position,
                'rotation': quaternion,
                'velocity': velocity,
                'angular_velocity': angular_velocity,
                'is_connected': True,
                'is_tracking': pose.bPoseIsValid,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.debug(f"Error extracting pose data for device {device_id}: {e}")
            return None
    
    def _matrix_to_quaternion(self, matrix: np.ndarray) -> List[float]:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
        try:
            trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (matrix[2, 1] - matrix[1, 2]) / s
                qy = (matrix[0, 2] - matrix[2, 0]) / s
                qz = (matrix[1, 0] - matrix[0, 1]) / s
            elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # s = 4 * qx
                qw = (matrix[2, 1] - matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (matrix[0, 1] + matrix[1, 0]) / s
                qz = (matrix[0, 2] + matrix[2, 0]) / s
            elif matrix[1, 1] > matrix[2, 2]:
                s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # s = 4 * qy
                qw = (matrix[0, 2] - matrix[2, 0]) / s
                qx = (matrix[0, 1] + matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (matrix[1, 2] + matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # s = 4 * qz
                qw = (matrix[1, 0] - matrix[0, 1]) / s
                qx = (matrix[0, 2] + matrix[2, 0]) / s
                qy = (matrix[1, 2] + matrix[2, 1]) / s
                qz = 0.25 * s
            
            return [qw, qx, qy, qz]
            
        except Exception as e:
            self.logger.debug(f"Error converting matrix to quaternion: {e}")
            return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    
    def _send_pose_data(self, pose_data: Dict):
        """Send pose data via UDP"""
        try:
            json_data = json.dumps(pose_data)
            data = json_data.encode('utf-8')
            self.socket.sendto(data, ('127.0.0.1', self.target_port))
            
        except Exception as e:
            self.logger.debug(f"Error sending pose data: {e}")

def main():
    """Main function to run the SteamVR bridge"""
    print("🎯 SteamVR Data Bridge - Headset Reference System")
    print("=" * 50)
    
    try:
        # Check if openvr is available
        import openvr
        print("✅ OpenVR library found")
    except ImportError:
        print("❌ OpenVR library not found!")
        print("Please install it with: pip install openvr")
        return 1
    
    # Create and start the bridge
    bridge = SteamVRDataBridge(target_port=6970)
    
    try:
        print("🚀 Starting SteamVR data extraction...")
        print("📡 Sending headset/controller data to TRDX on port 6970")
        print("🎮 Make sure SteamVR is running and your headset is connected")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        bridge.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping SteamVR bridge...")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        bridge.cleanup()
    
    print("✅ SteamVR bridge stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
