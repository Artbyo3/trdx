"""
SteamVR Driver - Custom VR Driver for TRDX
Implements communication with SteamVR for Full Body Tracking via UDP
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import socket
import struct
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
from .performance_monitor import performance_monitor

# Platform-specific imports
if sys.platform.startswith('win32'):
    import win32pipe
    import win32file
    import win32api
    import win32con

class TrackerState:
    """Represents the state of a VR tracker"""
    def __init__(self, tracker_id: int, role: str = "TrackerRole_Waist"):
        self.tracker_id = tracker_id
        self.role = role
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z quaternion
        self.confidence = 0.0
        self.last_update = 0.0
        self.is_active = False

class SteamVRDriver(QObject):
    """Custom SteamVR Driver for TRDX with UDP communication"""
    
    status_changed = Signal(str, bool)  # status_message, is_connected
    tracker_added = Signal(int, str)    # tracker_id, role
    error_occurred = Signal(str)        # error_message
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Driver state
        self.is_connected = False
        
        # UDP Communication
        self.udp_socket = None
        self.driver_host = 'localhost'
        self.driver_port = 9998  # Match the C++ driver port
        
        # Legacy pipe support (keeping for compatibility)
        self.pipe_name = r'\\.\pipe\TRDXPipeIn'
        self.pipe_handle = None
        self.driver_process = None
        
        # Tracker management
        self.trackers: Dict[int, TrackerState] = {}
        self.max_trackers = 10
        self.next_tracker_id = 0
        
        # Driver configuration
        self.driver_config = {
            'smoothing': 0.5,
            'additional_smoothing': 0.7,
            'latency_compensation': 0.0,
            'auto_calibrate': True,
            'test_mode': False  # Disable test mode - use real tracking data
        }
        
        # Communication
        self.communication_thread = None
        self.should_stop = False
        
        # Driver installation paths
        self.driver_path = self._get_driver_path()
        self.steamvr_path = self._find_steamvr_path()
        
    def _get_driver_path(self) -> Path:
        """Get the path where our driver should be installed"""
        # Create driver in project directory
        project_root = Path(__file__).parent.parent.parent
        driver_path = project_root / "steamvr_driver" / "trdx_driver"
        return driver_path
    
    def _find_steamvr_path(self) -> Optional[Path]:
        """Find SteamVR installation path"""
        possible_paths = [
            Path("C:/Program Files (x86)/Steam/steamapps/common/SteamVR"),
            Path("C:/Program Files/Steam/steamapps/common/SteamVR"),
            Path.home() / "Steam/steamapps/common/SteamVR"
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "bin").exists():
                return path
        
        # Try to find through Steam registry
        try:
            if sys.platform.startswith('win32'):
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"SOFTWARE\WOW6432Node\Valve\Steam")
                steam_path = winreg.QueryValueEx(key, "InstallPath")[0]
                steamvr_path = Path(steam_path) / "steamapps/common/SteamVR"
                if steamvr_path.exists():
                    return steamvr_path
        except Exception:
            pass
        
        return None
    
    def install_driver(self) -> bool:
        """Install the TRDX driver to SteamVR"""
        try:
            # Create driver files
            if not self._create_driver_files():
                return False
            
            # Install to SteamVR
            if not self._install_to_steamvr():
                return False
            
            self.logger.info("TRDX driver installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install driver: {e}")
            self.error_occurred.emit(f"Driver installation failed: {e}")
            return False
    
    def _create_driver_files(self) -> bool:
        """Create the driver files structure"""
        try:
            # Create driver directory structure
            driver_root = self.driver_path
            driver_root.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (driver_root / "bin" / "win64").mkdir(parents=True, exist_ok=True)
            (driver_root / "resources").mkdir(parents=True, exist_ok=True)
            
            # Create driver manifest
            manifest = {
                "name": "trdx_driver",
                "directory": str(driver_root),
                "resourcedir": "resources",
                "bindir": "bin"
            }
            
            with open(driver_root / "driver.vrdrivermanifest", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create driver DLL (stub for now - needs C++ implementation)
            self._create_driver_stub()
            
            self.logger.info(f"Driver files created at: {driver_root}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create driver files: {e}")
            return False
    
    def _create_driver_stub(self):
        """Create a stub driver implementation"""
        # For now, we'll use a Python-based communication system
        # In a production environment, this would be a C++ DLL
        
        stub_content = '''
# TRDX Driver Stub
# This would be implemented as a C++ DLL in production
# For development, we use Python-based communication
'''
        
        driver_stub_path = self.driver_path / "bin" / "win64" / "driver_trdx.dll"
        # Note: This creates a text file, not a real DLL
        # A real implementation would require C++ and OpenVR SDK
        with open(str(driver_stub_path).replace('.dll', '_stub.txt'), 'w') as f:
            f.write(stub_content)
    
    def _install_to_steamvr(self) -> bool:
        """Install driver to SteamVR drivers directory"""
        if not self.steamvr_path:
            self.logger.error("SteamVR path not found")
            return False
        
        try:
            steamvr_drivers = self.steamvr_path / "drivers"
            target_path = steamvr_drivers / "trdx"
            
            # Copy driver files
            import shutil
            if target_path.exists():
                shutil.rmtree(target_path)
            
            shutil.copytree(self.driver_path, target_path)
            
            self.logger.info(f"Driver installed to: {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install to SteamVR: {e}")
            return False
    
    def connect_to_steamvr(self) -> bool:
        """Connect to SteamVR via UDP"""
        self.logger.warning("[TRDX] Attempting to connect to SteamVR via UDP...")
        
        # First check if SteamVR is running
        if not self.is_steamvr_running():
            self.logger.error("[TRDX] SteamVR is not running! Please start SteamVR first.")
            self.error_occurred.emit("SteamVR is not running - please start SteamVR first")
            return False
        
        try:
            # Initialize UDP socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.settimeout(1.0)  # OPTIMIZED: Faster timeout
            
            # Test connection by sending a ping and waiting for response
            test_message = "PING"
            try:
                self.udp_socket.sendto(test_message.encode(), (self.driver_host, self.driver_port))
                self.logger.warning(f"[TRDX] Sent ping to {self.driver_host}:{self.driver_port}")
                
                # Try to receive a response (driver should respond)
                try:
                    self.udp_socket.settimeout(1.0)
                    data, addr = self.udp_socket.recvfrom(1024)
                    if data.decode().startswith("PONG"):
                        self.logger.warning(f"[TRDX] Received response from driver at {addr}")
                        self.is_connected = True
                        self.status_changed.emit("Connected to SteamVR Driver", True)
                        self._start_communication_thread()
                        self.logger.warning("[TRDX] Successfully connected to SteamVR driver via UDP")
                        return True
                    else:
                        self.logger.warning(f"[TRDX] Received unexpected response: {data.decode()}")
                except socket.timeout:
                    self.logger.warning("[TRDX] No response from driver (timeout) - driver may not be loaded")
                    # Continue anyway, but warn user
                    self.is_connected = True
                    self.status_changed.emit("Connected to SteamVR Driver (no response)", True)
                    self._start_communication_thread()
                    self.logger.warning("[TRDX] Connected to SteamVR via UDP (no driver response)")
                    return True
                    
            except Exception as e:
                self.logger.warning(f"[TRDX] UDP test failed: {e}")
                # Still continue, but warn
                self.is_connected = True
                self.status_changed.emit("Connected to SteamVR Driver (test failed)", True)
                self._start_communication_thread()
                self.logger.warning("[TRDX] Connected to SteamVR via UDP (test failed)")
                return True
                
        except Exception as e:
            self.logger.error(f"[TRDX] Failed to connect to SteamVR: {e}")
            self.error_occurred.emit(f"Connection failed: {e}")
            if self.udp_socket:
                self.udp_socket.close()
                self.udp_socket = None
            return False
    
    def disconnect(self):
        """Disconnect from SteamVR"""
        self.should_stop = True
        
        if self.communication_thread and self.communication_thread.is_alive():
            self.communication_thread.join(timeout=2.0)
        
        # Close UDP socket
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except:
                pass
            self.udp_socket = None
        
        # Close legacy pipe (if used)
        if self.pipe_handle:
            try:
                win32file.CloseHandle(self.pipe_handle)
            except:
                pass
            self.pipe_handle = None
        
        self.is_connected = False
        self.status_changed.emit("Disconnected", False)
        self.logger.info("Disconnected from SteamVR")
    
    def _start_communication_thread(self):
        """Start the communication thread"""
        self.should_stop = False
        self.communication_thread = threading.Thread(
            target=self._communication_loop,
            daemon=True
        )
        self.communication_thread.start()
    
    def _communication_loop(self):
        """Main communication loop - waits for real tracker data"""
        while not self.should_stop and self.is_connected:
            try:
                # Just wait for real data from pose processor
                time.sleep(0.005)  # OPTIMIZED: 200 Hz update rate for lower latency
                
            except Exception as e:
                self.logger.error(f"Communication error: {e}")
                break
    
    def add_tracker(self, role: str = "TrackerRole_Waist") -> int:
        """Add a new tracker"""
        if len(self.trackers) >= self.max_trackers:
            self.logger.error("Maximum number of trackers reached")
            return -1
        
        tracker_id = self.next_tracker_id
        self.next_tracker_id += 1
        
        tracker = TrackerState(tracker_id, role)
        self.trackers[tracker_id] = tracker
        
        self.tracker_added.emit(tracker_id, role)
        self.logger.info(f"Added tracker {tracker_id} with role {role}")
        
        return tracker_id
    
    def remove_tracker(self, tracker_id: int):
        """Remove a tracker"""
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]
            self.logger.info(f"Removed tracker {tracker_id}")
    
    def update_trackers(self, tracker_data_list: List):
        """Update multiple trackers with new pose data"""
        if not tracker_data_list:
            return
            
        # Only log every 100th update to reduce spam
        if hasattr(self, '_update_counter'):
            self._update_counter += 1
        else:
            self._update_counter = 1
            
        if self._update_counter % 100 == 1:
            self.logger.debug(f"Updating {len(tracker_data_list)} trackers")
        
        for tracker_data in tracker_data_list:
            self.update_tracker(
                tracker_data.tracker_id,
                tracker_data.position,
                tracker_data.rotation,
                tracker_data.confidence
            )
    
    def update_tracker(self, tracker_id: int, position: np.ndarray, 
                      rotation: np.ndarray, confidence: float = 1.0):
        """Update a specific tracker's pose"""
        if tracker_id not in self.trackers:
            # Auto-create tracker if it doesn't exist
            if tracker_id == 0:
                role = "TrackerRole_Waist"
            elif tracker_id == 1:
                role = "TrackerRole_LeftFoot"
            elif tracker_id == 2:
                role = "TrackerRole_RightFoot"
            else:
                role = "TrackerRole_Handed"
            self.add_tracker(role)
        tracker = self.trackers[tracker_id]
        # Always mark as active when receiving data
        tracker.is_active = True
        # NO smoothing in driver - use raw position to prevent teleporting
        tracker.position = position
        tracker.rotation = rotation
        tracker.confidence = confidence
        tracker.last_update = time.time()
        # Debug log
        self.logger.info(f"[DEBUG] Tracker {tracker_id} updated: pos={tracker.position}, rot={tracker.rotation}, conf={tracker.confidence}, active={tracker.is_active}")
        # Send to SteamVR (simulated for now)
        self._send_tracker_update(tracker_id, tracker)
    
    def _send_tracker_update(self, tracker_id: int, tracker: TrackerState):
        """Send tracker update to SteamVR via UDP"""
        if not self.is_connected or not self.udp_socket:
            # Only log connection errors occasionally to reduce spam
            if hasattr(self, '_connection_error_counter'):
                self._connection_error_counter += 1
            else:
                self._connection_error_counter = 1
                
            if self._connection_error_counter % 300 == 1:  # Every 300 attempts (about every 10 seconds)
                self.logger.warning(f"Cannot send tracker {tracker_id} data: not connected to SteamVR")
            return
        
        try:
            # Formato exacto requerido por el driver: trackerId,x,y,z,qw,qx,qy,qz,confidence
            message = f"{int(tracker_id)},{float(tracker.position[0])},{float(tracker.position[1])},{float(tracker.position[2])},{float(tracker.rotation[0])},{float(tracker.rotation[1])},{float(tracker.rotation[2])},{float(tracker.rotation[3])},{float(tracker.confidence)}"
            
            # DEBUG: Log the exact message being sent
            self.logger.info(f"[UDP SEND] Tracker {tracker_id}: {message}")
            
            self.udp_socket.sendto(message.encode(), (self.driver_host, self.driver_port))
            
            # OPTIMIZED: Record UDP send for performance monitoring
            performance_monitor.record_udp_send()

            # OPTIMIZED: Dramatically reduced logging for performance
            if not hasattr(self, '_send_log_counter'):
                self._send_log_counter = 0
            self._send_log_counter += 1
            
            # Only log every 300 frames (10 seconds at 30fps) instead of every 30
            if self._send_log_counter % 300 == 0:
                role_name = ['Hip/Waist', 'Left Foot', 'Right Foot'][tracker_id] if tracker_id < 3 else f'Tracker{tracker_id}'
                self.logger.debug(f"UDP {tracker_id}({role_name}): {tracker.confidence:.2f}")

        except Exception as e:
            self.logger.error(f"Failed to send tracker data via UDP: {e}")
            # Try to reconnect if connection was lost
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                self.logger.warning("UDP connection lost, attempting to reconnect...")
                self.disconnect()
                threading.Timer(2.0, self.connect_to_steamvr).start()
            else:
                self.logger.warning(f"UDP send error: {e}")
        # For now, we'll just log the update
        
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Tracker {tracker_id}: pos={tracker.position}, "
                f"rot={tracker.rotation}, conf={tracker.confidence}"
            )
    
    def get_hmd_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get HMD position and rotation"""
        # Simulated HMD pose for now
        # In a real implementation, this would query the actual HMD
        position = np.array([0.0, 1.8, 0.0])  # 1.8m height
        rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        return position, rotation
    
    def calibrate_tracking_space(self):
        """Calibrate the tracking space"""
        # TODO: Implement space calibration
        self.logger.info("Tracking space calibration started")
    
    def set_driver_settings(self, settings: dict):
        """Update driver settings"""
        self.driver_config.update(settings)
        self.logger.info(f"Driver settings updated: {settings}")
    
    def get_tracker_count(self) -> int:
        """Get number of active trackers"""
        return len([t for t in self.trackers.values() if t.is_active])
    
    def get_tracker_status(self, tracker_id: int) -> dict:
        """Get status of a specific tracker"""
        if tracker_id not in self.trackers:
            return {'exists': False}
        
        tracker = self.trackers[tracker_id]
        return {
            'exists': True,
            'active': tracker.is_active,
            'role': tracker.role,
            'position': tracker.position.tolist(),
            'rotation': tracker.rotation.tolist(),
            'confidence': tracker.confidence,
            'last_update': tracker.last_update
        }
    
    def get_all_tracker_status(self) -> dict:
        """Get status of all trackers"""
        status = {}
        for tracker_id in self.trackers:
            status[tracker_id] = self.get_tracker_status(tracker_id)
        return status
    
    def is_steamvr_running(self) -> bool:
        """Check if SteamVR is currently running"""
        try:
            # Check for SteamVR processes
            import psutil
            steamvr_processes = []
            for proc in psutil.process_iter(['name', 'pid']):
                if proc.info['name'] and 'vrserver' in proc.info['name'].lower():
                    steamvr_processes.append(proc.info['pid'])
            
            if steamvr_processes:
                self.logger.info(f"[TRDX] Found SteamVR processes: {steamvr_processes}")
                return True
            else:
                self.logger.warning("[TRDX] No SteamVR processes found")
                return False
        except Exception as e:
            self.logger.error(f"[TRDX] Error checking SteamVR processes: {e}")
            return False
    
    def start_steamvr(self) -> bool:
        """Start SteamVR if not running"""
        if self.is_steamvr_running():
            return True
        
        if not self.steamvr_path:
            self.logger.error("SteamVR path not found")
            return False
        
        try:
            # Start SteamVR
            steamvr_exe = self.steamvr_path / "bin" / "win64" / "vrstartup.exe"
            if steamvr_exe.exists():
                subprocess.Popen([str(steamvr_exe)], cwd=str(self.steamvr_path))
                self.logger.info("SteamVR startup initiated")
                return True
            else:
                self.logger.error(f"SteamVR executable not found: {steamvr_exe}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start SteamVR: {e}")
            return False
