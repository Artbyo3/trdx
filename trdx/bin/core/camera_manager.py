"""
Camera Manager - Multi-Camera Support for TRDX
Handles up to 4 cameras for Full Body Tracking
Supports physical cameras, IP cameras, and HTTP streams
"""

import cv2
import numpy as np
import threading
import time
import requests
import psutil
from typing import Dict, List, Optional, Tuple, Union
from PySide6.QtCore import QObject, Signal, QTimer, QThread
import logging
from urllib.parse import urlparse

class CameraThread(QThread):
    """Thread for handling individual camera capture"""
    
    frame_captured = Signal(int, np.ndarray)  # camera_id, frame
    error_occurred = Signal(int, str)  # camera_id, error_message
    
    def __init__(self, camera_id: int, source: Union[int, str]):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.cap = None
        self.running = False
        self.fps = 30
        self.resolution = (640, 480)
        self.mirror = False
        self.logger = logging.getLogger(f"Camera{camera_id}")
        self.is_url_source = isinstance(source, str)
        self.is_http_stream = False
        
        # OPTIMIZED: Frame buffering for smoother processing
        self.frame_buffer_size = 2  # Reduced buffer size for lower latency
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        self.skip_frames = 0  # Frame skipping for performance
        self.last_frame_time = 0  # Track frame timing
        self.frame_interval = 1.0 / self.fps  # Calculate frame interval
        
        if self.is_url_source:
            parsed = urlparse(str(source))
            self.is_http_stream = parsed.scheme in ['http', 'https']
            
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            if self.is_http_stream:
                return self._initialize_http_stream()
            else:
                return self._initialize_cv2_capture()
                
        except (cv2.error, OSError, ValueError) as e:
            self.error_occurred.emit(self.camera_id, f"Camera initialization error: {str(e)}")
            return False
        except Exception as e:
            self.error_occurred.emit(self.camera_id, f"Unexpected error during camera initialization: {str(e)}")
            return False
    
    def _initialize_cv2_capture(self) -> bool:
        """Initialize standard CV2 capture (physical cameras, RTSP, etc.) with timeout protection"""
        try:
            # Simple and direct camera initialization
            self.logger.info(f"Camera {self.camera_id}: Trying to open camera {self.source}...")
            
            # Try DirectShow first (most reliable on Windows)
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                self.logger.warning(f"Camera {self.camera_id}: DirectShow failed, trying default...")
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Camera {self.camera_id}: Failed to open camera {self.source}")
                self.error_occurred.emit(self.camera_id, f"Failed to open camera {self.source}")
                return False
            
            # Test if we can read a frame with timeout
            test_success = [False]
            test_error = [None]
            
            def test_frame_read():
                try:
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        test_success[0] = True
                    else:
                        test_error[0] = "Failed to read frame"
                except Exception as e:
                    test_error[0] = str(e)
            
            # Test frame reading with timeout
            import threading
            test_thread = threading.Thread(target=test_frame_read)
            test_thread.daemon = True
            test_thread.start()
            
            # Wait for test with timeout (5 seconds)
            test_thread.join(timeout=5.0)
            
            if test_thread.is_alive():
                self.logger.error(f"Camera {self.camera_id}: Frame read test timed out")
                self.cap.release()
                self.cap = None
                self.error_occurred.emit(self.camera_id, f"Camera {self.source} test timed out")
                return False
            
            if not test_success[0]:
                self.logger.error(f"Camera {self.camera_id}: Can't read frames from camera {self.source}")
                self.cap.release()
                self.cap = None
                
                # Check for common interfering processes
                self._kill_camera_processes()
                
                self.error_occurred.emit(self.camera_id, f"Camera {self.source} detected but can't read frames. Close other apps using camera!")
                return False
                
            # Set camera properties with error checking
            try:
                # Set buffer size to 1 for real-time capture
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Additional properties for better performance
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                
                # Verify the settings
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
                
                self.logger.info(f"Camera {self.camera_id} properties: {actual_width}x{actual_height} @ {actual_fps} FPS, buffer: {buffer_size}")
            except Exception as prop_e:
                self.logger.warning(f"Camera {self.camera_id}: Failed to set properties: {prop_e}")
            
            self.logger.info(f"Camera {self.camera_id} initialized successfully (source: {self.source})")
            return True
            
        except (cv2.error, OSError, ValueError) as e:
            self.error_occurred.emit(self.camera_id, f"CV2 capture initialization error: {str(e)}")
            return False
        except Exception as e:
            self.error_occurred.emit(self.camera_id, f"Unexpected error during CV2 capture initialization: {str(e)}")
            return False
    
    def _initialize_http_stream(self) -> bool:
        """Initialize HTTP/HTTPS stream (like IP camera MJPEG streams)"""
        try:
            # Test the URL first
            response = requests.get(self.source, stream=True, timeout=5)
            if response.status_code != 200:
                self.error_occurred.emit(self.camera_id, f"HTTP stream returned status {response.status_code}")
                return False
                
            # Try to open with OpenCV (works for many IP cameras)
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.logger.info(f"Camera {self.camera_id} initialized via OpenCV (HTTP: {self.source})")
                return True
            
            # If OpenCV fails, we'll handle it manually in the run loop
            self.logger.info(f"Camera {self.camera_id} will use manual HTTP handling (URL: {self.source})")
            return True
            
        except Exception as e:
            self.error_occurred.emit(self.camera_id, f"Failed to initialize HTTP stream: {e}")
            return False
    
    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        self.resolution = (width, height)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_fps(self, fps: int):
        """Set camera FPS"""
        self.fps = fps
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def run(self):
        """Main camera capture loop"""
        if not self.initialize_camera():
            return
            
        self.running = True
        frame_time = 1.0 / self.fps
        frame_skip_counter = 0
        
        while self.running:
            try:
                if self.is_http_stream and (not self.cap or not self.cap.isOpened()):
                    self._capture_http_frame()
                else:
                    # OPTIMIZED: Frame capture with buffering
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        # OPTIMIZED: Skip frames for performance if needed
                        frame_skip_counter += 1
                        if frame_skip_counter <= self.skip_frames:
                            continue
                        frame_skip_counter = 0
                        
                        # Apply mirror if enabled
                        if self.mirror:
                            frame = cv2.flip(frame, 1)
                        
                        # OPTIMIZED: Add to buffer instead of direct emit
                        with self.buffer_lock:
                            if len(self.frame_buffer) >= self.frame_buffer_size:
                                self.frame_buffer.pop(0)  # Remove oldest
                            self.frame_buffer.append(frame)
                        
                        # Emit latest frame
                        self.frame_captured.emit(self.camera_id, frame)
                    else:
                        # Try to recover the camera connection
                        if self.cap and self.cap.isOpened():
                            self.error_occurred.emit(self.camera_id, f"Camera {self.camera_id}: Failed to read frame (camera connected but no data)")
                        else:
                            self.error_occurred.emit(self.camera_id, f"Camera {self.camera_id}: Connection lost")
                        
                        # Try to reinitialize once
                        if not hasattr(self, '_recovery_attempted'):
                            self._recovery_attempted = True
                            self.logger.warning(f"Camera {self.camera_id}: Attempting recovery...")
                            if self.cap:
                                self.cap.release()
                            if self.initialize_camera():
                                self.logger.info(f"Camera {self.camera_id}: Recovery successful")
                                continue
                        
                        break
                        
                time.sleep(frame_time)
                
            except Exception as e:
                self.error_occurred.emit(self.camera_id, str(e))
                break
        
        self.cleanup()
    
    def _capture_http_frame(self):
        """Capture frame from HTTP stream manually"""
        try:
            response = requests.get(self.source, stream=True, timeout=2)
            if response.status_code == 200:
                # Convert response content to numpy array
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Apply mirror if enabled
                    if self.mirror:
                        frame = cv2.flip(frame, 1)  # Flip horizontally
                    self.frame_captured.emit(self.camera_id, frame)
                else:
                    self.error_occurred.emit(self.camera_id, "Failed to decode HTTP image")
            else:
                self.error_occurred.emit(self.camera_id, f"HTTP error: {response.status_code}")
                
        except Exception as e:
            self.error_occurred.emit(self.camera_id, f"HTTP capture error: {e}")
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Clear frame buffer
            with self.buffer_lock:
                self.frame_buffer.clear()
            
            self.logger.info(f"Camera {self.camera_id} cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up camera {self.camera_id}: {e}")
    
    def _kill_camera_processes(self):
        """Check for common processes that use cameras and suggest closing them"""
        camera_processes = [
            'Skype.exe', 'Teams.exe', 'Discord.exe', 'zoom.exe',
            'obs64.exe', 'obs32.exe', 'streamlabs obs.exe',
            'chrome.exe', 'firefox.exe', 'msedge.exe',
            'WindowsCamera.exe', 'Camera.exe'
        ]
        
        found_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'].lower() in [p.lower() for p in camera_processes]:
                    found_processes.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if found_processes:
            self.logger.warning(f"Camera {self.camera_id}: Found apps that may use camera: {found_processes}")
            self.logger.warning(f"Camera {self.camera_id}: Please close these apps manually and try again")

class CameraManager(QObject):
    """Manager for multiple cameras (up to 4)"""
    
    frame_ready = Signal(int, np.ndarray)  # camera_id, frame
    status_changed = Signal(int, str, bool)  # camera_id, status, is_active
    all_cameras_ready = Signal(dict)  # {camera_id: frame}
    
    def __init__(self, max_cameras: int = 4):
        super().__init__()
        self.max_cameras = max_cameras
        self.cameras: Dict[int, CameraThread] = {}
        self.active_cameras: Dict[int, bool] = {}
        self.latest_frames: Dict[int, np.ndarray] = {}
        self.camera_configs: Dict[int, dict] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize default configurations
        for i in range(max_cameras):
            self.camera_configs[i] = {
                'source': i,  # Default to camera index
                'resolution': (640, 480),
                'fps': 30,
                'enabled': False,
                'type': 'physical'  # 'physical', 'ip', 'http'
            }
    
    def get_available_cameras(self) -> List[int]:
        """Detect available physical cameras with timeout protection"""
        available = []
        
        for i in range(8):  # Check first 8 camera indices
            try:
                # Use timeout for camera detection
                import threading
                import time
                
                detection_success = [False]
                detection_error = [None]
                
                def detect_camera():
                    try:
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            # Quick test read to ensure camera is working
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                detection_success[0] = True
                            cap.release()
                        else:
                            detection_error[0] = "Camera not opened"
                    except Exception as e:
                        detection_error[0] = str(e)
                
                # Start detection in separate thread with timeout
                detect_thread = threading.Thread(target=detect_camera)
                detect_thread.daemon = True
                detect_thread.start()
                
                # Wait for detection with timeout (3 seconds per camera)
                detect_thread.join(timeout=3.0)
                
                if detect_thread.is_alive():
                    self.logger.warning(f"Camera {i} detection timed out - skipping")
                    continue
                
                if detection_success[0]:
                    available.append(i)
                    self.logger.debug(f"Camera {i} detected successfully")
                else:
                    if detection_error[0]:
                        self.logger.debug(f"Camera {i} not available: {detection_error[0]}")
                    
            except Exception as e:
                self.logger.debug(f"Exception detecting camera {i}: {e}")
                continue
        
        return available
    
    def add_camera(self, camera_id: int, source: Union[int, str], 
                   resolution: Tuple[int, int] = (640, 480), fps: int = 30) -> bool:
        """Add a camera to the manager
        
        Args:
            camera_id: Unique identifier for the camera (0-3)
            source: Camera source - int for physical camera index, str for URL/path
            resolution: Video resolution tuple
            fps: Frames per second
        
        Examples:
            add_camera(0, 0)  # Physical camera 0
            add_camera(1, "http://192.168.0.16:8080/img_stream.jpeg")  # IP camera
            add_camera(2, "rtsp://192.168.1.100:554/stream")  # RTSP stream
        """
        if camera_id >= self.max_cameras:
            self.logger.error(f"Camera ID {camera_id} exceeds maximum ({self.max_cameras})")
            return False
            
        if camera_id in self.cameras:
            self.logger.warning(f"Camera {camera_id} already exists")
            return False
        
        # Determine camera type
        camera_type = 'physical'
        if isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme in ['http', 'https']:
                camera_type = 'http'
            elif parsed.scheme in ['rtsp', 'rtmp']:
                camera_type = 'ip'
            else:
                camera_type = 'file'  # File path or other
        
        # Update configuration
        self.camera_configs[camera_id] = {
            'source': source,
            'resolution': resolution,
            'fps': fps,
            'enabled': True,
            'type': camera_type
        }
        
        # Create camera thread
        camera_thread = CameraThread(camera_id, source)
        camera_thread.set_resolution(*resolution)
        camera_thread.set_fps(fps)
        
        # Connect signals
        camera_thread.frame_captured.connect(self._on_frame_captured)
        camera_thread.error_occurred.connect(self._on_camera_error)
        
        self.cameras[camera_id] = camera_thread
        self.active_cameras[camera_id] = False
        
        self.logger.info(f"Camera {camera_id} added (source: {source}, type: {camera_type})")
        return True
    
    def start_camera(self, camera_id: int) -> bool:
        """Start a specific camera"""
        if camera_id not in self.cameras:
            self.logger.error(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        if not camera.isRunning():
            camera.start()
            self.active_cameras[camera_id] = True
            self.status_changed.emit(camera_id, "Starting", True)
            self.logger.info(f"Camera {camera_id} started")
            return True
        else:
            self.logger.warning(f"Camera {camera_id} already running")
            return False
    
    def stop_camera(self, camera_id: int) -> bool:
        """Stop a specific camera"""
        if camera_id not in self.cameras:
            return False
        
        camera = self.cameras[camera_id]
        if camera.isRunning():
            camera.stop()
            camera.wait()  # Wait for thread to finish
            self.active_cameras[camera_id] = False
            self.status_changed.emit(camera_id, "Stopped", False)
            self.logger.info(f"Camera {camera_id} stopped")
            return True
        return False
    
    def start_all_cameras(self):
        """Start all configured cameras"""
        for camera_id in self.cameras:
            if self.camera_configs[camera_id]['enabled']:
                self.start_camera(camera_id)
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        for camera_id in list(self.cameras.keys()):
            self.stop_camera(camera_id)
    
    def remove_camera(self, camera_id: int):
        """Remove a camera from the manager"""
        if camera_id in self.cameras:
            self.stop_camera(camera_id)
            del self.cameras[camera_id]
            del self.active_cameras[camera_id]
            if camera_id in self.latest_frames:
                del self.latest_frames[camera_id]
            self.camera_configs[camera_id]['enabled'] = False
            self.logger.info(f"Camera {camera_id} removed")
    
    def get_camera_status(self, camera_id: int) -> dict:
        """Get status of a specific camera"""
        if camera_id not in self.cameras:
            return {'exists': False}
        
        camera = self.cameras[camera_id]
        return {
            'exists': True,
            'running': camera.isRunning(),
            'active': self.active_cameras.get(camera_id, False),
            'config': self.camera_configs[camera_id],
            'has_frame': camera_id in self.latest_frames
        }
    
    def get_all_status(self) -> dict:
        """Get status of all cameras"""
        status = {}
        for camera_id in range(self.max_cameras):
            status[camera_id] = self.get_camera_status(camera_id)
        return status
    
    def _on_frame_captured(self, camera_id: int, frame: np.ndarray):
        """Handle frame captured from camera"""
        self.latest_frames[camera_id] = frame
        
        # DEBUG: Log frame capture occasionally
        if hasattr(self, '_frame_count'):
            self._frame_count[camera_id] = self._frame_count.get(camera_id, 0) + 1
        else:
            self._frame_count = {camera_id: 1}
        
        # OPTIMIZED: Reduced logging frequency
        if self._frame_count[camera_id] % 300 == 1:  # Much less frequent
            self.logger.debug(f"Cam {camera_id}: {self._frame_count[camera_id]} frames")
        
        self.frame_ready.emit(camera_id, frame)
        
        # Check if we have frames from all active cameras
        active_count = sum(1 for active in self.active_cameras.values() if active)
        if len(self.latest_frames) == active_count and active_count > 0:
            self.all_cameras_ready.emit(self.latest_frames.copy())
    
    def _on_camera_error(self, camera_id: int, error_message: str):
        """Handle camera error"""
        self.logger.error(f"Camera {camera_id} error: {error_message}")
        self.active_cameras[camera_id] = False
        self.status_changed.emit(camera_id, f"Error: {error_message}", False)
    
    def get_latest_frames(self) -> Dict[int, np.ndarray]:
        """Get latest frames from all cameras"""
        return self.latest_frames.copy()
    
    def configure_camera(self, camera_id: int, config: dict):
        """Configure camera settings"""
        if camera_id in self.camera_configs:
            self.camera_configs[camera_id].update(config)
            
            # Apply settings if camera is running
            if camera_id in self.cameras:
                camera = self.cameras[camera_id]
                if 'resolution' in config:
                    camera.set_resolution(*config['resolution'])
                if 'fps' in config:
                    camera.set_fps(config['fps'])
    
    def set_camera_mirror(self, camera_id: int, mirror: bool):
        """Set mirror setting for a camera"""
        if camera_id in self.cameras:
            camera_thread = self.cameras[camera_id]
            camera_thread.mirror = mirror
            self.logger.info(f"Camera {camera_id} mirror set to: {mirror}")
        else:
            self.logger.warning(f"Camera {camera_id} not found for mirror setting")
