"""
TRDX Application - Main Application Class
Multi-Camera Full Body Tracking for SteamVR
"""

import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QIcon

from .ui.main_window import MainWindow
from .core.camera_manager import CameraManager
from .core.pose_processor import PoseProcessor
from .core.steamvr_driver import SteamVRDriver
from .core.config_manager import ConfigManager
from .core.performance_monitor import performance_monitor

# Configure logging with filtering to reduce console spam
import os
class LessSpammyFilter(logging.Filter):
    def filter(self, record):
        # Only show INFO/DEBUG for TRDX core modules if enabled
        show_debug = os.environ.get('TRDX_DEBUG', '0') == '1'
        
        # Filtrar warnings específicos de MediaPipe
        if record.name == 'bin.core.pose_processor' and 'MediaPipe not available' in record.getMessage():
            return False  # No mostrar estos warnings
        
        if record.levelno >= logging.WARNING:
            return True
        if show_debug:
            return True
        # Suppress most INFO/DEBUG logs except for startup/config
        if record.name.startswith('trdx') and record.levelno == logging.INFO:
            message = record.getMessage().lower()
            if any(keyword in message for keyword in ['initialized', 'started', 'connected', 'ready']):
                return True
        return False

# Configure logging with better formatting
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.addFilter(LessSpammyFilter())

# Clear existing handlers and set up new one
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# Set specific logger levels for noisy modules
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Solo errores, no warnings
logging.getLogger('cv2').setLevel(logging.ERROR)  # Solo errores, no warnings

class TRDXApplication:
    """Main TRDX Application"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.app = None
        self.main_window = None
        self.camera_manager = None
        self.pose_processor = None
        self.steamvr_driver = None
        self.config_manager = None
        
    def initialize(self):
        """Initialize all application components"""
        try:
            self.logger.info("Initializing TRDX Application...")
            
            # Create Qt Application
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("TRDX")
            self.app.setApplicationVersion("1.0.0")
            self.app.setOrganizationName("TRDX Project")
            
            # Initialize core components
            self.config_manager = ConfigManager()
            self.camera_manager = CameraManager(max_cameras=4)
            self.pose_processor = PoseProcessor(self.config_manager)
            self.steamvr_driver = SteamVRDriver()
            
            # Connect to SteamVR automatically
            if self.steamvr_driver.connect_to_steamvr():
                self.logger.info("✓ Connected to SteamVR driver successfully")
                
                # Initialize default trackers
                tracker_roles = ["TrackerRole_Waist", "TrackerRole_LeftFoot", "TrackerRole_RightFoot"]
                for i, role in enumerate(tracker_roles):
                    tracker_id = self.steamvr_driver.add_tracker(role)
                    if tracker_id >= 0:
                        self.logger.info(f"✓ Added tracker {tracker_id} ({role})")
                    else:
                        self.logger.warning(f"⚠ Failed to add tracker {i} ({role})")
            else:
                self.logger.warning("⚠ Could not connect to SteamVR driver - trackers may not work")
            
            # Load configuration into components
            config = self.config_manager.config
            self.pose_processor.load_config(config)
            
            # Start headset tracking if available
            if hasattr(self.pose_processor, 'start_headset_tracking'):
                headset_started = self.pose_processor.start_headset_tracking()
                if headset_started:
                    self.logger.info("✓ Headset reference tracking started successfully")
                else:
                    self.logger.info("ℹ Headset reference tracking not available - using camera-only mode")
            
            # Start SteamVR integration as fallback
            if hasattr(self.pose_processor, 'start_steamvr_integration'):
                steamvr_started = self.pose_processor.start_steamvr_integration()
                if steamvr_started:
                    self.logger.info("✓ SteamVR integration started as secondary reference")
                else:
                    self.logger.info("ℹ SteamVR integration not available")
            
            # Create main window
            self.main_window = MainWindow(
                camera_manager=self.camera_manager,
                pose_processor=self.pose_processor,
                steamvr_driver=self.steamvr_driver,
                config_manager=self.config_manager
            )
            
            # Connect signals
            self._connect_signals()
            
            # Start cameras automatically
            self._start_cameras()
            
            # Start SteamVR data bridge if enabled
            self._start_vr_bridge()
            
            # OPTIMIZED: Start performance monitoring
            performance_monitor.start_monitoring()
            self.logger.info("TRDX Application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def initialize_without_qapp_safe(self):
        """Initialize all application components without creating QApplication - SAFE VERSION"""
        try:
            self.logger.info("Initializing TRDX Application (SAFE MODE)...")
            
            # Use existing QApplication instance
            self.app = QApplication.instance()
            if not self.app:
                self.logger.error("No QApplication instance found")
                return False
            
            # Initialize core components
            self.config_manager = ConfigManager()
            self.camera_manager = CameraManager(max_cameras=4)
            self.pose_processor = PoseProcessor(self.config_manager)
            self.steamvr_driver = SteamVRDriver()
            
            # Connect to SteamVR automatically (without threading)
            if self.steamvr_driver.connect_to_steamvr():
                self.logger.info("✓ Connected to SteamVR driver successfully")
                
                # Initialize default trackers
                tracker_roles = ["TrackerRole_Waist", "TrackerRole_LeftFoot", "TrackerRole_RightFoot"]
                for i, role in enumerate(tracker_roles):
                    tracker_id = self.steamvr_driver.add_tracker(role)
                    if tracker_id >= 0:
                        self.logger.info(f"✓ Added tracker {tracker_id} ({role})")
                    else:
                        self.logger.warning(f"⚠ Failed to add tracker {i} ({role})")
            else:
                self.logger.warning("⚠ Could not connect to SteamVR driver - trackers may not work")
            
            # Load configuration into components
            config = self.config_manager.config
            self.pose_processor.load_config(config)
            
            # Start headset tracking if available
            if hasattr(self.pose_processor, 'start_headset_tracking'):
                headset_started = self.pose_processor.start_headset_tracking()
                if headset_started:
                    self.logger.info("✓ Headset reference tracking started successfully")
                else:
                    self.logger.info("ℹ Headset reference tracking not available - using camera-only mode")
            
            # Start SteamVR integration as fallback
            if hasattr(self.pose_processor, 'start_steamvr_integration'):
                steamvr_started = self.pose_processor.start_steamvr_integration()
                if steamvr_started:
                    self.logger.info("✓ SteamVR integration started as secondary reference")
                else:
                    self.logger.info("ℹ SteamVR integration not available")
            
            # Create main window
            self.main_window = MainWindow(
                camera_manager=self.camera_manager,
                pose_processor=self.pose_processor,
                steamvr_driver=self.steamvr_driver,
                config_manager=self.config_manager
            )
            
            # Connect signals
            self._connect_signals()
            
            # CORRECCIÓN: NO iniciar cámaras automáticamente - esto causa problemas de threading
            # self._start_cameras()  # COMENTADO - causa problemas de UI
            
            # Start SteamVR data bridge if enabled
            self._start_vr_bridge()
            
            # OPTIMIZED: Start performance monitoring
            performance_monitor.start_monitoring()
            self.logger.info("TRDX Application initialized successfully (SAFE MODE)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def initialize_in_main_thread(self):
        """Initialize all application components in the main thread - CORREGIDO"""
        try:
            self.logger.info("Initializing TRDX Application in main thread...")
            
            # Use existing QApplication instance
            self.app = QApplication.instance()
            if not self.app:
                self.logger.error("No QApplication instance found")
                return False
            
            # Initialize core components
            self.config_manager = ConfigManager()
            self.camera_manager = CameraManager(max_cameras=4)
            self.pose_processor = PoseProcessor(self.config_manager)
            self.steamvr_driver = SteamVRDriver()
            
            # Connect to SteamVR automatically (without threading)
            if self.steamvr_driver.connect_to_steamvr():
                self.logger.info("✓ Connected to SteamVR driver successfully")
                
                # Initialize default trackers
                tracker_roles = ["TrackerRole_Waist", "TrackerRole_LeftFoot", "TrackerRole_RightFoot"]
                for i, role in enumerate(tracker_roles):
                    tracker_id = self.steamvr_driver.add_tracker(role)
                    if tracker_id >= 0:
                        self.logger.info(f"✓ Added tracker {tracker_id} ({role})")
                    else:
                        self.logger.warning(f"⚠ Failed to add tracker {i} ({role})")
            else:
                self.logger.warning("⚠ Could not connect to SteamVR driver - trackers may not work")
            
            # Load configuration into components
            config = self.config_manager.config
            self.pose_processor.load_config(config)
            
            # Start headset tracking if available
            if hasattr(self.pose_processor, 'start_headset_tracking'):
                headset_started = self.pose_processor.start_headset_tracking()
                if headset_started:
                    self.logger.info("✓ Headset reference tracking started successfully")
                else:
                    self.logger.info("ℹ Headset reference tracking not available - using camera-only mode")
            
            # Start SteamVR integration as fallback
            if hasattr(self.pose_processor, 'start_steamvr_integration'):
                steamvr_started = self.pose_processor.start_steamvr_integration()
                if steamvr_started:
                    self.logger.info("✓ SteamVR integration started as secondary reference")
                else:
                    self.logger.info("ℹ SteamVR integration not available")
            
            # Create main window
            self.main_window = MainWindow(
                camera_manager=self.camera_manager,
                pose_processor=self.pose_processor,
                steamvr_driver=self.steamvr_driver,
                config_manager=self.config_manager
            )
            
            # Connect signals
            self._connect_signals()
            
            # CORRECCIÓN: NO iniciar cámaras automáticamente - esto causa problemas de threading
            # self._start_cameras()  # COMENTADO - causa problemas de UI
            
            # Start SteamVR data bridge if enabled
            self._start_vr_bridge()
            
            # OPTIMIZED: Start performance monitoring
            performance_monitor.start_monitoring()
            self.logger.info("TRDX Application initialized successfully in main thread")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def initialize_minimal(self):
        """Initialize all application components in minimal mode - no SteamVR, no camera detection"""
        try:
            self.logger.info("Initializing TRDX Application in minimal mode...")
            
            # Use existing QApplication instance
            self.app = QApplication.instance()
            if not self.app:
                self.logger.error("No QApplication instance found")
                return False
            
            # Initialize core components
            self.config_manager = ConfigManager()
            self.camera_manager = CameraManager(max_cameras=4)
            self.pose_processor = PoseProcessor(self.config_manager)
            self.steamvr_driver = SteamVRDriver()
            
            # CORRECCIÓN: NO conectar a SteamVR automáticamente
            # if self.steamvr_driver.connect_to_steamvr():
            #     self.logger.info("✓ Connected to SteamVR driver successfully")
            # else:
            #     self.logger.warning("⚠ Could not connect to SteamVR driver - trackers may not work")
            
            # Load configuration into components
            config = self.config_manager.config
            self.pose_processor.load_config(config)
            
            # CORRECCIÓN: NO iniciar headset tracking automáticamente
            # if hasattr(self.pose_processor, 'start_headset_tracking'):
            #     headset_started = self.pose_processor.start_headset_tracking()
            #     if headset_started:
            #         self.logger.info("✓ Headset reference tracking started successfully")
            #     else:
            #         self.logger.info("ℹ Headset reference tracking not available - using camera-only mode")
            
            # CORRECCIÓN: NO iniciar SteamVR integration automáticamente
            # if hasattr(self.pose_processor, 'start_steamvr_integration'):
            #     steamvr_started = self.pose_processor.start_steamvr_integration()
            #     if steamvr_started:
            #         self.logger.info("✓ SteamVR integration started as secondary reference")
            #     else:
            #         self.logger.info("ℹ SteamVR integration not available")
            
            # Create main window
            self.main_window = MainWindow(
                camera_manager=self.camera_manager,
                pose_processor=self.pose_processor,
                steamvr_driver=self.steamvr_driver,
                config_manager=self.config_manager
            )
            
            # Connect signals
            self._connect_signals()
            
            # CORRECCIÓN: NO iniciar cámaras automáticamente
            # self._start_cameras()  # COMENTADO - causa problemas de UI
            
            # CORRECCIÓN: NO iniciar SteamVR bridge automáticamente
            # self._start_vr_bridge()
            
            # OPTIMIZED: Start performance monitoring
            performance_monitor.start_monitoring()
            self.logger.info("TRDX Application initialized successfully in minimal mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def run_in_main_thread(self):
        """Run the application in the main thread - CORREGIDO"""
        try:
            # Show main window
            self.main_window.show()
            
            # CORRECCIÓN: NO iniciar cámaras automáticamente - el usuario debe hacerlo manualmente
            # self._start_cameras_async()  # COMENTADO - causa problemas de UI
            
            # Start application event loop
            self.logger.info("Starting TRDX Application in main thread...")
            return self.app.exec()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            return 1
        finally:
            self.cleanup()
    
    def _start_cameras(self):
        """Start cameras automatically with improved error handling and timeouts"""
        try:
            self.logger.info("Starting cameras automatically...")
            
            # Get available cameras with timeout
            self.logger.info("Detecting available cameras...")
            available_cameras = self.camera_manager.get_available_cameras()
            self.logger.info(f"Found {len(available_cameras)} available cameras: {available_cameras}")
            
            if not available_cameras:
                self.logger.warning("⚠️ NO CAMERAS DETECTED - Running in camera-less mode")
                self.logger.info("ℹ️ Application will start without cameras")
                return  # Continue without cameras
            
            # Only add cameras that actually exist with timeout protection
            cameras_added = 0
            for i, camera_index in enumerate(available_cameras):
                if i >= 2:  # Limit to first 2 available cameras
                    break
                
                self.logger.info(f"🔍 Attempting to initialize camera {i} (physical index {camera_index})...")
                
                # Add timeout protection for camera initialization
                try:
                    import threading
                    import time
                    
                    # Initialize camera with timeout
                    init_success = [False]
                    init_error = [None]
                    
                    def init_camera():
                        try:
                            if self.camera_manager.add_camera(i, camera_index):
                                init_success[0] = True
                            else:
                                init_error[0] = "add_camera returned False"
                        except Exception as e:
                            init_error[0] = str(e)
                    
                    # Start initialization in separate thread with timeout
                    init_thread = threading.Thread(target=init_camera)
                    init_thread.daemon = True
                    init_thread.start()
                    
                    # Wait for initialization with timeout (10 seconds)
                    init_thread.join(timeout=10.0)
                    
                    if init_thread.is_alive():
                        self.logger.error(f"❌ Camera {i} initialization timed out after 10 seconds")
                        self.logger.error("💡 POSSIBLE CAUSES:")
                        self.logger.error("   • Camera is being used by another application")
                        self.logger.error("   • Camera driver issues")
                        self.logger.error("   • Camera hardware problems")
                        continue
                    
                    if init_success[0]:
                        self.logger.info(f"✓ Added camera {i} (physical camera {camera_index})")
                        cameras_added += 1
                    else:
                        self.logger.error(f"❌ Failed to add camera {i} (physical camera {camera_index})")
                        if init_error[0]:
                            self.logger.error(f"   Error: {init_error[0]}")
                        self.logger.error("💡 POSSIBLE CAUSES:")
                        self.logger.error("   • Another application is using the camera")
                        self.logger.error("   • Camera driver issues")
                        self.logger.error("   • Insufficient permissions")
                        
                except Exception as e:
                    self.logger.error(f"❌ Exception during camera {i} initialization: {e}")
                    continue
            
            # Start all cameras with timeout protection
            if cameras_added > 0:
                self.logger.info(f"Starting {cameras_added} cameras...")
                
                # Start cameras with timeout
                start_success = [False]
                start_error = [None]
                
                def start_cameras():
                    try:
                        self.camera_manager.start_all_cameras()
                        start_success[0] = True
                    except Exception as e:
                        start_error[0] = str(e)
                
                # Start cameras in separate thread with timeout
                start_thread = threading.Thread(target=start_cameras)
                start_thread.daemon = True
                start_thread.start()
                
                # Wait for camera start with timeout (15 seconds)
                start_thread.join(timeout=15.0)
                
                if start_thread.is_alive():
                    self.logger.error("❌ Camera start timed out after 15 seconds")
                    self.logger.warning("⚠️ Continuing without cameras - some features may not work")
                elif start_success[0]:
                    self.logger.info(f"✓ Started {cameras_added} cameras successfully")
                else:
                    self.logger.error(f"❌ Failed to start cameras: {start_error[0]}")
                    self.logger.warning("⚠️ Continuing without cameras - some features may not work")
            else:
                self.logger.warning("⚠️ NO CAMERAS WERE SUCCESSFULLY INITIALIZED!")
                self.logger.warning("ℹ️ Application will continue without cameras")
                self.logger.info("🛠️ MANUAL SOLUTIONS:")
                self.logger.info("   1. Close applications that might use camera:")
                self.logger.info("      - Skype, Microsoft Teams, Discord")
                self.logger.info("      - OBS Studio, Streamlabs")
                self.logger.info("      - Camera app, Zoom")
                self.logger.info("   2. Check Windows Settings > Privacy > Camera")
                self.logger.info("   3. Update camera drivers in Device Manager")
                self.logger.info("   4. Try different USB port if using external camera")
                
        except Exception as e:
            self.logger.error(f"Error starting cameras: {e}")
            self.logger.warning("⚠️ Continuing without cameras due to error")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def _start_vr_bridge(self):
        """Start the SteamVR data bridge in background"""
        try:
            import subprocess
            import os
            
            # Check if SteamVR bridge should be started
            steamvr_config = self.config_manager.config.get('steamvr', {})
            vr_integration = steamvr_config.get('vr_integration', {})
            
            if not vr_integration.get('enable_headset_reference', True):
                return
            
            # Start the bridge as a background process
            bridge_path = os.path.join(os.path.dirname(__file__), 'steamvr_bridge.py')
            if os.path.exists(bridge_path):
                self.logger.info("Starting SteamVR data bridge...")
                subprocess.Popen([sys.executable, bridge_path], 
                               creationflags=subprocess.CREATE_NO_WINDOW)
                self.logger.info("SteamVR bridge started in background")
            else:
                self.logger.warning(f"SteamVR bridge not found at {bridge_path}")
                
        except Exception as e:
            self.logger.warning(f"Could not start SteamVR bridge: {e}")
    
    def _connect_signals(self):
        """Connect signals between components"""
        # Camera to pose processor
        self.camera_manager.frame_ready.connect(self.pose_processor.process_frame)
        
        # Pose processor to SteamVR driver
        self.pose_processor.pose_data_ready.connect(self.steamvr_driver.update_trackers)
        
        # Status updates to main window
        self.camera_manager.status_changed.connect(self.main_window.update_camera_status)
        self.steamvr_driver.status_changed.connect(self.main_window.update_steamvr_status)
    
    def run(self):
        """Run the application"""
        if not self.initialize():
            return 1
            
        try:
            # Show main window
            self.main_window.show()
            
            # Start application event loop
            self.logger.info("Starting TRDX Application...")
            return self.app.exec()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            return 1
        finally:
            self.cleanup()
    
    def run_without_qapp_safe(self):
        """Run the application without creating new QApplication - SAFE VERSION"""
        try:
            # Show main window
            self.main_window.show()
            
            # CORRECCIÓN: NO iniciar cámaras automáticamente - el usuario debe hacerlo manualmente
            # self._start_cameras_async()  # COMENTADO - causa problemas de UI
            
            # Start application event loop
            self.logger.info("Starting TRDX Application (SAFE MODE)...")
            return self.app.exec()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            return 1
        finally:
            self.cleanup()
    
    def _start_cameras_async(self):
        """Start cameras asynchronously after main window is shown"""
        try:
            from PySide6.QtCore import QTimer
            
            def start_cameras_async():
                self.logger.info("🔄 Starting cameras asynchronously...")
                self._start_cameras()
            
            # Use QTimer.singleShot to run in main thread after a delay
            QTimer.singleShot(1000, start_cameras_async)
            
            self.logger.info("ℹ Camera initialization scheduled for main thread")
            
        except Exception as e:
            self.logger.error(f"Error scheduling camera initialization: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up TRDX Application...")
        
        if self.camera_manager:
            self.camera_manager.stop_all_cameras()
        
        if self.pose_processor:
            # Stop headset tracking
            if hasattr(self.pose_processor, 'stop_headset_tracking'):
                self.pose_processor.stop_headset_tracking()
            
            # Stop SteamVR integration
            if hasattr(self.pose_processor, 'stop_steamvr_integration'):
                self.pose_processor.stop_steamvr_integration()
        
        if self.steamvr_driver:
            self.steamvr_driver.disconnect()
            
        # OPTIMIZED: Stop performance monitoring
        performance_monitor.stop_monitoring()
        self.logger.info("TRDX Application cleanup complete")
