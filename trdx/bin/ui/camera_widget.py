"""
Camera Widget - Individual camera display and control
"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QCheckBox, QComboBox, QFrame, QLineEdit, QDialog,
    QDialogButtonBox, QFormLayout, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPixmap, QImage, QFont
import logging

class CameraWidget(QWidget):
    """Widget for displaying and controlling a single camera"""
    
    camera_enabled = Signal(int, bool)  # camera_id, enabled
    camera_configured = Signal(int, dict)  # camera_id, config
    
    def __init__(self, camera_id: int, camera_manager, config_manager):
        super().__init__()
        self.camera_id = camera_id
        self.camera_manager = camera_manager
        self.config_manager = config_manager
        self.logger = logging.getLogger(f"CameraWidget{camera_id}")
        
        # State
        self.is_active = False
        self.current_frame = None
        self.current_pose = None
        self.current_source = None
        
        # Initialize UI
        self.init_ui()
        self.load_config()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Camera group
        self.camera_group = QGroupBox(f"Camera {self.camera_id}")
        camera_layout = QVBoxLayout(self.camera_group)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setMaximumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 1px solid #555555;
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No Video")
        camera_layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Enable/disable checkbox
        self.enable_cb = QCheckBox("Enable")
        self.enable_cb.toggled.connect(self.on_enable_toggled)
        controls_layout.addWidget(self.enable_cb)
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(150)
        self.populate_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        controls_layout.addWidget(self.camera_combo)
        
        # Configure button for advanced source setup
        self.config_btn = QPushButton("Config")
        self.config_btn.clicked.connect(self.show_source_config)
        controls_layout.addWidget(self.config_btn)
        
        # Start/stop button
        self.start_stop_btn = QPushButton("Start")
        self.start_stop_btn.clicked.connect(self.on_start_stop_clicked)
        self.start_stop_btn.setEnabled(False)
        controls_layout.addWidget(self.start_stop_btn)
        
        # Mirror button
        self.mirror_btn = QPushButton("Mirror")
        self.mirror_btn.setCheckable(True)
        self.mirror_btn.clicked.connect(self.on_mirror_clicked)
        controls_layout.addWidget(self.mirror_btn)
        
        camera_layout.addLayout(controls_layout)
        
        # Status and info
        info_layout = QHBoxLayout()
        
        # Status indicator
        self.status_label = QLabel("Inactive")
        self.status_label.setStyleSheet("color: #888888;")
        info_layout.addWidget(self.status_label)
        
        info_layout.addStretch()
        
        # FPS counter
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setStyleSheet("color: #888888;")
        info_layout.addWidget(self.fps_label)
        
        # Frame counter
        self.frame_label = QLabel("0 frames")
        self.frame_label.setStyleSheet("color: #888888;")
        info_layout.addWidget(self.frame_label)
        
        camera_layout.addLayout(info_layout)
        
        layout.addWidget(self.camera_group)
        
        # Apply styling
        self.apply_style()
    
    def apply_style(self):
        """Apply widget styling"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                min-width: 50px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #666666;
            }
        """)
    
    def populate_camera_list(self):
        """Populate the camera combo box with available cameras"""
        self.camera_combo.clear()
        
        # CORRECCIÓN: Comentar detección automática de cámaras para evitar timeouts
        # Get available physical cameras
        # available_cameras = self.camera_manager.get_available_cameras()
        
        # for camera_index in available_cameras:
        #     self.camera_combo.addItem(f"Physical Camera {camera_index}", camera_index)
        
        # Agregar opciones básicas sin detección automática
        self.camera_combo.addItem("Camera 0", 0)
        self.camera_combo.addItem("Camera 1", 1)
        self.camera_combo.addItem("Camera 2", 2)
        self.camera_combo.addItem("Camera 3", 3)
            
        # Add IP camera options
        self.camera_combo.addItem("HTTP Stream...", "http://")
        self.camera_combo.addItem("RTSP Stream...", "rtsp://")
        self.camera_combo.addItem("Custom Source...", -1)
    
    def show_source_config(self):
        """Show camera source configuration dialog"""
        # Simple input dialog for now
        from PySide6.QtWidgets import QInputDialog
        
        current_text = ""
        if self.current_source:
            current_text = str(self.current_source)
        
        text, ok = QInputDialog.getText(
            self, 
            'Camera Source Configuration', 
            'Enter camera source (camera index or URL):\n\n'
            'Examples:\n'
            '• Physical camera: 0, 1, 2...\n'
            '• HTTP stream: http://192.168.0.16:8080/img_stream.jpeg\n'
            '• RTSP stream: rtsp://192.168.1.100:554/stream',
            text=current_text
        )
        
        if ok and text:
            # Try to convert to int for physical cameras
            try:
                source = int(text)
            except ValueError:
                source = text.strip()
            
            self.set_camera_source(source)
    
    def set_camera_source(self, source):
        """Set the camera source and update UI"""
        self.current_source = source
        
        # Add to combo box if it's a custom source
        if isinstance(source, str):
            # Check if already exists
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == source:
                    self.camera_combo.setCurrentIndex(i)
                    return
            
            # Add new item
            if source.startswith(('http://', 'https://')):
                display_name = f"HTTP: {source[:30]}..."
            elif source.startswith(('rtsp://', 'rtmp://')):
                display_name = f"RTSP: {source[:30]}..."
            else:
                display_name = f"Custom: {source[:30]}..."
            
            # Insert before "Custom Source..."
            self.camera_combo.insertItem(
                self.camera_combo.count() - 1, 
                display_name, 
                source
            )
            self.camera_combo.setCurrentIndex(self.camera_combo.count() - 2)
        else:
            # Physical camera
            index = self.camera_combo.findData(source)
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)
    
    def load_config(self):
        """Load camera configuration"""
        config = self.config_manager.get_camera_config(self.camera_id)
        
        if config:
            self.enable_cb.setChecked(config.get('enabled', False))
            
            # Load source
            if 'source' in config:
                self.set_camera_source(config['source'])
            elif 'index' in config:  # Legacy support
                camera_index = config.get('index', self.camera_id)
                self.set_camera_source(camera_index)
            
            # Load mirror setting
            mirror = config.get('mirror', False)
            self.mirror_btn.setChecked(mirror)
    
    def save_config(self):
        """Save camera configuration"""
        config = {
            'enabled': self.enable_cb.isChecked(),
            'source': self.current_source or self.camera_combo.currentData(),
            'name': f"Camera {self.camera_id}",
            'mirror': self.mirror_btn.isChecked()
        }
        
        self.config_manager.set_camera_config(self.camera_id, config)
    
    @Slot(bool)
    def on_enable_toggled(self, enabled):
        """Handle enable/disable toggle"""
        self.start_stop_btn.setEnabled(enabled)
        self.camera_combo.setEnabled(enabled)
        self.config_btn.setEnabled(enabled)
        
        if enabled:
            self.camera_group.setTitle(f"Camera {self.camera_id} (Enabled)")
        else:
            self.camera_group.setTitle(f"Camera {self.camera_id} (Disabled)")
            
            # Stop camera if it's running
            if self.is_active:
                self.stop_camera()
        
        self.save_config()
        self.camera_enabled.emit(self.camera_id, enabled)
    
    @Slot(int)
    def on_camera_changed(self, index):
        """Handle camera selection change"""
        current_data = self.camera_combo.currentData()
        
        if current_data == -1:  # Custom Source...
            self.show_source_config()
        elif isinstance(current_data, str) and current_data in ["http://", "rtsp://"]:
            # Show dialog for predefined types
            self.show_source_config()
        else:
            # Direct selection
            self.current_source = current_data
            self.save_config()
    
    @Slot()
    def on_start_stop_clicked(self):
        """Handle start/stop button click"""
        if self.is_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    @Slot()
    def on_mirror_clicked(self):
        """Handle mirror button click"""
        is_mirrored = self.mirror_btn.isChecked()
        self.logger.info(f"Camera {self.camera_id} mirror: {is_mirrored}")
        
        # Save mirror setting to config
        config_key = f"cameras.camera_{self.camera_id}.mirror"
        self.config_manager.set(config_key, is_mirrored)
        
        # Update camera manager if camera is active
        if self.is_active and self.camera_manager:
            self.camera_manager.set_camera_mirror(self.camera_id, is_mirrored)
    
    def start_camera(self):
        """Start the camera"""
        source = self.current_source or self.camera_combo.currentData()
        
        if source is None or source in ["http://", "rtsp://", -1]:
            self.logger.error("No valid camera source selected")
            return
        
        try:
            success = self.camera_manager.add_camera(self.camera_id, source)
            if success:
                success = self.camera_manager.start_camera(self.camera_id)
                
            if success:
                self.is_active = True
                self.start_stop_btn.setText("Stop")
                self.status_label.setText("Active")
                self.status_label.setStyleSheet("color: #00ff00;")
                self.logger.info(f"Camera {self.camera_id} started with source: {source}")
            else:
                self.logger.error(f"Failed to add camera {self.camera_id}")
                
        except Exception as e:
            self.logger.error(f"Error starting camera {self.camera_id}: {e}")
    
    def stop_camera(self):
        """Stop the camera"""
        try:
            self.camera_manager.stop_camera(self.camera_id)
            self.camera_manager.remove_camera(self.camera_id)
            
            self.is_active = False
            self.start_stop_btn.setText("Start")
            self.status_label.setText("Inactive")
            self.status_label.setStyleSheet("color: #888888;")
            self.video_label.setText("No Video")
            self.logger.info(f"Camera {self.camera_id} stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping camera {self.camera_id}: {e}")
    
    def update_frame(self, frame: np.ndarray):
        """Update the video display with a new frame"""
        if frame is None:
            return
            
        self.current_frame = frame.copy()
        
        # Draw pose landmarks if available
        display_frame = self._draw_pose_landmarks(self.current_frame)
        
        # Convert frame to QPixmap for display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(
            display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        
        # Scale to fit the label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def _draw_pose_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw pose landmarks on the frame"""
        if self.current_pose is None:
            return frame
        
        # Draw landmarks
        result_frame = frame.copy()
        
        try:
            # Draw 2D landmarks
            if hasattr(self.current_pose, 'landmarks_2d') and self.current_pose.landmarks_2d is not None:
                landmarks_2d = self.current_pose.landmarks_2d
                height, width = frame.shape[:2]
                
                self.logger.debug(f"Drawing {len(landmarks_2d)} landmarks on frame {width}x{height}")
                
                # Draw landmarks as circles
                for i, (x, y) in enumerate(landmarks_2d):
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        pixel_x = int(x * width)
                        pixel_y = int(y * height)
                        
                        # Color based on landmark importance
                        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face landmarks
                            color = (0, 255, 255)  # Yellow
                            radius = 4
                        elif i in [11, 12, 13, 14, 15, 16]:  # Arms
                            color = (0, 255, 0)  # Green
                            radius = 6
                        elif i in [23, 24, 25, 26, 27, 28]:  # Legs
                            color = (255, 0, 0)  # Red
                            radius = 6
                        elif i in [29, 30, 31, 32]:  # Feet (important for floor calibration)
                            color = (255, 0, 255)  # Magenta
                            radius = 8
                        else:
                            color = (255, 255, 255)  # White
                            radius = 3
                        
                        # Draw filled circle with black border
                        cv2.circle(result_frame, (pixel_x, pixel_y), radius, color, -1)
                        cv2.circle(result_frame, (pixel_x, pixel_y), radius + 1, (0, 0, 0), 2)
                        
                        # Draw landmark number for debugging (only important ones)
                        if i in [0, 11, 12, 23, 24, 29, 30, 31, 32]:  # Important landmarks
                            cv2.putText(result_frame, str(i), 
                                      (pixel_x + 8, pixel_y - 8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw pose connections
                self._draw_pose_connections(result_frame, landmarks_2d, width, height)
                
                # Draw info text with better visibility
                cv2.rectangle(result_frame, (5, 5), (250, 60), (0, 0, 0), -1)
                cv2.putText(result_frame, f"Pose: {self.current_pose.confidence:.2f}", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(result_frame, f"Landmarks: {len(landmarks_2d)}", 
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(result_frame, f"Camera: {self.current_pose.camera_id}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                self.logger.debug(f"Successfully drew pose with confidence {self.current_pose.confidence:.3f}")
            else:
                self.logger.debug("No landmarks_2d available in pose data")
            
        except Exception as e:
            self.logger.error(f"Error drawing pose landmarks: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return result_frame
    
    def _draw_pose_connections(self, frame: np.ndarray, landmarks_2d: np.ndarray, width: int, height: int):
        """Draw connections between pose landmarks"""
        # Define pose connections (simplified)
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # Arms
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            # Body
            (11, 23), (12, 24), (23, 24),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d):
                start_x, start_y = landmarks_2d[start_idx]
                end_x, end_y = landmarks_2d[end_idx]
                
                if (0 <= start_x <= 1 and 0 <= start_y <= 1 and 
                    0 <= end_x <= 1 and 0 <= end_y <= 1):
                    
                    start_pixel = (int(start_x * width), int(start_y * height))
                    end_pixel = (int(end_x * width), int(end_y * height))
                    
                    cv2.line(frame, start_pixel, end_pixel, (255, 255, 255), 2)
    
    def update_pose(self, pose_data):
        """Update pose overlay on the video"""
        self.current_pose = pose_data
        
        # If we have a current frame, redraw it with the new pose data
        if self.current_frame is not None:
            display_frame = self._draw_pose_landmarks(self.current_frame)
            
            # Convert frame to QPixmap for display
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            
            q_image = QImage(
                display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            
            # Scale to fit the label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
    
    def update_stats(self, fps: float, frame_count: int):
        """Update FPS and frame count display"""
        self.fps_label.setText(f"{fps:.1f} FPS")
        self.frame_label.setText(f"{frame_count} frames")
    
    def update_status(self, status: str, is_active: bool):
        """Update camera status from external sources"""
        if is_active:
            self.status_label.setText(f"Active - {status}")
            self.status_label.setStyleSheet("color: #00ff00;")
        else:
            self.status_label.setText(f"Inactive - {status}")
            self.status_label.setStyleSheet("color: #888888;")
    
    def get_config(self) -> dict:
        """Get current camera configuration"""
        return {
            'camera_id': self.camera_id,
            'enabled': self.enable_cb.isChecked(),
            'source': self.current_source or self.camera_combo.currentData(),
            'is_active': self.is_active
        }
