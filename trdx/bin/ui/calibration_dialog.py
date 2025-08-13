"""
Calibration Dialog - Pose tracking calibration interface
"""

import time
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QGroupBox,
    QSlider, QSpinBox, QDoubleSpinBox, QTextEdit,
    QWidget, QFrame
)
from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QFont, QPixmap, QImage
import logging
import cv2

class CalibrationDialog(QDialog):
    """Calibration dialog for pose tracking system"""
    
    calibration_started = Signal()
    calibration_completed = Signal(dict)  # calibration_data
    
    def __init__(self, pose_processor, steamvr_driver, parent=None):
        super().__init__(parent)
        self.pose_processor = pose_processor
        self.steamvr_driver = steamvr_driver
        self.logger = logging.getLogger(__name__)
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_step = 0
        self.calibration_data = {}
        self.countdown_timer = None
        self.countdown_remaining = 0
        
        # Update timer for detection info
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_detection_info)
        self.update_timer.start(500)  # Update every 500ms
        
        # Reference poses for calibration
        self.reference_poses = []
        self.calibration_poses = [
            "Stand straight with arms at your sides",
            "T-pose: Arms extended horizontally",
            "Hands on hips",
            "One foot forward, arms down"
        ]
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TRDX Calibration")
        self.setModal(True)
        self.resize(800, 600)
        
        # Configurar icono del diálogo
        try:
            from pathlib import Path
            from PySide6.QtGui import QIcon
            icon_path = Path(__file__).parent.parent.parent / "assets" / "logos" / "logo.ico"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                self.logger.info(f"✅ Icono de calibración cargado: {icon_path}")
            else:
                self.logger.warning(f"⚠️ Icono no encontrado en: {icon_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error al cargar icono de calibración: {e}")
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Pose Tracking Calibration")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Follow the instructions below to calibrate your pose tracking system.\n"
            "Make sure you are visible to all active cameras and have enough space to move."
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left panel - Controls
        controls_panel = self.create_controls_panel()
        content_layout.addWidget(controls_panel, 1)
        
        # Right panel - Preview
        preview_panel = self.create_preview_panel()
        content_layout.addWidget(preview_panel, 2)
        
        layout.addLayout(content_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Calibration")
        self.start_btn.clicked.connect(self.start_calibration)
        button_layout.addWidget(self.start_btn)
        
        self.skip_btn = QPushButton("Skip Step")
        self.skip_btn.clicked.connect(self.skip_step)
        self.skip_btn.setEnabled(False)
        button_layout.addWidget(self.skip_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_calibration)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Apply styling
        self.apply_style()
        
    def create_controls_panel(self):
        """Create the controls panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready to calibrate")
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.calibration_poses))
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        # Countdown
        self.countdown_label = QLabel("")
        self.countdown_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("color: #FF6B6B;")
        status_layout.addWidget(self.countdown_label)
        
        layout.addWidget(status_group)
        
        # Current pose instruction
        pose_group = QGroupBox("Current Pose")
        pose_layout = QVBoxLayout(pose_group)
        
        self.pose_instruction = QLabel("Click 'Start Calibration' to begin")
        self.pose_instruction.setFont(QFont("Arial", 12))
        self.pose_instruction.setWordWrap(True)
        self.pose_instruction.setAlignment(Qt.AlignCenter)
        self.pose_instruction.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 8px;
                padding: 20px;
                color: #333333;
            }
        """)
        pose_layout.addWidget(self.pose_instruction)
        
        layout.addWidget(pose_group)
        
        # Manual adjustments
        manual_group = QGroupBox("Manual Adjustments")
        manual_layout = QVBoxLayout(manual_group)
        
        # Scale adjustment
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(50, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        self.scale_value_label = QLabel("1.00")
        scale_layout.addWidget(self.scale_value_label)
        manual_layout.addLayout(scale_layout)
        
        # Height adjustment
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height (m):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 2.5)
        self.height_spin.setValue(1.8)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setDecimals(2)
        self.height_spin.valueChanged.connect(self.on_height_changed)
        height_layout.addWidget(self.height_spin)
        manual_layout.addLayout(height_layout)
        
        # Rotation adjustments
        for axis, name in [('X', 'Pitch'), ('Y', 'Yaw'), ('Z', 'Roll')]:
            rot_layout = QHBoxLayout()
            rot_layout.addWidget(QLabel(f"{name}:"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-180, 180)
            slider.setValue(0)
            slider.valueChanged.connect(lambda v, a=axis: self.on_rotation_changed(a, v))
            rot_layout.addWidget(slider)
            value_label = QLabel("0°")
            rot_layout.addWidget(value_label)
            manual_layout.addLayout(rot_layout)
            
            setattr(self, f'rotation_{axis.lower()}_slider', slider)
            setattr(self, f'rotation_{axis.lower()}_label', value_label)
        
        layout.addWidget(manual_group)
        
        layout.addStretch()
        
        return panel
        
    def create_preview_panel(self):
        """Create the preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Detection info
        detection_group = QGroupBox("Live Detection Info")
        detection_layout = QVBoxLayout(detection_group)
        
        # Detection info text area
        self.detection_info = QTextEdit()
        self.detection_info.setMaximumHeight(200)
        self.detection_info.setReadOnly(True)
        self.detection_info.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        self.detection_info.setText("Waiting for pose detection...\n\nPlease ensure:\n- Cameras are active\n- You are visible to cameras\n- Good lighting conditions")
        detection_layout.addWidget(self.detection_info)
        
        layout.addWidget(detection_group)
        
        # Calibration status
        status_group = QGroupBox("Calibration Status")
        status_layout = QVBoxLayout(status_group)
        
        self.calibration_status = QTextEdit()
        self.calibration_status.setMaximumHeight(150)
        self.calibration_status.setReadOnly(True)
        self.calibration_status.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffff00;
                border: 1px solid #555555;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        self.calibration_status.setText("Ready to calibrate\n\nClick 'Start Calibration' to begin")
        status_layout.addWidget(self.calibration_status)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        return panel
    
    def apply_style(self):
        """Apply dialog styling"""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
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
                padding: 8px;
                min-width: 100px;
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
    
    @Slot()
    def start_calibration(self):
        """Start the calibration process"""
        if self.is_calibrating:
            return
            
        self.logger.info("Starting calibration process")
        
        self.is_calibrating = True
        self.calibration_step = 0
        self.reference_poses.clear()
        
        # Update UI
        self.start_btn.setText("Calibrating...")
        self.start_btn.setEnabled(False)
        self.skip_btn.setEnabled(True)
        
        # Start first step
        self.next_calibration_step()
        
        self.calibration_started.emit()
    
    def next_calibration_step(self):
        """Move to the next calibration step"""
        if self.calibration_step >= len(self.calibration_poses):
            self.complete_calibration()
            return
            
        # Update UI for current step
        pose_instruction = self.calibration_poses[self.calibration_step]
        self.pose_instruction.setText(pose_instruction)
        self.status_label.setText(f"Step {self.calibration_step + 1} of {len(self.calibration_poses)}")
        self.progress_bar.setValue(self.calibration_step)
        
        # Start countdown
        self.start_countdown(3)
    
    def start_countdown(self, seconds):
        """Start countdown timer"""
        self.countdown_remaining = seconds
        self.update_countdown()
        
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.countdown_tick)
        self.countdown_timer.start(1000)  # 1 second intervals
    
    def countdown_tick(self):
        """Handle countdown tick"""
        self.countdown_remaining -= 1
        
        if self.countdown_remaining <= 0:
            self.countdown_timer.stop()
            self.capture_pose()
        else:
            self.update_countdown()
    
    def update_countdown(self):
        """Update countdown display"""
        if self.countdown_remaining > 0:
            self.countdown_label.setText(str(self.countdown_remaining))
        else:
            self.countdown_label.setText("CAPTURE!")
    
    def capture_pose(self):
        """Capture current pose for calibration"""
        self.logger.info(f"Capturing pose for step {self.calibration_step}")
        
        # Get current pose data from pose processor
        pose_data = None
        if hasattr(self.pose_processor, 'get_current_pose'):
            pose_data = self.pose_processor.get_current_pose()
        elif hasattr(self.pose_processor, 'last_processed_pose'):
            pose_data = self.pose_processor.last_processed_pose
        
        # If no pose data available, try to get from SteamVR driver
        if pose_data is None and hasattr(self.steamvr_driver, 'get_last_pose_data'):
            pose_data = self.steamvr_driver.get_last_pose_data()
        
        # Create calibration entry
        calibration_entry = {
            'step': self.calibration_step,
            'instruction': self.calibration_poses[self.calibration_step],
            'timestamp': time.time(),
            'pose': pose_data,
            'scale': self.scale_slider.value() / 100.0,
            'height': self.height_spin.value(),
            'rotation_x': getattr(self, 'rotation_x_slider', None),
            'rotation_y': getattr(self, 'rotation_y_slider', None),
            'rotation_z': getattr(self, 'rotation_z_slider', None)
        }
        
        self.reference_poses.append(calibration_entry)
        
        # Update detection info
        self.update_detection_info()
        
        # Move to next step
        self.calibration_step += 1
        
        # Brief pause before next step
        QTimer.singleShot(1000, self.next_calibration_step)
        
        # Clear countdown
        self.countdown_label.setText("")
    
    @Slot()
    def skip_step(self):
        """Skip current calibration step"""
        if not self.is_calibrating:
            return
            
        self.logger.info(f"Skipping calibration step {self.calibration_step}")
        
        # Stop countdown if running
        if self.countdown_timer:
            self.countdown_timer.stop()
        
        # Add empty pose data
        pose_data = {
            'step': self.calibration_step,
            'instruction': self.calibration_poses[self.calibration_step],
            'timestamp': time.time(),
            'pose': None,
            'skipped': True
        }
        
        self.reference_poses.append(pose_data)
        
        # Move to next step
        self.calibration_step += 1
        self.next_calibration_step()
    
    def complete_calibration(self):
        """Complete the calibration process"""
        self.logger.info("Calibration process completed")
        
        self.is_calibrating = False
        
        # Calculate calibration data
        calibration_data = self.calculate_calibration()
        
        # Update UI
        self.start_btn.setText("Start Calibration")
        self.start_btn.setEnabled(True)
        self.skip_btn.setEnabled(False)
        self.status_label.setText("Calibration completed!")
        self.pose_instruction.setText("Calibration completed successfully!")
        self.progress_bar.setValue(len(self.calibration_poses))
        self.countdown_label.setText("")
        
        # Emit completion signal
        self.calibration_completed.emit(calibration_data)
        
        # Apply calibration
        self.apply_calibration(calibration_data)
    
    def calculate_calibration(self):
        """Calculate calibration parameters from captured poses"""
        self.logger.info("Calculating calibration from captured poses")
        
        # Get current manual adjustments
        calibration_data = {
            'scale': self.scale_slider.value() / 100.0,
            'height': self.height_spin.value(),
            'rotation_x': 0,  # Default values
            'rotation_y': 0,
            'rotation_z': 0,
            'reference_poses': self.reference_poses,
            'timestamp': time.time()
        }
        
        # Try to get rotation values from sliders if they exist
        if hasattr(self, 'rotation_x_slider'):
            calibration_data['rotation_x'] = self.rotation_x_slider.value()
        if hasattr(self, 'rotation_y_slider'):
            calibration_data['rotation_y'] = self.rotation_y_slider.value()
        if hasattr(self, 'rotation_z_slider'):
            calibration_data['rotation_z'] = self.rotation_z_slider.value()
        
        # Analyze captured poses to improve calibration
        valid_poses = [pose for pose in self.reference_poses if pose.get('pose') is not None]
        
        if valid_poses:
            self.logger.info(f"Analyzing {len(valid_poses)} valid poses for calibration")
            
            # Calculate average height from poses
            heights = []
            for pose_entry in valid_poses:
                pose_data = pose_entry['pose']
                if isinstance(pose_data, dict) and 'landmarks' in pose_data:
                    landmarks = pose_data['landmarks']
                    if len(landmarks) > 0:
                        # Calculate height from landmarks (nose to ankle)
                        nose_y = landmarks[0][1] if len(landmarks) > 0 else 0
                        left_ankle_y = landmarks[27][1] if len(landmarks) > 27 else 0
                        right_ankle_y = landmarks[28][1] if len(landmarks) > 28 else 0
                        
                        if left_ankle_y > 0 and right_ankle_y > 0:
                            avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
                            height = abs(nose_y - avg_ankle_y)
                            heights.append(height)
            
            if heights:
                avg_height = sum(heights) / len(heights)
                # Convert to meters (assuming landmarks are in pixels)
                calibration_data['calculated_height'] = avg_height * 0.001  # Rough conversion
                self.logger.info(f"Calculated average height: {calibration_data['calculated_height']:.2f}m")
        
        return calibration_data
    
    def apply_calibration(self, calibration_data):
        """Apply calibration to the pose processor"""
        self.logger.info("Applying calibration data")
        
        try:
            # Apply height
            if hasattr(self.pose_processor, 'set_user_height'):
                height = calibration_data.get('calculated_height', calibration_data['height'])
                self.pose_processor.set_user_height(height)
                self.logger.info(f"Applied user height: {height:.2f}m")
            
            # Apply scale
            if hasattr(self.pose_processor, 'global_calibration'):
                if not hasattr(self.pose_processor.global_calibration, '__getitem__'):
                    self.pose_processor.global_calibration = {}
                self.pose_processor.global_calibration['scale'] = calibration_data['scale']
                self.logger.info(f"Applied scale: {calibration_data['scale']:.2f}")
            
            # Apply rotation
            if hasattr(self.pose_processor, 'global_calibration'):
                if not hasattr(self.pose_processor.global_calibration, '__getitem__'):
                    self.pose_processor.global_calibration = {}
                
                try:
                    from scipy.spatial.transform import Rotation as R
                    rx = calibration_data['rotation_x']
                    ry = calibration_data['rotation_y']
                    rz = calibration_data['rotation_z']
                    rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
                    self.pose_processor.global_calibration['rotation'] = rot
                    self.logger.info(f"Applied rotation: X={rx}°, Y={ry}°, Z={rz}°")
                except ImportError:
                    self.logger.warning("scipy not available, skipping rotation calibration")
                except Exception as e:
                    self.logger.error(f"Error applying rotation: {e}")
            
            # Apply to SteamVR driver if available
            if hasattr(self.steamvr_driver, 'apply_calibration'):
                self.steamvr_driver.apply_calibration(calibration_data)
                self.logger.info("Applied calibration to SteamVR driver")
            
            # Save calibration to config
            if hasattr(self.pose_processor, 'config_manager'):
                config_data = {
                    'calibration': calibration_data,
                    'timestamp': time.time()
                }
                self.pose_processor.config_manager.save_calibration(config_data)
                self.logger.info("Saved calibration to config")
            
        except Exception as e:
            self.logger.error(f"Error applying calibration: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    @Slot()
    def reset_calibration(self):
        """Reset calibration to defaults"""
        self.logger.info("Resetting calibration")
        
        # Stop any ongoing calibration
        if self.countdown_timer:
            self.countdown_timer.stop()
        
        self.is_calibrating = False
        self.calibration_step = 0
        self.reference_poses.clear()
        
        # Reset UI
        self.start_btn.setText("Start Calibration")
        self.start_btn.setEnabled(True)
        self.skip_btn.setEnabled(False)
        self.status_label.setText("Ready to calibrate")
        self.pose_instruction.setText("Click 'Start Calibration' to begin")
        self.progress_bar.setValue(0)
        self.countdown_label.setText("")
        
        # Reset manual adjustments
        self.scale_slider.setValue(100)
        self.height_spin.setValue(1.8)
        self.rotation_x_slider.setValue(0)
        self.rotation_y_slider.setValue(0)
        self.rotation_z_slider.setValue(0)
    
    @Slot(int)
    def on_scale_changed(self, value):
        """Handle scale adjustment"""
        scale = value / 100.0
        self.scale_value_label.setText(f"{scale:.2f}")
    
    @Slot(float)
    def on_height_changed(self, value):
        """Handle height adjustment"""
        pass  # Height is already displayed in the spin box
    
    def on_rotation_changed(self, axis, value):
        """Handle rotation adjustment"""
        label = getattr(self, f'rotation_{axis.lower()}_label')
        label.setText(f"{value}°")
    
    def update_detection_info(self):
        """Update detection information display"""
        try:
            # Get current pose data
            pose_data = None
            if hasattr(self.pose_processor, 'get_current_pose'):
                pose_data = self.pose_processor.get_current_pose()
            elif hasattr(self.pose_processor, 'last_processed_pose'):
                pose_data = self.pose_processor.last_processed_pose
            
            if pose_data:
                # Update detection info
                info_text = f"Pose detected: ✓\n"
                
                if isinstance(pose_data, dict):
                    if 'landmarks' in pose_data:
                        landmarks = pose_data['landmarks']
                        info_text += f"Landmarks: {len(landmarks)}\n"
                    
                    if 'confidence' in pose_data:
                        confidence = pose_data['confidence']
                        info_text += f"Confidence: {confidence:.2f}\n"
                    
                    if 'trackers' in pose_data:
                        trackers = pose_data['trackers']
                        info_text += f"Trackers: {len(trackers)}\n"
                        
                        for tracker_name, tracker_data in trackers.items():
                            if 'position' in tracker_data:
                                pos = tracker_data['position']
                                info_text += f"  {tracker_name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
                
                self.detection_info.setText(info_text)
            else:
                self.detection_info.setText("No pose detected\nPlease ensure you are visible to cameras")
            
            # Update calibration status
            self.update_calibration_status()
                
        except Exception as e:
            self.logger.error(f"Error updating detection info: {e}")
            self.detection_info.setText("Error updating detection info")
    
    def update_calibration_status(self):
        """Update calibration status display"""
        try:
            if self.is_calibrating:
                status_text = f"Calibrating...\n"
                status_text += f"Step: {self.calibration_step + 1}/{len(self.calibration_poses)}\n"
                status_text += f"Poses captured: {len(self.reference_poses)}\n"
                
                if self.countdown_remaining > 0:
                    status_text += f"Countdown: {self.countdown_remaining}\n"
                else:
                    status_text += "Capturing pose...\n"
            else:
                status_text = "Ready to calibrate\n\n"
                status_text += f"Captured poses: {len(self.reference_poses)}\n"
                status_text += "Click 'Start Calibration' to begin"
            
            self.calibration_status.setText(status_text)
            
        except Exception as e:
            self.logger.error(f"Error updating calibration status: {e}")
            self.calibration_status.setText("Error updating status")
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.is_calibrating:
            # Stop calibration if in progress
            self.reset_calibration()
        
        event.accept()
