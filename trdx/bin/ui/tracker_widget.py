"""
Tracker Widget - Individual VR tracker display and status
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QComboBox, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QFont, QPalette
import logging

class TrackerWidget(QWidget):
    """Widget for displaying VR tracker status and control"""
    
    tracker_role_changed = Signal(int, str)  # tracker_id, role
    
    def __init__(self, tracker_id: int, steamvr_driver, config_manager):
        super().__init__()
        self.tracker_id = tracker_id
        self.steamvr_driver = steamvr_driver
        self.config_manager = config_manager
        self.logger = logging.getLogger(f"TrackerWidget{tracker_id}")
        
        # State
        self.is_active = False
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.confidence = 0.0
        
        # Tracker names
        self.tracker_names = {
            0: "Hip",
            1: "Left Foot",
            2: "Right Foot",
            3: "Left Hand",
            4: "Right Hand"
        }
        
        # Initialize UI
        self.init_ui()
        self.load_config()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Tracker group
        tracker_name = self.tracker_names.get(self.tracker_id, f"Tracker {self.tracker_id}")
        self.tracker_group = QGroupBox(tracker_name)
        tracker_layout = QVBoxLayout(self.tracker_group)
        
        # Status indicator
        status_layout = QHBoxLayout()
        
        # Active indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setFont(QFont("Arial", 16))
        self.status_indicator.setStyleSheet("color: red;")
        status_layout.addWidget(self.status_indicator)
        
        # Status text
        self.status_label = QLabel("Inactive")
        self.status_label.setFont(QFont("Arial", 9, QFont.Bold))
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Confidence bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setMaximumHeight(15)
        status_layout.addWidget(self.confidence_bar)
        
        tracker_layout.addLayout(status_layout)
        
        # Role selection
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Role:"))
        
        self.role_combo = QComboBox()
        self.role_combo.addItems([
            "TrackerRole_Waist",
            "TrackerRole_LeftFoot", 
            "TrackerRole_RightFoot",
            "TrackerRole_LeftShoulder",
            "TrackerRole_RightShoulder",
            "TrackerRole_LeftElbow",
            "TrackerRole_RightElbow",
            "TrackerRole_LeftKnee",
            "TrackerRole_RightKnee",
            "TrackerRole_Handed"
        ])
        self.role_combo.currentTextChanged.connect(self.on_role_changed)
        role_layout.addWidget(self.role_combo)
        
        tracker_layout.addLayout(role_layout)
        
        # Position display
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout(pos_group)
        pos_layout.setContentsMargins(5, 5, 5, 5)
        
        self.pos_x_label = QLabel("X: 0.000")
        self.pos_y_label = QLabel("Y: 0.000") 
        self.pos_z_label = QLabel("Z: 0.000")
        
        for label in [self.pos_x_label, self.pos_y_label, self.pos_z_label]:
            label.setFont(QFont("Consolas", 8))
            pos_layout.addWidget(label)
        
        tracker_layout.addWidget(pos_group)
        
        # Rotation display
        rot_group = QGroupBox("Rotation")
        rot_layout = QVBoxLayout(rot_group)
        rot_layout.setContentsMargins(5, 5, 5, 5)
        
        self.rot_w_label = QLabel("W: 1.000")
        self.rot_x_label = QLabel("X: 0.000")
        self.rot_y_label = QLabel("Y: 0.000")
        self.rot_z_label = QLabel("Z: 0.000")
        
        for label in [self.rot_w_label, self.rot_x_label, self.rot_y_label, self.rot_z_label]:
            label.setFont(QFont("Consolas", 8))
            rot_layout.addWidget(label)
        
        tracker_layout.addWidget(rot_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        self.reset_btn.setMaximumWidth(60)
        button_layout.addWidget(self.reset_btn)
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.on_calibrate_clicked)
        self.calibrate_btn.setMaximumWidth(70)
        button_layout.addWidget(self.calibrate_btn)
        
        button_layout.addStretch()
        
        tracker_layout.addLayout(button_layout)
        
        layout.addWidget(self.tracker_group)
        
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
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)
    
    def load_config(self):
        """Load tracker configuration"""
        role = self.config_manager.get_tracker_role(self.tracker_id)
        
        # Set role in combo box
        index = self.role_combo.findText(role)
        if index >= 0:
            self.role_combo.setCurrentIndex(index)
    
    def save_config(self):
        """Save tracker configuration"""
        role = self.role_combo.currentText()
        self.config_manager.set_tracker_role(self.tracker_id, role)
    
    @Slot(str)
    def on_role_changed(self, role):
        """Handle role change"""
        self.save_config()
        self.tracker_role_changed.emit(self.tracker_id, role)
        self.logger.info(f"Tracker {self.tracker_id} role changed to: {role}")
    
    @Slot()
    def on_reset_clicked(self):
        """Handle reset button click"""
        # Reset tracker position to origin
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.update_displays()
        self.logger.info(f"Tracker {self.tracker_id} reset")
    
    @Slot()
    def on_calibrate_clicked(self):
        """Handle calibrate button click"""
        # Trigger calibration for this tracker
        self.logger.info(f"Calibrating tracker {self.tracker_id}")
        # TODO: Implement tracker-specific calibration
    
    def set_role(self, role: str):
        """Set tracker role programmatically"""
        index = self.role_combo.findText(role)
        if index >= 0:
            self.role_combo.setCurrentIndex(index)
    
    def update_status(self, status_data: dict):
        """Update tracker status from status data"""
        if not status_data.get('exists', False):
            self.set_inactive()
            return
        
        is_active = status_data.get('active', False)
        
        if is_active:
            self.set_active()
            
            # Update position
            position = status_data.get('position', [0.0, 0.0, 0.0])
            self.current_position = np.array(position)
            
            # Update rotation
            rotation = status_data.get('rotation', [1.0, 0.0, 0.0, 0.0])
            self.current_rotation = np.array(rotation)
            
            # Update confidence
            self.confidence = status_data.get('confidence', 0.0)
            
            # Update displays
            self.update_displays()
        else:
            self.set_inactive()
    
    def set_active(self):
        """Set tracker as active"""
        if not self.is_active:
            self.is_active = True
            self.status_indicator.setStyleSheet("color: green;")
            self.status_label.setText("Active")
            self.tracker_group.setEnabled(True)
    
    def set_inactive(self):
        """Set tracker as inactive"""
        if self.is_active:
            self.is_active = False
            self.status_indicator.setStyleSheet("color: red;")
            self.status_label.setText("Inactive")
            self.confidence = 0.0
            self.confidence_bar.setValue(0)
    
    def update_displays(self):
        """Update position and rotation displays"""
        # Update position labels
        self.pos_x_label.setText(f"X: {self.current_position[0]:7.3f}")
        self.pos_y_label.setText(f"Y: {self.current_position[1]:7.3f}")
        self.pos_z_label.setText(f"Z: {self.current_position[2]:7.3f}")
        
        # Update rotation labels (quaternion)
        self.rot_w_label.setText(f"W: {self.current_rotation[0]:7.3f}")
        self.rot_x_label.setText(f"X: {self.current_rotation[1]:7.3f}")
        self.rot_y_label.setText(f"Y: {self.current_rotation[2]:7.3f}")
        self.rot_z_label.setText(f"Z: {self.current_rotation[3]:7.3f}")
        
        # Update confidence bar
        confidence_percent = int(self.confidence * 100)
        self.confidence_bar.setValue(confidence_percent)
        
        # Color code confidence bar
        if confidence_percent >= 80:
            color = "#4CAF50"  # Green
        elif confidence_percent >= 60:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)
    
    def update_pose_data(self, position: np.ndarray, rotation: np.ndarray, confidence: float):
        """Update tracker with new pose data"""
        self.current_position = position
        self.current_rotation = rotation
        self.confidence = confidence
        
        if not self.is_active:
            self.set_active()
        
        self.update_displays()
    
    def get_tracker_info(self) -> dict:
        """Get tracker information"""
        return {
            'tracker_id': self.tracker_id,
            'name': self.tracker_names.get(self.tracker_id, f"Tracker {self.tracker_id}"),
            'role': self.role_combo.currentText(),
            'active': self.is_active,
            'position': self.current_position.tolist(),
            'rotation': self.current_rotation.tolist(),
            'confidence': self.confidence
        }
    
    def reset_tracker(self):
        """Reset tracker to default state"""
        self.set_inactive()
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.confidence = 0.0
        self.update_displays()
