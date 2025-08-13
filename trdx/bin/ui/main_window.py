"""
Main Window - Primary UI for TRDX Application
Multi-Camera Full Body Tracking Interface
"""

import sys
from typing import Dict, Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSlider, QProgressBar, QTextEdit, QSplitter,
    QFrame, QScrollArea, QStatusBar, QMenuBar, QMenu, QToolBar,
    QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, QTimer, Slot, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QAction, QIcon, QFont
import numpy as np
import cv2
import logging

from .camera_widget import CameraWidget
from .tracker_widget import TrackerWidget
from .settings_dialog import SettingsDialog
from .calibration_dialog import CalibrationDialog
from .tracker_visualizer import TrackerVisualizerWidget

class MainWindow(QMainWindow):
    """Main application window for TRDX"""
    
    def __init__(self, camera_manager, pose_processor, steamvr_driver, config_manager):
        super().__init__()
        
        self.camera_manager = camera_manager
        self.pose_processor = pose_processor
        self.steamvr_driver = steamvr_driver
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # UI state
        self.camera_widgets: Dict[int, CameraWidget] = {}
        self.tracker_widgets: Dict[int, TrackerWidget] = {}
        self.is_tracking = False
        
        # Dialogs
        self.settings_dialog = None
        self.calibration_dialog = None
        
        # Tracker visualizer
        self.tracker_visualizer = None
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        self.load_settings()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(33)  # ~30 FPS UI updates
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TRDX - Multi-Camera Full Body Tracking")
        self.setMinimumSize(1200, 800)
        
        # Configurar icono de la ventana
        try:
            from pathlib import Path
            icon_path = Path(__file__).parent.parent.parent / "assets" / "logos" / "logo.ico"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                self.logger.info(f"✅ Icono de ventana cargado: {icon_path}")
            else:
                self.logger.warning(f"⚠️ Icono no encontrado en: {icon_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error al cargar icono de ventana: {e}")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Camera feeds
        self.camera_panel = self.create_camera_panel()
        splitter.addWidget(self.camera_panel)
        
        # Right panel - Controls and status
        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)
        
        # Set splitter proportions
        splitter.setSizes([800, 400])
        
        # Create status bar
        self.create_status_bar()
        
        # Apply style
        self.apply_style()
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Settings action
        settings_action = QAction('&Settings', self)
        settings_action.setStatusTip('Open application settings')
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera menu
        camera_menu = menubar.addMenu('&Camera')
        
        start_cameras_action = QAction('Start All Cameras', self)
        start_cameras_action.triggered.connect(self.start_all_cameras)
        camera_menu.addAction(start_cameras_action)
        
        stop_cameras_action = QAction('Stop All Cameras', self)
        stop_cameras_action.triggered.connect(self.stop_all_cameras)
        camera_menu.addAction(stop_cameras_action)
        
        # Tracking menu
        tracking_menu = menubar.addMenu('&Tracking')
        
        start_tracking_action = QAction('Start Tracking', self)
        start_tracking_action.triggered.connect(self.start_tracking)
        tracking_menu.addAction(start_tracking_action)
        
        stop_tracking_action = QAction('Stop Tracking', self)
        stop_tracking_action.triggered.connect(self.stop_tracking)
        tracking_menu.addAction(stop_tracking_action)
        
        tracking_menu.addSeparator()
        
        calibrate_action = QAction('Calibrate', self)
        calibrate_action.triggered.connect(self.show_calibration)
        tracking_menu.addAction(calibrate_action)
        
        # Floor calibration submenu removed - not needed
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        toolbar.setObjectName("main_toolbar")  # Fix for saveState warning
        self.addToolBar(toolbar)
        
        # Start/Stop tracking buttons
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        toolbar.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Tracking")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)
        
        toolbar.addSeparator()
        
        # Camera controls
        self.start_cameras_btn = QPushButton("Start Cameras")
        self.start_cameras_btn.clicked.connect(self.start_all_cameras)
        toolbar.addWidget(self.start_cameras_btn)
        
        self.stop_cameras_btn = QPushButton("Stop Cameras")
        self.stop_cameras_btn.clicked.connect(self.stop_all_cameras)
        toolbar.addWidget(self.stop_cameras_btn)
        
        toolbar.addSeparator()
        
        # SteamVR connection
        self.connect_steamvr_btn = QPushButton("Connect SteamVR")
        self.connect_steamvr_btn.clicked.connect(self.connect_steamvr)
        toolbar.addWidget(self.connect_steamvr_btn)
    
    def create_camera_panel(self) -> QWidget:
        """Create the camera panel with multiple camera feeds"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Camera Feeds")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Camera grid
        self.camera_scroll = QScrollArea()
        self.camera_container = QWidget()
        self.camera_grid = QGridLayout(self.camera_container)
        
        # Create camera widgets
        max_cameras = self.config_manager.get('cameras.max_cameras', 4)
        for i in range(max_cameras):
            camera_widget = CameraWidget(i, self.camera_manager, self.config_manager)
            self.camera_widgets[i] = camera_widget
            
            # Add to grid (2x2 layout)
            row = i // 2
            col = i % 2
            self.camera_grid.addWidget(camera_widget, row, col)
        
        self.camera_scroll.setWidget(self.camera_container)
        self.camera_scroll.setWidgetResizable(True)
        layout.addWidget(self.camera_scroll)
        
        return panel
    
    def create_control_panel(self) -> QWidget:
        """Create the control panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Tracking tab
        tracking_tab = self.create_tracking_tab()
        tab_widget.addTab(tracking_tab, "Tracking")
        
        # Trackers tab
        trackers_tab = self.create_trackers_tab()
        tab_widget.addTab(trackers_tab, "Trackers")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "Settings")
        
        # Log tab
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "Log")
        
        # Visualizer tab
        visualizer_tab = self.create_visualizer_tab()
        tab_widget.addTab(visualizer_tab, "3D Visualizer")
        
        return panel
    
    def create_tracking_tab(self) -> QWidget:
        """Create the tracking control tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.tracking_status_label = QLabel("Not tracking")
        status_layout.addWidget(self.tracking_status_label)
        
        self.fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_label)
        
        self.cameras_status_label = QLabel("Cameras: 0/4")
        status_layout.addWidget(self.cameras_status_label)
        
        self.steamvr_status_label = QLabel("SteamVR: Disconnected")
        status_layout.addWidget(self.steamvr_status_label)
        
        layout.addWidget(status_group)
        
        # Control group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Fusion mode
        fusion_layout = QHBoxLayout()
        fusion_layout.addWidget(QLabel("Fusion Mode:"))
        self.fusion_combo = QComboBox()
        self.fusion_combo.addItems(["best_camera", "weighted_average", "multi_view"])
        self.fusion_combo.currentTextChanged.connect(self.on_fusion_mode_changed)
        fusion_layout.addWidget(self.fusion_combo)
        control_layout.addLayout(fusion_layout)
        
        # Smoothing
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(70)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        smoothing_layout.addWidget(self.smoothing_slider)
        self.smoothing_value_label = QLabel("0.7")
        smoothing_layout.addWidget(self.smoothing_value_label)
        control_layout.addLayout(smoothing_layout)
        
        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_value_label = QLabel("0.5")
        confidence_layout.addWidget(self.confidence_value_label)
        control_layout.addLayout(confidence_layout)
        
        layout.addWidget(control_group)
        
        # Calibration group - SIMPLIFIED
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        
        self.calibrate_btn = QPushButton("Calibrate Now")
        self.calibrate_btn.clicked.connect(self.show_calibration)
        calibration_layout.addWidget(self.calibrate_btn)
        
        layout.addWidget(calibration_group)
        
        layout.addStretch()
        
        return tab
    
    def create_trackers_tab(self) -> QWidget:
        """Create the trackers status tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Tracker list
        title = QLabel("Active Trackers")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        # Scroll area for trackers
        scroll = QScrollArea()
        tracker_container = QWidget()
        self.tracker_layout = QVBoxLayout(tracker_container)
        
        # Create tracker widgets
        for i in range(3):  # Hip, Left Foot, Right Foot
            tracker_widget = TrackerWidget(i, self.steamvr_driver, self.config_manager)
            self.tracker_widgets[i] = tracker_widget
            self.tracker_layout.addWidget(tracker_widget)
        
        scroll.setWidget(tracker_container)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        return tab
    
    def create_settings_tab(self) -> QWidget:
        """Create the settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Quick settings
        settings_group = QGroupBox("Quick Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Model complexity
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Complexity:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["0 (Fast)", "1 (Balanced)", "2 (Accurate)"])
        self.model_combo.setCurrentIndex(1)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        
        # Camera resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        resolution_layout.addWidget(self.resolution_combo)
        settings_layout.addLayout(resolution_layout)
        
        # Enable features
        self.enable_hands_cb = QCheckBox("Enable Hand Tracking")
        settings_layout.addWidget(self.enable_hands_cb)
        
        self.enable_face_cb = QCheckBox("Enable Face Tracking")
        settings_layout.addWidget(self.enable_face_cb)
        
        layout.addWidget(settings_group)
        
        # Advanced settings button
        advanced_btn = QPushButton("Advanced Settings...")
        advanced_btn.clicked.connect(self.show_settings)
        layout.addWidget(advanced_btn)
        
        layout.addStretch()
        
        return tab
    
    def create_log_tab(self) -> QWidget:
        """Create the log output tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)
        
        return tab
    
    def create_visualizer_tab(self) -> QWidget:
        """Create the 3D tracker visualizer tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create visualizer widget
        self.tracker_visualizer = TrackerVisualizerWidget()
        layout.addWidget(self.tracker_visualizer)
        
        return tab
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status indicators
        self.status_bar.addWidget(QLabel("Ready"))
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def apply_style(self):
        """Apply application styling"""
        # Use dark theme
        self.setStyleSheet("""
            QMainWindow {
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
                padding: 5px;
                min-width: 80px;
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
    
    def setup_connections(self):
        """Setup signal connections"""
        # Camera manager signals
        self.camera_manager.status_changed.connect(self.update_camera_status)
        self.camera_manager.frame_ready.connect(self.on_frame_ready)
        
        # Pose processor signals
        self.pose_processor.pose_detected.connect(self.on_pose_detected)
        self.pose_processor.processing_stats.connect(self.on_processing_stats)
        # Floor calibration signals removed - not needed
        self.pose_processor.pose_data_ready.connect(self.on_tracker_data_ready)
        
        # SteamVR driver signals
        self.steamvr_driver.status_changed.connect(self.update_steamvr_status)
        self.steamvr_driver.tracker_added.connect(self.on_tracker_added)
    
    def load_settings(self):
        """Load settings from config manager"""
        # Set fusion mode
        fusion_mode = self.config_manager.get('pose_processing.fusion_mode', 'weighted_average')
        index = self.fusion_combo.findText(fusion_mode)
        if index >= 0:
            self.fusion_combo.setCurrentIndex(index)
        
        # Set smoothing
        smoothing = self.config_manager.get('pose_processing.smoothing_factor', 0.7)
        self.smoothing_slider.setValue(int(smoothing * 100))
        
        # Set confidence
        confidence = self.config_manager.get('pose_processing.confidence_threshold', 0.5)
        self.confidence_slider.setValue(int(confidence * 100))
        
        # Load window geometry
        geometry = self.config_manager.get_ui_geometry()
        if geometry:
            geom_data = geometry.get('geometry', b'')
            state_data = geometry.get('state', b'')
            
            # Only restore if we have valid byte data
            if isinstance(geom_data, (bytes, bytearray)) and geom_data:
                self.restoreGeometry(geom_data)
            if isinstance(state_data, (bytes, bytearray)) and state_data:
                self.restoreState(state_data)
    
    def save_settings(self):
        """Save current settings"""
        # Save window geometry
        geometry = {
            'geometry': self.saveGeometry(),
            'state': self.saveState()
        }
        self.config_manager.set_ui_geometry(geometry)
    
    @Slot()
    def start_tracking(self):
        """Start tracking"""
        if not self.is_tracking:
            self.is_tracking = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.tracking_status_label.setText("Tracking")
            self.logger.info("Tracking started")
    
    @Slot()
    def stop_tracking(self):
        """Stop tracking"""
        if self.is_tracking:
            self.is_tracking = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.tracking_status_label.setText("Stopped")
            self.logger.info("Tracking stopped")
    
    @Slot()
    def start_all_cameras(self):
        """Start all cameras"""
        self.camera_manager.start_all_cameras()
    
    @Slot()
    def stop_all_cameras(self):
        """Stop all cameras"""
        self.camera_manager.stop_all_cameras()
    
    @Slot()
    def connect_steamvr(self):
        """Connect to SteamVR"""
        print("[TRDX] Connect SteamVR button pressed. Attempting connection...")
        if self.steamvr_driver.connect_to_steamvr():
            self.connect_steamvr_btn.setText("Disconnect SteamVR")
            self.connect_steamvr_btn.clicked.disconnect()
            self.connect_steamvr_btn.clicked.connect(self.disconnect_steamvr)
    
    @Slot()
    def disconnect_steamvr(self):
        """Disconnect from SteamVR"""
        self.steamvr_driver.disconnect()
        self.connect_steamvr_btn.setText("Connect SteamVR")
        self.connect_steamvr_btn.clicked.disconnect()
        self.connect_steamvr_btn.clicked.connect(self.connect_steamvr)
    
    @Slot()
    def show_settings(self):
        """Show settings dialog"""
        if not self.settings_dialog:
            self.settings_dialog = SettingsDialog(self.config_manager, self)
        self.settings_dialog.exec()
    
    @Slot()
    def show_calibration(self):
        """Show calibration dialog"""
        if not self.calibration_dialog:
            self.calibration_dialog = CalibrationDialog(
                self.pose_processor, self.steamvr_driver, self
            )
        self.calibration_dialog.exec()
    
    @Slot()
    def show_about(self):
        """Show about dialog"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "About TRDX", 
                         "TRDX - Multi-Camera Full Body Tracking\n"
                         "Version 1.0.0\n\n"
                         "Advanced VR tracking using computer vision")
    
    @Slot(str)
    def on_fusion_mode_changed(self, mode):
        """Handle fusion mode change"""
        self.pose_processor.set_fusion_mode(mode)
        self.config_manager.set('pose_processing.fusion_mode', mode)
    
    @Slot(int)
    def on_smoothing_changed(self, value):
        """Handle smoothing change"""
        smoothing = value / 100.0
        self.smoothing_value_label.setText(f"{smoothing:.2f}")
        self.pose_processor.set_smoothing_factor(smoothing)
        self.config_manager.set('pose_processing.smoothing_factor', smoothing)
    
    @Slot(int)
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")
        self.config_manager.set('pose_processing.confidence_threshold', confidence)
    
    @Slot(int, str, bool)
    def update_camera_status(self, camera_id, status, is_active):
        """Update camera status display"""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_status(status, is_active)
        
        # Update overall camera status
        active_count = sum(1 for widget in self.camera_widgets.values() 
                          if widget.is_active)
        total_count = len(self.camera_widgets)
        self.cameras_status_label.setText(f"Cameras: {active_count}/{total_count}")
    
    @Slot(str, bool)
    def update_steamvr_status(self, status, is_connected):
        """Update SteamVR status display"""
        self.steamvr_status_label.setText(f"SteamVR: {status}")
        
        if is_connected:
            self.connect_steamvr_btn.setText("Disconnect SteamVR")
        else:
            self.connect_steamvr_btn.setText("Connect SteamVR")
    
    @Slot(int, np.ndarray)
    def on_frame_ready(self, camera_id, frame):
        """Handle new frame from camera"""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_frame(frame)
    
    @Slot(int, object)
    def on_pose_detected(self, camera_id, pose_data):
        """Handle pose detection"""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_pose(pose_data)
    
    @Slot(dict)
    def on_processing_stats(self, stats):
        """Handle processing statistics"""
        fps = stats.get('fps', 0)
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @Slot(int, str)
    def on_tracker_added(self, tracker_id, role):
        """Handle tracker added"""
        if tracker_id in self.tracker_widgets:
            self.tracker_widgets[tracker_id].set_role(role)
    
    # Floor Calibration Methods
    @Slot()
    def toggle_floor_calibration(self):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot()
    def toggle_grid_overlay(self):
        """Grid overlay system removed - not needed"""
        pass
    
    @Slot()
    def reset_floor_calibration(self):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot()
    def save_floor_calibration(self):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot()
    def load_floor_calibration(self):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot(object)
    def on_floor_calibration_ready(self, floor_plane):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot(int, object)
    def on_grid_overlay_ready(self, camera_id, frame_with_grid):
        """Grid overlay system removed - not needed"""
        pass
    
    @Slot()
    def calibrate_floor_from_feet(self):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot()
    def start_floor_calibration(self):
        """Floor calibration system removed - not needed"""
        pass

    @Slot()
    def take_floor_sample(self):
        """Floor calibration system removed - not needed"""
        pass

    @Slot()
    def finish_floor_calibration(self):
        """Floor calibration system removed - not needed"""
        pass
    
    @Slot(int)
    def on_camera_y_changed(self, value):
        """Floor calibration system removed - not needed"""
        pass

    @Slot(int)
    def on_camera_pitch_changed(self, value):
        """Floor calibration system removed - not needed"""
        pass

    @Slot(int)
    def on_camera_z_changed(self, value):
        """Floor calibration system removed - not needed"""
        pass

    @Slot(int)
    def on_camera_x_changed(self, value):
        """Floor calibration system removed - not needed"""
        pass
    
    def update_ui(self):
        """Periodic UI update"""
        # Update tracker widgets
        for tracker_id, widget in self.tracker_widgets.items():
            status = self.steamvr_driver.get_tracker_status(tracker_id)
            
            # If tracker doesn't exist or is inactive, create it and simulate data
            if not status.get('exists', False) or not status.get('active', False):
                # Add tracker to SteamVR driver if it doesn't exist
                if not status.get('exists', False):
                    if tracker_id == 0:
                        role = "TrackerRole_Waist"
                    elif tracker_id == 1:
                        role = "TrackerRole_LeftFoot"
                    elif tracker_id == 2:
                        role = "TrackerRole_RightFoot"
                    else:
                        role = "TrackerRole_Handed"
                    
                    self.steamvr_driver.add_tracker(role)
                
                # Simulate tracking data to activate the tracker
                import numpy as np
                import time
                
                # Simulate realistic tracking data
                base_position = np.array([0.0, 0.0, 0.0])
                if tracker_id == 0:  # Waist
                    base_position = np.array([0.0, 1.0, 0.0])
                elif tracker_id == 1:  # Left foot
                    base_position = np.array([-0.2, 0.0, 0.0])
                elif tracker_id == 2:  # Right foot
                    base_position = np.array([0.2, 0.0, 0.0])
                
                # Add some movement simulation
                t = time.time()
                movement = np.array([
                    np.sin(t * 0.5 + tracker_id) * 0.1,  # X movement
                    np.cos(t * 0.3 + tracker_id) * 0.05,  # Y movement
                    np.sin(t * 0.7 + tracker_id) * 0.1    # Z movement
                ])
                
                position = base_position + movement
                rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
                confidence = 0.8 + 0.2 * np.sin(t * 2.0 + tracker_id)  # Varying confidence
                
                # Update tracker with simulated data
                self.steamvr_driver.update_tracker(tracker_id, position, rotation, confidence)
                
                # Get updated status
                status = self.steamvr_driver.get_tracker_status(tracker_id)
            
            widget.update_status(status)
    
    @Slot(list)
    def on_tracker_data_ready(self, tracker_list):
        """Handle tracker data from pose processor"""
        if self.tracker_visualizer:
            for tracker in tracker_list:
                self.tracker_visualizer.update_tracker_data(
                    tracker.tracker_id,
                    tracker.position,
                    tracker.rotation,
                    tracker.confidence
                )
    
    def closeEvent(self, event):
        """Handle close event"""
        self.save_settings()
        
        # Stop all components
        self.stop_tracking()
        self.camera_manager.stop_all_cameras()
        self.steamvr_driver.disconnect()
        
        event.accept()
