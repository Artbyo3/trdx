"""
Settings Dialog - Application configuration dialog
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QFormLayout, QDialogButtonBox, QWidget
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
import logging

class SettingsDialog(QDialog):
    """Settings dialog for TRDX application"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Track changes
        self.changes_made = False
        
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TRDX Settings")
        self.setModal(True)
        self.resize(600, 500)
        
        # Configurar icono del diálogo
        try:
            from pathlib import Path
            from PySide6.QtGui import QIcon
            icon_path = Path(__file__).parent.parent.parent / "assets" / "logos" / "logo.ico"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                self.logger.info(f"✅ Icono de diálogo cargado: {icon_path}")
            else:
                self.logger.warning(f"⚠️ Icono no encontrado en: {icon_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error al cargar icono de diálogo: {e}")
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Camera settings tab
        camera_tab = self.create_camera_tab()
        tab_widget.addTab(camera_tab, "Cameras")
        
        # Pose processing tab
        pose_tab = self.create_pose_tab()
        tab_widget.addTab(pose_tab, "Pose Processing")
        
        # SteamVR tab
        steamvr_tab = self.create_steamvr_tab()
        tab_widget.addTab(steamvr_tab, "SteamVR")
        
        # Performance tab
        performance_tab = self.create_performance_tab()
        tab_widget.addTab(performance_tab, "Performance")
        
        # UI tab
        ui_tab = self.create_ui_tab()
        tab_widget.addTab(ui_tab, "Interface")
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        layout.addWidget(button_box)
        
        # Add reset to defaults button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_box.addButton(reset_btn, QDialogButtonBox.ResetRole)
    
    def create_camera_tab(self):
        """Create camera settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # General camera settings
        general_group = QGroupBox("General")
        general_layout = QFormLayout(general_group)
        
        self.max_cameras_spin = QSpinBox()
        self.max_cameras_spin.setRange(1, 8)
        general_layout.addRow("Maximum Cameras:", self.max_cameras_spin)
        
        self.default_width_spin = QSpinBox()
        self.default_width_spin.setRange(320, 1920)
        self.default_width_spin.setSingleStep(160)
        general_layout.addRow("Default Width:", self.default_width_spin)
        
        self.default_height_spin = QSpinBox()
        self.default_height_spin.setRange(240, 1080)
        self.default_height_spin.setSingleStep(120)
        general_layout.addRow("Default Height:", self.default_height_spin)
        
        self.default_fps_spin = QSpinBox()
        self.default_fps_spin.setRange(15, 120)
        general_layout.addRow("Default FPS:", self.default_fps_spin)
        
        self.auto_detect_cb = QCheckBox("Auto-detect cameras")
        general_layout.addRow("", self.auto_detect_cb)
        
        layout.addWidget(general_group)
        
        layout.addStretch()
        return tab
    
    def create_pose_tab(self):
        """Create pose processing settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # MediaPipe settings
        mediapipe_group = QGroupBox("MediaPipe")
        mediapipe_layout = QFormLayout(mediapipe_group)
        
        self.model_complexity_combo = QComboBox()
        self.model_complexity_combo.addItems(["0 (Fast)", "1 (Balanced)", "2 (Accurate)"])
        mediapipe_layout.addRow("Model Complexity:", self.model_complexity_combo)
        
        self.min_detection_conf_spin = QDoubleSpinBox()
        self.min_detection_conf_spin.setRange(0.0, 1.0)
        self.min_detection_conf_spin.setSingleStep(0.1)
        self.min_detection_conf_spin.setDecimals(2)
        mediapipe_layout.addRow("Min Detection Confidence:", self.min_detection_conf_spin)
        
        self.min_tracking_conf_spin = QDoubleSpinBox()
        self.min_tracking_conf_spin.setRange(0.0, 1.0)
        self.min_tracking_conf_spin.setSingleStep(0.1)
        self.min_tracking_conf_spin.setDecimals(2)
        mediapipe_layout.addRow("Min Tracking Confidence:", self.min_tracking_conf_spin)
        
        layout.addWidget(mediapipe_group)
        
        # Processing settings
        processing_group = QGroupBox("Processing")
        processing_layout = QFormLayout(processing_group)
        
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 1.0)
        self.smoothing_spin.setSingleStep(0.1)
        self.smoothing_spin.setDecimals(2)
        processing_layout.addRow("Smoothing Factor:", self.smoothing_spin)
        
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.0, 1.0)
        self.confidence_threshold_spin.setSingleStep(0.1)
        self.confidence_threshold_spin.setDecimals(2)
        processing_layout.addRow("Confidence Threshold:", self.confidence_threshold_spin)
        
        self.fusion_mode_combo = QComboBox()
        self.fusion_mode_combo.addItems(["best_camera", "weighted_average", "multi_view"])
        processing_layout.addRow("Fusion Mode:", self.fusion_mode_combo)
        
        self.enable_hands_cb = QCheckBox("Enable hand tracking")
        processing_layout.addRow("", self.enable_hands_cb)
        
        self.enable_face_cb = QCheckBox("Enable face tracking")
        processing_layout.addRow("", self.enable_face_cb)
        
        layout.addWidget(processing_group)

        # --- Manual Body Offset ---
        offset_group = QGroupBox("Manual Body Offset (Ajuste de posición cuerpo)")
        offset_layout = QFormLayout(offset_group)
        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-2.0, 2.0)
        self.offset_x_spin.setSingleStep(0.01)
        self.offset_x_spin.setDecimals(3)
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-2.0, 2.0)
        self.offset_y_spin.setSingleStep(0.01)
        self.offset_y_spin.setDecimals(3)
        self.offset_z_spin = QDoubleSpinBox()
        self.offset_z_spin.setRange(-2.0, 2.0)
        self.offset_z_spin.setSingleStep(0.01)
        self.offset_z_spin.setDecimals(3)
        offset_layout.addRow("Offset X:", self.offset_x_spin)
        offset_layout.addRow("Offset Y:", self.offset_y_spin)
        offset_layout.addRow("Offset Z:", self.offset_z_spin)
        self.offset_reset_btn = QPushButton("Reset Offset")
        self.offset_reset_btn.clicked.connect(self.reset_offset)
        offset_layout.addRow(self.offset_reset_btn)
        layout.addWidget(offset_group)

        # --- Headset Reference Calibration ---
        self.recalibrate_btn = QPushButton("Recalibrar referencia del headset")
        self.recalibrate_btn.clicked.connect(self.recalibrate_headset_reference)
        layout.addWidget(self.recalibrate_btn)
        self.calib_status_label = QLabel("Estado de calibración: desconocido")
        layout.addWidget(self.calib_status_label)
        
        layout.addStretch()
        return tab
    
    def create_steamvr_tab(self):
        """Create SteamVR settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Driver settings
        driver_group = QGroupBox("Driver")
        driver_layout = QFormLayout(driver_group)
        
        self.auto_install_driver_cb = QCheckBox("Auto-install driver")
        driver_layout.addRow("", self.auto_install_driver_cb)
        
        self.auto_add_trackers_cb = QCheckBox("Auto-add trackers")
        driver_layout.addRow("", self.auto_add_trackers_cb)
        
        layout.addWidget(driver_group)
        
        # Tracking settings
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QFormLayout(tracking_group)
        
        self.steamvr_smoothing_spin = QDoubleSpinBox()
        self.steamvr_smoothing_spin.setRange(0.0, 1.0)
        self.steamvr_smoothing_spin.setSingleStep(0.1)
        self.steamvr_smoothing_spin.setDecimals(2)
        tracking_layout.addRow("Smoothing:", self.steamvr_smoothing_spin)
        
        self.additional_smoothing_spin = QDoubleSpinBox()
        self.additional_smoothing_spin.setRange(0.0, 1.0)
        self.additional_smoothing_spin.setSingleStep(0.1)
        self.additional_smoothing_spin.setDecimals(2)
        tracking_layout.addRow("Additional Smoothing:", self.additional_smoothing_spin)
        
        self.latency_compensation_spin = QDoubleSpinBox()
        self.latency_compensation_spin.setRange(0.0, 0.1)
        self.latency_compensation_spin.setSingleStep(0.01)
        self.latency_compensation_spin.setDecimals(3)
        self.latency_compensation_spin.setSuffix(" s")
        tracking_layout.addRow("Latency Compensation:", self.latency_compensation_spin)
        
        layout.addWidget(tracking_group)
        
        layout.addStretch()
        return tab
    
    def create_performance_tab(self):
        """Create performance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Performance settings
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout(performance_group)
        
        self.max_fps_spin = QSpinBox()
        self.max_fps_spin.setRange(15, 120)
        performance_layout.addRow("Max FPS:", self.max_fps_spin)
        
        self.use_gpu_cb = QCheckBox("Use GPU acceleration")
        performance_layout.addRow("", self.use_gpu_cb)
        
        self.multi_threading_cb = QCheckBox("Enable multi-threading")
        performance_layout.addRow("", self.multi_threading_cb)
        
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(512, 4096)
        self.memory_limit_spin.setSuffix(" MB")
        performance_layout.addRow("Memory Limit:", self.memory_limit_spin)
        
        layout.addWidget(performance_group)
        
        # Logging settings
        logging_group = QGroupBox("Logging")
        logging_layout = QFormLayout(logging_group)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        logging_layout.addRow("Log Level:", self.log_level_combo)
        
        layout.addWidget(logging_group)
        
        layout.addStretch()
        return tab
    
    def create_ui_tab(self):
        """Create UI settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # UI settings
        ui_group = QGroupBox("Interface")
        ui_layout = QFormLayout(ui_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light", "auto"])
        ui_layout.addRow("Theme:", self.theme_combo)
        
        self.show_debug_cb = QCheckBox("Show debug information")
        ui_layout.addRow("", self.show_debug_cb)
        
        self.show_skeleton_cb = QCheckBox("Show pose skeleton")
        ui_layout.addRow("", self.show_skeleton_cb)
        
        layout.addWidget(ui_group)
        
        layout.addStretch()
        return tab
    
    def load_settings(self):
        """Load settings from config manager"""
        # Camera settings
        self.max_cameras_spin.setValue(self.config_manager.get('cameras.max_cameras', 4))
        resolution = self.config_manager.get('cameras.default_resolution', [640, 480])
        self.default_width_spin.setValue(resolution[0])
        self.default_height_spin.setValue(resolution[1])
        self.default_fps_spin.setValue(self.config_manager.get('cameras.default_fps', 30))
        self.auto_detect_cb.setChecked(self.config_manager.get('cameras.auto_detect', True))
        
        # Pose processing settings
        complexity = self.config_manager.get('pose_processing.model_complexity', 1)
        self.model_complexity_combo.setCurrentIndex(complexity)
        
        self.min_detection_conf_spin.setValue(
            self.config_manager.get('pose_processing.min_detection_confidence', 0.5)
        )
        self.min_tracking_conf_spin.setValue(
            self.config_manager.get('pose_processing.min_tracking_confidence', 0.5)
        )
        self.smoothing_spin.setValue(
            self.config_manager.get('pose_processing.smoothing_factor', 0.7)
        )
        self.confidence_threshold_spin.setValue(
            self.config_manager.get('pose_processing.confidence_threshold', 0.5)
        )
        
        fusion_mode = self.config_manager.get('pose_processing.fusion_mode', 'weighted_average')
        index = self.fusion_mode_combo.findText(fusion_mode)
        if index >= 0:
            self.fusion_mode_combo.setCurrentIndex(index)
        
        self.enable_hands_cb.setChecked(
            self.config_manager.get('pose_processing.enable_hands', False)
        )
        self.enable_face_cb.setChecked(
            self.config_manager.get('pose_processing.enable_face', False)
        )
        
        # Manual body offset
        offset = self.config_manager.get('pose_processing.manual_body_offset', [0.0, 0.0, 0.0])
        if len(offset) == 3:
            self.offset_x_spin.setValue(offset[0])
            self.offset_y_spin.setValue(offset[1])
            self.offset_z_spin.setValue(offset[2])
        
        # SteamVR settings
        self.auto_install_driver_cb.setChecked(
            self.config_manager.get('steamvr.auto_install_driver', True)
        )
        self.auto_add_trackers_cb.setChecked(
            self.config_manager.get('steamvr.auto_add_trackers', True)
        )
        self.steamvr_smoothing_spin.setValue(
            self.config_manager.get('steamvr.smoothing', 0.5)
        )
        self.additional_smoothing_spin.setValue(
            self.config_manager.get('steamvr.additional_smoothing', 0.7)
        )
        self.latency_compensation_spin.setValue(
            self.config_manager.get('steamvr.latency_compensation', 0.0)
        )
        
        # Performance settings
        self.max_fps_spin.setValue(self.config_manager.get('performance.max_fps', 60))
        self.use_gpu_cb.setChecked(
            self.config_manager.get('performance.use_gpu_acceleration', True)
        )
        self.multi_threading_cb.setChecked(
            self.config_manager.get('performance.multi_threading', True)
        )
        self.memory_limit_spin.setValue(
            self.config_manager.get('performance.memory_limit_mb', 1024)
        )
        
        log_level = self.config_manager.get('performance.log_level', 'INFO')
        index = self.log_level_combo.findText(log_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        
        # UI settings
        theme = self.config_manager.get('ui.theme', 'dark')
        index = self.theme_combo.findText(theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        self.show_debug_cb.setChecked(self.config_manager.get('ui.show_debug_info', False))
        self.show_skeleton_cb.setChecked(self.config_manager.get('ui.show_skeleton', False))
    
    def save_settings(self):
        """Save settings to config manager"""
        # Camera settings
        self.config_manager.set('cameras.max_cameras', self.max_cameras_spin.value())
        self.config_manager.set('cameras.default_resolution', 
                               [self.default_width_spin.value(), self.default_height_spin.value()])
        self.config_manager.set('cameras.default_fps', self.default_fps_spin.value())
        self.config_manager.set('cameras.auto_detect', self.auto_detect_cb.isChecked())
        
        # Pose processing settings
        self.config_manager.set('pose_processing.model_complexity', 
                               self.model_complexity_combo.currentIndex())
        self.config_manager.set('pose_processing.min_detection_confidence',
                               self.min_detection_conf_spin.value())
        self.config_manager.set('pose_processing.min_tracking_confidence',
                               self.min_tracking_conf_spin.value())
        self.config_manager.set('pose_processing.smoothing_factor',
                               self.smoothing_spin.value())
        self.config_manager.set('pose_processing.confidence_threshold',
                               self.confidence_threshold_spin.value())
        self.config_manager.set('pose_processing.fusion_mode',
                               self.fusion_mode_combo.currentText())
        self.config_manager.set('pose_processing.enable_hands',
                               self.enable_hands_cb.isChecked())
        self.config_manager.set('pose_processing.enable_face',
                               self.enable_face_cb.isChecked())
        
        # Manual body offset
        offset = [self.offset_x_spin.value(), self.offset_y_spin.value(), self.offset_z_spin.value()]
        self.config_manager.set('pose_processing.manual_body_offset', offset)
        
        # SteamVR settings
        self.config_manager.set('steamvr.auto_install_driver',
                               self.auto_install_driver_cb.isChecked())
        self.config_manager.set('steamvr.auto_add_trackers',
                               self.auto_add_trackers_cb.isChecked())
        self.config_manager.set('steamvr.smoothing',
                               self.steamvr_smoothing_spin.value())
        self.config_manager.set('steamvr.additional_smoothing',
                               self.additional_smoothing_spin.value())
        self.config_manager.set('steamvr.latency_compensation',
                               self.latency_compensation_spin.value())
        
        # Performance settings
        self.config_manager.set('performance.max_fps', self.max_fps_spin.value())
        self.config_manager.set('performance.use_gpu_acceleration',
                               self.use_gpu_cb.isChecked())
        self.config_manager.set('performance.multi_threading',
                               self.multi_threading_cb.isChecked())
        self.config_manager.set('performance.memory_limit_mb',
                               self.memory_limit_spin.value())
        self.config_manager.set('performance.log_level',
                               self.log_level_combo.currentText())
        
        # UI settings
        self.config_manager.set('ui.theme', self.theme_combo.currentText())
        self.config_manager.set('ui.show_debug_info', self.show_debug_cb.isChecked())
        self.config_manager.set('ui.show_skeleton', self.show_skeleton_cb.isChecked())
        
        # Save to file
        self.config_manager.save_config()
        self.changes_made = True
    
    @Slot()
    def apply_settings(self):
        """Apply settings without closing dialog"""
        self.save_settings()
        QMessageBox.information(self, "Settings", "Settings applied successfully!")
    
    @Slot()
    def accept_settings(self):
        """Accept and save settings"""
        self.save_settings()
        self.accept()
    
    @Slot()
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings", 
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.reset_to_defaults()
            self.load_settings()
            QMessageBox.information(self, "Settings", "Settings reset to defaults!")

    def reset_offset(self):
        self.offset_x_spin.setValue(0.0)
        self.offset_y_spin.setValue(0.0)
        self.offset_z_spin.setValue(0.0)

    def recalibrate_headset_reference(self):
        if hasattr(self.parent(), 'pose_processor'):
            self.parent().pose_processor.recalibrate_headset_reference()
            self.calib_status_label.setText("Estado de calibración: recalibrado")
        else:
            self.calib_status_label.setText("No se pudo recalibrar (pose_processor no disponible)")
