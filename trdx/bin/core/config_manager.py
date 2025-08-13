"""
Configuration Manager - Handles settings and configuration for TRDX
Enhanced with robust backup and validation systems to prevent corruption
"""

import json
import logging
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from PySide6.QtCore import QObject, Signal

# Import configuration validator
try:
    from .config_validator import ConfigValidator, validate_config_file
    CONFIG_VALIDATOR_AVAILABLE = True
except ImportError:
    CONFIG_VALIDATOR_AVAILABLE = False

class ConfigManager(QObject):
    """Manages application configuration and settings with corruption protection"""
    
    config_changed = Signal(str, object)  # setting_name, new_value
    config_loaded = Signal()
    config_saved = Signal()
    
    def __init__(self, config_file: Optional[str] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Configuration file path
        if config_file:
            self.config_file = Path(config_file)
        else:
            # Default to project directory
            project_root = Path(__file__).parent.parent.parent
            self.config_file = project_root / "config" / "trdx_config.json"
        
        # Backup file paths
        self.backup_file = self.config_file.with_suffix('.json.backup')
        self.emergency_backup = self.config_file.with_suffix('.json.emergency')
        
        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.default_config = {
            # Application settings
            "app": {
                "version": "1.0.0",
                "auto_start_cameras": False,
                "auto_connect_steamvr": True,
                "minimize_to_tray": False,
                "language": "en"
            },
            
            # Camera settings
            "cameras": {
                "max_cameras": 4,
                "default_resolution": [640, 480],
                "default_fps": 30,
                "auto_detect": True,
                "camera_configs": {}
            },
            
            # Pose processing settings
            "pose_processing": {
                "model_complexity": 1,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "smoothing_factor": 0.7,
                "confidence_threshold": 0.5,
                "fusion_mode": "weighted_average",
                "enable_hands": False,
                "enable_face": False
            },
            
            # SteamVR settings
            "steamvr": {
                "auto_install_driver": True,
                "auto_add_trackers": True,
                "smoothing": 0.5,
                "additional_smoothing": 0.7,
                "latency_compensation": 0.0,
                "tracker_roles": {
                    0: "TrackerRole_Waist",
                    1: "TrackerRole_LeftFoot", 
                    2: "TrackerRole_RightFoot"
                },
                "headset_reference": {
                    "enable_headset_tracking": True,
                    "update_frequency": 90,
                    "pose_smoothing_factor": 0.7,
                    "confidence_threshold": 0.8,
                    "use_headset_for_alignment": True,
                    "auto_calibrate_user_height": True,
                    "enhanced_alignment": {
                        "stability_factor": 0.85,
                        "precision_factor": 0.15,
                        "height_adaptation_speed": 0.02,
                        "outlier_rejection_threshold": 0.3,
                        "min_calibration_samples": 120
                    }
                },
                "vr_integration": {
                    "enable_headset_reference": True,
                    "use_controller_height_reference": True,
                    "auto_detect_floor_from_controllers": True,
                    "stabilization_factor": 0.8,
                    "vr_data_port": 6970
                }
            },
            
            # Calibration settings
            "calibration": {
                "auto_calibrate": True,
                "calibration_time": 3.0,
                "scale": 1.0,
                "rotation": [0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0],
                "tracking_space_height": 1.8
            },
            
            # UI settings
            "ui": {
                "theme": "dark",
                "window_geometry": None,
                "window_state": None,
                "show_debug_info": False,
                "show_skeleton": False,
                "camera_preview_size": [320, 240]
            },
            
            # Performance settings
            "performance": {
                "max_fps": 60,
                "use_gpu_acceleration": True,
                "multi_threading": True,
                "memory_limit_mb": 1024,
                "log_level": "INFO"
            }
        }
        
        # Current configuration
        self.config = self.default_config.copy()
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file with robust validation and auto-restore"""
        try:
            if self.config_file.exists():
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            raise ValueError("Empty config file")
                        loaded_config = json.loads(content)
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(f"Config file corrupted: {e}")
                    # Try to restore from backup
                    backup_file = self.config_file.with_suffix('.json.backup')
                    if backup_file.exists():
                        try:
                            import shutil
                            shutil.copy2(backup_file, self.config_file)
                            self.logger.info("Restored config from backup")
                            with open(self.config_file, 'r', encoding='utf-8') as f:
                                loaded_config = json.load(f)
                        except Exception:
                            self.logger.error("Backup also corrupted, using defaults")
                            loaded_config = {}
                    else:
                        self.logger.warning("No backup found, using defaults")
                        loaded_config = {}
                
                self.config = self._merge_configs(self.default_config, loaded_config)
                
                # Validate configuration using the new validator
                if CONFIG_VALIDATOR_AVAILABLE:
                    validator = ConfigValidator()
                    if not validator.validate_config(self.config):
                        self.logger.warning("Configuration validation found issues:")
                        validator.print_validation_report()
                    else:
                        self.logger.info("Configuration validation passed")
                else:
                    # Fallback to old validation
                    validation_errors = self.validate_config()
                    if validation_errors:
                        self.logger.warning(f"Configuration validation found {len(validation_errors)} issues")
                        for error in validation_errors:
                            self.logger.warning(f"  - {error}")
                
                self.logger.info(f"Configuration loaded from: {self.config_file}")
            else:
                self.config = self.default_config.copy()
                if not self.config_file.with_suffix('.json.backup').exists():
                    self.save_config()
                self.logger.info("Created default configuration")
            self.config_loaded.emit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = self.default_config.copy()
            return False

    def save_config(self) -> bool:
        """Save configuration to file with atomic write and backup protection"""
        try:
            backup_file = self.config_file.with_suffix('.json.backup')
            if self.config_file.exists():
                import shutil
                shutil.copy2(self.config_file, backup_file)
            try:
                test_json = json.dumps(self.config, indent=2, ensure_ascii=False)
                json.loads(test_json)
            except Exception as e:
                self.logger.error(f"Configuration validation failed: {e}")
                return False
            import tempfile, os
            dirpath = os.path.dirname(self.config_file)
            with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False, encoding='utf-8') as tf:
                json.dump(self.config, tf, indent=2, ensure_ascii=False)
                tempname = tf.name
            os.replace(tempname, self.config_file)
            self.logger.info(f"Configuration saved to: {self.config_file}")
            self.config_saved.emit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            backup_file = self.config_file.with_suffix('.json.backup')
            if backup_file.exists():
                try:
                    import shutil
                    shutil.copy2(backup_file, self.config_file)
                    self.logger.info("Restored configuration from backup")
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore backup: {restore_error}")
            return False
    
    def _merge_configs(self, default: dict, loaded: dict) -> dict:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'cameras.max_cameras')"""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            if default is not None:
                return default
            
            # Try to get from defaults
            try:
                value = self.default_config
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return None
    
    def set(self, key_path: str, value: Any, save_immediately: bool = False):
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config_ref = self.config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the value
            old_value = config_ref.get(keys[-1])
            config_ref[keys[-1]] = value
            
            # Emit signal if value changed
            if old_value != value:
                self.config_changed.emit(key_path, value)
            
            # Save if requested
            if save_immediately:
                self.save_config()
            
            self.logger.debug(f"Config set: {key_path} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to set config {key_path}: {e}")
    
    def get_section(self, section: str) -> dict:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def set_section(self, section: str, config_dict: dict, save_immediately: bool = False):
        """Set entire configuration section"""
        self.config[section] = config_dict
        self.config_changed.emit(section, config_dict)
        
        if save_immediately:
            self.save_config()
    
    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to defaults"""
        if section:
            if section in self.default_config:
                self.config[section] = self.default_config[section].copy()
                self.config_changed.emit(section, self.config[section])
        else:
            self.config = self.default_config.copy()
            self.config_changed.emit("*", self.config)
        
        self.logger.info(f"Configuration reset to defaults: {section or 'all'}")
    
    def export_config(self, file_path: str) -> bool:
        """Export configuration to a file"""
        try:
            export_path = Path(file_path)
            with open(export_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, file_path: str) -> bool:
        """Import configuration from a file"""
        try:
            import_path = Path(file_path)
            if not import_path.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_path, 'r') as f:
                imported_config = json.load(f)
            
            # Merge with current config
            self.config = self._merge_configs(self.config, imported_config)
            self.config_changed.emit("*", self.config)
            
            self.logger.info(f"Configuration imported from: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of errors"""
        errors = []
        
        try:
            # Validate camera settings
            max_cameras = self.get('cameras.max_cameras')
            if not isinstance(max_cameras, int) or max_cameras < 1 or max_cameras > 8:
                errors.append("cameras.max_cameras must be between 1 and 8")
            
            # Validate resolution
            resolution = self.get('cameras.default_resolution')
            if not isinstance(resolution, list) or len(resolution) != 2:
                errors.append("cameras.default_resolution must be [width, height]")
            elif not all(isinstance(x, int) and x > 0 for x in resolution):
                errors.append("cameras.default_resolution values must be positive integers")
            
            # Validate FPS
            fps = self.get('cameras.default_fps')
            if not isinstance(fps, (int, float)) or fps <= 0 or fps > 120:
                errors.append("cameras.default_fps must be between 0 and 120")
            
            # Validate pose processing settings
            model_complexity = self.get('pose_processing.model_complexity')
            if model_complexity not in [0, 1, 2]:
                errors.append("pose_processing.model_complexity must be 0, 1, or 2")
            
            # Validate confidence thresholds
            for setting in ['min_detection_confidence', 'min_tracking_confidence', 'confidence_threshold']:
                value = self.get(f'pose_processing.{setting}')
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    errors.append(f"pose_processing.{setting} must be between 0 and 1")
            
            # Validate fusion mode
            fusion_mode = self.get('pose_processing.fusion_mode')
            valid_modes = ['best_camera', 'weighted_average', 'multi_view']
            if fusion_mode not in valid_modes:
                errors.append(f"pose_processing.fusion_mode must be one of: {valid_modes}")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors
    
    def get_camera_config(self, camera_id: int) -> dict:
        """Get configuration for a specific camera"""
        camera_configs = self.get('cameras.camera_configs', {})
        
        if str(camera_id) in camera_configs:
            return camera_configs[str(camera_id)]
        else:
            # Return default config for camera
            return {
                'enabled': False,
                'index': camera_id,
                'resolution': self.get('cameras.default_resolution'),
                'fps': self.get('cameras.default_fps'),
                'name': f"Camera {camera_id}"
            }
    
    def set_camera_config(self, camera_id: int, config: dict):
        """Set configuration for a specific camera"""
        camera_configs = self.get('cameras.camera_configs', {})
        camera_configs[str(camera_id)] = config
        self.set('cameras.camera_configs', camera_configs)
    
    def get_tracker_role(self, tracker_id: int) -> str:
        """Get SteamVR role for a tracker"""
        roles = self.get('steamvr.tracker_roles', {})
        return roles.get(tracker_id, "TrackerRole_Handed")
    
    def set_tracker_role(self, tracker_id: int, role: str):
        """Set SteamVR role for a tracker"""
        roles = self.get('steamvr.tracker_roles', {})
        roles[tracker_id] = role
        self.set('steamvr.tracker_roles', roles)
    
    def get_ui_geometry(self) -> Optional[dict]:
        """Get saved UI window geometry"""
        return self.get('ui.window_geometry')
    
    def set_ui_geometry(self, geometry: dict):
        """Save UI window geometry"""
        # Convert QByteArray to bytes for JSON serialization
        serializable_geometry = {}
        for key, value in geometry.items():
            if hasattr(value, 'data'):  # QByteArray
                serializable_geometry[key] = bytes(value).hex()  # Convert to hex string
            else:
                serializable_geometry[key] = value
        self.set('ui.window_geometry', serializable_geometry, save_immediately=True)
    
    def __str__(self) -> str:
        return f"ConfigManager(file={self.config_file})"
    
    def __repr__(self) -> str:
        return self.__str__()
