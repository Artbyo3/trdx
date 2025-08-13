"""
Configuration Validator for TRDX
Validates configuration files and provides helpful error messages
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class ConfigValidator:
    """Validates TRDX configuration files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate a configuration dictionary"""
        self.errors.clear()
        self.warnings.clear()
        
        # Validate required sections
        required_sections = ['app', 'cameras', 'pose_processing', 'steamvr', 'calibration', 'ui', 'performance']
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required configuration section: {section}")
        
        if self.errors:
            return False
        
        # Validate each section
        self._validate_app_section(config.get('app', {}))
        self._validate_cameras_section(config.get('cameras', {}))
        self._validate_pose_processing_section(config.get('pose_processing', {}))
        self._validate_steamvr_section(config.get('steamvr', {}))
        self._validate_calibration_section(config.get('calibration', {}))
        self._validate_ui_section(config.get('ui', {}))
        self._validate_performance_section(config.get('performance', {}))
        
        return len(self.errors) == 0
    
    def _validate_app_section(self, app_config: Dict[str, Any]):
        """Validate app configuration section"""
        if 'version' not in app_config:
            self.warnings.append("App version not specified")
        
        if 'auto_start_cameras' not in app_config:
            self.warnings.append("auto_start_cameras not specified, defaulting to False")
    
    def _validate_cameras_section(self, cameras_config: Dict[str, Any]):
        """Validate cameras configuration section"""
        if 'max_cameras' not in cameras_config:
            self.errors.append("max_cameras not specified in cameras section")
        elif not isinstance(cameras_config['max_cameras'], int):
            self.errors.append("max_cameras must be an integer")
        elif cameras_config['max_cameras'] <= 0:
            self.errors.append("max_cameras must be greater than 0")
        
        if 'default_resolution' in cameras_config:
            resolution = cameras_config['default_resolution']
            if not isinstance(resolution, list) or len(resolution) != 2:
                self.errors.append("default_resolution must be a list of 2 integers [width, height]")
            elif not all(isinstance(x, int) and x > 0 for x in resolution):
                self.errors.append("default_resolution values must be positive integers")
    
    def _validate_pose_processing_section(self, pose_config: Dict[str, Any]):
        """Validate pose processing configuration section"""
        confidence_fields = ['min_detection_confidence', 'min_tracking_confidence', 'confidence_threshold']
        for field in confidence_fields:
            if field in pose_config:
                value = pose_config[field]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    self.errors.append(f"{field} must be a number between 0 and 1")
        
        if 'model_complexity' in pose_config:
            complexity = pose_config['model_complexity']
            if not isinstance(complexity, int) or complexity not in [0, 1, 2]:
                self.errors.append("model_complexity must be 0, 1, or 2")
    
    def _validate_steamvr_section(self, steamvr_config: Dict[str, Any]):
        """Validate SteamVR configuration section"""
        if 'tracker_roles' in steamvr_config:
            roles = steamvr_config['tracker_roles']
            if not isinstance(roles, dict):
                self.errors.append("tracker_roles must be a dictionary")
            else:
                valid_roles = [
                    "TrackerRole_Waist", "TrackerRole_LeftFoot", "TrackerRole_RightFoot",
                    "TrackerRole_LeftShoulder", "TrackerRole_RightShoulder", "TrackerRole_LeftElbow",
                    "TrackerRole_RightElbow", "TrackerRole_LeftKnee", "TrackerRole_RightKnee"
                ]
                for tracker_id, role in roles.items():
                    if role not in valid_roles:
                        self.warnings.append(f"Unknown tracker role: {role} for tracker {tracker_id}")
    
    def _validate_calibration_section(self, calibration_config: Dict[str, Any]):
        """Validate calibration configuration section"""
        if 'scale' in calibration_config:
            scale = calibration_config['scale']
            if not isinstance(scale, (int, float)) or scale <= 0:
                self.errors.append("calibration scale must be a positive number")
        
        if 'tracking_space_height' in calibration_config:
            height = calibration_config['tracking_space_height']
            if not isinstance(height, (int, float)) or height <= 0:
                self.errors.append("tracking_space_height must be a positive number")
    
    def _validate_ui_section(self, ui_config: Dict[str, Any]):
        """Validate UI configuration section"""
        if 'theme' in ui_config:
            theme = ui_config['theme']
            if theme not in ['dark', 'light', 'auto']:
                self.warnings.append(f"Unknown UI theme: {theme}")
    
    def _validate_performance_section(self, performance_config: Dict[str, Any]):
        """Validate performance configuration section"""
        if 'max_fps' in performance_config:
            fps = performance_config['max_fps']
            if not isinstance(fps, int) or fps <= 0:
                self.errors.append("max_fps must be a positive integer")
        
        if 'memory_limit_mb' in performance_config:
            memory = performance_config['memory_limit_mb']
            if not isinstance(memory, int) or memory <= 0:
                self.errors.append("memory_limit_mb must be a positive integer")
    
    def get_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get list of validation warnings"""
        return self.warnings.copy()
    
    def print_validation_report(self):
        """Print validation errors and warnings"""
        if self.errors:
            self.logger.error("Configuration validation errors:")
            for error in self.errors:
                self.logger.error(f"  ❌ {error}")
        
        if self.warnings:
            self.logger.warning("Configuration validation warnings:")
            for warning in self.warnings:
                self.logger.warning(f"  ⚠️ {warning}")
        
        if not self.errors and not self.warnings:
            self.logger.info("✅ Configuration validation passed")
        elif not self.errors:
            self.logger.info("✅ Configuration validation passed (with warnings)")
        else:
            self.logger.error("❌ Configuration validation failed")

def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file"""
    validator = ConfigValidator()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        is_valid = validator.validate_config(config)
        validator.print_validation_report()
        return is_valid
        
    except FileNotFoundError:
        validator.logger.error(f"Configuration file not found: {config_path}")
        return False
    except json.JSONDecodeError as e:
        validator.logger.error(f"Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        validator.logger.error(f"Error reading configuration file: {e}")
        return False 