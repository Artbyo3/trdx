"""
Enhanced Tracker Alignment - Improved precision using headset reference
Provides advanced algorithms for aligning virtual trackers with user body
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
import math

class BodyModel:
    """Human body model for improved tracker placement"""
    
    def __init__(self, user_height: float = 1.75):
        self.user_height = user_height
        
        # Body proportions (based on human anatomy averages)
        self.proportions = {
            'head_to_neck': 0.12 / user_height,
            'neck_to_shoulders': 0.05 / user_height,
            'shoulders_to_chest': 0.15 / user_height,
            'chest_to_waist': 0.20 / user_height,
            'waist_to_hips': 0.10 / user_height,
            'hips_to_upper_thigh': 0.15 / user_height,
            'upper_thigh_to_knee': 0.25 / user_height,
            'knee_to_ankle': 0.25 / user_height,
            'ankle_to_ground': 0.08 / user_height,
            'shoulder_width': 0.45 / user_height,
            'hip_width': 0.35 / user_height
        }
        
        # Tracker optimal positions (relative to body parts)
        self.tracker_offsets = {
            'hip': {
                'position_offset': np.array([0.0, 0.0, 0.05]),  # Slightly back from actual hip
                'min_distance_to_ground': 0.8,  # Minimum height from ground
                'stability_weight': 1.0  # How much to prioritize stability
            },
            'left_foot': {
                'position_offset': np.array([0.0, 0.05, 0.0]),  # Slightly above actual foot
                'min_distance_to_ground': 0.05,
                'stability_weight': 0.8
            },
            'right_foot': {
                'position_offset': np.array([0.0, 0.05, 0.0]),
                'min_distance_to_ground': 0.05,
                'stability_weight': 0.8
            }
        }

class EnhancedTrackerAlignment:
    """Advanced system for aligning trackers using headset reference"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        # Configuration
        self.config = config or {}
        self.stability_factor = self.config.get('stability_factor', 0.75)
        self.precision_factor = self.config.get('precision_factor', 0.25)
        self.height_adaptation_speed = self.config.get('height_adaptation_speed', 0.02)
        # Body anchoring configuration
        self.enable_body_anchoring = self.config.get('enable_body_anchoring', True)
        self.max_headset_distance = self.config.get('max_headset_distance', 2.0)
        # Body model
        self.body_model = BodyModel()
        self.estimated_user_height = 1.75
        # Tracking state
        self.previous_tracker_positions = {}
        self.position_history = []
        self.max_history_length = 30
        # Escalado configurable para transformar landmarks a mundo VR
        self.scale_x = self.config.get('landmark_scale_x', 2.0)
        self.scale_y = self.config.get('landmark_scale_y', 2.0)
        self.scale_z = self.config.get('landmark_scale_z', 2.0)

    def set_landmark_scales(self, scale_x: float, scale_y: float, scale_z: float):
        """Permite ajustar los factores de escala en tiempo real"""
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        
        # Calibration data
        self.headset_to_body_offset = np.array([0.0, 0.0, 0.0])
        self.is_calibrated = False
        self.calibration_samples = []
        self.min_calibration_samples = 120  # 4 seconds at 30fps
        
        # Error correction - more permissive for natural movement
        self.drift_correction_enabled = True
        self.outlier_rejection_threshold = 0.4  # Increased from 0.3 to allow more natural movement
        
        self.logger.info("Enhanced Tracker Alignment initialized with body anchoring")
    
    def update_user_height(self, height: float):
        """Update user height and recalculate body model"""
        if height > 1.4 and height < 2.2:  # Reasonable height range
            self.estimated_user_height = height
            self.body_model = BodyModel(height)
            self.logger.info(f"Updated user height to {height:.2f}m")
    
    def add_calibration_sample(self, mediapipe_landmarks: np.ndarray, headset_pose: np.ndarray):
        """Add a sample for calibrating headset-to-body offset"""
        if headset_pose is None or len(mediapipe_landmarks) < 33:
            return
        
        try:
            # Calculate MediaPipe head position
            mp_nose = mediapipe_landmarks[0]
            mp_left_ear = mediapipe_landmarks[7]
            mp_right_ear = mediapipe_landmarks[8]
            mp_head_center = np.mean([mp_nose, mp_left_ear, mp_right_ear], axis=0)
            
            # Calculate offset
            offset = headset_pose - mp_head_center
            
            self.calibration_samples.append({
                'headset_pose': headset_pose.copy(),
                'mp_head_center': mp_head_center.copy(),
                'offset': offset.copy(),
                'timestamp': time.time()
            })
            
            # Keep only recent samples
            if len(self.calibration_samples) > self.min_calibration_samples * 2:
                self.calibration_samples = self.calibration_samples[-self.min_calibration_samples:]
            
            # Update calibration if we have enough samples
            if len(self.calibration_samples) >= self.min_calibration_samples and not self.is_calibrated:
                self._update_calibration()
                
        except Exception as e:
            self.logger.warning(f"Error adding calibration sample: {e}")
    
    def _update_calibration(self):
        """Update calibration based on collected samples"""
        if len(self.calibration_samples) < self.min_calibration_samples:
            return
        
        try:
            # Calculate stable offset
            recent_offsets = [sample['offset'] for sample in self.calibration_samples[-self.min_calibration_samples:]]
            
            # Remove outliers
            offsets_array = np.array(recent_offsets)
            median_offset = np.median(offsets_array, axis=0)
            distances = np.linalg.norm(offsets_array - median_offset, axis=1)
            
            # Keep only samples within reasonable range
            valid_indices = distances < self.outlier_rejection_threshold
            if np.sum(valid_indices) < len(recent_offsets) * 0.7:
                self.logger.warning("Too many outlier samples in calibration")
                return
            
            valid_offsets = offsets_array[valid_indices]
            self.headset_to_body_offset = np.mean(valid_offsets, axis=0)
            
            # Estimate user height from calibration data
            head_heights = [sample['headset_pose'][1] for sample in self.calibration_samples[-30:]]
            avg_head_height = np.mean(head_heights)
            
            # Typical eye-to-top-of-head distance is about 0.12m
            estimated_height = avg_head_height + 0.12
            
            if abs(estimated_height - self.estimated_user_height) > 0.1:
                self.update_user_height(estimated_height)
            
            self.is_calibrated = True
            self.logger.info(f"Calibration complete: offset={self.headset_to_body_offset}, height={self.estimated_user_height:.2f}m")
            
        except Exception as e:
            self.logger.error(f"Error updating calibration: {e}")
    
    def align_trackers_to_body(self, mediapipe_landmarks: np.ndarray, headset_pose: Optional[np.ndarray] = None, controller_poses: Optional[Dict] = None) -> Dict:
        """Align virtual trackers to user's body with ABSOLUTE headset anchoring + CONTROLLER FUSION for maximum precision"""
        
        try:
            # If no headset pose, we cannot do proper tracking
            if headset_pose is None:
                now = time.time()
                if not hasattr(self, '_last_no_headset_warning') or now - self._last_no_headset_warning > 5:
                    self.logger.warning("No headset pose available - cannot properly anchor trackers")
                    self._last_no_headset_warning = now
                return {}
            
            # ENHANCED: Use controller data if available for superior tracking
            if controller_poses and len(controller_poses) > 0:
                self.logger.debug(f"Using controller fusion with {len(controller_poses)} controllers")
                result = self._force_physical_body_constraints_with_controllers(headset_pose, mediapipe_landmarks, controller_poses)
            else:
                # Fallback to headset-only tracking
                result = self._force_physical_body_constraints(headset_pose, mediapipe_landmarks)
            
            # Store positions for next frame
            self.previous_tracker_positions = {name: data['position'] for name, data in result.items()}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in tracker alignment: {e}")
            return {}
    
    def _force_physical_body_constraints(self, headset_pose: np.ndarray, mediapipe_landmarks: Optional[np.ndarray] = None) -> Dict:
        """Force trackers to be in EXACT anatomically correct positions - HYBRID APPROACH"""
        
        headset_position = headset_pose[:3]
        headset_rotation = headset_pose[3:7] if len(headset_pose) > 3 else np.array([1.0, 0.0, 0.0, 0.0])
        
        result = {}
        
        # Get user facing direction from headset rotation (Y-axis rotation)
        # Convert quaternion to facing direction
        facing_forward = self._get_forward_direction_from_quaternion(headset_rotation)
        facing_right = self._get_right_direction_from_quaternion(headset_rotation)
        
        # If we have MediaPipe data, use it to improve positioning, but always constrain to headset
        if mediapipe_landmarks is not None and len(mediapipe_landmarks) >= 33:
            result = self._calculate_hybrid_positions(headset_position, mediapipe_landmarks, facing_forward, facing_right)
        else:
            result = self._calculate_fallback_positions(headset_position, facing_forward, facing_right)
        
        # Apply smoothing with previous positions
        result = self._apply_position_smoothing(result)
        
        # Log debug information
        self._log_debug_info(headset_position, result)
        
        return result
    
    def _get_forward_direction_from_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to forward direction vector"""
        try:
            w, x, y, z = quat
            # Forward direction (Z-axis) in world space
            forward = np.array([
                2 * (x*z + w*y),
                2 * (y*z - w*x),
                1 - 2 * (x*x + y*y)
            ])
            # Normalize
            norm = np.linalg.norm(forward)
            if norm > 0:
                forward = forward / norm
            else:
                forward = np.array([0.0, 0.0, 1.0])  # Default forward
            return forward
        except:
            return np.array([0.0, 0.0, 1.0])  # Default forward
    
    def _get_right_direction_from_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to right direction vector"""
        try:
            w, x, y, z = quat
            # Right direction (X-axis) in world space
            right = np.array([
                1 - 2 * (y*y + z*z),
                2 * (x*y + w*z),
                2 * (x*z - w*y)
            ])
            # Normalize
            norm = np.linalg.norm(right)
            if norm > 0:
                right = right / norm
            else:
                right = np.array([1.0, 0.0, 0.0])  # Default right
            return right
        except:
            return np.array([1.0, 0.0, 0.0])  # Default right
    
    def _calculate_hybrid_positions(self, headset_position: np.ndarray, mediapipe_landmarks: Optional[np.ndarray], 
                                   forward: np.ndarray, right: np.ndarray) -> Dict:
        """Calculate tracker positions using hybrid approach"""
        
        result = {}
        
        # Get user height for scaling
        user_height = getattr(self, 'user_height', 1.7)
        
        if mediapipe_landmarks is not None and len(mediapipe_landmarks) >= 33:
            # Calculate hip position from MediaPipe landmarks
            hip_landmarks = [23, 24]  # Left and right hip landmarks
            hip_positions = []
            
            for landmark_idx in hip_landmarks:
                if landmark_idx < len(mediapipe_landmarks):
                    landmark = mediapipe_landmarks[landmark_idx]
                    if landmark[2] > 0.5:  # Confidence threshold
                        # Convert to world coordinates
                        world_pos = self._landmark_to_world(landmark, headset_position, forward, right)
                        hip_positions.append(world_pos)
            
            if hip_positions:
                # Average hip positions
                hip_position = np.mean(hip_positions, axis=0)
                # FIXED: Ensure hip is at correct height relative to headset
                hip_position[1] = headset_position[1] - 0.3  # 30cm below headset
                result['hip'] = {
                    'position': hip_position,
                    'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                    'confidence': 0.9
                }
            else:
                # Fallback: Calculate hip from headset position
                hip_position = headset_position.copy()
                hip_position[1] -= 0.3  # 30cm below headset
                result['hip'] = {
                    'position': hip_position,
                    'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                    'confidence': 0.7
                }
        else:
            # Fallback: Calculate hip from headset position
            hip_position = headset_position.copy()
            hip_position[1] -= 0.3  # 30cm below headset
            result['hip'] = {
                'position': hip_position,
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.7
            }
        
        # Calculate foot positions
        if mediapipe_landmarks is not None and len(mediapipe_landmarks) >= 33:
            # Left foot
            left_foot_landmarks = [27, 29, 31]  # Left ankle, knee, hip
            left_foot_position = self._calculate_foot_position(left_foot_landmarks, mediapipe_landmarks, 
                                                             headset_position, forward, right, 'left')
            result['left_foot'] = {
                'position': left_foot_position,
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.8
            }
            
            # Right foot
            right_foot_landmarks = [28, 30, 32]  # Right ankle, knee, hip
            right_foot_position = self._calculate_foot_position(right_foot_landmarks, mediapipe_landmarks, 
                                                              headset_position, forward, right, 'right')
            result['right_foot'] = {
                'position': right_foot_position,
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.8
            }
        else:
            # Fallback: Calculate feet from hip position
            hip_pos = result['hip']['position']
            
            # FIXED: Correct foot positioning relative to hip
            result['left_foot'] = {
                'position': np.array([hip_pos[0] - 0.15, 0.05, hip_pos[2]]),  # 15cm left, 5cm above floor
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.6
            }
            
            result['right_foot'] = {
                'position': np.array([hip_pos[0] + 0.15, 0.05, hip_pos[2]]),  # 15cm right, 5cm above floor
                'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                'confidence': 0.6
            }
        
        return result
    
    def _calculate_foot_position(self, foot_landmarks: List[int], mediapipe_landmarks: np.ndarray,
                                headset_position: np.ndarray, forward: np.ndarray, right: np.ndarray,
                                foot_side: str) -> np.ndarray:
        """Calculate foot position from MediaPipe landmarks"""
        
        foot_positions = []
        
        for landmark_idx in foot_landmarks:
            if landmark_idx < len(mediapipe_landmarks):
                landmark = mediapipe_landmarks[landmark_idx]
                if landmark[2] > 0.5:  # Confidence threshold
                    # Convert to world coordinates
                    world_pos = self._landmark_to_world(landmark, headset_position, forward, right)
                    foot_positions.append(world_pos)
        
        if foot_positions:
            # Average foot positions
            foot_position = np.mean(foot_positions, axis=0)
            # FIXED: Ensure foot is at floor level
            foot_position[1] = 0.05  # 5cm above floor
            return foot_position
        else:
            # Fallback: Calculate from hip position
            hip_position = headset_position.copy()
            hip_position[1] -= 0.3  # 30cm below headset
            
            if foot_side == 'left':
                return np.array([hip_position[0] - 0.15, 0.05, hip_position[2]])
            else:  # right
                return np.array([hip_position[0] + 0.15, 0.05, hip_position[2]])
    
    def _landmark_to_world(self, landmark: np.ndarray, headset_position: np.ndarray, 
                          forward: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Convert MediaPipe landmark to world coordinates"""
        
        # FIXED: Correct coordinate transformation to fix inverted directions
        x, y, z = landmark[0], landmark[1], landmark[2]
        
        # Usar los factores de escala configurables
        scale_x = self.scale_x
        scale_y = self.scale_y
        scale_z = self.scale_z

        # Convert normalized coordinates to world coordinates
        world_x = (x - 0.5) * scale_x
        world_y = (0.5 - y) * scale_y  # Invert Y axis
        world_z = (z - 0.5) * scale_z

        # Apply coordinate system transformation
        world_pos = headset_position + forward * world_z + right * world_x + np.array([0, 1, 0]) * world_y

        return world_pos
    
    def _ensure_anatomical_correctness(self, result: Dict):
        """Ensure all tracker positions are anatomically correct with GENTLE constraints"""
        
        if 'hip' not in result:
            return
        
        hip_pos = result['hip']['position']
        
        # GENTLE anatomical constraints - prevent only extreme cases
        MAX_LEG_LENGTH = 1.2  # Increased maximum leg length for more freedom
        MIN_LEG_LENGTH = 0.4  # Reduced minimum leg length
        MAX_FOOT_SEPARATION = 1.0  # Increased foot separation allowance
        MIN_FOOT_SEPARATION = 0.05  # Reduced minimum foot separation
        
        # Only constrain feet if they're extremely far from anatomical positions
        for foot_name in ['left_foot', 'right_foot']:
            if foot_name in result:
                foot_pos = result[foot_name]['position']
                
                # Calculate distance from hip
                hip_to_foot = foot_pos - hip_pos
                distance = np.linalg.norm(hip_to_foot)
                
                # Only constrain if leg length is extremely unrealistic
                if distance > MAX_LEG_LENGTH:
                    # Gentle scaling down
                    foot_pos = hip_pos + (hip_to_foot / distance) * MAX_LEG_LENGTH * 0.95
                    self.logger.debug(f"Gently constrained {foot_name} leg length: {distance:.2f}m")
                
                # Ensure foot is below hip (but allow some variance)
                if foot_pos[1] > hip_pos[1] + 0.05:  # Allow 5cm above hip for natural movement
                    foot_pos[1] = hip_pos[1] - 0.05  # Place slightly below hip
                
                result[foot_name]['position'] = foot_pos
        
        # Only adjust foot separation if extremely unrealistic
        if 'left_foot' in result and 'right_foot' in result:
            left_foot = result['left_foot']['position']
            right_foot = result['right_foot']['position']
            
            foot_separation = np.linalg.norm(left_foot - right_foot)
            
            # Only intervene if separation is extremely large
            if foot_separation > MAX_FOOT_SEPARATION:
                # Gentle adjustment
                direction = (left_foot - right_foot) / foot_separation
                midpoint = (left_foot + right_foot) / 2
                target_separation = MAX_FOOT_SEPARATION * 0.9
                left_foot = midpoint + direction * (target_separation / 2)
                right_foot = midpoint - direction * (target_separation / 2)
                self.logger.debug(f"Gently adjusted foot separation: {foot_separation:.2f}m")
                
                result['left_foot']['position'] = left_foot
                result['right_foot']['position'] = right_foot
    
    def _force_physical_body_constraints_with_controllers(self, headset_pose: np.ndarray, mediapipe_landmarks: Optional[np.ndarray] = None, controller_poses: Optional[Dict] = None) -> Dict:
        """Force trackers to be in EXACT anatomically correct positions - STABLE CONTROLLER FUSION"""
        
        headset_position = headset_pose[:3]
        headset_rotation = headset_pose[3:7] if len(headset_pose) > 3 else np.array([1.0, 0.0, 0.0, 0.0])
        
        # Get forward and right directions from headset
        forward = self._get_forward_direction_from_quaternion(headset_rotation)
        right = self._get_right_direction_from_quaternion(headset_rotation)
        
        # Step 1: Calculate base positions using hybrid approach
        result = self._calculate_hybrid_positions(headset_position, mediapipe_landmarks, forward, right)
        
        # Step 2: Apply controller guidance BEFORE constraints (order matters!)
        if controller_poses and len(controller_poses) >= 2:
            self.logger.debug("Applying controller guidance for body anchoring")
            result = self._apply_controller_guidance(result, controller_poses, headset_position, forward, right)
        
        # Step 3: Apply ONLY gentle physical constraints to preserve controller guidance
        result = self._apply_stable_body_anchoring(result, headset_position, controller_poses)
        
        # Step 4: Apply smoothing LAST to eliminate jitter
        result = self._apply_position_smoothing(result)
        
        return result
    
    def _apply_stable_body_anchoring(self, result: Dict, headset_position: np.ndarray, controller_poses: Optional[Dict] = None) -> Dict:
        """Apply stable body anchoring that respects controller guidance and prevents drift"""
        
        # Very conservative constraints - only fix extreme cases
        MAX_HIP_DISTANCE = 1.0      # Hip max 1m from headset
        MAX_FOOT_DISTANCE = 1.8     # Feet max 1.8m from headset
        MIN_HIP_HEIGHT = 0.6        # Hip at least 60cm above ground
        MAX_HIP_HEIGHT = 1.3        # Hip max 1.3m above ground
        
        # Get controller data for guidance
        controller_available = False
        controller_midpoint = None
        if controller_poses and len(controller_poses) >= 2:
            left_pos = controller_poses.get('left', {}).get('position')
            right_pos = controller_poses.get('right', {}).get('position')
            if left_pos is not None and right_pos is not None:
                controller_available = True
                controller_midpoint = (np.array(left_pos) + np.array(right_pos)) / 2
        
        # Apply constraints to each tracker
        for tracker_name, tracker_data in result.items():
            position = tracker_data['position'].copy()
            constrained = False
            
            if tracker_name == 'hip':
                # Hip constraints
                distance_to_headset = np.linalg.norm(position - headset_position)
                
                # Distance constraint
                if distance_to_headset > MAX_HIP_DISTANCE:
                    direction = (position - headset_position) / distance_to_headset
                    position = headset_position + direction * MAX_HIP_DISTANCE
                    constrained = True
                
                # Height constraints
                if position[1] < MIN_HIP_HEIGHT:
                    position[1] = MIN_HIP_HEIGHT
                    constrained = True
                elif position[1] > MAX_HIP_HEIGHT:
                    position[1] = MAX_HIP_HEIGHT
                    constrained = True
                
                # If controllers available, ensure hip is reasonable relative to them
                if controller_available:
                    controller_height = controller_midpoint[1]
                    expected_hip_height = controller_height - 0.25  # 25cm below controllers
                    if abs(position[1] - expected_hip_height) > 0.3:  # Allow 30cm variance
                        position[1] = position[1] * 0.7 + expected_hip_height * 0.3  # Gentle correction
                        constrained = True
            
            elif 'foot' in tracker_name:
                # Foot constraints
                distance_to_headset = np.linalg.norm(position - headset_position)
                
                # Distance constraint
                if distance_to_headset > MAX_FOOT_DISTANCE:
                    direction = (position - headset_position) / distance_to_headset
                    position = headset_position + direction * MAX_FOOT_DISTANCE
                    constrained = True
                
                # Floor constraint - feet should be near ground
                if position[1] < 0.0:  # Below ground
                    position[1] = 0.05
                    constrained = True
                elif position[1] > 0.2:  # Too high for feet
                    position[1] = max(0.2, position[1] * 0.9)  # Gently bring down
                    constrained = True
                
                # Ensure feet are below hip
                if 'hip' in result:
                    hip_height = result['hip']['position'][1]
                    if position[1] > hip_height - 0.3:  # Foot should be at least 30cm below hip
                        position[1] = hip_height - 0.3
                        constrained = True
            
            if constrained:
                self.logger.debug(f"Applied stable constraint to {tracker_name}")
                result[tracker_name]['position'] = position
        
        return result
    
    def _apply_controller_guidance(self, base_result: Dict, controller_poses: Dict, headset_position: np.ndarray, 
                                  forward: np.ndarray, right: np.ndarray) -> Dict:
        """Apply controller guidance to improve body anchoring without overriding positions completely"""
        
        result = base_result.copy()
        
        # Get controller positions
        left_controller = controller_poses.get('left', {})
        right_controller = controller_poses.get('right', {})
        left_pos = left_controller.get('position')
        right_pos = right_controller.get('position')
        
        if left_pos is None or right_pos is None:
            return result
        
        # Convert to numpy arrays for calculations
        left_pos = np.array(left_pos)
        right_pos = np.array(right_pos)
        
        # Calculate body reference points from controllers
        controller_midpoint = (left_pos + right_pos) / 2
        controller_separation = np.linalg.norm(left_pos - right_pos)
        avg_controller_height = (left_pos[1] + right_pos[1]) / 2
        
        # Guide hip position (blend with existing position)
        if 'hip' in result:
            current_hip = result['hip']['position']
            
            # Calculate guided hip position
            guided_hip = controller_midpoint.copy()
            guided_hip[1] = avg_controller_height - 0.25  # 25cm below controllers
            
            # Blend current position with guided position (60% guided, 40% current)
            blend_factor = 0.6
            blended_hip = current_hip * (1 - blend_factor) + guided_hip * blend_factor
            result['hip']['position'] = blended_hipded_hip = current_hip * (1 - blend_factor) + guided_hip * blend_factor
            result['hip']['position'] = blended_hip
        
        # Guide foot positions (more subtle influence)
        foot_guidance_factor = 0.4  # Less aggressive guidance for feet
        
        if 'left_foot' in result:
            current_left_foot = result['left_foot']['position']
            
            # Calculate guided left foot position
            guided_left_foot = np.array([
                controller_midpoint[0] - controller_separation * 0.3,  # Left of center
                0.08,  # Floor level
                controller_midpoint[2] + 0.1  # Slightly forward
            ])
            
            # Blend with current position
            blended_left_foot = current_left_foot * (1 - foot_guidance_factor) + guided_left_foot * foot_guidance_factor
            result['left_foot']['position'] = blended_left_foot
        
        if 'right_foot' in result:
            current_right_foot = result['right_foot']['position']
            
            # Calculate guided right foot position
            guided_right_foot = np.array([
                controller_midpoint[0] + controller_separation * 0.3,  # Right of center
                0.08,  # Floor level
                controller_midpoint[2] + 0.1  # Slightly forward
            ])
            
            # Blend with current position
            blended_right_foot = current_right_foot * (1 - foot_guidance_factor) + guided_right_foot * foot_guidance_factor
            result['right_foot']['position'] = blended_right_foot
        
        return result
    
    def _apply_gentle_physical_constraints(self, result: Dict, headset_position: np.ndarray, controller_poses: Optional[Dict] = None) -> Dict:
        """Apply gentle physical constraints that preserve controller guidance"""
        """Apply STABLE physical constraints with smooth controller integration"""
        
        # Maximum distances for gentle validation
        MAX_HIPSET_DISTANCE = 0.8
        MAX_FOOT_HIP_DISTANCE = 1.2
        MAX_FOOT_HEADSET_DISTANCE = 1.8
        
        # Very gentle constraints - only fix extreme outliers
        for tracker_name, tracker_data in result.items():
            position = tracker_data['position']
            
            if tracker_name == 'hip':
                # Only constrain if hip is extremely far from headset
                distance_to_headset = np.linalg.norm(position - headset_position)
                if distance_to_headset > MAX_HIPSET_DISTANCE:
                    # Move just enough to be within range
                    direction = (headset_position - position) / distance_to_headset
                    position = headset_position - direction * MAX_HIPSET_DISTANCE
                    result[tracker_name]['position'] = position
                    self.logger.debug(f"Gently constrained {tracker_name} distance from headset")
            
            elif 'foot' in tracker_name:
                # Constrain foot only if extremely far
                distance_to_headset = np.linalg.norm(position - headset_position)
                
                if distance_to_headset > MAX_FOOT_HEADSET_DISTANCE:
                    direction = (headset_position - position) / distance_to_headset
                    position = headset_position - direction * MAX_FOOT_HEADSET_DISTANCE
                    self.logger.debug(f"Gently constrained {tracker_name} distance from headset")
                
                # Constrain relative to hip if available
                if 'hip' in result:
                    hip_pos = result['hip']['position']
                    distance_to_hip = np.linalg.norm(position - hip_pos)
                    
                    if distance_to_hip > MAX_FOOT_HIP_DISTANCE:
                        direction = (hip_pos - position) / distance_to_hip
                        position = hip_pos - direction * MAX_FOOT_HIP_DISTANCE
                        self.logger.debug(f"Gently constrained {tracker_name} distance from hip")
                
                # Gentle floor constraint
                if position[1] < 0.0:  # Below ground
                    position[1] = 0.05
                elif position[1] > 0.3:  # Too high for foot
                    position[1] = max(0.3, position[1] * 0.9)  # Gradually bring down
                
                result[tracker_name]['position'] = position
        
        return result
    
    # Keep the original _apply_strict_physical_constraints for backward compatibility
    def _apply_strict_physical_constraints(self, result: Dict, headset_position: np.ndarray, controller_poses: Optional[Dict] = None) -> Dict:
        """Apply strict physical constraints that enforce anatomical correctness"""
        
        # Maximum physically possible distances - More realistic
        MAX_HIPSET_DISTANCE = 0.7  # More realistic hip-headset distance
        MAX_FOOT_HIP_DISTANCE = 1.1  # More realistic leg length
        MAX_FOOT_HEADSET_DISTANCE = 1.6  # More realistic total distance
        
        # Constrain hip tracker position - SMOOTH POSITIONING with body mechanics
        if 'hip' in result:
            hip_pos = result['hip']['position'].copy()
            
            # Calculate distance from headset
            distance_to_headset = np.linalg.norm(hip_pos - headset_position)
            
            # Gentle constraint to headset distance
            if distance_to_headset > MAX_HIPSET_DISTANCE:
                direction_to_headset = (headset_position - hip_pos) / distance_to_headset
                hip_pos = headset_position - direction_to_headset * MAX_HIPSET_DISTANCE
                self.logger.debug(f"Gently constrained hip distance: {distance_to_headset:.2f}m -> {MAX_HIPSET_DISTANCE:.2f}m")
            
            result['hip']['position'] = hip_pos
        
        # Get controller positions if available for guidance
        left_controller_pos = None
        right_controller_pos = None
        if controller_poses:
            left_controller_pos = controller_poses.get('left', {}).get('position')
            right_controller_pos = controller_poses.get('right', {}).get('position')
        
        # Constrain foot tracker positions - ANATOMICAL FOOT TRACKING
        for foot_name in ['left_foot', 'right_foot']:
            if foot_name in result:
                foot_pos = result[foot_name]['position'].copy()
                
                # Get hip position for anatomical reference
                hip_pos = result['hip']['position'] if 'hip' in result else headset_position
                
                # Calculate distances for validation
                distance_to_headset = np.linalg.norm(foot_pos - headset_position)
                distance_to_hip = np.linalg.norm(foot_pos - hip_pos)
                
                # Gentle constraint from headset
                if distance_to_headset > MAX_FOOT_HEADSET_DISTANCE:
                    direction_to_headset = (headset_position - foot_pos) / distance_to_headset
                    foot_pos = headset_position - direction_to_headset * MAX_FOOT_HEADSET_DISTANCE
                    self.logger.debug(f"Gently constrained {foot_name} headset distance")
                
                # Gentle constraint from hip (leg length)
                if distance_to_hip > MAX_FOOT_HIP_DISTANCE:
                    direction_to_hip = (hip_pos - foot_pos) / distance_to_hip
                    foot_pos = hip_pos - direction_to_hip * MAX_FOOT_HIP_DISTANCE
                    self.logger.debug(f"Gently constrained {foot_name} leg length")
                
                # IMPROVED: Use controller positions as ANATOMICAL GUIDANCE
                if left_controller_pos is not None and right_controller_pos is not None:
                    # Calculate foot position based on body mechanics, not direct controller position
                    if foot_name == 'left_foot':
                        # Left foot should be influenced by left controller but anatomically positioned
                        anatomical_foot_x = hip_pos[0] - 0.18  # 18cm left of hip center
                        anatomical_foot_z = hip_pos[2] + 0.1   # 10cm forward of hip
                        
                        # Blend with controller influence (reduced to 30%)
                        controller_influence = 0.3
                        foot_pos[0] = foot_pos[0] * (1 - controller_influence) + (anatomical_foot_x * 0.7 + left_controller_pos[0] * 0.3) * controller_influence
                        foot_pos[2] = foot_pos[2] * (1 - controller_influence) + (anatomical_foot_z * 0.7 + left_controller_pos[2] * 0.3) * controller_influence
                    else:  # right_foot
                        # Right foot should be influenced by right controller but anatomically positioned
                        anatomical_foot_x = hip_pos[0] + 0.18  # 18cm right of hip center
                        anatomical_foot_z = hip_pos[2] + 0.1   # 10cm forward of hip
                        
                        # Blend with controller influence (reduced to 30%)
                        controller_influence = 0.3
                        foot_pos[0] = foot_pos[0] * (1 - controller_influence) + (anatomical_foot_x * 0.7 + right_controller_pos[0] * 0.3) * controller_influence
                        foot_pos[2] = foot_pos[2] * (1 - controller_influence) + (anatomical_foot_z * 0.7 + right_controller_pos[2] * 0.3) * controller_influence
                    
                    # Set foot height anatomically (at floor level)
                    target_foot_height = 0.08  # 8cm above floor for foot tracker
                    foot_pos[1] = foot_pos[1] * 0.8 + target_foot_height * 0.2  # Smooth transition to floor
                else:
                    # Fallback: Anatomical positioning based on hip
                    if foot_name == 'left_foot':
                        fallback_foot = np.array([hip_pos[0] - 0.18, 0.08, hip_pos[2] + 0.1])
                    else:  # right_foot
                        fallback_foot = np.array([hip_pos[0] + 0.18, 0.08, hip_pos[2] + 0.1])
                    
                    # Blend with fallback
                    blend_factor = 0.2
                    foot_pos = foot_pos * (1 - blend_factor) + fallback_foot * blend_factor
                
                result[foot_name]['position'] = foot_pos
        
        return result
    
    def _fuse_controller_data(self, base_result: Dict, controller_poses: Dict, headset_position: np.ndarray, 
                              facing_forward: np.ndarray, facing_right: np.ndarray) -> Dict:
        """Fuse controller data with MediaPipe landmarks for superior tracking accuracy"""
        
        result = base_result.copy()
        
        # Get controller positions and rotations
        left_controller = controller_poses.get('left', {})
        right_controller = controller_poses.get('right', {})
        
        left_pos = left_controller.get('position')
        right_pos = right_controller.get('position')
        left_rot = left_controller.get('rotation')
        right_rot = right_controller.get('rotation')
        
        if left_pos is None or right_pos is None:
            return result
        
        # Calculate shoulder positions from controllers
        # Shoulders should be roughly at controller height but closer to body center
        shoulder_height = (left_pos[1] + right_pos[1]) / 2
        shoulder_width = np.linalg.norm(left_pos - right_pos) * 0.6  # Shoulders are closer than controllers
        
        # Calculate body center from controllers
        body_center_x = (left_pos[0] + right_pos[0]) / 2
        body_center_z = (left_pos[2] + right_pos[2]) / 2
        
        # Calculate shoulder positions
        left_shoulder_pos = np.array([
            body_center_x - shoulder_width / 2,
            shoulder_height,
            body_center_z
        ])
        
        right_shoulder_pos = np.array([
            body_center_x + shoulder_width / 2,
            shoulder_height,
            body_center_z
        ])
        
        # Calculate hip position using shoulder positions and headset
        hip_height = shoulder_height - 0.3  # Hip is ~30cm below shoulders
        hip_pos = np.array([
            body_center_x,
            hip_height,
            body_center_z
        ])
        
        # Refine hip position using headset data
        # Hip should be roughly below headset but at calculated height
        hip_pos[0] = (hip_pos[0] + headset_position[0]) / 2  # Blend with headset X
        hip_pos[2] = (hip_pos[2] + headset_position[2]) / 2  # Blend with headset Z
        
        # Calculate foot positions using shoulder positions and controller data
        left_foot_pos = np.array([
            left_shoulder_pos[0],  # Same X as left shoulder
            hip_height - 0.8,      # ~80cm below hip
            left_shoulder_pos[2]   # Same Z as left shoulder
        ])
        
        right_foot_pos = np.array([
            right_shoulder_pos[0],  # Same X as right shoulder
            hip_height - 0.8,       # ~80cm below hip
            right_shoulder_pos[2]   # Same Z as right shoulder
        ])
        
        # Apply controller-based refinements
        if left_pos is not None and right_pos is not None:
            # Use controller positions to refine foot positions
            left_foot_pos[0] = (left_foot_pos[0] + left_pos[0]) / 2
            left_foot_pos[2] = (left_foot_pos[2] + left_pos[2]) / 2
            
            right_foot_pos[0] = (right_foot_pos[0] + right_pos[0]) / 2
            right_foot_pos[2] = (right_foot_pos[2] + right_pos[2]) / 2
        
        # Update result with controller-fused positions
        if 'hip' in result:
            result['hip']['position'] = hip_pos
            # Use controller rotations to influence hip rotation
            if left_rot is not None and right_rot is not None:
                # Average controller rotations for hip rotation
                avg_rotation = self._average_quaternions(left_rot, right_rot)
                result['hip']['rotation'] = avg_rotation
        
        if 'left_foot' in result:
            result['left_foot']['position'] = left_foot_pos
            # Use left controller rotation for left foot
            if left_rot is not None:
                result['left_foot']['rotation'] = left_rot
        
        if 'right_foot' in result:
            result['right_foot']['position'] = right_foot_pos
            # Use right controller rotation for right foot
            if right_rot is not None:
                result['right_foot']['rotation'] = right_rot
        
        return result
    
    def _average_quaternions(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Average two quaternions for smooth rotation blending"""
        # Simple linear interpolation for quaternions
        avg_q = (q1 + q2) / 2
        # Normalize the result
        norm = np.linalg.norm(avg_q)
        if norm > 0:
            avg_q = avg_q / norm
        return avg_q
    
    def _estimate_shoulders_from_controllers(self, left_controller: Optional[Dict], right_controller: Optional[Dict],
                                           headset_position: np.ndarray, facing_forward: np.ndarray, 
                                           facing_right: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Estimate shoulder positions from controller positions"""
        
        try:
            shoulders = []
            
            # Process left controller
            if left_controller and 'position' in left_controller:
                left_pos = np.array(left_controller['position'])
                # Estimate left shoulder: controller position + offset toward body center
                left_shoulder = left_pos + facing_right * 0.05 + np.array([0, 0.15, -0.1])  # Adjust to shoulder
                shoulders.append(('left', left_shoulder))
            
            # Process right controller
            if right_controller and 'position' in right_controller:
                right_pos = np.array(right_controller['position'])
                # Estimate right shoulder: controller position + offset toward body center
                right_shoulder = right_pos - facing_right * 0.05 + np.array([0, 0.15, -0.1])  # Adjust to shoulder
                shoulders.append(('right', right_shoulder))
            
            # If we have both shoulders, return them
            if len(shoulders) == 2:
                left_shoulder = shoulders[0][1] if shoulders[0][0] == 'left' else shoulders[1][1]
                right_shoulder = shoulders[1][1] if shoulders[1][0] == 'right' else shoulders[0][1]
                return left_shoulder, right_shoulder
            
            # If we only have one shoulder, estimate the other
            elif len(shoulders) == 1:
                side, shoulder_pos = shoulders[0]
                shoulder_center = headset_position + np.array([0, -0.2, 0])  # Approximate shoulder level
                
                if side == 'left':
                    left_shoulder = shoulder_pos
                    # Estimate right shoulder using typical shoulder width
                    right_shoulder = shoulder_center + facing_right * 0.2
                else:
                    right_shoulder = shoulder_pos
                    # Estimate left shoulder using typical shoulder width
                    left_shoulder = shoulder_center - facing_right * 0.2
                
                return left_shoulder, right_shoulder
            
        except Exception as e:
            self.logger.error(f"Error estimating shoulders from controllers: {e}")
        
        return None
    
    def _refine_hip_with_shoulders(self, result: Dict, left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                                  headset_position: np.ndarray) -> Dict:
        """Refine hip tracker position using shoulder reference for better body alignment"""
        
        try:
            # Calculate shoulder center
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Calculate shoulder width for body proportion validation
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            
            # Estimate hip position based on shoulder center and headset
            # Hip should be roughly 0.3-0.4m below shoulder center
            hip_offset_down = np.array([0.0, -0.35, 0.0])  # 35cm below shoulders
            
            # Calculate hip position
            estimated_hip = shoulder_center + hip_offset_down
            
            # Blend with existing hip position (70% shoulder-based, 30% existing)
            if 'hip' in result:
                current_hip = result['hip']['position']
                refined_hip = 0.7 * estimated_hip + 0.3 * current_hip
                
                # Apply anatomical constraints
                refined_hip[1] = max(0.8, min(1.2, refined_hip[1]))  # Keep within reasonable waist height
                
                result['hip']['position'] = refined_hip
                result['hip']['confidence'] = min(1.0, result['hip']['confidence'] + 0.2)  # Boost confidence
                
                self.logger.debug(f"Hip refined: shoulder_center={shoulder_center}, estimated_hip={estimated_hip}, final_hip={refined_hip}")
            
        except Exception as e:
            self.logger.warning(f"Error refining hip with shoulders: {e}")
        
        return result
    
    def _refine_feet_with_arm_tracking(self, result: Dict, left_controller: Optional[Dict], right_controller: Optional[Dict],
                                      left_shoulder: np.ndarray, right_shoulder: np.ndarray, headset_position: np.ndarray) -> Dict:
        """Use arm tracking data to improve foot placement accuracy"""
        
        try:
            # Calculate body orientation from shoulder line
            shoulder_vector = right_shoulder - left_shoulder
            shoulder_forward = np.cross(shoulder_vector, np.array([0, 1, 0]))
            shoulder_forward = shoulder_forward / np.linalg.norm(shoulder_forward) if np.linalg.norm(shoulder_forward) > 0 else np.array([0, 0, 1])
            
            # Refine foot positions based on shoulder orientation
            if 'left_foot' in result and 'right_foot' in result:
                # Calculate ideal foot separation based on shoulder width
                shoulder_width = np.linalg.norm(shoulder_vector)
                foot_separation = shoulder_width * 0.8  # Feet typically narrower than shoulders
                
                # Get current foot center
                current_left = result['left_foot']['position']
                current_right = result['right_foot']['position']
                foot_center = (current_left + current_right) / 2
                
                # Calculate new foot positions aligned with shoulder orientation
                foot_right_vector = shoulder_vector / np.linalg.norm(shoulder_vector) if np.linalg.norm(shoulder_vector) > 0 else np.array([1, 0, 0])
                
                refined_left_foot = foot_center - foot_right_vector * (foot_separation / 2)
                refined_right_foot = foot_center + foot_right_vector * (foot_separation / 2)
                
                # Keep feet on ground
                refined_left_foot[1] = 0.05
                refined_right_foot[1] = 0.05
                
                # Blend with existing positions
                fusion_weight = 0.2  # Subtle influence to maintain stability
                
                result['left_foot']['position'] = (1 - fusion_weight) * current_left + fusion_weight * refined_left_foot
                result['right_foot']['position'] = (1 - fusion_weight) * current_right + fusion_weight * refined_right_foot
                
                # Boost confidence
                result['left_foot']['confidence'] = min(1.0, result['left_foot']['confidence'] + 0.05)
                result['right_foot']['confidence'] = min(1.0, result['right_foot']['confidence'] + 0.05)
                
                self.logger.debug("Feet refined with arm tracking data")
            
        except Exception as e:
            self.logger.error(f"Error refining feet with arm tracking: {e}")
        
        return result
    
    def _apply_controller_confidence_boost(self, result: Dict, controller_poses: Dict):
        """Apply confidence boost when controllers are tracking well"""
        
        try:
            # Count valid controllers
            valid_controllers = 0
            for controller_data in controller_poses.values():
                if controller_data and 'position' in controller_data:
                    # Check if controller seems to be tracking (not at origin)
                    pos = np.array(controller_data['position'])
                    if np.linalg.norm(pos) > 0.1:  # Not at origin
                        valid_controllers += 1
            
            # Apply confidence boost based on number of valid controllers
            confidence_boost = min(0.15, valid_controllers * 0.075)  # Max 15% boost
            
            for tracker_name in result:
                result[tracker_name]['confidence'] = min(1.0, result[tracker_name]['confidence'] + confidence_boost)
            
            if confidence_boost > 0:
                self.logger.debug(f"Applied controller confidence boost: {confidence_boost:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error applying controller confidence boost: {e}")
    
    def _log_debug_info_with_controllers(self, headset_position: np.ndarray, result: Dict, controller_poses: Optional[Dict] = None):
        """Enhanced debug logging that includes controller information"""
        
        # Debug logging every 120 frames (about every 4 seconds)
        if not hasattr(self, '_controller_debug_counter'):
            self._controller_debug_counter = 0
        
        self._controller_debug_counter += 1
        if self._controller_debug_counter % 120 == 0:
            self.logger.info("=== CONTROLLER-ENHANCED TRACKER POSITIONING ===")
            self.logger.info(f"Headset: ({headset_position[0]:.3f}, {headset_position[1]:.3f}, {headset_position[2]:.3f})")
            
            # Log controller information
            if controller_poses:
                for controller_name, controller_data in controller_poses.items():
                    if controller_data and 'position' in controller_data:
                        pos = controller_data['position']
                        self.logger.info(f"{controller_name.upper()} Controller: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    else:
                        self.logger.info(f"{controller_name.upper()} Controller: NO DATA")
            else:
                self.logger.info("Controllers: NO DATA")
            
            # Log tracker results
            for name, data in result.items():
                pos = data['position']
                distance = np.linalg.norm(pos - headset_position)
                vertical_offset = pos[1] - headset_position[1]
                confidence = data.get('confidence', 0.0)
                
                self.logger.info(f"{name.upper()}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) [conf={confidence:.2f}]")
                self.logger.info(f"  Distance from headset: {distance:.3f}m")
                self.logger.info(f"  Vertical offset: {vertical_offset:.3f}m")
            
            # Verify positioning
            if 'hip' in result and 'left_foot' in result and 'right_foot' in result:
                hip_y = result['hip']['position'][1]
                left_foot_y = result['left_foot']['position'][1]
                right_foot_y = result['right_foot']['position'][1]
                
                if hip_y > left_foot_y and hip_y > right_foot_y:
                    self.logger.info("✓ Correct: Hip is above feet")
                else:
                    self.logger.error("✗ ERROR: Hip is not above feet!")
            
            # Check controller enhancement
            controller_count = len([c for c in (controller_poses or {}).values() if c and 'position' in c])
            if controller_count > 0:
                self.logger.info(f"✓ Controller enhancement active with {controller_count} controllers")
            else:
                self.logger.info("○ No controller enhancement (headset-only tracking)")
            
            self.logger.info("=== End Controller-Enhanced Positioning ===")
    
    def _calculate_fallback_positions(self, headset_position: np.ndarray, forward: np.ndarray, right: np.ndarray) -> Dict:
        """Calculate fallback positions when MediaPipe data is not available"""
        result = {}
        
        # HIP TRACKER: 55cm below headset, slightly forward
        hip_position = headset_position.copy()
        hip_position[1] -= 0.55  # 55cm below headset
        hip_position += forward * 0.05  # 5cm forward
        
        result['hip'] = {
            'position': hip_position,
            'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
            'confidence': 0.8
        }
        
        # LEFT FOOT: At floor level, 20cm left, 10cm forward
        left_foot_position = headset_position.copy()
        left_foot_position[1] = 0.05  # 5cm above floor
        left_foot_position -= right * 0.20  # 20cm to the left
        left_foot_position += forward * 0.10  # 10cm forward
        
        result['left_foot'] = {
            'position': left_foot_position,
            'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
            'confidence': 0.7
        }
        
        # RIGHT FOOT: At floor level, 20cm right, 10cm forward
        right_foot_position = headset_position.copy()
        right_foot_position[1] = 0.05  # 5cm above floor
        right_foot_position += right * 0.20  # 20cm to the right
        right_foot_position += forward * 0.10  # 10cm forward
        
        result['right_foot'] = {
            'position': right_foot_position,
            'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
            'confidence': 0.7
        }
        
        return result
    
    def _apply_position_smoothing(self, result: Dict) -> Dict:
        """Apply ROBUST position smoothing to eliminate jitter and erratic movement"""
        
        # Initialize smoothing buffers if not exists
        if not hasattr(self, '_position_buffers'):
            self._position_buffers = {}
            self._last_positions = {}
            self._stable_positions = {}
        
        smoothed_result = {}
        
        for tracker_name, tracker_data in result.items():
            current_position = tracker_data['position'].copy()
            
            # Initialize buffer for this tracker
            if tracker_name not in self._position_buffers:
                self._position_buffers[tracker_name] = []
                self._last_positions[tracker_name] = current_position.copy()
                self._stable_positions[tracker_name] = current_position.copy()
            
            # CRITICAL FIX: Outlier detection and smoothing
            last_pos = self._last_positions[tracker_name]
            stable_pos = self._stable_positions[tracker_name]
            
            # Calculate movement from last frame
            position_delta = np.linalg.norm(current_position - last_pos)
            
            # Different max movement based on tracker type
            if tracker_name == 'hip':
                max_movement = 0.15  # Hip should move slowly - 15cm max per frame
                smoothing_factor = 0.1  # Very heavy smoothing for hip
            elif 'foot' in tracker_name:
                max_movement = 0.25  # Feet can move a bit more - 25cm max per frame
                smoothing_factor = 0.2  # Medium smoothing for feet
            else:
                max_movement = 0.3   # Other trackers - 30cm max per frame
                smoothing_factor = 0.3  # Light smoothing for others
            
            # OUTLIER REJECTION: If movement is too large, limit it
            if position_delta > max_movement:
                # Limit the movement to prevent jumps
                direction = (current_position - last_pos) / position_delta if position_delta > 0 else np.array([0, 0, 0])
                current_position = last_pos + direction * max_movement
                self.logger.debug(f"Limited {tracker_name} movement: {position_delta:.3f}m -> {max_movement:.3f}m")
            
            # EXPONENTIAL MOVING AVERAGE for smooth tracking
            smoothed_position = stable_pos * (1 - smoothing_factor) + current_position * smoothing_factor
            
            # Special constraints for feet - keep on ground
            if 'foot' in tracker_name:
                # Keep feet near ground level with gentle transitions
                target_foot_height = 0.08  # 8cm above ground
                if smoothed_position[1] < 0.0:  # Below ground
                    smoothed_position[1] = target_foot_height * 0.5  # Gentle transition
                elif smoothed_position[1] > 0.25:  # Too high
                    smoothed_position[1] = smoothed_position[1] * 0.8 + target_foot_height * 0.2
                else:
                    # Gentle pull towards target height
                    smoothed_position[1] = smoothed_position[1] * 0.9 + target_foot_height * 0.1
            
            # Update tracking variables
            self._last_positions[tracker_name] = smoothed_position.copy()
            self._stable_positions[tracker_name] = smoothed_position.copy()
            
            # Create smoothed tracker data
            smoothed_result[tracker_name] = {
                'position': smoothed_position,
                'rotation': tracker_data.get('rotation', np.array([1.0, 0.0, 0.0, 0.0])),
                'confidence': tracker_data.get('confidence', 0.8)
            }
        
        return smoothed_result
    
    def _log_debug_info(self, headset_position: np.ndarray, result: Dict):
        """Log debug information about tracker positioning"""
        # Debug logging every 120 frames (about every 4 seconds)
        if not hasattr(self, '_simple_debug_counter'):
            self._simple_debug_counter = 0
        
        self._simple_debug_counter += 1
        if self._simple_debug_counter % 120 == 0:
            self.logger.info("=== HYBRID TRACKER POSITIONING ===")
            self.logger.info(f"Headset: ({headset_position[0]:.3f}, {headset_position[1]:.3f}, {headset_position[2]:.3f})")
            
            for name, data in result.items():
                pos = data['position']
                distance = np.linalg.norm(pos - headset_position)
                vertical_offset = pos[1] - headset_position[1]
                confidence = data.get('confidence', 0.0)
                
                self.logger.info(f"{name.upper()}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) [conf={confidence:.2f}]")
                self.logger.info(f"  Distance from headset: {distance:.3f}m")
                self.logger.info(f"  Vertical offset: {vertical_offset:.3f}m")
            
            # Verify positioning
            if 'hip' in result and 'left_foot' in result and 'right_foot' in result:
                hip_y = result['hip']['position'][1]
                left_foot_y = result['left_foot']['position'][1]
                right_foot_y = result['right_foot']['position'][1]
                
                if hip_y > left_foot_y and hip_y > right_foot_y:
                    self.logger.info("✓ Correct: Hip is above feet")
                else:
                    self.logger.error("✗ ERROR: Hip is not above feet!")
            
            self.logger.info("=== End Hybrid Positioning ===")
        
        return result
    
    def _apply_headset_alignment(self, landmarks: np.ndarray, headset_pose: np.ndarray) -> np.ndarray:
        """Apply headset-based alignment to landmarks"""
        
        # Calculate MediaPipe head center
        mp_head_landmarks = [0, 7, 8, 9, 10]  # nose, ears, eyes
        mp_head_center = np.mean([landmarks[i] for i in mp_head_landmarks], axis=0)
        
        # Calculate expected head position from headset
        expected_head_pos = headset_pose + self.headset_to_body_offset
        
        # Apply translation
        translation_offset = expected_head_pos - mp_head_center
        aligned_landmarks = landmarks + translation_offset
        
        # Apply height scaling if needed
        height_scale = self.estimated_user_height / 1.75  # MediaPipe default height
        if abs(height_scale - 1.0) > 0.05:  # Only apply if significant difference
            # Scale relative to head position
            aligned_landmarks = (aligned_landmarks - expected_head_pos) * height_scale + expected_head_pos
        
        return aligned_landmarks
    
    def _calculate_optimal_tracker_positions(self, landmarks: np.ndarray) -> List[np.ndarray]:
        """Calculate optimal positions for each tracker with corrected body positioning"""
        
        if len(landmarks) < 33:
            return []
        
        positions = []
        
        # Hip tracker (using hip landmarks 23, 24) - should be at WAIST level, NOT below feet
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_center = (left_hip + right_hip) / 2
        
        # Adjust hip position for optimal tracking
        hip_offset = self.body_model.tracker_offsets['hip']['position_offset']
        hip_position = hip_center + hip_offset
        
        # CRITICAL: Ensure hip tracker is ABOVE foot trackers (fix upside-down issue)
        # Hip should be at least 0.8m above ground level
        min_hip_height = 0.8  # Fixed minimum height for hip tracker
        hip_position[1] = max(hip_position[1], min_hip_height)
        
        positions.append(hip_position)
        
        # Left foot tracker (using ankle landmark 27 and foot landmark 31) - should be NEAR GROUND
        left_ankle = landmarks[27]
        left_foot = landmarks[31] if len(landmarks) > 31 else left_ankle
        left_foot_pos = (left_ankle + left_foot) / 2
        
        left_foot_offset = self.body_model.tracker_offsets['left_foot']['position_offset']
        left_foot_position = left_foot_pos + left_foot_offset
        
        # CRITICAL: Foot trackers should be near ground level (NOT in the air)
        max_foot_height = 0.15  # Maximum 15cm above ground
        left_foot_position[1] = min(left_foot_position[1], max_foot_height)
        left_foot_position[1] = max(left_foot_position[1], 0.02)  # At least 2cm above ground
        
        positions.append(left_foot_position)
        
        # Right foot tracker (using ankle landmark 28 and foot landmark 32) - should be NEAR GROUND
        right_ankle = landmarks[28]
        right_foot = landmarks[32] if len(landmarks) > 32 else right_ankle
        right_foot_pos = (right_ankle + right_foot) / 2
        
        right_foot_offset = self.body_model.tracker_offsets['right_foot']['position_offset']
        right_foot_position = right_foot_pos + right_foot_offset
        
        # CRITICAL: Foot trackers should be near ground level (NOT in the air)
        right_foot_position[1] = min(right_foot_position[1], max_foot_height)
        right_foot_position[1] = max(right_foot_position[1], 0.02)  # At least 2cm above ground
        
        positions.append(right_foot_position)
        
        # Debug logging for position verification
        if not hasattr(self, '_position_debug_logged'):
            self.logger.info("=== TRACKER POSITION DEBUG ===")
            self.logger.info(f"Hip position: ({hip_position[0]:.3f}, {hip_position[1]:.3f}, {hip_position[2]:.3f}) - Should be ABOVE feet")
            self.logger.info(f"Left foot: ({left_foot_position[0]:.3f}, {left_foot_position[1]:.3f}, {left_foot_position[2]:.3f}) - Should be NEAR ground")
            self.logger.info(f"Right foot: ({right_foot_position[0]:.3f}, {right_foot_position[1]:.3f}, {right_foot_position[2]:.3f}) - Should be NEAR ground")
            
            # Verify correct ordering
            if hip_position[1] > left_foot_position[1] and hip_position[1] > right_foot_position[1]:
                self.logger.info("✓ Correct positioning: Hip is above feet")
            else:
                self.logger.warning("✗ INCORRECT positioning: Hip is NOT above feet - this indicates upside-down issue")
            
            self._position_debug_logged = True
        
        return positions
    
    def _apply_stability_filtering(self, current_positions: List[np.ndarray]) -> List[np.ndarray]:
        """Apply stability filtering to reduce jitter while allowing natural movement"""
        
        if not self.previous_tracker_positions:
            # Log first time initialization
            if not hasattr(self, '_first_positions_logged'):
                self.logger.info("Initializing tracker positions (no previous positions)")
                for i, pos in enumerate(current_positions):
                    tracker_name = ['hip', 'left_foot', 'right_foot'][i]
                    self.logger.info(f"Initial {tracker_name} position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                self._first_positions_logged = True
            return current_positions
        
        stable_positions = []
        tracker_names = ['hip', 'left_foot', 'right_foot']
        
        # Debug logging for tracker movement
        log_movement = not hasattr(self, '_movement_log_counter')
        if not log_movement:
            self._movement_log_counter += 1
            log_movement = self._movement_log_counter % 90 == 0  # Log every 90 frames
        else:
            self._movement_log_counter = 1
        
        for i, (current_pos, tracker_name) in enumerate(zip(current_positions, tracker_names)):
            if tracker_name in self.previous_tracker_positions:
                prev_pos = self.previous_tracker_positions[tracker_name]
                
                # Calculate movement distance
                movement_distance = np.linalg.norm(current_pos - prev_pos)
                
                # Dynamic stability based on movement and tracker type
                if movement_distance > self.outlier_rejection_threshold:
                    # Large movement detected - might be tracking error or teleportation
                    stability_weight = 0.85  # Increased to allow more responsiveness
                    
                    if log_movement:
                        self.logger.info(f"Large {tracker_name} movement: {movement_distance:.3f}m > {self.outlier_rejection_threshold:.3f}m")
                        self.logger.info(f"Using outlier rejection stability: {stability_weight}")
                elif movement_distance > 0.1:
                    # Medium movement - natural walking/moving
                    stability_weight = 0.7  # Allow responsive movement
                    
                    if log_movement and tracker_name == 'hip':
                        self.logger.info(f"Natural {tracker_name} movement: {movement_distance:.3f}m, stability: {stability_weight}")
                elif movement_distance > 0.02:
                    # Small movement - small adjustments
                    stability_weight = 0.8  # Balanced smoothing
                else:
                    # Very small movement - probably jitter
                    stability_weight = self.stability_factor  # Use configured smoothing
                    
                    if log_movement and tracker_name == 'hip':
                        self.logger.info(f"Small {tracker_name} jitter: {movement_distance:.3f}m, stability: {stability_weight}")
                
                # Apply weighted smoothing with adaptive response
                stable_pos = prev_pos * stability_weight + current_pos * (1 - stability_weight)
                stable_positions.append(stable_pos)
                
                if log_movement:
                    self.logger.info(f"{tracker_name.title()} - Movement: {movement_distance:.3f}m, Weight: {stability_weight:.2f}")
                    self.logger.info(f"{tracker_name.title()} - Final: ({stable_pos[0]:.3f}, {stable_pos[1]:.3f}, {stable_pos[2]:.3f})")
            else:
                stable_positions.append(current_pos)
                if log_movement:
                    self.logger.info(f"No previous {tracker_name} position, using current: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
        
        return stable_positions
    
    def _calculate_tracker_rotations(self, landmarks: np.ndarray, positions: List[np.ndarray]) -> List[np.ndarray]:
        """Calculate rotations for each tracker"""
        
        rotations = []
        
        if len(landmarks) < 33 or len(positions) < 3:
            # Return default rotations
            return [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(3)]
        
        # Hip rotation (based on hip and shoulder orientation)
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Calculate hip forward direction
            hip_vector = right_hip - left_hip
            shoulder_vector = right_shoulder - left_shoulder
            
            # Average the orientations
            combined_vector = (hip_vector + shoulder_vector) / 2
            combined_vector = combined_vector / np.linalg.norm(combined_vector)
            
            # Convert to rotation (simplified)
            forward = np.array([0, 0, -1])
            axis = np.cross(forward, combined_vector)
            angle = np.arccos(np.clip(np.dot(forward, combined_vector), -1, 1))
            
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                hip_rotation = R.from_rotvec(axis * angle).as_quat()
                hip_rotation = np.array([hip_rotation[3], hip_rotation[0], hip_rotation[1], hip_rotation[2]])  # [w,x,y,z]
            else:
                hip_rotation = np.array([1.0, 0.0, 0.0, 0.0])
            
            rotations.append(hip_rotation)
            
        except Exception:
            rotations.append(np.array([1.0, 0.0, 0.0, 0.0]))
        
        # Foot rotations (simplified - point forward)
        for i in range(2):
            rotations.append(np.array([1.0, 0.0, 0.0, 0.0]))
        
        return rotations
    
    def _calculate_tracker_confidence(self, tracker_name: str, position: np.ndarray, landmarks: np.ndarray) -> float:
        """Calculate confidence score for tracker position"""
        
        base_confidence = 0.8
        
        # Check if position is reasonable
        if position[1] < 0:  # Below ground
            base_confidence *= 0.5
        
        # Check landmark visibility if available
        if hasattr(landmarks, 'visibility'):
            relevant_landmarks = {
                'hip': [23, 24],
                'left_foot': [27, 31],
                'right_foot': [28, 32]
            }
            
            if tracker_name in relevant_landmarks:
                indices = relevant_landmarks[tracker_name]
                if hasattr(landmarks, 'visibility') and len(landmarks.visibility) > max(indices):
                    visibilities = [landmarks.visibility[i] for i in indices if i < len(landmarks.visibility)]
                    if visibilities:
                        avg_visibility = np.mean(visibilities)
                        base_confidence *= avg_visibility
        
        # Check stability with previous positions
        if tracker_name in self.previous_tracker_positions:
            prev_pos = self.previous_tracker_positions[tracker_name]
            movement = np.linalg.norm(position - prev_pos)
            
            # Penalize large movements
            if movement > 0.1:  # 10cm movement between frames
                movement_penalty = max(0.5, 1.0 - movement)
                base_confidence *= movement_penalty
        
        return np.clip(base_confidence, 0.0, 1.0)
    
    def reset_tracker_positions(self):
        """Reset all tracker positions to allow fresh tracking"""
        self.previous_tracker_positions.clear()
        self.position_history.clear()
        self.is_calibrated = False
        self.calibration_samples.clear()
        self.logger.info("Reset all tracker positions and calibration data")

    def force_recalibration(self):
        """Force recalibration of the tracking system"""
        self.is_calibrated = False
        self.calibration_samples.clear()
        self.headset_to_body_offset = np.array([0.0, 0.0, 0.0])
        self.logger.info("Forced recalibration - tracking system will recalibrate")
    
    def _apply_headset_anchoring(self, landmarks: np.ndarray, headset_pose: np.ndarray) -> np.ndarray:
        """Apply headset anchoring to keep all landmarks near the user's body in VR space"""
        
        # CRITICAL FIX: Apply coordinate correction for upside-down tracker issue
        # MediaPipe uses a different coordinate system than SteamVR
        corrected_landmarks = landmarks.copy()
        
        # Apply 180-degree rotation around X-axis to fix upside-down issue
        # This flips Y and Z coordinates to match SteamVR coordinate system
        for i in range(len(corrected_landmarks)):
            original_y = corrected_landmarks[i][1]
            original_z = corrected_landmarks[i][2]
            # Flip Y and Z coordinates (180° rotation around X-axis)
            corrected_landmarks[i][1] = -original_z  # New Y = -old Z
            corrected_landmarks[i][2] = original_y   # New Z = old Y
        
        # Define maximum reasonable distances from headset to body parts
        max_distances = {
            'head': 0.3,      # Head should be very close to headset
            'torso': 0.8,     # Torso within 80cm of headset
            'hips': 1.2,      # Hips within 1.2m of headset  
            'feet': 2.0       # Feet within 2m of headset (standing height)
        }
        
        # Ensure landmarks don't drift too far from headset position
        headset_position = headset_pose[:3] if len(headset_pose) >= 3 else headset_pose
        
        # Apply constraints to key landmarks
        if len(corrected_landmarks) >= 33:
            # Head landmarks (0-10): Keep very close to headset
            for i in range(11):
                distance = np.linalg.norm(corrected_landmarks[i] - headset_position)
                if distance > max_distances['head']:
                    # Pull landmark back toward headset
                    direction = (corrected_landmarks[i] - headset_position) / distance
                    corrected_landmarks[i] = headset_position + direction * max_distances['head']
            
            # Torso landmarks (11-22): Keep within torso range
            for i in range(11, 23):
                distance = np.linalg.norm(corrected_landmarks[i] - headset_position)
                if distance > max_distances['torso']:
                    direction = (corrected_landmarks[i] - headset_position) / distance
                    corrected_landmarks[i] = headset_position + direction * max_distances['torso']
            
            # Hip landmarks (23-24): Keep within hip range
            for i in range(23, 25):
                distance = np.linalg.norm(corrected_landmarks[i] - headset_position)
                if distance > max_distances['hips']:
                    direction = (corrected_landmarks[i] - headset_position) / distance
                    corrected_landmarks[i] = headset_position + direction * max_distances['hips']
            
            # Leg and foot landmarks (25-32): Keep within feet range
            for i in range(25, min(33, len(corrected_landmarks))):
                distance = np.linalg.norm(corrected_landmarks[i] - headset_position)
                if distance > max_distances['feet']:
                    direction = (corrected_landmarks[i] - headset_position) / distance
                    corrected_landmarks[i] = headset_position + direction * max_distances['feet']
        
        return corrected_landmarks
    
    def _apply_body_anchoring_constraints(self, tracker_positions: List[np.ndarray], headset_pose: np.ndarray) -> List[np.ndarray]:
        """Apply body anchoring constraints to keep trackers reasonably close to headset and in correct positions"""
        
        if not tracker_positions or len(headset_pose) < 3:
            return tracker_positions
        
        headset_position = headset_pose[:3]
        constrained_positions = []
        
        # CRITICAL: Ensure correct body positioning relative to headset
        # Hip should be BELOW headset, feet should be BELOW hip
        
        tracker_names = ['hip', 'left_foot', 'right_foot']
        
        for i, (position, tracker_name) in enumerate(zip(tracker_positions, tracker_names)):
            corrected_position = position.copy()
            
            # Apply body-relative positioning constraints
            if tracker_name == 'hip':
                # Hip should be approximately 0.6-0.8m below headset Y level
                expected_hip_y = headset_position[1] - 0.7  # 70cm below headset
                if abs(corrected_position[1] - expected_hip_y) > 0.3:  # Allow 30cm variance
                    corrected_position[1] = expected_hip_y
                    
                # Hip should be within 0.5m horizontally from headset (X,Z)
                horizontal_offset = np.array([corrected_position[0] - headset_position[0], 0, corrected_position[2] - headset_position[2]])
                horizontal_distance = np.linalg.norm(horizontal_offset)
                if horizontal_distance > 0.5:
                    direction = horizontal_offset / horizontal_distance
                    corrected_position[0] = headset_position[0] + direction[0] * 0.5
                    corrected_position[2] = headset_position[2] + direction[2] * 0.5
            
            elif 'foot' in tracker_name:
                # Feet should be approximately 1.0-1.2m below headset Y level
                expected_foot_y = headset_position[1] - 1.4  # 140cm below headset (near ground)
                expected_foot_y = max(expected_foot_y, 0.02)  # Never below 2cm from ground
                
                if abs(corrected_position[1] - expected_foot_y) > 0.2:  # Allow 20cm variance
                    corrected_position[1] = expected_foot_y
                
                # Feet should be within 0.8m horizontally from headset
                horizontal_offset = np.array([corrected_position[0] - headset_position[0], 0, corrected_position[2] - headset_position[2]])
                horizontal_distance = np.linalg.norm(horizontal_offset)
                if horizontal_distance > 0.8:
                    direction = horizontal_offset / horizontal_distance
                    corrected_position[0] = headset_position[0] + direction[0] * 0.8
                    corrected_position[2] = headset_position[2] + direction[2] * 0.8
            
            # Final distance check from headset
            distance_from_headset = np.linalg.norm(corrected_position - headset_position)
            max_distance = 1.0 if tracker_name == 'hip' else 1.8
            
            if distance_from_headset > max_distance:
                # Tracker is too far from headset - pull it back
                direction = (corrected_position - headset_position) / distance_from_headset
                corrected_position = headset_position + direction * max_distance
                
                # Log the constraint application
                if not hasattr(self, f'_constraint_log_{tracker_name}'):
                    self.logger.warning(f"Applying body anchoring constraint to {tracker_name} tracker")
                    self.logger.warning(f"Distance from headset: {distance_from_headset:.3f}m > max: {max_distance:.3f}m")
                    self.logger.warning(f"Headset: ({headset_position[0]:.3f}, {headset_position[1]:.3f}, {headset_position[2]:.3f})")
                    self.logger.warning(f"Original: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
                    self.logger.warning(f"Constrained: ({corrected_position[0]:.3f}, {corrected_position[1]:.3f}, {corrected_position[2]:.3f})")
                    setattr(self, f'_constraint_log_{tracker_name}', True)
            
            constrained_positions.append(corrected_position)
        
        # Final sanity check: Ensure hip is above feet
        if len(constrained_positions) >= 3:
            hip_pos = constrained_positions[0]
            left_foot_pos = constrained_positions[1]
            right_foot_pos = constrained_positions[2]
            
            # If hip is below feet, this indicates upside-down issue - force correction
            min_foot_y = min(left_foot_pos[1], right_foot_pos[1])
            if hip_pos[1] <= min_foot_y:
                self.logger.warning("CRITICAL: Hip below feet detected - applying emergency correction")
                hip_pos[1] = min_foot_y + 0.7  # Force hip 70cm above feet
                constrained_positions[0] = hip_pos
        
        return constrained_positions
    
    def _apply_final_sanity_check(self, tracker_results: Dict, headset_pose: Optional[np.ndarray]) -> Dict:
        """Final sanity check to ensure tracker positions are reasonable and not upside-down"""
        
        if headset_pose is None:
            return tracker_results
        
        headset_position = headset_pose[:3]
        sanity_checked_results = {}
        
        # Extract positions for upside-down check
        hip_pos = None
        foot_positions = []
        
        for name, data in tracker_results.items():
            position = data['position'].copy()
            
            if 'hip' in name:
                hip_pos = position
            elif 'foot' in name:
                foot_positions.append(position)
        
        # CRITICAL: Check for upside-down configuration
        if hip_pos is not None and foot_positions:
            avg_foot_y = np.mean([fp[1] for fp in foot_positions])
            
            if hip_pos[1] < avg_foot_y:
                self.logger.error("CRITICAL UPSIDE-DOWN DETECTION: Hip is below feet!")
                self.logger.error(f"Hip Y: {hip_pos[1]:.3f}, Average Foot Y: {avg_foot_y:.3f}")
                self.logger.error("Applying emergency coordinate correction...")
                
                # Emergency fix: Force correct body positioning relative to headset
                corrected_hip_y = headset_position[1] - 0.7    # 70cm below headset
                corrected_foot_y = headset_position[1] - 1.4   # 140cm below headset (near ground)
                corrected_foot_y = max(corrected_foot_y, 0.02) # Never below 2cm from ground
                
                # Apply corrections to all tracker results
                for name, data in tracker_results.items():
                    position = data['position'].copy()
                    
                    if 'hip' in name:
                        position[1] = corrected_hip_y
                        self.logger.error(f"Corrected hip Y from {data['position'][1]:.3f} to {position[1]:.3f}")
                    elif 'foot' in name:
                        position[1] = corrected_foot_y
                        self.logger.error(f"Corrected {name} Y from {data['position'][1]:.3f} to {position[1]:.3f}")
                    
                    tracker_results[name]['position'] = position
        
        # Apply individual tracker checks
        for name, data in tracker_results.items():
            position = data['position'].copy()
            
            # Calculate distance from headset
            distance_from_headset = np.linalg.norm(position - headset_position)
            
            # Define expected positions relative to headset
            if 'hip' in name:
                # Hip should be 60-80cm below headset
                expected_y = headset_position[1] - 0.7
                if abs(position[1] - expected_y) > 0.3:
                    self.logger.warning(f"Hip Y position seems wrong: {position[1]:.3f}, expected: {expected_y:.3f}")
                    position[1] = expected_y
                
                max_distance = 1.0
            elif 'foot' in name:
                # Feet should be 120-150cm below headset (near ground)
                expected_y = headset_position[1] - 1.4
                expected_y = max(expected_y, 0.02)  # Never below ground
                if abs(position[1] - expected_y) > 0.2:
                    self.logger.warning(f"{name} Y position seems wrong: {position[1]:.3f}, expected: {expected_y:.3f}")
                    position[1] = expected_y
                
                max_distance = 1.8
            else:
                max_distance = 2.0
            
            # Distance check
            if distance_from_headset > max_distance:
                direction = (position - headset_position) / distance_from_headset
                sane_position = headset_position + direction * max_distance
                
                self.logger.warning(f"Sanity check: {name} tracker repositioned")
                self.logger.warning(f"Distance from headset: {distance_from_headset:.3f}m > max: {max_distance:.3f}m")
                self.logger.warning(f"Repositioned: ({sane_position[0]:.3f}, {sane_position[1]:.3f}, {sane_position[2]:.3f})")
                
                position = sane_position
            
            # Ensure height is reasonable (not too low or high)
            position[1] = np.clip(position[1], 0.0, 2.5)  # Ground level to 2.5m height limit
            
            sanity_checked_results[name] = {
                'position': position,
                'rotation': data['rotation'],
                'confidence': data['confidence']
            }
        
        # Final verification log
        if hip_pos is not None and foot_positions:
            final_hip = sanity_checked_results.get('hip', {}).get('position', hip_pos)
            final_feet = [sanity_checked_results.get(name, {}).get('position', pos) 
                         for name, pos in zip(['left_foot', 'right_foot'], foot_positions)]
            
            if len(final_feet) > 0:
                final_avg_foot_y = np.mean([fp[1] for fp in final_feet])
                if final_hip[1] > final_avg_foot_y:
                    self.logger.info(f"✓ Final check: Hip ({final_hip[1]:.3f}) is above feet ({final_avg_foot_y:.3f})")
                else:
                    self.logger.error(f"✗ Final check: Hip ({final_hip[1]:.3f}) is still below feet ({final_avg_foot_y:.3f})")
        
        return sanity_checked_results
