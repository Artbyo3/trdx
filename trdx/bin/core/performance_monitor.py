"""
TRDX Performance Monitor - Real-time performance tracking and optimization
Provides lightweight performance monitoring for the TRDX tracking system
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import logging

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    frame_rate: float
    processing_latency: float
    camera_fps: Dict[int, float]
    pose_detection_rate: float
    udp_send_rate: float
    # NEW: Precision tracking metrics
    tracker_precision: Dict[int, float]  # Tracker ID -> precision score
    coordinate_consistency: float  # Measure of coordinate transformation consistency
    mediapipe_confidence: float  # Average MediaPipe confidence

class PerformanceMonitor:
    """Lightweight performance monitoring system"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.logger = logging.getLogger(__name__)
        
        # Performance counters
        self.frame_counts = {}
        self.pose_detection_count = 0
        self.udp_send_count = 0
        self.last_reset_time = time.time()
        
        # Timing measurements
        self.processing_times = deque(maxlen=50)
        self.camera_frame_times = {}
        
        # System monitoring
        self.process = psutil.Process()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'min_fps': 15.0,
            'max_cpu_percent': 80.0,
            'max_memory_mb': 1024.0,
            'max_latency_ms': 50.0
        }
    
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics every 1 second
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.debug(f"Performance monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_reset_time
        
        # System metrics
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # Frame rate calculation
        total_frames = sum(self.frame_counts.values())
        frame_rate = total_frames / time_delta if time_delta > 0 else 0
        
        # Camera FPS
        camera_fps = {}
        for camera_id, count in self.frame_counts.items():
            camera_fps[camera_id] = count / time_delta if time_delta > 0 else 0
        
        # Processing latency (average of recent measurements)
        avg_latency = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # Detection and UDP rates
        pose_rate = self.pose_detection_count / time_delta if time_delta > 0 else 0
        udp_rate = self.udp_send_count / time_delta if time_delta > 0 else 0
        
        return PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            frame_rate=frame_rate,
            processing_latency=avg_latency * 1000,  # Convert to ms
            camera_fps=camera_fps,
            pose_detection_rate=pose_rate,
            udp_send_rate=udp_rate
        )
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and log warnings"""
        alerts = []
        
        if metrics.frame_rate < self.thresholds['min_fps']:
            alerts.append(f"Low FPS: {metrics.frame_rate:.1f}")
        
        if metrics.cpu_percent > self.thresholds['max_cpu_percent']:
            alerts.append(f"High CPU: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_mb > self.thresholds['max_memory_mb']:
            alerts.append(f"High Memory: {metrics.memory_mb:.1f}MB")
        
        if metrics.processing_latency > self.thresholds['max_latency_ms']:
            alerts.append(f"High Latency: {metrics.processing_latency:.1f}ms")
        
        if alerts:
            # Only log performance alerts occasionally to reduce spam
            if hasattr(self, '_performance_alert_counter'):
                self._performance_alert_counter += 1
            else:
                self._performance_alert_counter = 1
                
            if self._performance_alert_counter % 60 == 1:  # Every 60 seconds
                self.logger.warning(f"Performance alerts: {', '.join(alerts)}")
    
    def record_frame(self, camera_id: int):
        """Record a frame from a camera"""
        if camera_id not in self.frame_counts:
            self.frame_counts[camera_id] = 0
        self.frame_counts[camera_id] += 1
    
    def record_pose_detection(self):
        """Record a pose detection"""
        self.pose_detection_count += 1
    
    def record_udp_send(self):
        """Record a UDP packet send"""
        self.udp_send_count += 1
    
    def record_processing_time(self, start_time: float):
        """Record processing time for latency calculation"""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
    
    def record_tracker_precision(self, tracker_id: int, precision_score: float):
        """Record precision score for a specific tracker"""
        if not hasattr(self, 'tracker_precision_scores'):
            self.tracker_precision_scores = {}
        
        if tracker_id not in self.tracker_precision_scores:
            self.tracker_precision_scores[tracker_id] = []
        
        self.tracker_precision_scores[tracker_id].append(precision_score)
        
        # Keep only last 100 precision scores per tracker
        if len(self.tracker_precision_scores[tracker_id]) > 100:
            self.tracker_precision_scores[tracker_id].pop(0)
    
    def record_mediapipe_confidence(self, confidence: float):
        """Record MediaPipe confidence score"""
        if not hasattr(self, 'mediapipe_confidences'):
            self.mediapipe_confidences = []
        
        self.mediapipe_confidences.append(confidence)
        
        # Keep only last 100 confidence scores
        if len(self.mediapipe_confidences) > 100:
            self.mediapipe_confidences.pop(0)
    
    def record_coordinate_consistency(self, consistency_score: float):
        """Record coordinate transformation consistency"""
        if not hasattr(self, 'coordinate_consistencies'):
            self.coordinate_consistencies = []
        
        self.coordinate_consistencies.append(consistency_score)
        
        # Keep only last 100 consistency scores
        if len(self.coordinate_consistencies) > 100:
            self.coordinate_consistencies.pop(0)
    
    def reset_counters(self):
        """Reset performance counters"""
        self.frame_counts.clear()
        self.pose_detection_count = 0
        self.udp_send_count = 0
        self.last_reset_time = time.time()
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, seconds: int = 30) -> Dict:
        """Get average metrics over the last N seconds"""
        if not self.metrics_history:
            return {}
        
        cutoff_time = time.time() - seconds
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_mb': sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            'avg_frame_rate': sum(m.frame_rate for m in recent_metrics) / len(recent_metrics),
            'avg_latency_ms': sum(m.processing_latency for m in recent_metrics) / len(recent_metrics),
            'avg_pose_rate': sum(m.pose_detection_rate for m in recent_metrics) / len(recent_metrics),
            'avg_udp_rate': sum(m.udp_send_rate for m in recent_metrics) / len(recent_metrics)
        }
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary"""
        current = self.get_current_metrics()
        if not current:
            return "No performance data available"
        
        avg_30s = self.get_average_metrics(30)
        
        summary = [
            f"🔄 FPS: {current.frame_rate:.1f}",
            f"💻 CPU: {current.cpu_percent:.1f}%",
            f"🧠 RAM: {current.memory_mb:.1f}MB",
            f"⏱️ Latency: {current.processing_latency:.1f}ms",
            f"👁️ Poses/s: {current.pose_detection_rate:.1f}",
            f"📡 UDP/s: {current.udp_send_rate:.1f}"
        ]
        
        if avg_30s:
            summary.append(f"📊 30s Avg: {avg_30s['avg_frame_rate']:.1f}fps, {avg_30s['avg_cpu_percent']:.1f}% CPU")
        
        return " | ".join(summary)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()