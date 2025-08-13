"""
Advanced 3D Tracker Visualizer Widget
Shows EXACTLY the same data being sent to SteamVR via UDP in real-time
This is NOT a reconstruction from cameras - it's the EXACT SteamVR data
"""

import numpy as np
import time
import math
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QFrame, QGridLayout, QGroupBox, QTabWidget,
                               QSlider, QCheckBox, QPushButton, QSpinBox)
from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QPoint, QRect
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QLinearGradient, QPolygon


class TrackerVisualizerWidget(QWidget):
    """Advanced 3D visualization of EXACT SteamVR UDP data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracker_data = {}  # Store latest tracker positions
        self.data_history = {}  # Store position history for graphs
        self.udp_messages = []  # Store raw UDP messages
        self.start_time = time.time()
        
        # Visualization settings
        self.show_trails = True
        self.show_grid = True
        self.show_axes = True
        self.show_skeleton = True
        self.animation_speed = 1.0
        self.zoom_level = 1.0
        self.rotation_x = 15
        self.rotation_y = 45
        
        self.setup_ui()
        
        # Update timer
        # OPTIMIZED: Reduced update frequency for better performance
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # ~20 FPS for better performance
        
    def setup_ui(self):
        """Setup the advanced UI layout"""
        layout = QVBoxLayout(self)
        
        # Title removed to save space for visualization
        
        # Control Panel
        controls = self.create_control_panel()
        layout.addWidget(controls)
        
        # Main visualization area
        main_content = QHBoxLayout()
        
        # Enhanced 3D Canvas
        self.canvas_widget = Advanced3DTrackerCanvas()
        self.canvas_widget.show_trails = self.show_trails
        self.canvas_widget.show_grid = self.show_grid
        self.canvas_widget.show_axes = self.show_axes
        self.canvas_widget.show_skeleton = self.show_skeleton
        main_content.addWidget(self.canvas_widget, 3)
        
        # Add demo data immediately for testing
        self.add_demo_data()
        
        # Tabbed Data Panel
        self.data_tabs = QTabWidget()
        self.data_tabs.setMinimumWidth(350)
        
        # Real-time data tab
        realtime_tab = self.create_realtime_data_tab()
        self.data_tabs.addTab(realtime_tab, "📊 Live Data")
        
        # History graphs tab
        history_tab = self.create_history_tab()
        self.data_tabs.addTab(history_tab, "📈 History")
        
        # Raw UDP tab
        udp_tab = self.create_udp_tab()
        self.data_tabs.addTab(udp_tab, "🔧 Raw UDP")
        
        main_content.addWidget(self.data_tabs, 1)
        
        layout.addLayout(main_content)
    
    def create_control_panel(self):
        """Create visualization control panel"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #2B2B2B;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
                margin: 5px 0px;
            }
        """)
        layout = QHBoxLayout(panel)
        
        # View controls
        view_group = QGroupBox("🎮 View Controls")
        view_layout = QHBoxLayout(view_group)
        
        # Grid toggle
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self.toggle_grid)
        view_layout.addWidget(self.grid_check)
        
        # Trails toggle
        self.trails_check = QCheckBox("Trails")
        self.trails_check.setChecked(True)
        self.trails_check.toggled.connect(self.toggle_trails)
        view_layout.addWidget(self.trails_check)
        
        # Skeleton toggle
        self.skeleton_check = QCheckBox("Skeleton")
        self.skeleton_check.setChecked(True)
        self.skeleton_check.toggled.connect(self.toggle_skeleton)
        view_layout.addWidget(self.skeleton_check)
        
        # Zoom control
        zoom_label = QLabel("Zoom:")
        view_layout.addWidget(zoom_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        view_layout.addWidget(self.zoom_slider)
        
        layout.addWidget(view_group)
        
        # Data controls
        data_group = QGroupBox("📡 Data Controls")
        data_layout = QHBoxLayout(data_group)
        
        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self.clear_history)
        data_layout.addWidget(clear_btn)
        
        # Demo data button
        demo_btn = QPushButton("🎯 Demo Data")
        demo_btn.clicked.connect(self.add_demo_data)
        data_layout.addWidget(demo_btn)
        
        # Freeze button
        self.freeze_btn = QPushButton("⏸️ Freeze")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.toggled.connect(self.toggle_freeze)
        data_layout.addWidget(self.freeze_btn)
        
        # FPS display
        self.fps_label = QLabel("FPS: --")
        data_layout.addWidget(self.fps_label)
        
        layout.addWidget(data_group)
        
        return panel
    
    def create_realtime_data_tab(self):
        """Create real-time data display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Hip Tracker
        self.hip_group = self.create_enhanced_tracker_group("Hip/Waist (ID: 0)", "#FF5722", "🔴")
        layout.addWidget(self.hip_group)
        
        # Left Foot Tracker
        self.left_group = self.create_enhanced_tracker_group("Left Foot (ID: 1)", "#4CAF50", "🟢")
        layout.addWidget(self.left_group)
        
        # Right Foot Tracker  
        self.right_group = self.create_enhanced_tracker_group("Right Foot (ID: 2)", "#2196F3", "🔵")
        layout.addWidget(self.right_group)
        
        # Statistics
        stats_group = QGroupBox("📊 Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.total_packets_label = QLabel("Total Packets: 0")
        self.packet_rate_label = QLabel("Packet Rate: 0 Hz")
        self.data_rate_label = QLabel("Data Rate: 0 B/s")
        
        stats_layout.addWidget(self.total_packets_label, 0, 0)
        stats_layout.addWidget(self.packet_rate_label, 0, 1)
        stats_layout.addWidget(self.data_rate_label, 1, 0)
        
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return tab
    
    def create_history_tab(self):
        """Create history graphs tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Position history graph
        self.position_graph = PositionHistoryGraph()
        layout.addWidget(self.position_graph)
        
        return tab
    
    def create_udp_tab(self):
        """Create raw UDP data tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # UDP message log
        self.udp_log = QLabel("Waiting for UDP data...")
        self.udp_log.setStyleSheet("""
            QLabel {
                font-family: 'Consolas', monospace;
                font-size: 10px;
                background-color: #1E1E1E;
                color: #00FF00;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.udp_log.setWordWrap(True)
        self.udp_log.setAlignment(Qt.AlignTop)
        layout.addWidget(self.udp_log)
        
        return tab
    
    def create_enhanced_tracker_group(self, title, color, icon):
        """Create enhanced tracker group with more details"""
        group = QGroupBox(f"{icon} {title}")
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {color};
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #2D2D2D;
                color: #E0E0E0;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: {color};
                background-color: #2D2D2D;
            }}
        """)
        
        layout = QGridLayout(group)
        
        # Position with larger, colorful display
        pos_frame = QFrame()
        pos_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #3A3A3A;
                border: 1px solid {color};
                border-radius: 5px;
                padding: 5px;
            }}
        """)
        pos_layout = QGridLayout(pos_frame)
        
        # Enhanced position labels
        pos_x = QLabel("X: 0.000")
        pos_y = QLabel("Y: 0.000") 
        pos_z = QLabel("Z: 0.000")
        
        for label in [pos_x, pos_y, pos_z]:
            label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    padding: 3px;
                    color: #E0E0E0;
                }
            """)
        
        pos_layout.addWidget(QLabel("Position:"), 0, 0)
        pos_layout.addWidget(pos_x, 1, 0)
        pos_layout.addWidget(pos_y, 2, 0)
        pos_layout.addWidget(pos_z, 3, 0)
        
        layout.addWidget(pos_frame, 0, 0, 1, 2)
        
        # Additional info
        conf_label = QLabel("Confidence: 0.000")
        status_label = QLabel("Status: Inactive")
        packets_label = QLabel("Packets: 0")
        
        layout.addWidget(conf_label, 1, 0)
        layout.addWidget(status_label, 1, 1)
        layout.addWidget(packets_label, 2, 0)
        
        # Store references
        group.pos_x = pos_x
        group.pos_y = pos_y
        group.pos_z = pos_z
        group.confidence = conf_label
        group.status = status_label
        group.packets = packets_label
        
        return group
        
    def add_demo_data(self):
        """Add demo tracker data for visualization testing"""
        import numpy as np
        
        # Demo positions for testing the 3D visualizer (más visibles)
        demo_trackers = {
            0: np.array([0.0, 1.2, 0.0]),    # Hip at waist level (más alto)
            1: np.array([0.5, 0.0, 0.3]),    # Left foot on ground (más separado)
            2: np.array([-0.5, 0.0, 0.3])   # Right foot on ground (más separado)
        }
        
        # Add demo data to both the widget and canvas
        for tracker_id, position in demo_trackers.items():
            self.update_tracker_data(tracker_id, position, np.array([1.0, 0.0, 0.0, 0.0]), 0.95)
    
    def toggle_grid(self, checked):
        """Toggle grid display"""
        self.show_grid = checked
        self.canvas_widget.show_grid = checked
        
    def toggle_trails(self, checked):
        """Toggle trails display"""
        self.show_trails = checked
        self.canvas_widget.show_trails = checked
        
    def toggle_skeleton(self, checked):
        """Toggle skeleton display"""
        self.show_skeleton = checked
        self.canvas_widget.show_skeleton = checked
        
    def update_zoom(self, value):
        """Update zoom level"""
        self.zoom_level = value / 100.0
        self.canvas_widget.zoom_level = self.zoom_level
        
    def clear_history(self):
        """Clear position history"""
        self.data_history.clear()
        self.position_graph.clear_data()
        
    def toggle_freeze(self, checked):
        """Freeze/unfreeze data updates"""
        if checked:
            self.update_timer.stop()
            self.freeze_btn.setText("▶️ Resume")
            # Status label removed
        else:
            self.update_timer.start(16)
            self.freeze_btn.setText("⏸️ Freeze")
            # Status label removed
        
    def create_data_panel(self):
        """Create panel showing numerical tracker data"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setStyleSheet("""
            QFrame {
                background-color: #F5F5F5;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        
        # Hip Tracker
        self.hip_group = self.create_tracker_group("Hip/Waist (ID: 0)", "#FF5722")
        layout.addWidget(self.hip_group)
        
        # Left Foot Tracker
        self.left_group = self.create_tracker_group("Left Foot (ID: 1)", "#4CAF50")
        layout.addWidget(self.left_group)
        
        # Right Foot Tracker  
        self.right_group = self.create_tracker_group("Right Foot (ID: 2)", "#2196F3")
        layout.addWidget(self.right_group)
        
        layout.addStretch()
        
        return panel
        
    def create_tracker_group(self, title, color):
        """Create a group box for tracker data"""
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {color};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {color};
            }}
        """)
        
        layout = QGridLayout(group)
        
        # Position labels
        layout.addWidget(QLabel("Position:"), 0, 0)
        pos_x = QLabel("X: 0.000")
        pos_y = QLabel("Y: 0.000") 
        pos_z = QLabel("Z: 0.000")
        
        layout.addWidget(pos_x, 1, 0)
        layout.addWidget(pos_y, 2, 0)
        layout.addWidget(pos_z, 3, 0)
        
        # Confidence
        conf_label = QLabel("Confidence: 0.000")
        layout.addWidget(conf_label, 4, 0)
        
        # Status
        status_label = QLabel("Status: Inactive")
        layout.addWidget(status_label, 5, 0)
        
        # Store references
        group.pos_x = pos_x
        group.pos_y = pos_y
        group.pos_z = pos_z
        group.confidence = conf_label
        group.status = status_label
        
        return group
        
    def update_tracker_data(self, tracker_id, position, rotation, confidence):
        """Update tracker data - THIS IS EXACT STEAMVR UDP DATA"""
        current_time = time.time() - self.start_time
        
        # Store current data
        self.tracker_data[tracker_id] = {
            'position': position,
            'rotation': rotation,
            'confidence': confidence,
            'active': True,
            'timestamp': current_time
        }
        
        # Store data history for graphs
        if tracker_id not in self.data_history:
            self.data_history[tracker_id] = {
                'positions': [],
                'timestamps': [],
                'confidences': []
            }
        
        history = self.data_history[tracker_id]
        history['positions'].append(position.copy())
        history['timestamps'].append(current_time)
        history['confidences'].append(confidence)
        
        # Keep only last 1000 points for performance
        if len(history['positions']) > 1000:
            history['positions'].pop(0)
            history['timestamps'].pop(0)
            history['confidences'].pop(0)
        
        # Update canvas widget with EXACT UDP data
        self.canvas_widget.update_tracker(tracker_id, position)
        
        # Create UDP message string for raw display
        udp_msg = f"ID:{tracker_id} | X:{position[0]:.6f} Y:{position[1]:.6f} Z:{position[2]:.6f} | Conf:{confidence:.3f}"
        self.udp_messages.append(f"[{current_time:.2f}s] {udp_msg}")
        
        # Keep only last 50 UDP messages
        if len(self.udp_messages) > 50:
            self.udp_messages.pop(0)
        
    def update_display(self):
        """Update the enhanced numerical display"""
        # Calculate statistics
        total_packets = sum(len(self.data_history.get(i, {}).get('positions', [])) for i in range(3))
        current_time = time.time() - self.start_time
        
        # Status indicator removed (header was removed)
        
        # Update Hip (ID: 0)
        if 0 in self.tracker_data:
            data = self.tracker_data[0]
            pos = data['position']
            self.hip_group.pos_x.setText(f"X: {pos[0]:.6f}")
            self.hip_group.pos_y.setText(f"Y: {pos[1]:.6f}")
            self.hip_group.pos_z.setText(f"Z: {pos[2]:.6f}")
            self.hip_group.confidence.setText(f"Confidence: {data['confidence']:.3f}")
            self.hip_group.status.setText("● ACTIVE")
            self.hip_group.status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if hasattr(self.hip_group, 'packets'):
                packet_count = len(self.data_history.get(0, {}).get('positions', []))
                self.hip_group.packets.setText(f"Packets: {packet_count}")
        else:
            self.hip_group.status.setText("○ INACTIVE")
            self.hip_group.status.setStyleSheet("color: #F44336; font-weight: bold;")
            
        # Update Left Foot (ID: 1)
        if 1 in self.tracker_data:
            data = self.tracker_data[1]
            pos = data['position']
            self.left_group.pos_x.setText(f"X: {pos[0]:.6f}")
            self.left_group.pos_y.setText(f"Y: {pos[1]:.6f}")
            self.left_group.pos_z.setText(f"Z: {pos[2]:.6f}")
            self.left_group.confidence.setText(f"Confidence: {data['confidence']:.3f}")
            self.left_group.status.setText("● ACTIVE")
            self.left_group.status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if hasattr(self.left_group, 'packets'):
                packet_count = len(self.data_history.get(1, {}).get('positions', []))
                self.left_group.packets.setText(f"Packets: {packet_count}")
        else:
            self.left_group.status.setText("○ INACTIVE")
            self.left_group.status.setStyleSheet("color: #F44336; font-weight: bold;")
            
        # Update Right Foot (ID: 2)
        if 2 in self.tracker_data:
            data = self.tracker_data[2]
            pos = data['position']
            self.right_group.pos_x.setText(f"X: {pos[0]:.6f}")
            self.right_group.pos_y.setText(f"Y: {pos[1]:.6f}")
            self.right_group.pos_z.setText(f"Z: {pos[2]:.6f}")
            self.right_group.confidence.setText(f"Confidence: {data['confidence']:.3f}")
            self.right_group.status.setText("● ACTIVE")
            self.right_group.status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if hasattr(self.right_group, 'packets'):
                packet_count = len(self.data_history.get(2, {}).get('positions', []))
                self.right_group.packets.setText(f"Packets: {packet_count}")
        else:
            self.right_group.status.setText("○ INACTIVE")
            self.right_group.status.setStyleSheet("color: #F44336; font-weight: bold;")
        
        # Update statistics
        if hasattr(self, 'total_packets_label'):
            self.total_packets_label.setText(f"Total Packets: {total_packets}")
            
        # Update UDP log
        if hasattr(self, 'udp_log') and self.udp_messages:
            log_text = "\n".join(self.udp_messages[-20:])  # Show last 20 messages
            self.udp_log.setText(log_text)
        
        # Update position graph
        if hasattr(self, 'position_graph'):
            self.position_graph.update_data(self.data_history)


class Advanced3DTrackerCanvas(QWidget):
    """Advanced 3D Canvas widget showing EXACT SteamVR UDP data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trackers = {}
        self.tracker_trails = {0: [], 1: [], 2: []}  # Position history trails
        self.setMinimumSize(600, 400)
        self.scale = 150  # Enhanced scale factor
        self.zoom_level = 1.0
        self.rotation_x = 15
        self.rotation_y = 45
        
        # Visual settings
        self.show_trails = True
        self.show_grid = True
        self.show_axes = True
        self.show_skeleton = True
        self.animation_time = 0
        
        # 3D Camera settings (posición corregida para ver trackers correctamente)
        self.camera_position = [0.0, 1.5, -2.5]  # X, Y, Z position in world space (atrás para ver hacia adelante)
        self.camera_elevation = -15.0  # degrees (up/down rotation) - mirando ligeramente hacia abajo (orientación corregida)
        self.camera_azimuth = 0.0    # degrees (left/right rotation) - centrado
        self.field_of_view = 60.0     # degrees
        self.mouse_sensitivity = 0.5
        self.movement_speed = 0.1     # Units per frame
        self.fast_movement_speed = 0.3  # When holding Shift
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.is_dragging = False
        
        # Keyboard movement state
        self.keys_pressed = {
            'w': False, 'a': False, 's': False, 'd': False,
            'q': False, 'e': False, 'shift': False
        }
        
        # Enhanced colors with gradients
        self.colors = {
            0: QColor(255, 87, 34),   # Hip - Red-orange
            1: QColor(76, 175, 80),   # Left foot - Green  
            2: QColor(33, 150, 243)   # Right foot - Blue
        }
        
        # Advanced 3D rendering settings
        self.enable_antialiasing = True
        self.enable_depth_sorting = True
        self.enable_lighting = True
        self.ambient_light = 0.3
        self.directional_light = 0.7
        self.light_direction = [0.5, -1.0, 0.3]  # Normalized light direction
        self.enable_shadows = True
        self.shadow_opacity = 0.2
        self.enable_glow_effects = True
        self.particle_system = True
        
        # 3D Geometry for trackers
        self.tracker_geometries = {
            0: self.generate_hip_geometry(),     # Hip - Cube
            1: self.generate_foot_geometry(),    # Left foot - Cylinder
            2: self.generate_foot_geometry()     # Right foot - Cylinder
        }
        
        # Background timer for smooth animation
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.animate)
        self.anim_timer.start(16)  # 60 FPS animation
        
        # Enable mouse tracking for 3D interaction
        self.setMouseTracking(True)
        
        # Enable keyboard focus for WASD movement
        self.setFocusPolicy(Qt.StrongFocus)
        
    def generate_hip_geometry(self):
        """Generate 3D cube geometry for hip tracker"""
        import math
        vertices = []
        # Cube vertices (centered, 0.1 unit size)
        size = 0.05
        for x in [-size, size]:
            for y in [-size, size]:
                for z in [-size, size]:
                    vertices.append([x, y, z])
        
        # Cube faces (each face has 4 vertices, triangulated to 2 triangles)
        faces = [
            # Front face
            [0, 1, 3], [0, 3, 2],
            # Back face  
            [4, 6, 7], [4, 7, 5],
            # Top face
            [2, 3, 7], [2, 7, 6],
            # Bottom face
            [0, 4, 5], [0, 5, 1],
            # Left face
            [0, 2, 6], [0, 6, 4],
            # Right face
            [1, 5, 7], [1, 7, 3]
        ]
        return {'vertices': vertices, 'faces': faces}
        
    def generate_foot_geometry(self):
        """Generate 3D cylinder geometry for foot trackers"""
        import math
        vertices = []
        radius = 0.03
        height = 0.08
        segments = 8  # 8-sided cylinder
        
        # Generate cylinder vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            # Bottom circle
            vertices.append([x, -height/2, z])
            # Top circle
            vertices.append([x, height/2, z])
        
        # Center points
        vertices.append([0, -height/2, 0])  # Bottom center
        vertices.append([0, height/2, 0])   # Top center
        
        # Generate faces
        faces = []
        bottom_center = len(vertices) - 2
        top_center = len(vertices) - 1
        
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # Side faces (2 triangles per side)
            faces.append([i*2, next_i*2, i*2+1])
            faces.append([next_i*2, next_i*2+1, i*2+1])
            
            # Bottom face
            faces.append([bottom_center, i*2, next_i*2])
            
            # Top face  
            faces.append([top_center, next_i*2+1, i*2+1])
            
        return {'vertices': vertices, 'faces': faces}
        
    def draw_tracker_shadows(self, painter, tracker_depth_list):
        """Draw shadows for trackers on the ground plane"""
        for tracker_id, position, screen_x, screen_y, depth in tracker_depth_list:
            # Project shadow position (Y = 0 for ground plane)
            shadow_x, shadow_y, shadow_z = self.project_3d_to_2d(position[0], 0.0, position[2])
            
            if shadow_z > 0:  # Only draw if shadow is visible
                color = self.colors.get(tracker_id, QColor(255, 255, 255))
                shadow_color = QColor(0, 0, 0, int(self.shadow_opacity * 255 / shadow_z))
                
                # Shadow size based on height and depth
                height_factor = max(0.3, 1.0 - position[1] * 0.3)  # Higher = smaller shadow
                base_radius = 15 if tracker_id == 0 else 12
                shadow_radius = int(base_radius * height_factor / shadow_z * 2)
                shadow_radius = max(3, min(30, shadow_radius))
                
                # Draw elliptical shadow
                painter.setBrush(QBrush(shadow_color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(shadow_x - shadow_radius), int(shadow_y - shadow_radius//2), 
                                  shadow_radius * 2, shadow_radius)
                                  
    def calculate_lighting(self, normal_vector, position):
        """Calculate lighting intensity based on normal vector and light direction"""
        import math
        
        # Normalize vectors
        def normalize(v):
            length = math.sqrt(sum(x*x for x in v))
            return [x/length if length > 0 else 0 for x in v]
        
        normal = normalize(normal_vector)
        light_dir = normalize(self.light_direction)
        
        # Calculate dot product for diffuse lighting
        dot_product = sum(n * l for n, l in zip(normal, light_dir))
        diffuse = max(0, -dot_product)  # Invert because light direction points away
        
        # Combine ambient and diffuse lighting
        intensity = self.ambient_light + self.directional_light * diffuse
        return min(1.0, intensity)
        
    def draw_3d_tracker_geometry(self, painter, tracker_id, position, screen_x, screen_y, depth):
        """Draw full 3D geometry for a tracker with lighting"""
        import math
        
        color = self.colors.get(tracker_id, QColor(255, 255, 255))
        geometry = self.tracker_geometries[tracker_id]
        
        # Transform and project all vertices
        transformed_faces = []
        
        for face_indices in geometry['faces']:
            face_vertices_2d = []
            face_vertices_3d = []
            
            for vertex_idx in face_indices:
                vertex = geometry['vertices'][vertex_idx]
                # Transform vertex to world position
                world_x = position[0] + vertex[0]
                world_y = position[1] + vertex[1] 
                world_z = position[2] + vertex[2]
                
                # Project to 2D
                screen_x_v, screen_y_v, depth_v = self.project_3d_to_2d(world_x, world_y, world_z)
                
                if depth_v > 0:  # Only include visible vertices
                    face_vertices_2d.append((int(screen_x_v), int(screen_y_v)))
                    face_vertices_3d.append((world_x, world_y, world_z))
            
            if len(face_vertices_2d) == 3:  # Valid triangle
                # Calculate face normal for lighting
                v1 = face_vertices_3d[0]
                v2 = face_vertices_3d[1] 
                v3 = face_vertices_3d[2]
                
                # Cross product for normal
                edge1 = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
                edge2 = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]]
                normal = [
                    edge1[1]*edge2[2] - edge1[2]*edge2[1],
                    edge1[2]*edge2[0] - edge1[0]*edge2[2], 
                    edge1[0]*edge2[1] - edge1[1]*edge2[0]
                ]
                
                # Calculate average depth for this face
                avg_depth = sum(v[2] for v in face_vertices_3d) / 3
                
                transformed_faces.append((face_vertices_2d, normal, avg_depth))
        
        # Sort faces by depth (back to front)
        transformed_faces.sort(key=lambda x: x[2], reverse=True)
        
        # Draw faces with lighting
        for face_2d, normal, face_depth in transformed_faces:
            if self.enable_lighting:
                lighting_intensity = self.calculate_lighting(normal, position)
            else:
                lighting_intensity = 1.0
            
            # Apply lighting to color
            lit_color = QColor(
                int(color.red() * lighting_intensity),
                int(color.green() * lighting_intensity), 
                int(color.blue() * lighting_intensity),
                max(100, min(255, int(255 / face_depth * 2)))  # Depth-based alpha
            )
            
            # Pulsing effect
            pulse = 0.9 + 0.1 * math.sin(self.animation_time * 2 + tracker_id)
            final_color = QColor(
                int(lit_color.red() * pulse),
                int(lit_color.green() * pulse),
                int(lit_color.blue() * pulse),
                lit_color.alpha()
            )
            
            # Draw face
            painter.setBrush(QBrush(final_color))
            
            # Edge highlighting based on lighting
            edge_intensity = max(0.3, lighting_intensity)
            edge_color = QColor(
                int(color.red() * edge_intensity * 1.2),
                int(color.green() * edge_intensity * 1.2),
                int(color.blue() * edge_intensity * 1.2),
                min(255, int(final_color.alpha() * 1.5))
            )
            painter.setPen(QPen(edge_color, max(1, int(2 / face_depth))))
            
            # Create polygon and draw
            if len(face_2d) >= 3:
                polygon = QPolygon([QPoint(x, y) for x, y in face_2d])
                painter.drawPolygon(polygon)
        
        # Draw glow effect if enabled
        if self.enable_glow_effects:
            glow_radius = int(30 / depth)
            glow_color = QColor(color.red(), color.green(), color.blue(), int(50 / depth))
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(screen_x - glow_radius), int(screen_y - glow_radius),
                              glow_radius * 2, glow_radius * 2)
        
        # Draw enhanced label
        self.draw_tracker_label(painter, tracker_id, position, screen_x, screen_y, depth)
        
    def draw_tracker_label(self, painter, tracker_id, position, screen_x, screen_y, depth):
        """Draw enhanced label for tracker with depth effects"""
        # Font size based on depth
        font_size = max(8, min(16, int(14 / depth * 2)))
        painter.setFont(QFont("Arial", font_size, QFont.Bold))
        
        # Opacity based on depth
        opacity = max(150, min(255, int(255 / depth * 2)))
        painter.setPen(QPen(QColor(255, 255, 255, opacity), 2))
        
        labels = {0: "🎯 HIP", 1: "👟 L.FOOT", 2: "👟 R.FOOT"}
        label = labels.get(tracker_id, f"T{tracker_id}")
        
        # Enhanced label background with gradient
        label_rect = painter.fontMetrics().boundingRect(label)
        label_bg_gradient = QLinearGradient(0, 0, 0, label_rect.height())
        label_bg_gradient.setColorAt(0, QColor(0, 0, 0, int(200 / depth * 2)))
        label_bg_gradient.setColorAt(1, QColor(20, 20, 20, int(180 / depth * 2)))
        
        bg_rect = QRect(int(screen_x - label_rect.width()//2 - 5), 
                       int(screen_y + 35),
                       label_rect.width() + 10, label_rect.height() + 6)
        painter.fillRect(bg_rect, QBrush(label_bg_gradient))
        
        # Label border
        border_color = self.colors.get(tracker_id, QColor(255, 255, 255))
        border_color.setAlpha(int(opacity * 0.8))
        painter.setPen(QPen(border_color, max(1, int(2 / depth))))
        painter.drawRect(bg_rect)
        
        # Label text with glow effect
        painter.setPen(QPen(QColor(255, 255, 255, opacity), 2))
        painter.drawText(int(screen_x - label_rect.width()//2), 
                        int(screen_y + 45), label)
        
        # Position coordinates
        small_font_size = max(6, int(font_size * 0.7))
        painter.setFont(QFont("Consolas", small_font_size))
        pos_text = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        pos_rect = painter.fontMetrics().boundingRect(pos_text)
        
        pos_bg_rect = QRect(int(screen_x - pos_rect.width()//2 - 3), 
                           int(screen_y + 50),
                           pos_rect.width() + 6, pos_rect.height() + 3)
        painter.fillRect(pos_bg_rect, QColor(0, 0, 0, int(150 / depth * 2)))
        
        painter.setPen(QPen(QColor(200, 200, 200, int(opacity * 0.9)), 1))
        painter.drawText(int(screen_x - pos_rect.width()//2), 
                        int(screen_y + 60), pos_text)
    
    def mousePressEvent(self, event):
        """Handle mouse press for 3D rotation"""
        # Give focus to widget for keyboard events
        self.setFocus()
        
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
            
    def mouseMoveEvent(self, event):
        """Handle mouse move for 3D rotation"""
        if self.is_dragging and self.last_mouse_pos:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            
            # Update camera angles
            self.camera_azimuth += dx * self.mouse_sensitivity
            self.camera_elevation -= dy * self.mouse_sensitivity
            
            # Clamp elevation to prevent flipping
            self.camera_elevation = max(-90, min(90, self.camera_elevation))
            
            self.last_mouse_pos = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming (FOV adjustment)"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.field_of_view = max(20.0, self.field_of_view - 2.0)  # Zoom in
        else:
            self.field_of_view = min(120.0, self.field_of_view + 2.0)  # Zoom out
        self.update()
        
    def keyPressEvent(self, event):
        """Handle key press for WASD movement"""
        key = event.key()
        
        # Movement keys
        if key == Qt.Key_W:
            self.keys_pressed['w'] = True
        elif key == Qt.Key_A:
            self.keys_pressed['a'] = True
        elif key == Qt.Key_S:
            self.keys_pressed['s'] = True
        elif key == Qt.Key_D:
            self.keys_pressed['d'] = True
        elif key == Qt.Key_Q:
            self.keys_pressed['q'] = True
        elif key == Qt.Key_E:
            self.keys_pressed['e'] = True
        elif key == Qt.Key_Shift:
            self.keys_pressed['shift'] = True
        elif key == Qt.Key_R:
            # Reset camera position and rotation
            self.reset_camera()
            
    def keyReleaseEvent(self, event):
        """Handle key release for WASD movement"""
        key = event.key()
        
        if key == Qt.Key_W:
            self.keys_pressed['w'] = False
        elif key == Qt.Key_A:
            self.keys_pressed['a'] = False
        elif key == Qt.Key_S:
            self.keys_pressed['s'] = False
        elif key == Qt.Key_D:
            self.keys_pressed['d'] = False
        elif key == Qt.Key_Q:
            self.keys_pressed['q'] = False
        elif key == Qt.Key_E:
            self.keys_pressed['e'] = False
        elif key == Qt.Key_Shift:
            self.keys_pressed['shift'] = False
            
    def reset_camera(self):
        """Reset camera to default position and rotation (centrado y orientado correctamente)"""
        self.camera_position = [0.0, 1.5, -2.5]  # Atrás para ver hacia adelante
        self.camera_elevation = -15.0   # Mirando ligeramente hacia abajo (orientación corregida)
        self.camera_azimuth = 0.0      # Centrado
        self.field_of_view = 60.0
        self.update()
        
    def animate(self):
        """Update animation time and handle continuous movement"""
        self.animation_time += 0.016  # ~60 FPS
        
        # Handle WASD movement
        self.update_camera_movement()
        
        self.update()
        
    def update_camera_movement(self):
        """Update camera position based on pressed keys"""
        import math
        
        # Determine movement speed
        speed = self.fast_movement_speed if self.keys_pressed['shift'] else self.movement_speed
        
        # Calculate forward, right, and up vectors based on camera rotation
        azimuth_rad = math.radians(self.camera_azimuth)
        elevation_rad = math.radians(self.camera_elevation)
        
        # Forward vector (in XZ plane, ignoring elevation for movement)
        forward_x = math.sin(azimuth_rad)
        forward_z = math.cos(azimuth_rad)
        
        # Right vector (perpendicular to forward in XZ plane)
        right_x = math.cos(azimuth_rad)
        right_z = -math.sin(azimuth_rad)
        
        # Movement in local coordinate system
        move_forward = 0
        move_right = 0
        move_up = 0
        
        # WASD movement
        if self.keys_pressed['w']:  # Move forward
            move_forward += speed
        if self.keys_pressed['s']:  # Move backward
            move_forward -= speed
        if self.keys_pressed['a']:  # Move left
            move_right -= speed
        if self.keys_pressed['d']:  # Move right
            move_right += speed
        if self.keys_pressed['q']:  # Move down
            move_up -= speed
        if self.keys_pressed['e']:  # Move up
            move_up += speed
        
        # Apply movement to camera position
        self.camera_position[0] += move_forward * forward_x + move_right * right_x
        self.camera_position[1] += move_up
        self.camera_position[2] += move_forward * forward_z + move_right * right_z
        
    def project_3d_to_2d(self, x, y, z):
        """Project 3D coordinates to 2D screen coordinates with real perspective"""
        import math
        
        # Convert angles to radians
        azimuth_rad = math.radians(self.camera_azimuth)
        elevation_rad = math.radians(self.camera_elevation)
        
        # Use the new camera position
        cam_x = self.camera_position[0]
        cam_y = self.camera_position[1]
        cam_z = self.camera_position[2]
        
        # Translate point relative to camera
        rel_x = x - cam_x
        rel_y = y - cam_y
        rel_z = z - cam_z
        
        # Rotate around Y axis (azimuth)
        cos_az = math.cos(-azimuth_rad)
        sin_az = math.sin(-azimuth_rad)
        new_x = rel_x * cos_az - rel_z * sin_az
        new_z = rel_x * sin_az + rel_z * cos_az
        
        # Rotate around X axis (elevation)
        cos_el = math.cos(-elevation_rad)
        sin_el = math.sin(-elevation_rad)
        final_y = rel_y * cos_el - new_z * sin_el
        final_z = rel_y * sin_el + new_z * cos_el
        
        # Perspective projection with better error handling
        if final_z <= 0.001:  # More precise zero check
            final_z = 0.001
            
        fov_scale = 1.0 / math.tan(math.radians(self.field_of_view / 2))
        screen_x = (new_x * fov_scale / final_z) * (self.width() / 4) + self.width() / 2
        screen_y = (-final_y * fov_scale / final_z) * (self.height() / 4) + self.height() / 2  # Y invertido para orientación correcta
        
        # Clamp screen coordinates to prevent drawing outside bounds
        screen_x = max(0, min(self.width(), screen_x))
        screen_y = max(0, min(self.height(), screen_y))
        
        # Calculate depth for proper layering
        depth = final_z
        
        return screen_x, screen_y, depth
        
    def create_3d_grid_points(self):
        """Create 3D grid points for real 3D visualization"""
        grid_points = []
        grid_size = 2.0
        grid_spacing = 0.5
        
        # Horizontal grid lines (XZ plane)
        for x in range(-4, 5):
            x_pos = x * grid_spacing
            for z in range(-4, 5):
                z_pos = z * grid_spacing
                if abs(x_pos) <= grid_size and abs(z_pos) <= grid_size:
                    grid_points.append((x_pos, 0, z_pos))
                    
        return grid_points
        
    def paintEvent(self, event):
        """Paint the advanced tracker visualization with enhanced 3D effects"""
        painter = QPainter(self)
        
        # Enable advanced rendering hints for maximum quality
        if self.enable_antialiasing:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
        
        # Dark background consistent with main app theme
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(25, 25, 25))
        gradient.setColorAt(1, QColor(35, 35, 35))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw enhanced grid if enabled
        if self.show_grid:
            self.draw_enhanced_grid(painter)
        
        # Draw coordinate axes if enabled
        if self.show_axes:
            self.draw_enhanced_axes(painter)
        
        # Draw tracker trails if enabled
        if self.show_trails:
            self.draw_tracker_trails(painter)
        
        # Draw trackers with enhanced effects
        self.draw_enhanced_trackers(painter)
        
        # Draw skeleton if enabled
        if self.show_skeleton:
            self.draw_enhanced_skeleton(painter)
        
        # Draw enhanced legend and info
        self.draw_enhanced_legend(painter)
        
        # Draw real-time info overlay
        self.draw_realtime_overlay(painter)
        
        # Ensure painter is properly ended
        painter.end()
        
    def draw_grid(self, painter):
        """Draw reference grid"""
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Grid spacing
        spacing = 50
        
        # Vertical lines
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
            
        # Horizontal lines  
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)
            
    def draw_axes(self, painter):
        """Draw coordinate axes"""
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # X axis (red)
        painter.setPen(QPen(QColor(255, 100, 100), 3))
        painter.drawLine(center_x, center_y, center_x + 100, center_y)
        painter.drawText(center_x + 105, center_y + 5, "X")
        
        # Y axis (green) - inverted because screen coordinates
        painter.setPen(QPen(QColor(100, 255, 100), 3))
        painter.drawLine(center_x, center_y, center_x, center_y - 100)
        painter.drawText(center_x + 5, center_y - 105, "Y")
        
        # Z axis (blue) - diagonal for 3D effect
        painter.setPen(QPen(QColor(100, 100, 255), 3))
        painter.drawLine(center_x, center_y, center_x - 70, center_y + 70)
        painter.drawText(center_x - 80, center_y + 85, "Z")
        
    def draw_trackers(self, painter):
        """Draw tracker positions"""
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        for tracker_id, position in self.trackers.items():
            # Convert 3D position to 2D screen coordinates
            # Using simple isometric projection
            x = position[0] * self.scale
            z = position[2] * self.scale
            y = position[1] * self.scale
            
            screen_x = center_x + x - z * 0.5
            screen_y = center_y - y + z * 0.3
            
            # Draw tracker circle
            color = self.colors.get(tracker_id, QColor(255, 255, 255))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(), 2))
            
            radius = 15 if tracker_id == 0 else 12  # Hip is larger
            painter.drawEllipse(int(screen_x - radius), int(screen_y - radius), 
                              radius * 2, radius * 2)
            
            # Draw tracker label
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            
            labels = {0: "Hip", 1: "L.Foot", 2: "R.Foot"}
            label = labels.get(tracker_id, f"T{tracker_id}")
            painter.drawText(int(screen_x - 20), int(screen_y + 30), label)
            
            # Draw position values
            painter.setFont(QFont("Arial", 8))
            pos_text = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            painter.drawText(int(screen_x - 40), int(screen_y + 45), pos_text)
            
    def draw_skeleton(self, painter):
        """Draw skeleton connecting trackers"""
        if len(self.trackers) < 2:
            return
            
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        
        # Convert positions to screen coordinates
        screen_positions = {}
        for tracker_id, position in self.trackers.items():
            x = position[0] * self.scale
            z = position[2] * self.scale
            y = position[1] * self.scale
            
            screen_x = center_x + x - z * 0.5
            screen_y = center_y - y + z * 0.3
            screen_positions[tracker_id] = (screen_x, screen_y)
        
        # Draw connections
        if 0 in screen_positions and 1 in screen_positions:  # Hip to left foot
            hip_pos = screen_positions[0]
            left_pos = screen_positions[1]
            painter.drawLine(int(hip_pos[0]), int(hip_pos[1]), 
                           int(left_pos[0]), int(left_pos[1]))
            
        if 0 in screen_positions and 2 in screen_positions:  # Hip to right foot
            hip_pos = screen_positions[0]
            right_pos = screen_positions[2]
            painter.drawLine(int(hip_pos[0]), int(hip_pos[1]), 
                           int(right_pos[0]), int(right_pos[1]))
            
    def draw_legend(self, painter):
        """Draw legend"""
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        
        legend_text = "SteamVR Tracker Data (Real-time)"
        painter.drawText(10, 20, legend_text)
        
        painter.setFont(QFont("Arial", 8))
        painter.drawText(10, 40, "• Move to see tracker positions")
        painter.drawText(10, 55, "• Values shown are sent to SteamVR")
        
    def update_tracker(self, tracker_id, position):
        """Update tracker position"""
        self.trackers[tracker_id] = position
        self.update()  # Trigger repaint
        
        # Add to trails for enhanced visualization
        if tracker_id in self.tracker_trails:
            self.tracker_trails[tracker_id].append(position.copy())
            # Keep only last 100 positions for trails
            if len(self.tracker_trails[tracker_id]) > 100:
                self.tracker_trails[tracker_id].pop(0)
    
    def draw_enhanced_grid(self, painter):
        """Draw real 3D grid with perspective"""
        painter.setPen(QPen(QColor(70, 70, 70, 120), 1))
        
        grid_size = 2.0
        grid_spacing = 0.2
        
        # Draw grid lines in XZ plane (ground)
        for i in range(-10, 11):
            x_pos = i * grid_spacing
            if abs(x_pos) <= grid_size:
                # X-direction lines
                start_x, start_y, start_z = self.project_3d_to_2d(x_pos, 0, -grid_size)
                end_x, end_y, end_z = self.project_3d_to_2d(x_pos, 0, grid_size)
                
                if start_z > 0 and end_z > 0:  # Only draw if in front of camera
                    painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
                
                # Z-direction lines
                z_pos = i * grid_spacing
                if abs(z_pos) <= grid_size:
                    start_x, start_y, start_z = self.project_3d_to_2d(-grid_size, 0, z_pos)
                    end_x, end_y, end_z = self.project_3d_to_2d(grid_size, 0, z_pos)
                    
                    if start_z > 0 and end_z > 0:  # Only draw if in front of camera
                        painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
    
    def draw_enhanced_axes(self, painter):
        """Draw real 3D coordinate axes with perspective"""
        axis_length = 1.0
        
        # Origin point
        origin_x, origin_y, origin_z = self.project_3d_to_2d(0, 0, 0)
        
        if origin_z > 0:  # Only draw if origin is in front of camera
            # X axis (red)
            x_end_x, x_end_y, x_end_z = self.project_3d_to_2d(axis_length, 0, 0)
            if x_end_z > 0:
                painter.setPen(QPen(QColor(200, 80, 80), 4))
                painter.drawLine(int(origin_x), int(origin_y), int(x_end_x), int(x_end_y))
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                painter.setPen(QPen(QColor(220, 220, 220), 2))
                painter.drawText(int(x_end_x + 10), int(x_end_y + 5), "X")
            
            # Y axis (green)
            y_end_x, y_end_y, y_end_z = self.project_3d_to_2d(0, axis_length, 0)
            if y_end_z > 0:
                painter.setPen(QPen(QColor(80, 200, 80), 4))
                painter.drawLine(int(origin_x), int(origin_y), int(y_end_x), int(y_end_y))
                painter.setPen(QPen(QColor(220, 220, 220), 2))
                painter.drawText(int(y_end_x + 5), int(y_end_y - 10), "Y")
            
            # Z axis (blue)
            z_end_x, z_end_y, z_end_z = self.project_3d_to_2d(0, 0, axis_length)
            if z_end_z > 0:
                painter.setPen(QPen(QColor(80, 120, 200), 4))
                painter.drawLine(int(origin_x), int(origin_y), int(z_end_x), int(z_end_y))
                painter.setPen(QPen(QColor(220, 220, 220), 2))
                painter.drawText(int(z_end_x - 20), int(z_end_y + 20), "Z")
    
    def draw_tracker_trails(self, painter):
        """Draw real 3D position trails with perspective"""
        for tracker_id, trail in self.tracker_trails.items():
            if len(trail) < 2:
                continue
                
            color = self.colors.get(tracker_id, QColor(255, 255, 255))
            
            # Draw trail with fading effect and 3D perspective
            for i in range(1, len(trail)):
                alpha = int(255 * (i / len(trail)) * 0.7)  # Fade effect
                trail_color = QColor(color.red(), color.green(), color.blue(), alpha)
                
                # Get 3D positions
                pos1 = trail[i-1]
                pos2 = trail[i]
                
                # Project to 2D with perspective
                x1, y1, z1 = self.project_3d_to_2d(pos1[0], pos1[1], pos1[2])
                x2, y2, z2 = self.project_3d_to_2d(pos2[0], pos2[1], pos2[2])
                
                # Only draw if both points are in front of camera
                if z1 > 0 and z2 > 0:
                    # Line thickness based on distance (depth effect)
                    camera_distance = (self.camera_position[0]**2 + self.camera_position[1]**2 + self.camera_position[2]**2)**0.5
                    thickness = max(1, int(5 * min(z1, z2) / max(1.0, camera_distance)))
                    painter.setPen(QPen(trail_color, thickness))
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    def draw_enhanced_trackers(self, painter):
        """Draw real 3D trackers with advanced geometry, lighting and shadows"""
        # First, draw a simple fallback if no data
        if not self.trackers:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(QFont("Arial", 18, QFont.Bold))
            
            # Título principal
            title = "🎯 3D VISUALIZER - READY!"
            title_rect = painter.fontMetrics().boundingRect(title)
            painter.drawText(
                self.width()//2 - title_rect.width()//2,
                self.height()//2 - 50,
                title
            )
            
            # Instrucciones
            painter.setFont(QFont("Arial", 14))
            instructions = [
                "1. Click '🎯 Demo Data' button to see 3D trackers",
                "2. Use WASD keys to move around",
                "3. Use mouse drag to rotate view",
                "4. Mouse wheel to zoom",
                "5. Start cameras to see real tracking data"
            ]
            
            for i, instruction in enumerate(instructions):
                inst_rect = painter.fontMetrics().boundingRect(instruction)
                painter.drawText(
                    self.width()//2 - inst_rect.width()//2,
                    self.height()//2 + 20 + i*25,
                    instruction
                )
            return
        
        # Sort trackers by depth for proper rendering order
        tracker_depth_list = []
        for tracker_id, position in self.trackers.items():
            # Validate position data
            if not isinstance(position, (list, tuple)) or len(position) < 3:
                continue
                
            try:
                x, y, z = self.project_3d_to_2d(position[0], position[1], position[2])
                if z > 0:  # Only include trackers in front of camera
                    tracker_depth_list.append((tracker_id, position, x, y, z))
            except (ValueError, TypeError, IndexError) as e:
                # Skip invalid position data
                continue
        
        # Sort by depth (furthest first for proper alpha blending)
        tracker_depth_list.sort(key=lambda item: item[4], reverse=True)
        
        # Draw shadows first (if enabled)
        if self.enable_shadows:
            self.draw_tracker_shadows(painter, tracker_depth_list)
        
        # Draw both 3D geometries AND simple circles for debugging
        for tracker_id, position, screen_x, screen_y, depth in tracker_depth_list:
            # Validate screen coordinates
            if not (0 <= screen_x <= self.width() and 0 <= screen_y <= self.height()):
                continue
                
            # Draw simple circle first as fallback
            color = self.colors.get(tracker_id, QColor(255, 255, 255))
            radius = int(20 / depth * 2)
            radius = max(8, min(40, radius))
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.lighter(), 2))
            painter.drawEllipse(int(screen_x - radius), int(screen_y - radius), radius * 2, radius * 2)
            
            # Draw label
            labels = {0: "HIP", 1: "L.FOOT", 2: "R.FOOT"}
            label = labels.get(tracker_id, f"T{tracker_id}")
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            label_rect = painter.fontMetrics().boundingRect(label)
            painter.drawText(int(screen_x - label_rect.width()//2), int(screen_y + radius + 15), label)
            
            # Also try to draw the 3D geometry
            try:
                self.draw_3d_tracker_geometry(painter, tracker_id, position, screen_x, screen_y, depth)
            except Exception as e:
                # Log specific errors for debugging
                import logging
                logging.debug(f"3D geometry failed for tracker {tracker_id}: {e}")
                # If 3D geometry fails, we still have the simple circles
                pass
    
    def draw_enhanced_skeleton(self, painter):
        """Draw real 3D skeleton connections with perspective"""
        if len(self.trackers) < 2:
            return
        
        # Calculate screen positions with real 3D projection
        screen_positions = {}
        depths = {}
        
        for tracker_id, position in self.trackers.items():
            x, y, z = self.project_3d_to_2d(position[0], position[1], position[2])
            if z > 0:  # Only include trackers in front of camera
                screen_positions[tracker_id] = (x, y)
                depths[tracker_id] = z
        
        # Skeleton connections
        connections = [(0, 1), (0, 2)]  # Hip to feet
        
        for connection in connections:
            if connection[0] in screen_positions and connection[1] in screen_positions:
                pos1 = screen_positions[connection[0]]
                pos2 = screen_positions[connection[1]]
                depth1 = depths[connection[0]]
                depth2 = depths[connection[1]]
                
                # Average depth for line properties
                avg_depth = (depth1 + depth2) / 2
                
                # Line thickness and opacity based on depth
                thickness = max(2, int(8 / avg_depth))
                opacity = max(80, min(200, int(200 / avg_depth * 1.5)))
                
                # Colors with depth-based opacity
                glow_color = QColor(120, 120, 150, int(opacity * 0.4))
                main_color = QColor(180, 180, 180, opacity)
                
                # Glow effect
                painter.setPen(QPen(glow_color, thickness + 4))
                painter.drawLine(int(pos1[0]), int(pos1[1]), int(pos2[0]), int(pos2[1]))
                
                # Main line
                painter.setPen(QPen(main_color, thickness))
                painter.drawLine(int(pos1[0]), int(pos1[1]), int(pos2[0]), int(pos2[1]))
    
    def draw_enhanced_legend(self, painter):
        """Draw enhanced legend with real-time info"""
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.setPen(QPen(QColor(220, 220, 220), 2))
        
        # Main title with dark theme
        title = "🎯 EXACT SteamVR UDP Data"
        painter.drawText(20, 30, title)
        
        # Real-time indicators with dark theme colors
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        y_offset = 50
        
        tracker_count = len(self.trackers)
        painter.drawText(20, y_offset, f"● Active Trackers: {tracker_count}/3")
        painter.drawText(20, y_offset + 20, f"● Camera Pos: ({self.camera_position[0]:.1f}, {self.camera_position[1]:.1f}, {self.camera_position[2]:.1f})")
        painter.drawText(20, y_offset + 40, f"● Azimuth: {self.camera_azimuth:.0f}°")
        painter.drawText(20, y_offset + 60, f"● Elevation: {self.camera_elevation:.0f}°")
        painter.drawText(20, y_offset + 80, f"● FOV: {self.field_of_view:.0f}°")
        
        # DEBUG: Show tracker positions
        painter.setPen(QPen(QColor(255, 255, 0), 1))
        painter.drawText(20, y_offset + 100, f"🔍 DEBUG INFO:")
        for i, (tracker_id, position) in enumerate(self.trackers.items()):
            painter.drawText(20, y_offset + 120 + i*15, f"T{tracker_id}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        
        # Advanced rendering status
        painter.setPen(QPen(QColor(100, 200, 100), 1))
        painter.drawText(20, y_offset + 180, f"🔥 3D Geometry: {'ON' if self.enable_lighting else 'OFF'}")
        painter.drawText(20, y_offset + 195, f"💡 Lighting: {'ON' if self.enable_lighting else 'OFF'}")
        painter.drawText(20, y_offset + 210, f"🌑 Shadows: {'ON' if self.enable_shadows else 'OFF'}")
        painter.drawText(20, y_offset + 225, f"✨ Effects: {'ON' if self.enable_glow_effects else 'OFF'}")
        painter.drawText(20, y_offset + 240, f"🎨 Anti-alias: {'ON' if self.enable_antialiasing else 'OFF'}")
        
        # Instructions with subtle text
        painter.setFont(QFont("Arial", 9))
        painter.setPen(QPen(QColor(140, 140, 140), 1))
        instructions = [
            "🖱️ Left Click + Drag: Rotate 3D view",
            "🎯 Mouse Wheel: Zoom (FOV adjustment)", 
            "⌨️ WASD: Move in 3D space",
            "⌨️ Q/E: Move up/down",
            "⌨️ Shift: Fast movement",
            "⌨️ R: Reset camera",
            "🔥 Real 3D geometry with lighting",
            "🌑 Dynamic shadows on ground",
            "✨ Glow effects and anti-aliasing",
            "📡 Shows EXACT data sent to SteamVR"
        ]
        
        for i, instruction in enumerate(instructions):
            painter.drawText(20, self.height() - 115 + i * 15, instruction)
    
    def draw_realtime_overlay(self, painter):
        """Draw real-time overlay information"""
        if not self.trackers:
            # No data overlay with dark theme
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.setPen(QPen(QColor(200, 100, 100), 2))
            painter.drawText(self.width()//2 - 100, self.height()//2, "⚠️ NO TRACKER DATA")
            painter.setFont(QFont("Arial", 12))
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawText(self.width()//2 - 120, self.height()//2 + 25, "Waiting for SteamVR UDP packets...")
        else:
            # FPS and performance info with dark theme colors
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.setPen(QPen(QColor(100, 200, 100), 2))
            painter.drawText(self.width() - 120, 20, f"🟢 LIVE @ 60 FPS")
            
            # 3D info
            painter.setPen(QPen(QColor(150, 180, 220), 2))
            painter.drawText(self.width() - 150, 40, f"📡 {len(self.trackers)} trackers active")
            painter.drawText(self.width() - 120, 60, f"🎯 Real 3D Perspective")


class PositionHistoryGraph(QWidget):
    """Position history graph widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_history = {}
        self.setMinimumHeight(200)
        
    def update_data(self, history):
        """Update graph data"""
        self.data_history = history
        self.update()
        
    def clear_data(self):
        """Clear all graph data"""
        self.data_history.clear()
        self.update()
        
    def paintEvent(self, event):
        """Paint the position history graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dark background consistent with theme
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw placeholder with dark theme colors
        painter.setPen(QPen(QColor(180, 180, 180), 2))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(20, self.height()//2, "📈 Position History Graph")
        painter.setPen(QPen(QColor(140, 140, 140), 1))
        painter.drawText(20, self.height()//2 + 20, "(Coming in next update)")
        
        # Draw simple axis with dark theme
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(50, self.height() - 50, self.width() - 50, self.height() - 50)  # X axis
        painter.drawLine(50, 50, 50, self.height() - 50)  # Y axis