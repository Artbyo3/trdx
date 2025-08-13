# # # # 99.9 of code Is AI # # # 


# TRDX - Technical Documentation

## Project Overview
TRDX is a full-body tracking system for VR that uses computer vision and MediaPipe to provide SteamVR tracking without physical hardware. The system processes camera input to generate pose data that can be used in SteamVR and VRChat.

## Architecture

### Core Components

#### 1. Main Application (`trdx_app.py`)
- **Class**: `TRDXApplication`
- **Purpose**: Main application controller that orchestrates all components
- **Responsibilities**: 
  - Camera management
  - Pose processing
  - UI coordination
  - Configuration management

#### 2. Pose Processing (`pose_processor.py`)
- **Purpose**: Core MediaPipe pose detection and processing
- **Features**:
  - Real-time pose estimation
  - Landmark filtering and smoothing
  - 3D coordinate transformation
  - Confidence scoring

#### 3. Camera Management (`camera_manager.py`)
- **Purpose**: Handles camera input and configuration
- **Supported Sources**:
  - USB webcams
  - IP cameras
  - Multiple camera support (up to 4)
- **Features**:
  - Auto-detection
  - Resolution configuration
  - Frame rate optimization

#### 4. SteamVR Integration (`steamvr_driver.py`)
- **Purpose**: Bridges pose data to SteamVR
- **Protocol**: UDP communication on localhost:9998
- **Data Format**: `trackerId,x,y,z,qw,qx,qy,qz,confidence`
- **Features**:
  - Automatic tracker activation
  - Real-time pose updates
  - Multiple tracker support

### Data Flow

```
Camera Input → MediaPipe Processing → Pose Data → SteamVR Driver → SteamVR
     ↓              ↓                    ↓           ↓
  Frame Capture → Landmark Detection → 3D Transform → UDP Broadcast
```

## Configuration

### Key Configuration Files

#### `trdx_config.json`
```json
{
  "auto_connect_steamvr": true,
  "camera": {
    "resolution": [640, 480],
    "fps": 30,
    "camera_id": 0
  },
  "tracking": {
    "landmark_scale_x": 2.0,
    "landmark_scale_y": 2.0,
    "landmark_scale_z": 2.0,
    "smoothing_factor": 0.8
  },
  "steamvr": {
    "udp_port": 9998,
    "tracker_count": 3
  }
}
```

#### `tracking_config_optimized.json`
- Advanced tracking parameters
- Performance optimization settings
- Calibration data

### Environment Variables

- `TRDX_DEBUG=1`: Enable verbose logging
- `TRDX_CAMERA_ID`: Override default camera ID
- `TRDX_UDP_PORT`: Override UDP port

## SteamVR Driver

### Installation
1. Build the driver using `build_driver.bat`
2. Install using `install_driver.bat`
3. Restart SteamVR completely
4. Configure trackers in SteamVR settings

### Driver Features
- **Automatic Detection**: Self-registers with SteamVR
- **Multiple Trackers**: Supports up to 3 virtual trackers
- **Real-time Updates**: 60Hz pose updates
- **Fallback Handling**: Graceful degradation on tracking loss

### Tracker Configuration
- **Tracker 0**: Hip/Waist
- **Tracker 1**: Left Foot
- **Tracker 2**: Right Foot

## Performance Optimization

### Camera Settings
- **Resolution**: 640x480 recommended for performance
- **Frame Rate**: 30 FPS minimum, 60 FPS optimal
- **Exposure**: Auto-exposure with manual override option

### Processing Pipeline
- **Landmark Filtering**: Kalman filter for smooth tracking
- **Coordinate Scaling**: Configurable X, Y, Z scaling factors
- **Confidence Thresholding**: Minimum confidence for pose updates

### Memory Management
- **Frame Buffering**: Minimal frame buffering
- **Garbage Collection**: Automatic cleanup of processed frames
- **Resource Pooling**: Reusable pose data structures

## Troubleshooting

### Common Issues

#### Trackers Appear "Inactive"
1. Verify camera is working and MediaPipe is detecting poses
2. Check UDP connection to port 9998
3. Ensure SteamVR is completely restarted
4. Verify driver installation in SteamVR drivers folder

#### Poor Tracking Quality
1. Improve lighting conditions
2. Ensure camera has clear view of full body
3. Adjust landmark scaling factors in configuration
4. Check camera resolution and frame rate settings

#### High CPU Usage
1. Reduce camera resolution
2. Lower frame rate
3. Disable debug logging
4. Check for multiple camera instances

### Debug Mode
Enable debug mode to see detailed logs:
```bash
set TRDX_DEBUG=1
python trdx/main.py
```

## Development

### Project Structure
```
trdx/
├── main.py                 # Entry point
├── bin/                    # Core modules
│   ├── trdx_app.py        # Main application
│   ├── core/              # Core processing modules
│   └── ui/                # User interface
├── config/                 # Configuration files
└── assets/                 # UI assets and resources
```

### Building from Source
1. Install Python dependencies: `pip install -r requirements.txt`
2. Run main application: `python trdx/main.py`
3. For SteamVR driver: Use provided batch files in `steamvr_driver/`

### Dependencies
- **Python 3.8+**
- **MediaPipe**
- **OpenCV**
- **PyQt5/PySide2** (for UI)
- **NumPy**
- **OpenVR SDK** (for SteamVR integration)

## API Reference

### Core Classes

#### TRDXApplication
```python
class TRDXApplication:
    def __init__(self):
        """Initialize the main application"""
    
    def run(self) -> int:
        """Run the application and return exit code"""
    
    def shutdown(self):
        """Gracefully shutdown the application"""
```

#### PoseProcessor
```python
class PoseProcessor:
    def process_frame(self, frame) -> PoseData:
        """Process camera frame and return pose data"""
    
    def get_landmarks(self) -> List[Landmark]:
        """Get current pose landmarks"""
    
    def set_smoothing_factor(self, factor: float):
        """Set pose smoothing factor (0.0 to 1.0)"""
```

#### CameraManager
```python
class CameraManager:
    def get_cameras(self) -> List[Camera]:
        """Get list of available cameras"""
    
    def start_camera(self, camera_id: int) -> bool:
        """Start camera with specified ID"""
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
```



