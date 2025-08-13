#include "trdx_driver.h"
#include <openvr_driver.h>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace vr;

namespace trdx_driver
{
    // Interface versions array
    static const char* k_InterfaceVersions[] = {
        IServerTrackedDeviceProvider_Version,
        nullptr
    };

    // Global driver instance
    TRDXDriverProvider g_driverProvider;

    //-----------------------------------------------------------------------------
    // UDP Communication Implementation
    //-----------------------------------------------------------------------------
    UDPCommunication::UDPCommunication(int port)
        : m_socket(INVALID_SOCKET)
        , m_port(port)
        , m_isConnected(false)
    {
        ZeroMemory(&m_serverAddr, sizeof(m_serverAddr));
    }

    UDPCommunication::~UDPCommunication()
    {
        Shutdown();
    }

    bool UDPCommunication::Initialize()
    {
        // Initialize Winsock
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0) {
            return false;
        }

        // Create socket
        m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (m_socket == INVALID_SOCKET) {
            WSACleanup();
            return false;
        }

        // Set socket to non-blocking mode
        u_long mode = 1;
        ioctlsocket(m_socket, FIONBIO, &mode);

        // Set socket options for reuse
        int reuseAddr = 1;
        setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, (char*)&reuseAddr, sizeof(reuseAddr));

        // Bind socket
        m_serverAddr.sin_family = AF_INET;
        m_serverAddr.sin_addr.s_addr = INADDR_ANY;
        m_serverAddr.sin_port = htons(m_port);

        if (bind(m_socket, (SOCKADDR*)&m_serverAddr, sizeof(m_serverAddr)) == SOCKET_ERROR) {
            CleanupSocket();
            return false;
        }

        m_isConnected = true;
        return true;
    }

    void UDPCommunication::Shutdown()
    {
        m_isConnected = false;
        CleanupSocket();
        WSACleanup();
    }

    void UDPCommunication::CleanupSocket()
    {
        if (m_socket != INVALID_SOCKET) {
            closesocket(m_socket);
            m_socket = INVALID_SOCKET;
        }
    }

    bool UDPCommunication::ReceiveTrackerData(TrackerPose& pose)
    {
        if (!m_isConnected || m_socket == INVALID_SOCKET) {
            return false;
        }

        std::lock_guard<std::mutex> lock(m_socketMutex);

        char buffer[1024];
        sockaddr_in clientAddr;
        int clientAddrSize = sizeof(clientAddr);

        int bytesReceived = recvfrom(m_socket, buffer, sizeof(buffer) - 1, 0, 
                                   (SOCKADDR*)&clientAddr, &clientAddrSize);

        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            
            // Check if this is a PING message
            if (strcmp(buffer, "PING") == 0) {
                // Send PONG response
                const char* response = "PONG";
                sendto(m_socket, response, strlen(response), 0, 
                      (SOCKADDR*)&clientAddr, clientAddrSize);
                return false; // Not tracker data
            }
            
            // Parse the received data
            // Expected format: "trackerId,x,y,z,qw,qx,qy,qz,confidence"
            int trackerId;
            float x, y, z, qw, qx, qy, qz, confidence;
            
            int parsed = sscanf_s(buffer, "%d,%f,%f,%f,%f,%f,%f,%f,%f",
                                &trackerId, &x, &y, &z, &qw, &qx, &qy, &qz, &confidence);
            
            if (parsed == 9) {
                // Validate data ranges and sanity checks
                if (trackerId >= 0 && trackerId < 10 && 
                    confidence >= 0.0f && confidence <= 1.0f &&
                    !std::isnan(x) && !std::isnan(y) && !std::isnan(z) &&
                    !std::isnan(qw) && !std::isnan(qx) && !std::isnan(qy) && !std::isnan(qz)) {
                    
                    pose.trackerId = trackerId;
                    pose.position[0] = x;
                    pose.position[1] = y;
                    pose.position[2] = z;
                    pose.rotation[0] = qw;
                    pose.rotation[1] = qx;
                    pose.rotation[2] = qy;
                    pose.rotation[3] = qz;
                    pose.confidence = confidence;
                    pose.isValid = true;
                    pose.timestamp = GetTickCount64();
                    
                    return true;
                }
            }
        }

        return false;
    }

    //-----------------------------------------------------------------------------
    // TRDX Tracker Device Implementation
    //-----------------------------------------------------------------------------
    TRDXTrackerDevice::TRDXTrackerDevice(int trackerId, const std::string& serialNumber)
        : m_trackerId(trackerId)
        , m_serialNumber(serialNumber)
        , m_deviceIndex(k_unTrackedDeviceIndexInvalid)
        , m_isActive(false)
        , m_role(TrackedControllerRole_Invalid)
        , m_propertyContainer(k_ulInvalidPropertyContainer)
    {
        // Initialize pose
        m_driverPose = { 0 };
        m_driverPose.poseIsValid = false;
        m_driverPose.result = TrackingResult_Uninitialized;
        m_driverPose.deviceIsConnected = true;
        m_driverPose.poseTimeOffset = 0.0;

        // Initialize last pose
        m_lastPose = { 0 };
        m_lastPose.trackerId = trackerId;
        m_lastPose.isValid = false;
    }

    TRDXTrackerDevice::~TRDXTrackerDevice()
    {
        Deactivate();
    }

    EVRInitError TRDXTrackerDevice::Activate(TrackedDeviceIndex_t unObjectId)
    {
        m_deviceIndex = unObjectId;
        m_propertyContainer = VRProperties()->TrackedDeviceToPropertyContainer(m_deviceIndex);

        // Set device properties
        VRProperties()->SetStringProperty(m_propertyContainer, Prop_ModelNumber_String, "TRDX-FBT-1.0");
        VRProperties()->SetStringProperty(m_propertyContainer, Prop_RenderModelName_String, "{htc}vr_tracker_vive_1_0");
        VRProperties()->SetStringProperty(m_propertyContainer, Prop_SerialNumber_String, m_serialNumber.c_str());
        VRProperties()->SetStringProperty(m_propertyContainer, Prop_ManufacturerName_String, "TRDX Project");
        
        VRProperties()->SetBoolProperty(m_propertyContainer, Prop_WillDriftInYaw_Bool, false);
        VRProperties()->SetBoolProperty(m_propertyContainer, Prop_DeviceIsWireless_Bool, true);
        VRProperties()->SetBoolProperty(m_propertyContainer, Prop_DeviceIsCharging_Bool, false);
        VRProperties()->SetFloatProperty(m_propertyContainer, Prop_DeviceBatteryPercentage_Float, 1.0f);

        // Set device class
        VRProperties()->SetInt32Property(m_propertyContainer, Prop_DeviceClass_Int32, TrackedDeviceClass_GenericTracker);
        
        // Set additional properties for better SteamVR recognition
        VRProperties()->SetBoolProperty(m_propertyContainer, Prop_DeviceProvidesBatteryStatus_Bool, false);
        VRProperties()->SetBoolProperty(m_propertyContainer, Prop_DeviceCanPowerOff_Bool, false);
        VRProperties()->SetBoolProperty(m_propertyContainer, Prop_DeviceIsCharging_Bool, false);
        VRProperties()->SetFloatProperty(m_propertyContainer, Prop_DeviceBatteryPercentage_Float, 1.0f);

        // Set tracker role based on tracker ID - CORRECTED ROLE ASSIGNMENT
        switch (m_trackerId) {
            case 0: // Hip/Waist
                VRProperties()->SetInt32Property(m_propertyContainer, Prop_ControllerRoleHint_Int32, TrackedControllerRole_Invalid);
                VRProperties()->SetStringProperty(m_propertyContainer, Prop_ControllerType_String, "vive_tracker");
                break;
            case 1: // Left Foot
                VRProperties()->SetInt32Property(m_propertyContainer, Prop_ControllerRoleHint_Int32, TrackedControllerRole_Invalid);
                VRProperties()->SetStringProperty(m_propertyContainer, Prop_ControllerType_String, "vive_tracker");
                break;
            case 2: // Right Foot
                VRProperties()->SetInt32Property(m_propertyContainer, Prop_ControllerRoleHint_Int32, TrackedControllerRole_Invalid);
                VRProperties()->SetStringProperty(m_propertyContainer, Prop_ControllerType_String, "vive_tracker");
                break;
        }

        // Set input profile
        VRProperties()->SetStringProperty(m_propertyContainer, Prop_InputProfilePath_String, "{trdx_tracker}/input/trdx_tracker_profile.json");

        m_isActive = true;
        return VRInitError_None;
    }

    void TRDXTrackerDevice::Deactivate()
    {
        m_isActive = false;
        m_deviceIndex = k_unTrackedDeviceIndexInvalid;
    }

    void TRDXTrackerDevice::EnterStandby()
    {
        // Nothing to do
    }

    void* TRDXTrackerDevice::GetComponent(const char* pchComponentNameAndVersion)
    {
        // Return nullptr for all components - this is a simple tracker without additional components
        return nullptr;
    }

    void TRDXTrackerDevice::DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize)
    {
        // Handle debug requests from SteamVR
        if (unResponseBufferSize >= 1) {
            pchResponseBuffer[0] = 0;
        }
        
        // Could implement specific debug commands here if needed
        // For now, just return empty response
    }

    DriverPose_t TRDXTrackerDevice::GetPose()
    {
        std::lock_guard<std::mutex> lock(m_poseMutex);
        return m_driverPose;
    }

    void TRDXTrackerDevice::UpdatePose(const TrackerPose& pose)
    {
        if (!m_isActive || pose.trackerId != m_trackerId) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_poseMutex);
        
        m_lastPose = pose;
        UpdateDriverPose();

        // Send pose update to SteamVR
        if (m_deviceIndex != k_unTrackedDeviceIndexInvalid) {
            VRServerDriverHost()->TrackedDevicePoseUpdated(m_deviceIndex, m_driverPose, sizeof(DriverPose_t));
        }
        
        // Mark device as connected and active
        m_driverPose.deviceIsConnected = true;
        m_driverPose.poseIsValid = true;
    }

    void TRDXTrackerDevice::UpdateDriverPose()
    {
        if (!m_lastPose.isValid) {
            m_driverPose.poseIsValid = false;
            m_driverPose.result = TrackingResult_Running_OutOfRange;
            m_driverPose.deviceIsConnected = true;
            return;
        }

        // Apply stability filtering and anatomical constraints
        ApplyStabilityFiltering();
        ApplyAnatomicalConstraints();

        // Set position with improved stability
        m_driverPose.vecPosition[0] = m_lastPose.position[0];
        m_driverPose.vecPosition[1] = m_lastPose.position[1];
        m_driverPose.vecPosition[2] = m_lastPose.position[2];

        // Set rotation (convert quaternion to matrix)
        double w = m_lastPose.rotation[0];
        double x = m_lastPose.rotation[1];
        double y = m_lastPose.rotation[2];
        double z = m_lastPose.rotation[3];

        // Normalize quaternion
        double norm = sqrt(w*w + x*x + y*y + z*z);
        if (norm > 0.0) {
            w /= norm; x /= norm; y /= norm; z /= norm;
        }

        // Convert to rotation matrix
        m_driverPose.qRotation.w = w;
        m_driverPose.qRotation.x = x;
        m_driverPose.qRotation.y = y;
        m_driverPose.qRotation.z = z;

        // Calculate velocity for better tracking
        CalculateVelocity();

        // Set tracking result based on confidence with improved thresholds
        if (m_lastPose.confidence > 0.6f) {
            m_driverPose.result = TrackingResult_Running_OK;
        } else if (m_lastPose.confidence > 0.3f) {
            m_driverPose.result = TrackingResult_Running_OutOfRange;
        } else {
            m_driverPose.result = TrackingResult_Running_OutOfRange;
        }

        m_driverPose.poseIsValid = true;
        m_driverPose.deviceIsConnected = true;
        m_driverPose.poseTimeOffset = 0.0;
    }

    void TRDXTrackerDevice::ApplyStabilityFiltering()
    {
        // MINIMAL filtering for maximum MediaPipe precision
        static const int FILTER_SIZE = 2;  // Reduced from 3 to 2 for minimal filtering
        static std::vector<TrackerPose> poseHistory;
        
        poseHistory.push_back(m_lastPose);
        if (poseHistory.size() > FILTER_SIZE) {
            poseHistory.erase(poseHistory.begin());
        }
        
        if (poseHistory.size() >= 2) {
            // Calculate filtered position with minimal filtering
            double filteredX = 0.0, filteredY = 0.0, filteredZ = 0.0;
            for (const auto& pose : poseHistory) {
                filteredX += pose.position[0];
                filteredY += pose.position[1];
                filteredZ += pose.position[2];
            }
            
            // Apply minimal filtering (95% new data, 5% filtered) for maximum precision
            double newWeight = 0.95;
            double filterWeight = 0.05;
            
            m_lastPose.position[0] = newWeight * m_lastPose.position[0] + filterWeight * (filteredX / poseHistory.size());
            m_lastPose.position[1] = newWeight * m_lastPose.position[1] + filterWeight * (filteredY / poseHistory.size());
            m_lastPose.position[2] = newWeight * m_lastPose.position[2] + filterWeight * (filteredZ / poseHistory.size());
        }
    }

    void TRDXTrackerDevice::ApplyAnatomicalConstraints()
    {
        // Apply basic anatomical constraints based on tracker role
        switch (m_trackerId) {
            case 0: // Hip/Waist - should be roughly at waist level
                // Constrain Y position to reasonable waist height (0.8-1.2 meters)
                if (m_lastPose.position[1] < 0.8) m_lastPose.position[1] = 0.8;
                if (m_lastPose.position[1] > 1.2) m_lastPose.position[1] = 1.2;
                break;
                
            case 1: // Left Foot - should be near ground level
            case 2: // Right Foot - should be near ground level
                // Constrain Y position to ground level (0.0-0.3 meters)
                if (m_lastPose.position[1] < 0.0) m_lastPose.position[1] = 0.0;
                if (m_lastPose.position[1] > 0.3) m_lastPose.position[1] = 0.3;
                break;
        }
    }

    void TRDXTrackerDevice::CalculateVelocity()
    {
        static TrackerPose lastPose = m_lastPose;
        static uint64_t lastTimestamp = GetTickCount64();
        
        uint64_t currentTimestamp = GetTickCount64();
        double deltaTime = (currentTimestamp - lastTimestamp) / 1000.0; // Convert to seconds
        
        if (deltaTime > 0.0 && deltaTime < 0.1) { // Reasonable time delta
            // Calculate linear velocity
            m_driverPose.vecVelocity[0] = (m_lastPose.position[0] - lastPose.position[0]) / deltaTime;
            m_driverPose.vecVelocity[1] = (m_lastPose.position[1] - lastPose.position[1]) / deltaTime;
            m_driverPose.vecVelocity[2] = (m_lastPose.position[2] - lastPose.position[2]) / deltaTime;
            
            // Limit maximum velocity to prevent unrealistic movement
            double maxVelocity = 5.0; // 5 m/s
            double currentVelocity = sqrt(
                m_driverPose.vecVelocity[0] * m_driverPose.vecVelocity[0] +
                m_driverPose.vecVelocity[1] * m_driverPose.vecVelocity[1] +
                m_driverPose.vecVelocity[2] * m_driverPose.vecVelocity[2]
            );
            
            if (currentVelocity > maxVelocity) {
                double scale = maxVelocity / currentVelocity;
                m_driverPose.vecVelocity[0] *= scale;
                m_driverPose.vecVelocity[1] *= scale;
                m_driverPose.vecVelocity[2] *= scale;
            }
        } else {
            // Reset velocity if time delta is too large
            m_driverPose.vecVelocity[0] = 0.0;
            m_driverPose.vecVelocity[1] = 0.0;
            m_driverPose.vecVelocity[2] = 0.0;
        }
        
        // Angular velocity (simplified)
        m_driverPose.vecAngularVelocity[0] = 0.0;
        m_driverPose.vecAngularVelocity[1] = 0.0;
        m_driverPose.vecAngularVelocity[2] = 0.0;
        
        lastPose = m_lastPose;
        lastTimestamp = currentTimestamp;
    }

    //-----------------------------------------------------------------------------
    // Driver Provider Implementation  
    //-----------------------------------------------------------------------------
    TRDXDriverProvider::TRDXDriverProvider()
        : m_isRunning(false)
    {
    }

    TRDXDriverProvider::~TRDXDriverProvider()
    {
        Cleanup();
    }

    EVRInitError TRDXDriverProvider::Init(IVRDriverContext* pDriverContext)
    {
        VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);

        // Create tracker devices first
        CreateTrackerDevices();

        // Initialize UDP communication (don't fail if UDP fails)
        m_udpComm = std::make_unique<UDPCommunication>(9998);
        if (!m_udpComm->Initialize()) {
            // Don't fail the driver if UDP fails - just log it
            // The driver can still work without UDP initially
        }

        // Start update thread
        m_isRunning = true;
        m_updateThread = std::thread(&TRDXDriverProvider::UpdateTrackersFromUDP, this);

        return VRInitError_None;
    }

    void TRDXDriverProvider::Cleanup()
    {
        // Stop update thread
        m_isRunning = false;
        if (m_updateThread.joinable()) {
            m_updateThread.join();
        }

        // Cleanup UDP communication
        if (m_udpComm) {
            m_udpComm->Shutdown();
            m_udpComm.reset();
        }

        // Cleanup tracker devices
        m_trackerDevices.clear();
    }

    const char* const* TRDXDriverProvider::GetInterfaceVersions()
    {
        return k_InterfaceVersions;
    }

    void TRDXDriverProvider::RunFrame()
    {
        // This is called every frame by SteamVR
        // We handle updates in a separate thread, so nothing to do here
    }

    bool TRDXDriverProvider::ShouldBlockStandbyMode()
    {
        return false;
    }

    void TRDXDriverProvider::EnterStandby()
    {
        // Nothing to do
    }

    void TRDXDriverProvider::LeaveStandby()
    {
        // Nothing to do
    }

    void TRDXDriverProvider::CreateTrackerDevices()
    {
        // Create tracker devices
        for (int i = 0; i < MAX_TRACKERS; ++i) {
            std::string serialNumber = "TRDX_TRACKER_" + std::to_string(i);
            auto tracker = std::make_unique<TRDXTrackerDevice>(i, serialNumber);
            
            // Add device to SteamVR
            bool added = VRServerDriverHost()->TrackedDeviceAdded(
                serialNumber.c_str(),
                TrackedDeviceClass_GenericTracker,
                tracker.get()
            );

            if (added) {
                m_trackerDevices.push_back(std::move(tracker));
            }
            // Note: Even if TrackedDeviceAdded fails, we keep the tracker
            // SteamVR might still recognize it later
        }
    }

    void TRDXDriverProvider::UpdateTrackersFromUDP()
    {
        while (m_isRunning) {
            TrackerPose pose;
            if (m_udpComm && m_udpComm->ReceiveTrackerData(pose)) {
                // Find corresponding tracker device
                for (auto& tracker : m_trackerDevices) {
                    if (tracker->GetTrackerId() == pose.trackerId) {
                        tracker->UpdatePose(pose);
                        break;
                    }
                }
            }

            // Sleep for a longer time to reduce CPU usage and improve performance
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    //-----------------------------------------------------------------------------
    // Driver Factory Function
    //-----------------------------------------------------------------------------
    extern "C" __declspec(dllexport) void* HmdDriverFactory(const char* pInterfaceName, int* pReturnCode)
    {
        if (0 == strcmp(IServerTrackedDeviceProvider_Version, pInterfaceName)) {
            if (pReturnCode) {
                *pReturnCode = VRInitError_None;
            }
            return &g_driverProvider;
        }

        if (pReturnCode) {
            *pReturnCode = VRInitError_Init_InterfaceNotFound;
        }

        return nullptr;
    }
}
