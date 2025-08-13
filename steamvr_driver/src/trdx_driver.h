#pragma once

#include <openvr_driver.h>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <winsock2.h>
#include <ws2tcpip.h>

// Link with ws2_32.lib
#pragma comment(lib, "ws2_32.lib")

namespace trdx_driver
{
    // Forward declarations
    class TRDXTrackerDevice;
    class UDPCommunication;

    // Tracker pose data structure
    struct TrackerPose
    {
        int trackerId;
        float position[3];      // x, y, z
        float rotation[4];      // quaternion: w, x, y, z
        float confidence;
        bool isValid;
        uint64_t timestamp;
    };

    // UDP Communication Handler
    class UDPCommunication
    {
    public:
        UDPCommunication(int port = 9998);
        ~UDPCommunication();

        bool Initialize();
        void Shutdown();
        bool ReceiveTrackerData(TrackerPose& pose);
        bool IsConnected() const { return m_isConnected; }

    private:
        void CleanupSocket();
        
        SOCKET m_socket;
        sockaddr_in m_serverAddr;
        int m_port;
        std::atomic<bool> m_isConnected;
        std::mutex m_socketMutex;
    };

    // TRDX Tracker Device Class
    class TRDXTrackerDevice : public vr::ITrackedDeviceServerDriver
    {
    public:
        TRDXTrackerDevice(int trackerId, const std::string& serialNumber);
        virtual ~TRDXTrackerDevice();

        // ITrackedDeviceServerDriver interface
        virtual vr::EVRInitError Activate(vr::TrackedDeviceIndex_t unObjectId) override;
        virtual void Deactivate() override;
        virtual void EnterStandby() override;
        virtual void* GetComponent(const char* pchComponentNameAndVersion) override;
        virtual void DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize) override;
        virtual vr::DriverPose_t GetPose() override;

        // TRDX specific methods
        void UpdatePose(const TrackerPose& pose);
        void SetTrackerRole(vr::ETrackedControllerRole role);
        int GetTrackerId() const { return m_trackerId; }
        bool IsActive() const { return m_isActive; }

    private:
        void UpdateDriverPose();
        void ApplyStabilityFiltering();
        void ApplyAnatomicalConstraints();
        void CalculateVelocity();
        
        int m_trackerId;
        std::string m_serialNumber;
        vr::TrackedDeviceIndex_t m_deviceIndex;
        vr::DriverPose_t m_driverPose;
        TrackerPose m_lastPose;
        vr::ETrackedControllerRole m_role;
        
        std::atomic<bool> m_isActive;
        std::mutex m_poseMutex;
        
        // Property container
        vr::PropertyContainerHandle_t m_propertyContainer;
    };

    // Main Driver Provider Class
    class TRDXDriverProvider : public vr::IServerTrackedDeviceProvider
    {
    public:
        TRDXDriverProvider();
        virtual ~TRDXDriverProvider();

        // IServerTrackedDeviceProvider interface
        virtual vr::EVRInitError Init(vr::IVRDriverContext* pDriverContext) override;
        virtual void Cleanup() override;
        virtual const char* const* GetInterfaceVersions() override;
        virtual void RunFrame() override;
        virtual bool ShouldBlockStandbyMode() override;
        virtual void EnterStandby() override;
        virtual void LeaveStandby() override;

    private:
        void CreateTrackerDevices();
        void UpdateTrackersFromUDP();
        
        std::vector<std::unique_ptr<TRDXTrackerDevice>> m_trackerDevices;
        std::unique_ptr<UDPCommunication> m_udpComm;
        std::thread m_updateThread;
        std::atomic<bool> m_isRunning;
        
        static const int MAX_TRACKERS = 3; // Hip, Left Foot, Right Foot
    };

    // Driver entry point
    extern "C" __declspec(dllexport) void* HmdDriverFactory(const char* pInterfaceName, int* pReturnCode);
}
