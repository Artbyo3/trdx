cmake_minimum_required(VERSION 3.16)
project(trdx_driver)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenVR SDK
set(OPENVR_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/openvr")
set(OPENVR_INCLUDE_DIR "${OPENVR_SDK_DIR}/headers")
set(OPENVR_LIB_DIR "${OPENVR_SDK_DIR}/lib/win64")
set(OPENVR_BIN_DIR "${OPENVR_SDK_DIR}/bin/win64")

# Include directories
include_directories(${OPENVR_INCLUDE_DIR})

# Source files
set(SOURCES
    trdx_driver.cpp
)

# Create shared library
add_library(driver_trdx_tracker SHARED ${SOURCES})

# Link libraries
target_link_libraries(driver_trdx_tracker
    "${OPENVR_LIB_DIR}/openvr_api.lib"
    ws2_32.lib
    winmm.lib
)

# Set output directory
set_target_properties(driver_trdx_tracker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../install/trdx_tracker/bin/win64"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../install/trdx_tracker/bin/win64"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../install/trdx_tracker/bin/win64"
)

# Copy OpenVR DLL
add_custom_command(TARGET driver_trdx_tracker POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OPENVR_BIN_DIR}/openvr_api.dll"
    "${CMAKE_CURRENT_SOURCE_DIR}/../install/trdx_tracker/bin/win64/"
    COMMENT "Copying OpenVR API DLL"
) 