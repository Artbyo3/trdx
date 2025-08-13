@echo off
echo Compiling TRDX driver for SteamVR...

REM Find Visual Studio
set "VS_PATH="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo ERROR: Visual Studio not found
    echo Installing Visual Studio Build Tools...
    pause
    exit /b 1
)

echo Using Visual Studio at: %VS_PATH%

REM Create build directory
if exist "build" rmdir /s /q "build"
mkdir "build"

REM Compile with CMake
echo Configuring CMake...
call "%VS_PATH%"
cmake -S src -B build -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed
    pause
    exit /b 1
)

echo Compiling driver...
cmake --build build --config Release

if %ERRORLEVEL% neq 0 (
    echo ERROR: Compilation failed
    pause
    exit /b 1
)

REM Create installation directory
if exist "install\trdx_tracker" rmdir /s /q "install\trdx_tracker"
mkdir "install\trdx_tracker\bin\win64"

REM Find the compiled driver
if exist "build\Release\driver_trdx_tracker.dll" (
    copy "build\Release\driver_trdx_tracker.dll" "install\trdx_tracker\bin\win64\" >nul
    echo ✓ Driver copied from build\Release\
) else if exist "build\x64\Release\driver_trdx_tracker.dll" (
    copy "build\x64\Release\driver_trdx_tracker.dll" "install\trdx_tracker\bin\win64\" >nul
    echo ✓ Driver copied from build\x64\Release\
) else (
    echo ✗ ERROR: Compiled driver not found
    echo Searching for compiled files...
    dir /s build\*.dll
    pause
    exit /b 1
)

REM Copy OpenVR files
if exist "third_party\openvr\bin\win64\openvr_api.dll" (
    copy "third_party\openvr\bin\win64\openvr_api.dll" "install\trdx_tracker\bin\win64\" >nul
    echo ✓ OpenVR API copied
)

REM Copy manifest and resources
if exist "trdx_tracker\driver.vrdrivermanifest" (
    copy "trdx_tracker\driver.vrdrivermanifest" "install\trdx_tracker\" >nul
    echo ✓ Manifest copied
)

if exist "trdx_tracker\resources" (
    xcopy "trdx_tracker\resources" "install\trdx_tracker\resources\" /e /i /y >nul
    echo ✓ Resources copied
)

echo.
echo ========================================
echo Driver compiled successfully!
echo Location: install\trdx_tracker\
echo ========================================
echo.
echo Driver files:
dir install\trdx_tracker\bin\win64\
echo.
pause 