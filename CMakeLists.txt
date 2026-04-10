cmake_minimum_required(VERSION 3.20)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Detect if being built as a subproject (FetchContent, add_subdirectory)
get_directory_property(FLANN_IS_SUBPROJECT PARENT_DIRECTORY)
if(FLANN_IS_SUBPROJECT)
    message(STATUS "Building FLANN as a subproject")
endif()

project(flann VERSION 1.9.2 LANGUAGES C CXX)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

# Set C++17 standard for modern features (shared_mutex, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "Using C++17 standard")

include(${PROJECT_SOURCE_DIR}/cmake/flann_utils.cmake)
set(FLANN_VERSION ${PROJECT_VERSION})
DISSECT_VERSION()
GET_OS_INFO()
ENABLE_TESTING()


# detect if using the Clang compiler
if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_COMPILER_IS_CLANG 1)
endif ()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add an "uninstall" target
CONFIGURE_FILE ("${PROJECT_SOURCE_DIR}/cmake/uninstall_target.cmake.in"
    "${PROJECT_BINARY_DIR}/uninstall_target.cmake" IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET (uninstall "${CMAKE_COMMAND}" -P
    "${PROJECT_BINARY_DIR}/uninstall_target.cmake")

# Set the build type.  Options are:
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries

if (NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    #set(CMAKE_BUILD_TYPE Debug)
endif()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
# set output path for tests
set(TEST_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test)

option(BUILD_C_BINDINGS "Build C bindings" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_MATLAB_BINDINGS "Build Matlab bindings" ON)
option(BUILD_CUDA_LIB "Build CUDA library" OFF)
option(BUILD_OpenCL_LIB "Build OpenCL library" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOC "Build documentation" ON)
option(USE_OPENMP "Use OpenMP multi-threading" ON)
option(USE_MPI "Use MPI" OFF)

set(NVCC_COMPILER_BINDIR "" CACHE PATH  "Directory where nvcc should look for C++ compiler. This is passed to nvcc through the --compiler-bindir option.")

if (NOT BUILD_C_BINDINGS)
    set(BUILD_PYTHON_BINDINGS OFF)
    set(BUILD_MATLAB_BINDINGS OFF)
endif()


# find python
if (BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter)
    if (NOT Python_EXECUTABLE)
        set(BUILD_PYTHON_BINDINGS OFF)
    else()
        set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})  # For backward compatibility
    endif()
endif()

find_hdf5()
if (NOT HDF5_FOUND)
	message(WARNING "hdf5 library not found, some tests will not be run")
endif()

if (USE_MPI OR HDF5_IS_PARALLEL)
    find_package(MPI)
endif()
if (HDF5_IS_PARALLEL)
    if (NOT MPI_FOUND)
        message(WARNING "Found the parallel HDF5 library, but could not find the MPI library. Define the MPI_COMPILER variable to the path of your MPI compiler.")
    endif()
endif()


if (USE_MPI)
    if (NOT MPI_FOUND)
        message(WARNING "Could not find an MPI library. Define the MPI_COMPILER variable to the path of your MPI compiler.")
        set(USE_MPI OFF)
    endif()

    if (NOT HDF5_IS_PARALLEL)
        message(WARNING "For MPI support the Parallel HDF5 library is required.")
        set(USE_MPI OFF)
    endif()
endif(USE_MPI)


if (USE_MPI AND HDF5_IS_PARALLEL)
    find_package(Boost COMPONENTS mpi system serialization thread REQUIRED)
    add_definitions("-DHAVE_MPI")
endif()


if (BUILD_TESTS)
# find_package(GTest)
# if (NOT GTEST_FOUND)
# 	message(WARNING "gtest library not found, some tests will not be run")
# endif()
endif()


if (USE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(WARNING "OpenMP NOT found")
    endif()
endif()


# OpenCL support
if (BUILD_OpenCL_LIB)
    find_package(OpenCL)
    if (OpenCL_FOUND)
        message(STATUS "OpenCL found (include: ${OpenCL_INCLUDE_DIRS}, lib: ${OpenCL_LIBRARIES})")
    else(OpenCL_FOUND)
        message(STATUS "OpenCL not found, OpenCL library will not be built")
        set(BUILD_OpenCL_LIB OFF)
    endif(OpenCL_FOUND)
endif(BUILD_OpenCL_LIB)

# CUDA support (modern CMake configuration)
if (BUILD_CUDA_LIB)
    # Modern CUDA support requires CMake 3.18+
    if(CMAKE_VERSION VERSION_LESS "3.18")
        message(WARNING "CUDA support requires CMake 3.18+, found ${CMAKE_VERSION}. Disabling CUDA.")
        set(BUILD_CUDA_LIB OFF)
    else()
        # Enable CUDA as a first-class language
        enable_language(CUDA)

        # Detect Jetson platform (works in Docker containers too)
        set(FLANN_IS_JETSON FALSE)
        if(EXISTS "/usr/lib/aarch64-linux-gnu" AND EXISTS "/usr/local/cuda/bin/nvcc")
            set(FLANN_IS_JETSON TRUE)
            message(STATUS "NVIDIA Jetson platform detected")
        endif()

        # CUDA Architecture Configuration
        # Supports: native (auto-detect), environment variable, or explicit list
        #
        # Architecture reference:
        #   75  = Tesla T4, RTX 20-series (Turing)
        #   80  = A100, RTX 30-series laptop (Ampere)
        #   86  = RTX 30-series desktop, A10 (Ampere)
        #   87  = Jetson Orin AGX/NX (Ampere)
        #   89  = RTX 40-series, L4 (Ada Lovelace)
        #   90  = H100 (Hopper)
        #
        # Usage:
        #   cmake -DCMAKE_CUDA_ARCHITECTURES=native ..  # Auto-detect (fastest build)
        #   cmake -DCMAKE_CUDA_ARCHITECTURES=87 ..      # Jetson Orin only
        #   cmake -DCMAKE_CUDA_ARCHITECTURES="75;86" .. # Multiple architectures
        #   CUDA_ARCHITECTURES=75 cmake ..              # Environment variable

        if(DEFINED ENV{CUDA_ARCHITECTURES})
            # User-specified architectures via environment variable
            set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures" FORCE)
            message(STATUS "CUDA architectures from environment: ${CMAKE_CUDA_ARCHITECTURES}")
        elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
            # Auto-detect current GPU architecture
            # This produces the smallest binary and fastest build
            execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE GPU_COMPUTE_CAP
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE NVIDIA_SMI_RESULT
                ERROR_QUIET
            )
            if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
                # Convert "7.5" to "75"
                string(REPLACE "." "" GPU_ARCH "${GPU_COMPUTE_CAP}")
                # Take first GPU if multiple
                string(REGEX REPLACE "\n.*" "" GPU_ARCH "${GPU_ARCH}")
                set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH} CACHE STRING "CUDA architectures")
                message(STATUS "CUDA native architecture detected: ${CMAKE_CUDA_ARCHITECTURES}")
            else()
                if(FLANN_IS_JETSON)
                    # Jetson native detection failed (common in Docker)
                    # Default to Orin (sm_87) - most common modern Jetson
                    message(STATUS "Jetson detected, native detection failed - defaulting to SM 87 (Orin)")
                    message(STATUS "For Xavier: cmake -DCMAKE_CUDA_ARCHITECTURES=72 ..")
                    message(STATUS "For TX2:    cmake -DCMAKE_CUDA_ARCHITECTURES=62 ..")
                    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")
                else()
                    message(WARNING "Could not detect GPU architecture, falling back to defaults")
                    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 CACHE STRING "CUDA architectures")
                endif()
            endif()
        elseif(NOT CMAKE_CUDA_ARCHITECTURES)
            # Default: auto-detect current GPU (fastest build, smallest binary)
            # For multi-arch builds, use: cmake -DCMAKE_CUDA_ARCHITECTURES="75;86;87" ..
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
            message(STATUS "CUDA architectures (default): native (auto-detect)")
        else()
            message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()

        # CUDA C++ standard (match project C++ standard)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # CUDA compiler flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra")

        # Debug flags
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

        # Release flags
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

        # Note: FLANN_USE_CUDA is added per-target in src/cpp/CMakeLists.txt and
        # cmake/flann_utils.cmake to avoid polluting non-CUDA targets (like OpenCL tests)

        # Find CUDA Toolkit (for cudart, etc.)
        find_package(CUDAToolkit 11.2)
        if (CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found")
        else()
            message(WARNING "CUDA Toolkit 11.0+ not found, CUDA library will not be built")
            set(BUILD_CUDA_LIB OFF)
        endif()
    endif()
endif(BUILD_CUDA_LIB)

# PkgConfig is required for standalone builds, optional for subprojects
if(NOT FLANN_IS_SUBPROJECT)
    find_package(PkgConfig REQUIRED)
else()
    find_package(PkgConfig QUIET)
endif()
pkg_check_modules(LZ4 REQUIRED liblz4)

# Compiler warning flags are now set per-target in src/cpp/CMakeLists.txt

# install and export variables
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(version_config "${generated_dir}/flann-config-version.cmake")
set(project_config "${generated_dir}/flann-config.cmake")
set(targets_export_name "flann-targets")
set(namespace "flann::")

add_subdirectory( cmake )
add_subdirectory( src )

# Skip examples/tests/docs when built as a subproject
if (BUILD_EXAMPLES AND NOT FLANN_IS_SUBPROJECT)
  add_subdirectory( examples )
endif()
if (BUILD_TESTS AND NOT FLANN_IS_SUBPROJECT)
  add_subdirectory( test )
endif()
if (BUILD_DOC AND NOT FLANN_IS_SUBPROJECT)
  add_subdirectory( doc )
endif()

# CMake configuration file creation
# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure 'flann-config-version.cmake'
# Note: FLANN_VERSION is used as a VERSION
write_basic_package_version_file(
    "${version_config}"
    VERSION ${FLANN_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure 'flann-config.cmake'
# Use variables:
#   * targets_export_name
#   * PROJECT_NAME
configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

# Config
#   * <prefix>/lib/cmake/flann/flann-config.cmake
#   * <prefix>/lib/cmake/flann/flann-config-version.cmake
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)
# Config
#   * <prefix>/lib/cmake/flann/flann-targets.cmake
install(
    EXPORT "${targets_export_name}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)


# CPACK options

# RPM
find_program(RPM_PROGRAM rpm)
if(EXISTS ${RPM_PROGRAM})
  list(APPEND CPACK_GENERATOR "RPM")
endif(EXISTS ${RPM_PROGRAM})
# DEB
find_program(DPKG_PROGRAM dpkg)
if(EXISTS ${DPKG_PROGRAM})
  list(APPEND CPACK_GENERATOR "DEB")
endif(EXISTS ${DPKG_PROGRAM})
# NSIS
find_program(NSIS_PROGRAM makensis MakeNSIS)
if(EXISTS ${NSIS_PROGRAM})
  list(APPEND CPACK_GENERATOR "NSIS")
endif(EXISTS ${NSIS_PROGRAM})
# dpkg
find_program(PACKAGE_MAKER_PROGRAM PackageMaker
	    HINTS /Developer/Applications/Utilities)
if(EXISTS ${PACKAGE_MAKER_PROGRAM})
  list(APPEND CPACK_GENERATOR "PackageMaker")
endif(EXISTS ${PACKAGE_MAKER_PROGRAM})

set(CPACK_GENERATOR "${CPACK_GENERATOR}")
set(CPACK_MONOLITHIC_INSTALL 1)
set(CPACK_SET_DESTDIR ON)
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_CONTACT "Marius Muja")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
SET(CPACK_PACKAGE_VERSION ${FLANN_VERSION})
SET(CPACK_PACKAGE_VERSION_MAJOR ${FLANN_VERSION_MAJOR})
SET(CPACK_PACKAGE_VERSION_MINOR ${FLANN_VERSION_MINOR})
SET(CPACK_PACKAGE_VERSION_PATCH ${FLANN_VERSION_PATCH})
include(CPack)


message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building C bindings: ${BUILD_C_BINDINGS}")
message(STATUS "Building examples: ${BUILD_EXAMPLES}")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building documentation: ${BUILD_DOC}")
message(STATUS "Building python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Building matlab bindings: ${BUILD_MATLAB_BINDINGS}")
message(STATUS "Building OpenCL library: ${BUILD_OpenCL_LIB}")
message(STATUS "Building CUDA library: ${BUILD_CUDA_LIB}")
message(STATUS "Using OpenMP support: ${USE_OPENMP}")
message(STATUS "Using MPI support: ${USE_MPI}")
