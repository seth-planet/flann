# Per-target compiler warning flags (moved from root CMakeLists.txt)
set(FLANN_COMPILE_OPTIONS_GCC -Wall -Wno-unknown-pragmas -Wno-unused-function)
set(FLANN_COMPILE_OPTIONS_MSVC /W1 -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS /bigobj)

add_definitions(-D_FLANN_VERSION=${FLANN_VERSION})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/flann/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/flann/config.h)

file(GLOB_RECURSE C_SOURCES flann.cpp)
file(GLOB_RECURSE CPP_SOURCES flann_cpp.cpp)
file(GLOB_RECURSE CU_SOURCES *.cu)

add_library(flann_cpp_s STATIC ${CPP_SOURCES})
target_link_libraries(flann_cpp_s PUBLIC ${LZ4_LINK_LIBRARIES})
target_include_directories(flann_cpp_s
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LZ4_INCLUDE_DIRS}
)
target_compile_options(flann_cpp_s PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${FLANN_COMPILE_OPTIONS_GCC}>
    $<$<CXX_COMPILER_ID:MSVC>:${FLANN_COMPILE_OPTIONS_MSVC}>
)
set_target_properties(flann_cpp_s PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    set_target_properties(flann_cpp_s PROPERTIES COMPILE_FLAGS -fPIC)
endif()
if (CMAKE_BUILD_STATIC_LIBS)
    list(APPEND flann_install_targets flann_cpp_s)
else()
    set_target_properties(flann_cpp_s PROPERTIES EXCLUDE_FROM_ALL true)
endif()

add_library(flann_cpp SHARED ${CPP_SOURCES})
target_link_libraries(flann_cpp PUBLIC ${LZ4_LINK_LIBRARIES})
target_include_directories(flann_cpp
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LZ4_INCLUDE_DIRS}
)
target_compile_options(flann_cpp PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${FLANN_COMPILE_OPTIONS_GCC}>
    $<$<CXX_COMPILER_ID:MSVC>:${FLANN_COMPILE_OPTIONS_MSVC}>
)
set_target_properties(flann_cpp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
# export lz4 headers, so that MSVC to creates flann_cpp.lib
set_target_properties(flann_cpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS YES)

# Alias targets for modern CMake usage (FetchContent, add_subdirectory)
add_library(flann::flann_cpp ALIAS flann_cpp)
add_library(flann::flann_cpp_s ALIAS flann_cpp_s)

set(flann_install_targets flann_cpp)

if (BUILD_CUDA_LIB)
    # Modern CMake CUDA: Use add_library with .cu sources directly
    add_library(flann_cuda_s STATIC ${CU_SOURCES})
    add_library(flann_cuda SHARED ${CU_SOURCES})

    # Set CUDA properties
    # Note: CUDA_SEPARABLE_COMPILATION not needed - all device code is self-contained
    # in headers included by each .cu file. Removing it improves build time and optimization.
    set_target_properties(flann_cuda_s flann_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    # Link CUDA runtime
    target_link_libraries(flann_cuda_s PUBLIC CUDA::cudart)
    target_link_libraries(flann_cuda PUBLIC CUDA::cudart)

    # Target-specific include directories for CUDA libraries
    target_include_directories(flann_cuda_s
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${LZ4_INCLUDE_DIRS}
    )
    target_include_directories(flann_cuda
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${LZ4_INCLUDE_DIRS}
    )

    # Set compile definitions
    # CUDA library targets need FLANN_USE_CUDA for enum definitions in defines.h
    set_property(TARGET flann_cuda_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC FLANN_USE_CUDA)
    set_property(TARGET flann_cuda PROPERTY COMPILE_DEFINITIONS FLANN_USE_CUDA)
    set_property(TARGET flann_cpp PROPERTY COMPILE_DEFINITIONS FLANN_USE_CUDA)
    set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC FLANN_USE_CUDA)

    # Add CUDA include directories for C++ library targets
    target_include_directories(flann_cpp PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(flann_cpp_s PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Link CUDA library for CUDA index implementations
    target_link_libraries(flann_cpp PRIVATE flann_cuda)
    target_link_libraries(flann_cpp_s PRIVATE flann_cuda_s)

    # Alias targets for modern CMake usage
    add_library(flann::flann_cuda ALIAS flann_cuda)
    add_library(flann::flann_cuda_s ALIAS flann_cuda_s)

    # Install targets
    if (CMAKE_BUILD_STATIC_LIBS)
        list(APPEND flann_install_targets flann_cuda_s)
    else()
        set_target_properties(flann_cuda_s PROPERTIES EXCLUDE_FROM_ALL true)
    endif()
    list(APPEND flann_install_targets flann_cuda)
else()
    set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC)
endif()

set_target_properties(flann_cpp PROPERTIES
   VERSION ${FLANN_VERSION}
   SOVERSION ${FLANN_SOVERSION}
   DEFINE_SYMBOL FLANN_EXPORTS
)

if (BUILD_CUDA_LIB)
    set_target_properties(flann_cuda PROPERTIES
       VERSION ${FLANN_VERSION}
       SOVERSION ${FLANN_SOVERSION}
       DEFINE_SYMBOL FLANN_EXPORTS
    )
endif()


if (USE_MPI AND HDF5_IS_PARALLEL)
    add_executable(flann_mpi_server flann/mpi/flann_mpi_server.cpp)
    target_link_libraries(flann_mpi_server flann_cpp ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})

    add_executable(flann_mpi_client flann/mpi/flann_mpi_client.cpp)
    target_link_libraries(flann_mpi_client flann_cpp ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})

    install (TARGETS flann_mpi_client flann_mpi_server
        DESTINATION bin)
endif()


if (BUILD_C_BINDINGS)
    add_library(flann_s STATIC ${C_SOURCES})
    target_link_libraries(flann_s PUBLIC ${LZ4_LINK_LIBRARIES})
    target_include_directories(flann_s
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${LZ4_INCLUDE_DIRS}
    )
    target_compile_options(flann_s PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${FLANN_COMPILE_OPTIONS_GCC}>
        $<$<CXX_COMPILER_ID:MSVC>:${FLANN_COMPILE_OPTIONS_MSVC}>
    )
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
        set_target_properties(flann_s PROPERTIES COMPILE_FLAGS -fPIC)
    endif()
    set_property(TARGET flann_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC)
    if (CMAKE_BUILD_STATIC_LIBS)
        list(APPEND flann_install_targets flann_s)
    else()
        set_target_properties(flann_s PROPERTIES EXCLUDE_FROM_ALL true)
    endif()

    add_library(flann SHARED ${C_SOURCES})
    target_link_libraries(flann PUBLIC ${LZ4_LINK_LIBRARIES})
    target_include_directories(flann
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${LZ4_INCLUDE_DIRS}
    )
    target_compile_options(flann PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${FLANN_COMPILE_OPTIONS_GCC}>
        $<$<CXX_COMPILER_ID:MSVC>:${FLANN_COMPILE_OPTIONS_MSVC}>
    )
    list(APPEND flann_install_targets flann)

    # Alias targets for modern CMake usage
    add_library(flann::flann ALIAS flann)
    add_library(flann::flann_s ALIAS flann_s)

    if(MINGW AND OPENMP_FOUND)
        target_link_libraries(flann PUBLIC gomp)
    endif()

    set_target_properties(flann PROPERTIES
       VERSION ${FLANN_VERSION}
       SOVERSION ${FLANN_SOVERSION}
       DEFINE_SYMBOL FLANN_EXPORTS
    )
endif()

# Note: C bindings do NOT use CUDA - they don't expose CUDA-specific APIs
# So we don't set FLANN_USE_CUDA or link against flann_cuda for C bindings

if (BUILD_OpenCL_LIB)
    set_property(TARGET flann_cpp PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    set_property(TARGET flann PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    set_property(TARGET flann_s PROPERTY COMPILE_DEFINITIONS FLANN_USE_OPENCL)
    # OpenCL is an implementation detail, link as PRIVATE (not PUBLIC)
    target_link_libraries(flann_cpp PRIVATE ${OpenCL_LIBRARIES})
    target_link_libraries(flann_cpp_s PRIVATE ${OpenCL_LIBRARIES})
    target_link_libraries(flann PRIVATE ${OpenCL_LIBRARIES})
    target_link_libraries(flann_s PRIVATE ${OpenCL_LIBRARIES})
    # Add OpenCL include directories
    target_include_directories(flann_cpp PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_include_directories(flann_cpp_s PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_include_directories(flann PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_include_directories(flann_s PRIVATE ${OpenCL_INCLUDE_DIRS})
endif()

if(WIN32)
if (BUILD_C_BINDINGS AND BUILD_MATLAB_BINDINGS)
    install (
        TARGETS flann
        RUNTIME DESTINATION share/flann/matlab
    )
endif()
endif(WIN32)


install (
    TARGETS ${flann_install_targets}
    EXPORT ${targets_export_name}
    INCLUDES DESTINATION include
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION ${FLANN_LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${FLANN_LIB_INSTALL_DIR}
)


install (
    DIRECTORY flann
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
