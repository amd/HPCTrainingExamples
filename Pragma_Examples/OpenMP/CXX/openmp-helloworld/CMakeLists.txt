project(openmp_helloworld)

cmake_minimum_required(VERSION 3.16)

# Search for rocm in common locations
if(WIN32)
  list(APPEND CMAKE_PREFIX_PATH "C:/hip")
  list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/AMD HIP SDK/hip")
else()
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
endif()

# Find HIP.
# The user may override AMDGPU_TARGETS defined in the HIP config file
# to select the AMDGPU archs to compile for.
# ex. set(AMDGPU_TARGETS "gfx803;gfx900;gfx906")
find_package(hip REQUIRED)

# Find OpenMP.
find_package(OpenMP REQUIRED)

# Set compiler and linker.
if(NOT WIN32)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
endif()

set(CMAKE_BUILD_TYPE Release)

if(WIN32)
  # Compile for OpenMP code (Windows requires this).
  set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  # Tell CMake where to find the OpenMP libraries (libomp.lib).
  link_directories("C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\Llvm\\x64\\lib")
endif()

# Source files.
set(CPP_SOURCES ${CMAKE_SOURCE_DIR}/openmp_helloworld.cpp)

# Preparing the executable.
add_executable(test_openmp_helloworld ${CPP_SOURCES})

# Link Libraries - HIP Device and OpenMP.
target_compile_options(test_openmp_helloworld PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(test_openmp_helloworld PRIVATE hip::device ${OpenMP_CXX_FLAGS})

if(WIN32)
  target_link_libraries(test_openmp_helloworld PRIVATE OpenMP::OpenMP_CXX)
endif()
