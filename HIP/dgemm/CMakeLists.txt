# /***************************************************************************
#  Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
# ***************************************************************************/

cmake_minimum_required(VERSION 2.8)
project(xGEMM)

# ----------------------------------------------------------------SET CXX STANDARD--
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------------------------------INCLUDES--
include(GNUInstallDirs)
# ------------------------------------------------------------------------------HIP--
if (NOT DEFINED HIP_PATH)
  if (NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "HIP path")
  else()
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "HIP path")
  endif()
endif()

include(${HIP_PATH}/cmake/FindHIP.cmake)

find_package(HIP REQUIRED)

if (HIP_FOUND)
# List any other hip libs needed
  find_package(hipblas REQUIRED)
  if (hipblas_FOUND)
    message(STATUS "Found ${hipblas_LIBRARIES} in ${hipblas_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Cannot find hipblas")
  endif()

# set compiler settings
  set(CMAKE_CXX_COMPILER hipcc)
  message(STATUS "CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")

# set compiler flags
  set(xDGEMM_HIP_FLAGS "-fno-gpu-rdc -munsafe-fp-atomics -fPIC -w" CACHE STRING "Compiler flags for HIP")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${xDGEMM_HIP_FLAGS}" CACHE STRING "Flags used by the CXX compiler during all build types." FORCE)
else ()
  message(FATAL_ERROR "Cannot find hip")
endif()
# ------------------------------------------------------------------------------SRC--
add_subdirectory("src")

# EOF
