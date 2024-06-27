cmake_minimum_required(VERSION 3.21.0)

#--------------------------------------------
#
#   CMAKE CONFIG FILE FOR  VECTORADD
#
#--------------------------------------------

#--------------------
# Project definition
#--------------------
project (vectoradd)


# Set the preprocessor directives for compilation
set (PREPROC_DIRECTIVES "${PREPROC_DIRECTIVES}")

# Colored output
string(ASCII 27 Esc) # Esc character for colored output
set(ColourReset "${Esc}[m")
set(Red         "${Esc}[31m")
set(Yellow      "${Esc}[33m")
set(Green       "${Esc}[32m")

#------------------------------
# Enable GPU_RUNTIME
#------------------------------
if (NOT CMAKE_GPU_RUNTIME)
   set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
else (NOT CMAKE_GPU_RUNTIME)
   set(GPU_RUNTIME "${CMAKE_GPU_RUNTIME}" CACHE STRING "Switches between HIP and CUDA")
endif (NOT CMAKE_GPU_RUNTIME)

#------------------------------
# Enable HIP or CUDA support
#------------------------------
enable_language (HIP)

#-------------------
# Build environment
#-------------------
if (NOT CMAKE_BUILD_TYPE)
   set (CMAKE_BUILD_TYPE Release)
endif ()

#------------------------------
# SET HIP OPTIONS
#------------------------------
if (${GPU_RUNTIME} MATCHES "HIP")
   set (VECTORADD_CXX_FLAGS "-fPIC")
elseif (${GPU_RUNTIME} MATCHES "CUDA")
   #------------------------------
   # SET FLAGS
   #------------------------------
   set (VECTORADD_CXX_FLAGS "-I$ENV{ROCM_PATH}/include/ ")
else ()
   message(FATAL_ERROR "Wrong GPU_RUNTIME choice! Available options: HIP or CUDA. Exiting...")
endif ()

message(STATUS "VECTORADD_CXX_FLAGS: ${VECTORADD_CXX_FLAGS}")

#------------------------------
# SET HIP COMPILER OPTIONS
#------------------------------
set(CMAKE_CXX_COMPILER hipcc)

#CMAKE_CXX_RELEASE_FLAGS "-w" to ignore all warnings
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTORADD_CXX_FLAGS}")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTORADD_CXX_FLAGS}")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${VECTORADD_CXX_FLAGS} -ggdb")

message(STATUS "CXX Compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX Compiler flags Release: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX Compiler flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}")

#------------------------
# Define the source path
#------------------------
set(SOURCE vectorAdd_hip.cpp)

#-------------------------
# Define the build & link
#-------------------------
add_executable (vectoradd ${SOURCE})
if (${GPU_RUNTIME} MATCHES "HIP")
   target_link_libraries (vectoradd "-L$ENV{ROCM_PATH}/lib" amdhip64)
elseif (${GPU_RUNTIME} MATCHES "CUDA")
   target_link_libraries (vectoradd cudadevrt cudart_static rt)   
endif ()
