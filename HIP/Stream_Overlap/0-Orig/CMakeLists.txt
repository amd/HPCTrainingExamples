cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(compute_comm_overlap LANGUAGES CXX)
include(CTest)

set (CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

string(REPLACE -O2 -O3 CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

if (NOT CMAKE_GPU_RUNTIME)
   set(GPU_RUNTIME "ROCM" CACHE STRING "Switches between ROCM and CUDA")
else (NOT CMAKE_GPU_RUNTIME)
   set(GPU_RUNTIME "${CMAKE_GPU_RUNTIME}" CACHE STRING "Switches between ROCM and CUDA")
endif (NOT CMAKE_GPU_RUNTIME)
# Really should only be ROCM or CUDA, but allowing HIP because it is the currently built-in option
set(GPU_RUNTIMES "ROCM" "CUDA" "HIP")
if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP, ROCM, or CUDA.")
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()
# GPU_RUNTIME for AMD GPUs should really be ROCM, if selecting AMD GPUs
# so manually resetting to HIP if ROCM is selected
if (${GPU_RUNTIME} MATCHES "ROCM")
   set(GPU_RUNTIME "HIP")
endif (${GPU_RUNTIME} MATCHES "ROCM")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

set(CMAKE_${GPU_RUNTIME}_FLAGS_DEBUG "-ggdb")

set(COMPUTE_COMM_OVERLAP_CXX_SRCS "")

set(COMPUTE_COMM_OVERLAP_HIP_SRCS compute_comm_overlap.hip)

if (DEFINED ENV{HIP_PATH})
   set(HIP_PATH $ENV{HIP_PATH})
else (DEFINED ENV{HIP_PATH})
   execute_process(COMMAND hipconfig --path OUTPUT_VARIABLE HIP_PATH ERROR_QUIET)
endif (DEFINED ENV{HIP_PATH})

add_executable(compute_comm_overlap ${COMPUTE_COMM_OVERLAP_CXX_SRCS} ${COMPUTE_COMM_OVERLAP_HIP_SRCS} )

# Make example runnable using ctest
add_test(NAME Compute_comm_overlap COMMAND compute_comm_overlap )
set_property(TEST Compute_comm_overlap
             PROPERTY PASS_REGULAR_EXPRESSION "PASSED!")

option(MI300A_RUN "Run on MI300A?" OFF)
message(STATUS "MI300A_RUN=${MI300A_RUN}")

if (NOT MI300A_RUN)
   set(ROCMCC_FLAGS "${ROCMCC_FLAGS} -munsafe-fp-atomics")
else (NOT MI300A_RUN)
   set(ROCMCC_FLAGS "${ROCMCC_FLAGS} -munsafe-fp-atomics -DMI300A_RUN")
endif()
set(CUDACC_FLAGS "${CUDACC_FLAGS} ")

if (${GPU_RUNTIME} MATCHES "HIP")
   set(HIPCC_FLAGS "${ROCMCC_FLAGS}")
else (${GPU_RUNTIME} MATCHES "HIP")
   set(HIPCC_FLAGS "${CUDACC_FLAGS} -I/${HIP_PATH}/include")
endif (${GPU_RUNTIME} MATCHES "HIP")

set_source_files_properties(${COMPUTE_COMM_OVERLAP_HIP_SRCS} PROPERTIES LANGUAGE ${GPU_RUNTIME})
set_source_files_properties(compute_comm_overlap.hip PROPERTIES COMPILE_FLAGS "${HIPCC_FLAGS}")

install(TARGETS compute_comm_overlap)
