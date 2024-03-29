cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Jacobi LANGUAGES CXX)

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

find_package(MPI REQUIRED)

set(JACOBI_CXX_SRCS "")

set(JACOBI_HIP_SRCS HaloExchange.hip Input.hip JacobiIteration.hip JacobiMain.hip JacobiRun.hip JacobiSetup.hip Laplacian.hip Norm.hip)

add_executable(Jacobi_hip ${JACOBI_CXX_SRCS} ${JACOBI_HIP_SRCS} )

set(CXXFLAGS "")
set(ROCMCC_FLAGS "${ROCMCC_FLAGS} -munsafe-fp-atomics -I$ENV{ROCM_PATH}/include/roctracer -I${MPI_INCLUDE_PATH}")
set(CUDACC_FLAGS "${CUDACC_FLAGS} ${MPI_INCLUDE_PATH}")

if (${GPU_RUNTIME} MATCHES "HIP")
   set(HIPCC_FLAGS "${ROCMCC_FLAGS} -D__HIP_PLATFORM_AMD__")
   set(HIPCC_FLAGS "${ROCMCC_FLAGS} -D__HIP_PLATFORM_HCC__")
   set(HIPCC_LIBS "-lroctracer64 -lroctx64")
else (${GPU_RUNTIME} MATCHES "HIP")
   set(HIPCC_FLAGS "${CUDACC_FLAGS} -D__HIP_PLATFORM_NVIDIA__")
   set(HIPCC_LIBS "")
endif (${GPU_RUNTIME} MATCHES "HIP")

set_source_files_properties(${JACOBI_HIP_SRCS} PROPERTIES LANGUAGE ${GPU_RUNTIME})
set_source_files_properties(${JACOBI_HIP_SRCS} PROPERTIES COMPILE_FLAGS ${HIPCC_FLAGS})
target_link_libraries(Jacobi_hip PRIVATE ${MPI_LIBRARIES} ${HIPCC_LIBS})

install(TARGETS Jacobi_hip)
