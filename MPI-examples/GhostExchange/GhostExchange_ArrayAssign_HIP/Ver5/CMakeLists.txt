cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(GhostExchange LANGUAGES C CXX HIP)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 11)
set (GPU_RUNTIME "HIP")
enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

find_package(hip REQUIRED IMPORTED CONFIG HINTS ${ROCM_PATH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(Vector)

if (CMAKE_VECTOR_VERBOSE)
    set(VECTOR_C_FLAGS "${VECTOR_C_FLAGS} ${VECTOR_C_VERBOSE}")
    set(VECTOR_CXX_FLAGS "${VECTOR_CXX_FLAGS} ${VECTOR_CXX_VERBOSE}")
endif (CMAKE_VECTOR_VERBOSE)

if (CMAKE_VECTOR_NOVEC)
    set(VECTOR_C_FLAGS "${VECTOR_NOVEC_C_FLAGS}")
    set(VECTOR_CXX_FLAGS "${VECTOR_NOVEC_CXX_FLAGS}")
endif (CMAKE_VECTOR_NOVEC)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VECTOR_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTOR_CXX_FLAGS}")

# Require MPI for this project:
find_package(MPI REQUIRED)

execute_process(COMMAND rocminfo COMMAND grep -m 1 -E gfx[^0]{1} COMMAND sed -e "s/ *Name: *//" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROCM_GPU)

set(CMAKE_HIP_ARCHITECTURES ${ROCM_GPU})
set(GPU_TARGETS ${ROCM_GPU})


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   # using Clang
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   # using GCC
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Cray")
endif()

add_executable(GhostExchange GhostExchange.hip timer.c timer.h)

set_source_files_properties(GhostExchange.hip PROPERTIES LANGUAGE ${GPU_RUNTIME})
set_target_properties(GhostExchange PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH};$ENV{ROCM_PATH}/include/roctracer")
set_target_properties(GhostExchange PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
target_link_libraries(GhostExchange PRIVATE "${MPI_CXX_LIBRARIES};-L$ENV{ROCM_PATH}/lib -lroctx64")

# Add a test:
enable_testing()
add_test(SimpleTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/GhostExchange ${MPIEXEC_POSTFLAGS})

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake CTestTestfile.cmake Testing)
