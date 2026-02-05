cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(GhostExchange C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")

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

if(USE_PNETCDF)

   add_compile_definitions(USE_PNETCDF)

   find_program(PNETCDF_CONFIG pnetcdf-config REQUIRED)

   execute_process(
       COMMAND ${PNETCDF_CONFIG} --version
       OUTPUT_VARIABLE PNETCDF_VERSION
       OUTPUT_STRIP_TRAILING_WHITESPACE
   )

   execute_process(
       COMMAND ${PNETCDF_CONFIG} --includedir
       OUTPUT_VARIABLE PNETCDF_INCLUDEDIR
       OUTPUT_STRIP_TRAILING_WHITESPACE
   )

   execute_process(
       COMMAND ${PNETCDF_CONFIG} --libdir
       OUTPUT_VARIABLE PNETCDF_LIBDIR
       OUTPUT_STRIP_TRAILING_WHITESPACE
   )

   message(STATUS "Found PnetCDF ${PNETCDF_VERSION}")
   message(STATUS "PnetCDF Include Dir: ${PNETCDF_INCLUDEDIR}")
   message(STATUS "PnetCDF Lib Dir: ${PNETCDF_LIBDIR}")
endif()

execute_process(COMMAND rocminfo COMMAND grep -m 1 -E gfx[^0]{1} COMMAND sed -e "s/ *Name: *//" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROCM_GPU)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   # using Clang
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp --offload-arch=${ROCM_GPU} -fstrict-aliasing")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp --offload-arch=${ROCM_GPU} -fstrict-aliasing")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   # using GCC
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -foffload=-march=native -fstrict-aliasing -fopt-info-optimized-omp")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -foffload=-march=native -fstrict-aliasing -fopt-info-optimized-omp")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "CrayClang")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()

add_executable(GhostExchange GhostExchange.cc timer.c timer.h)

set_target_properties(GhostExchange PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH};$ENV{ROCM_PATH}/include/roctracer")
set_target_properties(GhostExchange PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
target_link_libraries(GhostExchange PRIVATE "${MPI_CXX_LIBRARIES};-L$ENV{ROCM_PATH}/lib -lroctx64")

if(USE_PNETCDF)
   target_include_directories(GhostExchange PRIVATE ${PNETCDF_INCLUDEDIR})
   target_link_directories(GhostExchange PRIVATE ${PNETCDF_LIBDIR})
   target_link_libraries(GhostExchange PRIVATE pnetcdf)
endif()

# Add a test:
enable_testing()
add_test(SimpleTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/GhostExchange ${MPIEXEC_POSTFLAGS})

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake CTestTestfile.cmake Testing)
