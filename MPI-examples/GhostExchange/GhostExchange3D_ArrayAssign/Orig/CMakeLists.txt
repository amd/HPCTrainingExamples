cmake_minimum_required(VERSION 3.0)

# Pin clang to the newest GCC dir that actually ships libstdc++ (must run
# before project(); FLAGS_INIT is what its compiler checks consult).
if(NOT DEFINED GCC_LIBSTDCXX_PINNED)
  set(_cxx "$ENV{CXX}")
  if(_cxx STREQUAL "")
    set(_cxx "${CMAKE_CXX_COMPILER}")
  endif()
  if(_cxx MATCHES "clang")
    file(GLOB _gcc_dirs LIST_DIRECTORIES true /usr/lib/gcc/x86_64-linux-gnu/*)
    list(SORT _gcc_dirs COMPARE NATURAL ORDER DESCENDING)
    foreach(_d ${_gcc_dirs})
      if(EXISTS "${_d}/libstdc++.so" OR EXISTS "${_d}/libstdc++.a")
        set(_pin "--gcc-install-dir=${_d}")
        string(APPEND CMAKE_C_FLAGS_INIT   " ${_pin}")
        string(APPEND CMAKE_CXX_FLAGS_INIT " ${_pin}")
        string(APPEND CMAKE_HIP_FLAGS_INIT " ${_pin}")
        message(STATUS "Pinning clang libstdc++ to ${_d}")
        break()
      endif()
    endforeach()
  endif()
  set(GCC_LIBSTDCXX_PINNED TRUE)
endif()

project(GhostExchange)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 98)

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

add_executable(GhostExchange GhostExchange.cc timer.c timer.h malloc3D.c malloc3D.h)

set_target_properties(GhostExchange PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
set_target_properties(GhostExchange PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
set_target_properties(GhostExchange PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
target_link_libraries(GhostExchange PRIVATE "${MPI_CXX_LIBRARIES}")

# Add a test:
enable_testing()
add_test(SimpleTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/GhostExchange ${MPIEXEC_POSTFLAGS})

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake CTestTestfile.cmake Testing)
