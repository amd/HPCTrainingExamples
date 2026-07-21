cmake_minimum_required(VERSION 3.16)
project (ShallowWater)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

set(GPU_TARGETS "gfx942" CACHE STRING "GPU targets" FORCE)
set(AMDGPU_TARGETS "gfx942" CACHE STRING "AMD GPU targets" FORCE)

find_package(Kokkos REQUIRED)

add_executable(kokkos_code kokkos_code.cc)
target_link_libraries(kokkos_code Kokkos::kokkos)
