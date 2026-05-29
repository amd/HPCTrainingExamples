cmake_minimum_required(VERSION 3.16)
project(KokkosAPUCheck CXX)

# Prevent ROCm CMake from adding a second --offload-arch with xnack/sramecc
# suffixes that conflicts with the plain gfx942 arch from Kokkos targets.
set(GPU_TARGETS "gfx942" CACHE STRING "GPU targets" FORCE)
set(AMDGPU_TARGETS "gfx942" CACHE STRING "AMD GPU targets" FORCE)

find_package(Kokkos REQUIRED)

# Use the built-in kokkos_check() to verify the APU arch was configured
kokkos_check(ARCH AMD_GFX942_APU RETURN_VALUE KOKKOS_HAS_APU)

if(NOT KOKKOS_HAS_APU)
  message(FATAL_ERROR "Kokkos was NOT built with AMD_GFX942_APU architecture support")
else()
  message(STATUS "kokkos_check: AMD_GFX942_APU architecture confirmed")
endif()

# Also verify HIP backend is present (required for GPU-side APU usage)
kokkos_check(DEVICES HIP RETURN_VALUE KOKKOS_HAS_HIP)
if(NOT KOKKOS_HAS_HIP)
  message(FATAL_ERROR "Kokkos was NOT built with HIP backend (required for APU)")
else()
  message(STATUS "kokkos_check: HIP backend confirmed")
endif()

add_executable(kokkos_check_apu_enabled kokkos_check_apu_enabled.cpp)
target_link_libraries(kokkos_check_apu_enabled Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosAPUCheck COMMAND kokkos_check_apu_enabled)
