cmake_minimum_required(VERSION 3.16)
project(KokkosAPUCheck CXX)

# Prevent ROCm CMake from adding a second --offload-arch with xnack/sramecc
# suffixes that conflicts with the plain gfx942 arch from Kokkos targets.
set(GPU_TARGETS "gfx942" CACHE STRING "GPU targets" FORCE)
set(AMDGPU_TARGETS "gfx942" CACHE STRING "AMD GPU targets" FORCE)

# Which Kokkos AMD GPU arch to verify. MI300A (APU) -> AMD_GFX942_APU,
# MI300X (discrete) -> AMD_GFX942; both are gfx942 and differ only by the
# _APU suffix. The test harness passes -DEXPECTED_KOKKOS_ARCH=<arch> based on
# the detected GPU (or the installed Kokkos config). Default to the APU arch so
# a standalone configure still behaves like the original APU-only test.
if(NOT DEFINED EXPECTED_KOKKOS_ARCH)
  set(EXPECTED_KOKKOS_ARCH "AMD_GFX942_APU")
endif()

find_package(Kokkos REQUIRED)

# Use the built-in kokkos_check() to verify the expected arch was configured
kokkos_check(ARCH ${EXPECTED_KOKKOS_ARCH} RETURN_VALUE KOKKOS_HAS_ARCH)

if(NOT KOKKOS_HAS_ARCH)
  message(FATAL_ERROR "Kokkos was NOT built with ${EXPECTED_KOKKOS_ARCH} architecture support")
else()
  message(STATUS "kokkos_check: ${EXPECTED_KOKKOS_ARCH} architecture confirmed")
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

# Tell the source which arch family to assert at compile time. APU builds must
# expose KOKKOS_ARCH_AMD_GFX942_APU; discrete builds expose KOKKOS_ARCH_AMD_GFX942.
if(EXPECTED_KOKKOS_ARCH MATCHES "_APU$")
  target_compile_definitions(kokkos_check_apu_enabled PRIVATE KOKKOS_CHECK_EXPECT_APU=1)
endif()

enable_testing()
add_test(NAME KokkosAPUCheck COMMAND kokkos_check_apu_enabled)
