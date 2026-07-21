#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdlib>

// Fail at compile time if the expected gfx942 arch macro is missing — a
// successful compilation already proves the Kokkos install has the matching
// arch support. The harness/CMake selects which arch to assert via
// KOKKOS_CHECK_EXPECT_APU: defined for MI300A (APU, KOKKOS_ARCH_AMD_GFX942_APU),
// undefined for MI300X / discrete gfx942 (KOKKOS_ARCH_AMD_GFX942). Both are
// gfx942 and differ only by the _APU suffix.
#if defined(KOKKOS_CHECK_EXPECT_APU)
  #if !defined(KOKKOS_ARCH_AMD_GFX942_APU)
  #error "KOKKOS_ARCH_AMD_GFX942_APU is NOT defined (APU build expected)"
  #endif
  #define KOKKOS_CHECK_ARCH_LABEL "KOKKOS_ARCH_AMD_GFX942_APU"
#else
  #if !defined(KOKKOS_ARCH_AMD_GFX942)
  #error "KOKKOS_ARCH_AMD_GFX942 is NOT defined (discrete gfx942 build expected)"
  #endif
  #define KOKKOS_CHECK_ARCH_LABEL "KOKKOS_ARCH_AMD_GFX942"
#endif

#if !defined(KOKKOS_ENABLE_HIP)
#error "KOKKOS_ENABLE_HIP is NOT defined"
#endif

#if !defined(KOKKOS_ARCH_AMD_GPU)
#error "KOKKOS_ARCH_AMD_GPU is NOT defined"
#endif

int main(int argc, char* argv[]) {
  std::cout << "PASS: " << KOKKOS_CHECK_ARCH_LABEL << " is defined" << std::endl;
  std::cout << "PASS: KOKKOS_ENABLE_HIP is defined" << std::endl;
  std::cout << "PASS: KOKKOS_ARCH_AMD_GPU = \"" << KOKKOS_ARCH_AMD_GPU << "\"" << std::endl;

  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nKokkos runtime configuration:" << std::endl;
    Kokkos::print_configuration(std::cout, true);
  }
  Kokkos::finalize();

  std::cout << "\nAll Kokkos gfx942 checks PASSED (" << KOKKOS_CHECK_ARCH_LABEL << ")." << std::endl;
  return EXIT_SUCCESS;
}
