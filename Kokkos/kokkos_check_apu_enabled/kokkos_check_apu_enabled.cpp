#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdlib>

// Fail at compile time if the APU macros are missing — a successful
// compilation already proves the Kokkos install has APU support.
#if !defined(KOKKOS_ARCH_AMD_GFX942_APU)
#error "KOKKOS_ARCH_AMD_GFX942_APU is NOT defined"
#endif

#if !defined(KOKKOS_ENABLE_HIP)
#error "KOKKOS_ENABLE_HIP is NOT defined"
#endif

#if !defined(KOKKOS_ARCH_AMD_GPU)
#error "KOKKOS_ARCH_AMD_GPU is NOT defined"
#endif

int main(int argc, char* argv[]) {
  std::cout << "PASS: KOKKOS_ARCH_AMD_GFX942_APU is defined" << std::endl;
  std::cout << "PASS: KOKKOS_ENABLE_HIP is defined" << std::endl;
  std::cout << "PASS: KOKKOS_ARCH_AMD_GPU = \"" << KOKKOS_ARCH_AMD_GPU << "\"" << std::endl;

  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nKokkos runtime configuration:" << std::endl;
    Kokkos::print_configuration(std::cout, true);
  }
  Kokkos::finalize();

  std::cout << "\nAll Kokkos APU checks PASSED." << std::endl;
  return EXIT_SUCCESS;
}
