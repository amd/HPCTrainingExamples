#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos version: "
              << KOKKOS_VERSION_MAJOR << "."
              << KOKKOS_VERSION_MINOR << "."
              << KOKKOS_VERSION_PATCH << std::endl;

    std::cout << "Kokkos Default Execution Space :" << Kokkos::DefaultExecutionSpace::name() << "\n";
    std::cout << "Kokkos Host Execution Space :" << Kokkos::DefaultHostExecutionSpace::name() << "\n";

    Kokkos::print_configuration(std::cout, true); // verbose = true
  }
  Kokkos::finalize();
  return 0;
}
