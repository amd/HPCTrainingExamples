cmake_minimum_required(VERSION 3.16)
project (ShallowWater)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

find_package(Kokkos REQUIRED)

if (ENABLE_GRAPHICS)
   add_subdirectory(graphics)
endif (ENABLE_GRAPHICS)

# Adds build target of ShallowWater with source code files
add_executable(ShallowWater ShallowWater.cc memory.c timer.c)
target_link_libraries(ShallowWater -lm)
if (ENABLE_GRAPHICS)
   set_source_files_properties(ShallowWater.cc PROPERTIES COMPILE_FLAGS "-DHAVE_GRAPHICS=1")
   set_source_files_properties(ShallowWater.cc PROPERTIES INCLUDE_DIRECTORIES graphics)
   target_link_libraries(ShallowWater graphics ${FREEGLUT_LIBRARIES})
endif (ENABLE_GRAPHICS)

# Adds build target of ShallowWater_par1 with source code files
add_executable(ShallowWater_par1 ShallowWater_par1.cc timer.c)
target_link_libraries(ShallowWater_par1 Kokkos::kokkos -lm)

# Adds build target of ShallowWater_par2 with source code files
add_executable(ShallowWater_par2 ShallowWater_par2.cc timer.c)
target_link_libraries(ShallowWater_par2 Kokkos::kokkos -lm)

# Adds build target of ShallowWater_par3 with source code files
add_executable(ShallowWater_par3 ShallowWater_par3.cc timer.c)
target_link_libraries(ShallowWater_par3 Kokkos::kokkos -lm)

# Adds build target of ShallowWater_par4 with source code files
add_executable(ShallowWater_par4 ShallowWater_par4.cc)
target_link_libraries(ShallowWater_par4 Kokkos::kokkos -lm)

# Adds build target of ShallowWater_par5 with source code files
add_executable(ShallowWater_par5 ShallowWater_par5.cc)
target_link_libraries(ShallowWater_par5 Kokkos::kokkos -lm)

set_target_properties(ShallowWater ShallowWater_par1 ShallowWater_par2 ShallowWater_par3 ShallowWater_par4 ShallowWater_par5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake ShallowWater.dSYM ipo_out.optrpt)
