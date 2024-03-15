cmake_minimum_required(VERSION 3.10)
project (ShallowWater)

set (CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

# Adds build target of ShallowWater with source code files
add_executable(ShallowWater ShallowWater.cpp Var2D.hpp)
target_link_libraries(ShallowWater -lm)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake ShallowWater.dSYM ipo_out.optrpt)
