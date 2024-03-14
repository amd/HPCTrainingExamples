cmake_minimum_required(VERSION 3.10)
project (ShallowWater)

set (CMAKE_C_STANDARD 99)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")

# Adds build target of ShallowWater with source code files
add_executable(ShallowWater ShallowWater.c memory.c timer.c)
target_link_libraries(ShallowWater -lm)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake ShallowWater.dSYM ipo_out.optrpt)
