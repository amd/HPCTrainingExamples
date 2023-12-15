cmake_minimum_required (VERSION 3.0)
project (Raja_code)

find_package(Raja REQUIRED)

add_executable(raja_code raja_code.cc)
target_link_libraries(raja_code PUBLIC RAJA)

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles Raja Raja_build/build
    Makefile cmake_install.cmake raja_code.dSYM ipo_out.optrpt)
