cmake_minimum_required (VERSION 3.10)
project (StreamTriad)

find_package(Kokkos REQUIRED)

add_executable(StreamTriad StreamTriad.cc timer.c)
target_link_libraries(StreamTriad Kokkos::kokkos)

set_target_properties(StreamTriad PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
