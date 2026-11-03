cmake_minimum_required (VERSION 3.10)
project (StreamTriad)

add_executable(StreamTriad StreamTriad.cc timer.c)

set_target_properties(StreamTriad PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
