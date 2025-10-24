cmake_minimum_required(VERSION 3.20)
project(NumaWindowsTools LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Executable
add_executable(NumaWindowsTools
	main.cpp
	NumaWindowsTools.cpp
	NumaWindowsTools.h
)

# Windows-specific libraries
target_link_libraries(NumaWindowsTools PRIVATE SetupAPI)


# add HIP, used for this demo.
find_package(hip CONFIG REQUIRED)
target_link_libraries(NumaWindowsTools PRIVATE hip::amdhip64)
target_compile_definitions(NumaWindowsTools PRIVATE __HIP_PLATFORM_AMD__)


# Compiler warnings
if(MSVC)
	target_compile_options(NumaWindowsTools PRIVATE /W4 /permissive-)
endif()

# set Visual Studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NumaWindowsTools)
