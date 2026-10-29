cmake_minimum_required(VERSION 2.9)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
# --- custom targets: ---
#if (NOT TARGET distclean)
#   INCLUDE( cmake/Modules/TargetDistclean.cmake OPTIONAL)
#endif (NOT TARGET distclean)

# enable @rpath in the install name for any shared library being built
set(CMAKE_MACOSX_RPATH 1)

project (GRAPHICS)
# The version number.
set (GRAPHICS_VERSION_MAJOR 2)
set (GRAPHICS_VERSION_MINOR 0.7)
set (GRAPHICS_VERSION "${GRAPHICS_VERSION_MAJOR}.${GRAPHICS_VERSION_MINOR}")

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

# uncomment to build debug
#set(CMAKE_BUILD_TYPE Debug)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
   # using Clang
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
   # using GCC
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
   # using Intel C
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
   # using Visual Studio C++
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "XL")
   # using IBM XL Compiler
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
endif()

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic")

#find_package(ImageMagick COMPONENTS MagickCore MagickWand)
find_package(ImageMagick COMPONENTS MagickWand)
#   ImageMagick_FOUND                    - TRUE if all components are found.
#   ImageMagick_INCLUDE_DIRS             - Full paths to all include dirs.
#   ImageMagick_LIBRARIES                - Full paths to all libraries.
#   ImageMagick_<component>_FOUND        - TRUE if <component> is found.
#   ImageMagick_<component>_INCLUDE_DIRS - Full path to <component> include dirs.
#   ImageMagick_<component>_LIBRARIES    - Full path to <component> libraries.
if (ImageMagick_FOUND)
   if (ImageMagick_MagickWand_FOUND)
      set(ImageMagick_MagickWand_FOUND on PARENT_SCOPE)
      add_definitions(-DHAVE_MAGICKWAND)
      add_definitions(-DMAGICKCORE_HDRI_ENABLE=0)
      add_definitions(-DMAGICKCORE_QUANTUM_DEPTH=16)
#     add_definitions(-DMAGICKCORE_HDRI_ENABLE=0)
#     add_definitions(-DMAGICKCORE_QUANTUM_DEPTH=16)
      include_directories(${ImageMagick_INCLUDE_DIRS})
      foreach(v ${ImageMagick_INCLUDE_DIRS})
         find_path(ImageMagick_INCLUDE_SUBDIR MagickWand.h ${v}/wand ${v}/MagickWand)
      endforeach()
      get_filename_component(ImageMagick_INCLUDE_SUBDIR ${ImageMagick_INCLUDE_SUBDIR} NAME)
      if (${ImageMagick_INCLUDE_SUBDIR} STREQUAL wand)
          add_definitions(-DImageMagickWand=1)
      endif (${ImageMagick_INCLUDE_SUBDIR} STREQUAL wand)
   endif (ImageMagick_MagickWand_FOUND)

#   message("ImageMagick_FOUND                   ${ImageMagick_FOUND}")
#   message("ImageMagick_INCLUDE_DIRS            ${ImageMagick_INCLUDE_DIRS}")
#   message("ImageMagick_LIBRARIES               ${ImageMagick_LIBRARIES}")
##  message("ImageMagick_MagickCore_FOUND        ${ImageMagick_MagickCore_FOUND}")
##  message("ImageMagick_MagickCore_INCLUDE_DIRS ${ImageMagick_MagickCore_INCLUDE_DIRS}")
##  message("ImageMagick_MagickCore_LIBRARIES    ${ImageMagick_MagickCore_LIBRARIES}")
#   message("ImageMagick_MagickWand_FOUND        ${ImageMagick_MagickWand_FOUND}")
#   message("ImageMagick_MagickWand_INCLUDE_DIRS ${ImageMagick_MagickWand_INCLUDE_DIRS}")
#   message("ImageMagick_MagickWand_LIBRARIES    ${ImageMagick_MagickWand_LIBRARIES}")
endif (ImageMagick_FOUND)

if (NOT ENABLE_GRAPHICS)
   set (ENABLE_GRAPHICS false)
endif (NOT ENABLE_GRAPHICS)

if (ENABLE_GRAPHICS)
   find_package(OpenGL REQUIRED)
   find_package(FREEGLUT REQUIRED)

   set (ENABLE_GRAPHICS on PARENT_SCOPE)
   set (HAVE_OPENGL on PARENT_SCOPE)
   add_definitions(-DHAVE_GRAPHICS)
   add_definitions(-DHAVE_OPENGL)
   include_directories(${OPENGL_INCLUDE_DIR})
   include_directories(${FREEGLUT_INCLUDE_DIR})
endif (ENABLE_GRAPHICS)

########### global settings ###############
set(H_SRCS graphics.h display.h)
set(C_SRCS graphics.c display.c)

set(graphics_LIB_SRCS ${F_SRCS} ${C_SRCS} ${H_SRCS})

########### graphics target ###############

add_library(graphics STATIC ${graphics_LIB_SRCS})

set_target_properties(graphics PROPERTIES VERSION ${GRAPHICS_VERSION} SOVERSION ${GRAPHICS_VERSION_MAJOR})

target_link_libraries(graphics ${OPENGL_LIBRARIES} ${FREEGLUT_LIBRARIES})
if (ImageMagick_MagickWand_FOUND)
   target_link_libraries(graphics ${ImageMagick_LIBRARIES})
endif (ImageMagick_MagickWand_FOUND)

install(TARGETS graphics DESTINATION lib)

########### install files ###############

install(FILES  display.h DESTINATION include/graphics)
install(FILES  graphics.h DESTINATION include/graphics)

########### clean files ################
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
   "install_manifest.txt;docs/_build;docs/htmldoc.out;docs/pdfdoc.out;docs/singlehtmldoc.out")

