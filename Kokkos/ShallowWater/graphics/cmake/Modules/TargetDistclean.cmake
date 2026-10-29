# add custom target distclean
# cleans and removes cmake generated files etc.
# Jan Woetzel 04/2003
#

IF (UNIX)
  ADD_CUSTOM_TARGET (distclean @echo cleaning for source distribution)
  SET(DISTCLEANED
   cmake.depends
   cmake.check_depends
   CMakeCache.txt
   */CMakeCache.txt
   cmake.check_cache
   *.cmake
   *.a
   */*.a
   Makefile
   */Makefile
   */*/Makefile
   core core.*
   gmon.out
   *~
   CMakeFiles
   */CMakeFiles
   */*/CMakeFiles
   cmake_install.cmake
   */cmake_install.cmake
   */*/cmake_install.cmake
  )
  
  ADD_CUSTOM_COMMAND(
    DEPENDS clean
    COMMENT "distribution clean"
    COMMAND rm
    ARGS    -Rf CMakeTmp ${DISTCLEANED}
    TARGET  distclean
  )
ENDIF(UNIX)
