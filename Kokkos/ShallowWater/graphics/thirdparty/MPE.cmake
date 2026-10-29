#include(CheckIncludeFileCXX)

# Set the name of the directory where to compile MPE
set(MPE_PREFIX mpe2-2.4.9b)

# set a variable to point to the URL of the HDF5 source.
# since we manually downloaded this, it will look like below
set(MPE_URL ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mpe2-2.4.9b.tar.gz)

# calculate the MD5 sum of the file downloaded and set it in a variable
set(MPE_URL_MD5 b2f66c2ed9628f13ef0b3cbe9734c543)

message(STATUS "Could NOT find MPE. Will build it.")

## add instructions to build the MPE source
set (NUMPROCS 8)
set(MPE_CONFIGURE "")

ExternalProject_Add(${MPE_PREFIX}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${MPE_PREFIX}
    URL ftp://ftp.mcs.anl.gov/pub/mpi/mpe/mpe2.tar.gz
    URL_MD5 ${MPE_URL_MD5}
    #SOURCE_DIR ${MPE_PREFIX}
    #SOURCE_SUBDIR ${MPE_PREFIX}/${MPE_PREFIX}
    CONFIGURE_COMMAND ./${MPE_PREFIX}/configure #${MPE_CONFIGURE}
    BUILD_COMMAND  make -j ${NUMPROCS} mpe_build_prefix=${MPE_PREFIX}
	BUILD_IN_SOURCE 1
	INSTALL_COMMAND make install
	LOG_DOWNLOAD 1
	LOG_CONFIGURE 1
	LOG_BUILD 1
	LOG_INSTALL 1
)

# get the unpacked source directory path
ExternalProject_Get_Property(${MPE_PREFIX} SOURCE_DIR)
message(STATUS "Source directory of ${MPE_PREFIX} ${SOURCE_DIR}")

set(HAVE_MPE true)
# set the include directory variable and include it
set(MPE_INCLUDE_DIRS ${SOURCE_DIR}/mpe/include)
#include_directories(${MPE_INCLUDE_DIRS})

# link the correct MPE directory when the project is in Debug or Release mode
set(MPE_LIBS mpe)
set(MPE_LIBRARY_DIRS ${SOURCE_DIR}/mpe/lib)
set(MPE_LIBRARIES "${MPE_LIBRARY_DIRS}/libmpe.a")
set(MPE_VERSION 2.2.4.9b)

message(STATUS "MPE libs are ${MPE_LIBS}")
message(STATUS "MPE include dirs are ${MPE_INCLUDE_DIRS}")
message(STATUS "MPE library dirs are ${MPE_LIBRARY_DIRS}")
message(STATUS "MPE version is ${MPE_VERSION}")
