#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   if [ -z "$HIPCC" ]; then
      export HIPCC=`which hipcc`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module load cray-netcdf-hdf5parallel
else
   module load netcdf-c
fi
module load openmpi

if [[ ${HDF5_ENABLE_PARALLEL} == "OFF" ]]; then
   # NETCDF has not been built with parallel I/O support
   echo "Skip"
fi


# use the compiler used to build netcdf-c
CC=`nc-config --cc`

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://github.com/Unidata/netcdf-c.git
$CC -O2 ./netcdf-c/examples/C/parallel_vara.c -o parallel_vara -L${NETCDF_C_ROOT}/lib -lnetcdf -L${PNETCDF_ROOT}/lib -lpnetcdf
mpirun -n 4 ./parallel_vara testfile.nc
ncdump testfile.nc
