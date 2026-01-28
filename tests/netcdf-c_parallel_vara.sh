#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
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

rm -rf netcdf-c_test
mkdir netcdf-c_test
cd netcdf-c_test
git clone https://github.com/Unidata/netcdf-c.git
$CC -O2 ./netcdf-c/examples/C/parallel_vara.c -o parallel_vara -L${NETCDF_C_ROOT}/lib -lnetcdf -L${PNETCDF_ROOT}/lib -lpnetcdf
mpirun -n 4 ./parallel_vara testfile.nc
ncdump testfile.nc
cd ..
rm -rf netcdf-c_test



