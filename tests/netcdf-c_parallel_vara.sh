#!/bin/bash

module purge
module load netcdf-c
module load openmpi

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



