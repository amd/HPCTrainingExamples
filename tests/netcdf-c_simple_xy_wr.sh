#!/bin/bash

module purge
module load netcdf-c

# use the compiler used to build netcdf-c
CC=`nc-config --cc`

rm -rf netcdf-c_test
mkdir netcdf-c_test
cd netcdf-c_test
wget https://www.unidata.ucar.edu/software/netcdf/examples/programs/simple_xy_wr.c
$CC simple_xy_wr.c -lnetcdf -L${NETCDF_C_ROOT}/lib -o simple_xy_wr
./simple_xy_wr
rm -rf netcdf-c_test



