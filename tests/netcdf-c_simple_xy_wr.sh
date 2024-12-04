#!/bin/bash

module purge
module load netcdf-c
module load rocm

# use the compiler used to build netcdf-c
CC=`nc-config --cc`

rm -rf netcdf-c_test
mkdir netcdf-c_test
cd netcdf-c_test
wget https://people.sc.fsu.edu/~jburkardt/c_src/netcdf_test/simple_xy_wr.c
$CC simple_xy_wr.c -lnetcdf -L/opt/rocmplus-6.2.4/netcdf/netcdf-c/lib -o simple_xy_wr
./simple_xy_wr
rm -rf netcdf-c_test



