#!/bin/bash

module purge
module load netcdf-fortran

# use the compiler used to build netcdf-c
FC=`nf-config --fc`

rm -rf netcdf-fortran_test
mkdir netcdf-fortran_test
cd netcdf-fortran_test
wget https://www.unidata.ucar.edu/software/netcdf/examples/programs/simple_xy_wr.f 
$FC simple_xy_wr.f -I${NETCDF_FC_ROOT}/include -L${NETCDF_FC_ROOT}/lib -lnetcdff -o simple_xy_wr
./simple_xy_wr
rm -rf netcdf-fortran_test



