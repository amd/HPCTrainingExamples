#!/bin/bash

module purge
module load netcdf-fortran

# use the compiler used to build netcdf-c
FC=`nf-config --fc`

rm -rf netcdf-fortran_test
mkdir netcdf-fortran_test
cd netcdf-fortran_test
wget https://www.unidata.ucar.edu/software/netcdf/examples/programs/pres_temp_4D_wr.f90
$FC pres_temp_4D_wr.f90 -I${NETCDF_FC_ROOT}/include -L${NETCDF_FC_ROOT}/lib -lnetcdff -o pres_temp_4D_wr
./pres_temp_4D_wr
cd ..
rm -rf netcdf-fortran_test



