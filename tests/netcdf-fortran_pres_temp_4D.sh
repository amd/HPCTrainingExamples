#!/bin/bash

module purge
module load netcdf-fortran

# use the compiler used to build netcdf-fortran
FC=`nf-config --fc`

rm -rf netcdf-fortran_test
mkdir netcdf-fortran_test
cd netcdf-fortran_test
git clone https://github.com/Unidata/netcdf-fortran.git
$FC ./netcdf-fortran/examples/F90/pres_temp_4D_wr.F90 -I${NETCDF_FC_ROOT}/include -L${NETCDF_FC_ROOT}/lib -lnetcdff -o pres_temp_4D_wr
$FC ./netcdf-fortran/examples/F90/pres_temp_4D_rd.F90 -I${NETCDF_FC_ROOT}/include -L${NETCDF_FC_ROOT}/lib -lnetcdff -o pres_temp_4D_rd
./pres_temp_4D_wr
./pres_temp_4D_rd
cd ..
rm -rf netcdf-fortran_test



