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
   module load netcdf-fortran
fi

# use the compiler used to build netcdf-fortran
FC=`nf-config --fc`

rm -rf netcdf-fortran_test
mkdir netcdf-fortran_test
cd netcdf-fortran_test
git clone https://github.com/Unidata/netcdf-fortran.git
$FC ./netcdf-fortran/examples/F90/pres_temp_4D_wr.F90 -I${NETCDF_F_ROOT}/include -L${NETCDF_F_ROOT}/lib -lnetcdff -L${NETCDF_C_ROOT}/lib -lnetcdf -o pres_temp_4D_wr
$FC ./netcdf-fortran/examples/F90/pres_temp_4D_rd.F90 -I${NETCDF_F_ROOT}/include -L${NETCDF_F_ROOT}/lib -lnetcdff -L${NETCDF_C_ROOT}/lib -lnetcdf -o pres_temp_4D_rd
./pres_temp_4D_wr
./pres_temp_4D_rd
cd ..
rm -rf netcdf-fortran_test



