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
   module list 2>&1 | grep -q -w "rocm"
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



