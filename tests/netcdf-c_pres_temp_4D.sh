#!/bin/bash


module load netcdf-c

# use the compiler used to build netcdf-c
CC=`nc-config --cc`

rm -rf netcdf-c_test
mkdir netcdf-c_test
cd netcdf-c_test
git clone https://github.com/Unidata/netcdf-c.git
$CC ./netcdf-c/examples/C/pres_temp_4D_wr.c -lnetcdf -L${NETCDF_C_ROOT}/lib -o pres_temp_4D_wr
$CC ./netcdf-c/examples/C/pres_temp_4D_rd.c -lnetcdf -L${NETCDF_C_ROOT}/lib -o pres_temp_4D_rd
./pres_temp_4D_wr
./pres_temp_4D_rd
cd ..
rm -rf netcdf-c_test



