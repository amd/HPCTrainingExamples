#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
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
   module -t list 2>&1 | grep -q "^rocm"
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

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   module load cray-netcdf-hdf5parallel
   MPIRUN=srun
   NETCDF_LIBS=""
   # Cray's cray-netcdf-hdf5parallel is built without PnetCDF (nc-config
   # --has-pnetcdf = no), so it cannot create classic-format files in parallel.
   # Force the example onto the NetCDF-4/HDF5 parallel path it does support.
   FORCE_NETCDF4=1
else
   module load netcdf-c
   module load openmpi
   MPIRUN=mpirun
   # The netcdf-c and pnetcdf modules put their lib dirs on LIBRARY_PATH, so
   # -lnetcdf / -lpnetcdf resolve without explicit -L paths.
   NETCDF_LIBS="-lnetcdf -lpnetcdf"
   FORCE_NETCDF4=0
fi

if [[ ${HDF5_ENABLE_PARALLEL} == "OFF" ]]; then
   # NETCDF has not been built with parallel I/O support
   echo "Skip"
fi


# use the compiler used to build netcdf-c
CC=`nc-config --cc`

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://github.com/Unidata/netcdf-c.git
if [[ "${FORCE_NETCDF4}" == "1" ]]; then
   # Add NC_NETCDF4 to the create mode so nc_create_par uses parallel HDF5
   # instead of the classic/PnetCDF path (unavailable in the Cray build).
   sed -i -E 's/cmode[[:space:]]*=[[:space:]]*NC_CLOBBER[[:space:]]*;/cmode = NC_CLOBBER | NC_NETCDF4;/' ./netcdf-c/examples/C/parallel_vara.c
   sed -i -E 's/nc_var_par_access\(ncid, NC_GLOBAL,/nc_var_par_access(ncid, varid,/g' ./netcdf-c/examples/C/parallel_vara.c
fi
$CC -O2 ./netcdf-c/examples/C/parallel_vara.c -o parallel_vara ${NETCDF_LIBS}
${MPIRUN} -n 4 ./parallel_vara testfile.nc
ncdump testfile.nc
