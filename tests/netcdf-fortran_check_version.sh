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

IS_CRAY=0
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module load cray-netcdf-hdf5parallel
   IS_CRAY=1
else
   module load netcdf-fortran
fi

nf-config --version

# Strict module-to-binary version link check; see rationale in
# hdf5_check_version.sh. nf-config --version emits a single line
# like "netCDF-Fortran X.Y.Z".
if [ "${IS_CRAY}" -eq 1 ]; then
   echo "[MODULE_VERSION_CHECK_SKIPPED_CRAY] netcdf-fortran"
else
   LOADED_VER=$(module -t list 2>&1 | grep '^netcdf-fortran/' | head -1 | awk -F/ '{print $2}')
   REPORTED_VER=$(nf-config --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
   if [ -z "${LOADED_VER}" ]; then
      echo "[MODULE_VERSION_MISMATCH] no netcdf-fortran module loaded"
      exit 1
   elif [ -z "${REPORTED_VER}" ]; then
      echo "[MODULE_VERSION_MISMATCH] netcdf-fortran/${LOADED_VER}: could not extract version from nf-config --version"
      exit 1
   elif [ "${LOADED_VER}" = "${REPORTED_VER}" ]; then
      echo "[MODULE_VERSION_MATCH] netcdf-fortran/${LOADED_VER} matches nf-config version ${REPORTED_VER}"
   else
      echo "[MODULE_VERSION_MISMATCH] netcdf-fortran/${LOADED_VER} != nf-config reported ${REPORTED_VER}"
      exit 1
   fi
fi
