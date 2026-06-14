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

IS_CRAY=0
if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   module load cray-netcdf-hdf5parallel
   IS_CRAY=1
else
   module load netcdf-c
fi

ncdump -h

# Strict module-to-binary version link check; see rationale in
# hdf5_check_version.sh. ncdump -h with no file argument emits the
# usage banner which includes the netcdf-c library version on its
# first line ("netcdf library version X.Y.Z of ...").
if [ "${IS_CRAY}" -eq 1 ]; then
   echo "[MODULE_VERSION_CHECK_SKIPPED_CRAY] netcdf-c"
else
   LOADED_VER=$(module -t list 2>&1 | grep '^netcdf-c/' | head -1 | awk -F/ '{print $2}')
   REPORTED_VER=$(ncdump -h 2>&1 | grep -oE 'netcdf library version [0-9]+\.[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
   if [ -z "${LOADED_VER}" ]; then
      echo "[MODULE_VERSION_MISMATCH] no netcdf-c module loaded"
      exit 1
   elif [ -z "${REPORTED_VER}" ]; then
      echo "[MODULE_VERSION_MISMATCH] netcdf-c/${LOADED_VER}: could not extract version from ncdump -h"
      exit 1
   elif [ "${LOADED_VER}" = "${REPORTED_VER}" ]; then
      echo "[MODULE_VERSION_MATCH] netcdf-c/${LOADED_VER} matches ncdump version ${REPORTED_VER}"
   else
      echo "[MODULE_VERSION_MISMATCH] netcdf-c/${LOADED_VER} != ncdump reported ${REPORTED_VER}"
      exit 1
   fi
fi

