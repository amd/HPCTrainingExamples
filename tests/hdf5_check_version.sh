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
   module load cray-hdf5-parallel
   IS_CRAY=1
else
   module load hdf5
fi

h5dump --version

# Strict module-to-binary version link check. The "loose" regex in
# CTest only proves *some* h5dump emitted *some* semver -- it does
# not prove the loaded hdf5 modulefile and the resolved h5dump
# binary point at the same install. That mismatch class is what
# bit us on 2026-05-20 when bare `module load hdf5` started
# resolving to 2.1.1 while leaf modulefiles still chained to 1.x
# .so SONAMES. Capture the loaded module version and the tool's
# self-reported version and assert they agree.
if [ "${IS_CRAY}" -eq 1 ]; then
   echo "[MODULE_VERSION_CHECK_SKIPPED_CRAY] hdf5"
else
   LOADED_VER=$(module -t list 2>&1 | grep '^hdf5/' | head -1 | awk -F/ '{print $2}')
   REPORTED_VER=$(h5dump --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
   if [ -z "${LOADED_VER}" ]; then
      echo "[MODULE_VERSION_MISMATCH] no hdf5 module loaded"
      exit 1
   elif [ -z "${REPORTED_VER}" ]; then
      echo "[MODULE_VERSION_MISMATCH] hdf5/${LOADED_VER}: could not extract version from h5dump --version"
      exit 1
   elif [ "${LOADED_VER}" = "${REPORTED_VER}" ]; then
      echo "[MODULE_VERSION_MATCH] hdf5/${LOADED_VER} matches h5dump version ${REPORTED_VER}"
   else
      echo "[MODULE_VERSION_MISMATCH] hdf5/${LOADED_VER} != h5dump reported ${REPORTED_VER}"
      exit 1
   fi
fi

