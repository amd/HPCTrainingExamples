#!/bin/bash

# This test checks that hpcviewer
# returns the version

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
fi

module load hpctoolkit

if ! command -v xvfb-run &> /dev/null; then
  echo "Skip -- xvfb-run not found"
  exit 0
fi

HPCVIEWER_CONFIG=$(mktemp -d)
trap "rm -rf ${HPCVIEWER_CONFIG}" EXIT

xvfb-run -a hpcviewer -configuration "${HPCVIEWER_CONFIG}" --version
