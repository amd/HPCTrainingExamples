#!/bin/bash

# This test checks that hpcviewer
# returns the version

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load hpctoolkit

HPCVIEWER_CONFIG=$(mktemp -d)
trap "rm -rf ${HPCVIEWER_CONFIG}" EXIT

xvfb-run hpcviewer -configuration "${HPCVIEWER_CONFIG}" --version
