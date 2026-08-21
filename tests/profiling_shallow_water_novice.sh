#!/bin/bash
# Build and run one stage of the Profiling-by-example shallow-water novice track.
# The stage directory name is the only argument, e.g.
#   ./profiling_shallow_water_novice.sh 3_block_32x32

STAGE=$1

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/Profiling-by-example/shallow-water/novice/${STAGE}

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

cmake ${SRC_DIR}
make
./shallow
