#!/bin/bash
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cp ${REPO_DIR}/HIP/vectorAdd/hip_cmakelists_batch.sh hip_cmakelists_batch.sh
chmod +x hip_cmakelists_batch.sh
./hip_cmakelists_batch.sh

rm -r hip_cmakelists_batch.sh
