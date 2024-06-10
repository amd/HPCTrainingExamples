#!/bin/bash
cp ${REPO_DIR}/HIP/vectorAdd/hip_cmakelists_batch.sh hip_cmakelists_batch.sh
chmod +x hip_cmakelists_batch.sh
./hip_cmakelists_batch.sh

rm -r hip_cmakelists_batch.sh
