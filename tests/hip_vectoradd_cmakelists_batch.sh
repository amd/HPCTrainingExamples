#!/bin/bash
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
if [[ `sinfo | grep unk | wc -l` == 1 ]]; then
   echo "SLURM not configured properly -- SKIPPING"
else
   sbatch --wait ${REPO_DIR}/HIP/vectorAdd/hip_cmakelists_batch.sh

   grep PASSED! slurm-*.out
   rm  slurm-*.out
fi
