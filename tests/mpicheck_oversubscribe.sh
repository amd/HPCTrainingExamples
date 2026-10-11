#!/bin/bash

# This test checks that the mpicheck launch wrapper catches a genuinely
# oversubscribed job. Six MPI ranks confined to one GPU hold 30 compute queues
# at the default GPU_MAX_HW_QUEUES=4, over the amdgpu runlist budget on both
# calibrated families (24 queues on gfx942, 21 on gfx90a). The wrapper must
# warn, lower GPU_MAX_HW_QUEUES to the largest value that fits, and still run
# the job. The value it picks is family-dependent (3 on gfx942, 2 on gfx90a),
# so the check is that it dropped below the default rather than a fixed number.

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Prefer the installed mpicheck module; fall back to the copy carried in this
# repo so the test still runs where mpicheck has not been installed yet.
module load mpicheck >/dev/null 2>&1
MPIRUN_CHECK=$(command -v mpirun_check)
if [ -z "${MPIRUN_CHECK}" ]; then
   MPIRUN_CHECK=${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_Oversubscription_HIP/mpirun_check.sh
fi
if [ ! -x "${MPIRUN_CHECK}" ]; then
   echo "Skip: no mpirun_check found; load the mpicheck module or run from a tree that carries it"
   exit 0
fi
echo "wrapper: ${MPIRUN_CHECK}"

if [ ! -d /sys/class/kfd/kfd/topology/nodes ]; then
   echo "Skip: no kernel fusion driver topology, so there is no GPU to oversubscribe"
   exit 0
fi

SRC_DIR=${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver6
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cd ${BUILD_DIR}

cmake ${SRC_DIR}
make -j

# Ver6 calls hipSetDevice(0), so masking the node down to one device is what
# puts all six ranks on a single GPU instead of spreading them over the node.
export ROCR_VISIBLE_DEVICES=0
# Ver6 allocates with hipMalloc and does not rely on page migration.
export HSA_XNACK=0
export GPU_MAX_HW_QUEUES=4
# The wrapper reads the launcher's rank flags first but falls back to Slurm's
# view of the job. An allocation sized differently from this launch line would
# otherwise decide the rank count for it.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE

LOG=${BUILD_DIR}/mpirun_check.log
${MPIRUN_CHECK} --recommended-settings -n 6 --map-by core:OVERSUBSCRIBE --bind-to core \
    ./GhostExchange -x 3 -y 2 -i 2000 -j 2000 -h 1 -t -c -I 100 > ${LOG} 2>&1
cat ${LOG}

FAILED=0

grep -q "WARNING: the amdgpu runlist will be oversubscribed" ${LOG} || {
   echo "FAILED: the wrapper did not report the oversubscription"
   FAILED=1
}

QUEUES=$(sed -n 's/.*setting GPU_MAX_HW_QUEUES=\([0-9]*\).*/\1/p' ${LOG} | tail -1)
case "${QUEUES}" in
   1|2|3) echo "the wrapper lowered GPU_MAX_HW_QUEUES to ${QUEUES}" ;;
   *)     echo "FAILED: the wrapper did not lower GPU_MAX_HW_QUEUES (got '${QUEUES}')"
          FAILED=1 ;;
esac

grep -q "Printing Timings" ${LOG} || {
   echo "FAILED: the job did not run to completion"
   FAILED=1
}

if [ ${FAILED} -eq 0 ]; then
   echo "PASSED!"
fi
exit ${FAILED}
