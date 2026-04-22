#!/usr/bin/env bash

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# This script launches the python executable with fixed arguments:

# to be updated and overruled in job launcher scripts, but 
# in case they aren't set, declare these here:

if [[ -z "${MASTER_ADDR}" ]]; then
    export MASTER_ADDR=`hostname`
fi

if [[ -z "${MASTER_PORT}" ]];
then
    export MASTER_PORT=1234
fi

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

pushd ${PROFILER_TOP_DIR}
if [ ! -d data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Create the configuration for the system profiler:
export RSP_CFG=${PROFILER_TOP_DIR}/rocm-system-profiler/rocprofiler-systems.cfg
rocprof-sys-avail -G $RSP_CFG

# Execute the python script:
srun --ntasks 4 \
rocprof-sys-sample -c $RSP_CFG \
-I  all -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 20 \
--data-path ${PROFILER_TOP_DIR}/data

rm $RSP_CFG
