#!/bin/bash
#SBATCH --job-name=pytorch-training
#SBATCH --nodes=1
#SBATCH --ntasks=4 # N GPUs per node * number of nodes
#SBATCH --partition=LocalQ

# This script launches the python executable with fixed arguments:

# to be updated and overruled in job launcher scripts, but 
# in case they aren't set, declare these here:

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

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
if [ ! -f data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Profile the workload with PyTorch Profiler:
srun --nodes 1 --ntasks 4 python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --data-path ${PROFILER_TOP_DIR}/data/ --torch-profile

