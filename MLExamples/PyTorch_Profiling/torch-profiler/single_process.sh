#!/usr/bin/env bash

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

if [[ -z "${NPROCS}" ]];
then
    export NPROCS=1
fi

SRCDIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
BUILDDIR=$(mktemp -d)
trap 'rm -rf ${BUILDDIR}' EXIT
cp -r ${SRCDIR}/* ${BUILDDIR}/
cd ${BUILDDIR}
PROFILER_TOP_DIR=$PWD

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

pushd ${PROFILER_TOP_DIR}
if [ ! -d data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Profile the workload with PyTorch Profiler:
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --data-path ${PROFILER_TOP_DIR}/data/ --torch-profile
