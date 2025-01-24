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

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

TOOL_ORIGIN="AMD Research"
TOOL_NAME="omnitrace"
TOOL_COMMAND="omnitrace"

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1>6.1.2'`

if [[ "${result}" ]]; then
   TOOL_ORIGIN="ROCm"
fi
result=`echo ${ROCM_VERSION} | awk '$1>6.2.9'`
if [[ "${result}" ]]; then
   TOOL_NAME="rocprofiler-systems"
   TOOL_COMMAND="rocprof-sys"
fi

if [[ "${VERSION}" != "" ]]; then
   VERSION="/${VERSION}"
else
   VERSION=${ROCM_VERSION}
   VERSION="/${VERSION}"
fi

module load ${TOOL_NAME}${VERSION}

# Set the configuration for the system profiler:
export RSP_CFG=${PROFILER_TOP_DIR}/rocm-system-profiler/${TOOL_NAME}.cfg

pushd ${PROFILER_TOP_DIR}
if [ ! -f data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Execute the python script:
${TOOL_COMMAND}-sample -c $RSP_CFG \
-I  kokkosp mpip ompt rocm-smi rocprofiler roctracer roctx rw-locks spin-locks -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 20 \
--data-path ${PROFILER_TOP_DIR}/data
