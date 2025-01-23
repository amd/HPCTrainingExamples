#!/usr/bin/env bash

# This script launches the python executable with fixed arguments:

# to be updated and overruled in job launcher scripts, but 
# in case they aren't set, declare these here:

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--version : specifies the desired version"
    echo ""
    exit
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
   last() { send-error "Unsupported argument :: ${1}"; }
}

n=0
while [[ $# -gt 0 ]]
do
   case "${1}" in
      "--version")
          shift
          VERSION=${1}
          reset-last
          ;;
     "--help")
          usage
          ;;
      "--*")
          send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
          ;;
      *)
         last ${1}
         ;;
   esac
   n=$((${n} + 1))
   shift
done

TOOL_ORIGIN="AMD Research"
TOOL_NAME="omnitrace"
TOOL_COMMAND="omnitrace"

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1>6.1.2'` && echo $result

if [[ "${result}" ]]; then
   TOOL_ORIGIN="ROCm"
fi
result=`echo ${ROCM_VERSION} | awk '$1>6.2.9'` && echo $result
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

if [[ -z "${MASTER_ADDR}" ]]; then
    export MASTER_ADDR=`hostname`
fi

if [[ -z "${MASTER_PORT}" ]];
then
    export MASTER_PORT=1234
fi

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

# Set the configuration for the system profiler:
export RSP_CFG=${PROFILER_TOP_DIR}/rocm-system-profiler/${TOOL_NAME}.cfg

pushd ${PROFILER_TOP_DIR}
if [ ! -f data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Execute the python script:
srun --ntasks=4 \
${TOOL_COMMAND}-sample -c $RSP_CFG \
-I roctracer -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 20 \
--data-path ${PROFILER_TOP_DIR}/data
