#!/usr/bin/env bash

# For the ROCm Compute Profiler, formerly Omniperf, some manipulation
# must be done to avoid an issues currently being fixed.  omniperf
# expects a single executable with no arguments, so we call it with the `single_process.sh`
# argument to avoid this.
#

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
TOOL_NAME="omniperf"
TOOL_COMMAND="omniperf"

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

export NPROCS=1

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1>6.1.2'`

if [[ "${result}" ]]; then
   TOOL_ORIGIN="ROCm"
fi
result=`echo ${ROCM_VERSION} | awk '$1>6.2.9'`
if [[ "${result}" ]]; then
   TOOL_NAME="rocprofiler-compute"
   TOOL_COMMAND="rocprof-compute"
fi

if [[ "${VERSION}" != "" ]]; then
   VERSION="/${VERSION}"
else
   VERSION=${ROCM_VERSION}
   VERSION="/${VERSION}"
fi

module load rocm
if [ ! -f "`which rocprof-compute`.bin" ]; then
   module load ${TOOL_NAME}${VERSION}
fi

pushd ${PROFILER_TOP_DIR}
if [ ! -f data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Execute the python script:
${TOOL_COMMAND} profile --no-roof --name cifar_100_single_proc -- \
${PROFILER_TOP_DIR}/no-profiling/single_process.sh

${TOOL_COMMAND} analyze -p workloads/cifar_100_single_proc/MI* -b 2.1.2 2.1.3 2.1.4 2.1.5
