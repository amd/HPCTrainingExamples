#!/bin/bash

# This test checks that 
# rocprofiler-compute (formerly omniperf) profile runs

VERSION=""
TOOL_NAME="omniperf"
TOOL_COMMAND="omniperf"

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

module purge

module load rocm
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
pushd ${REPO_DIR}/HIP/saxpy
rm -rf build_for_test
mkdir build_for_test
cd build_for_test
cmake ..
make

result=`echo ${ROCM_VERSION} | awk '$1<=6.1.2'` && echo $result
result2=`echo ${ROCM_VERSION} | awk '$1>=6.3.0'` && echo $result2

if [[ "${VERSION}" != "" ]]; then
   VERSION="/${VERSION}"
fi

if [[ "${result}" ]]; then
   echo " ------------------------------- "
   echo " "
   echo "loaded ${TOOL_NAME} from AMD Research"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "module load ${TOOL_NAME}${VERSION}"
   echo " "
   echo " ------------------------------- "
   module show ${TOOL_NAME}${VERSION}
   module load ${TOOL_NAME}${VERSION}
else
   if [[ "${result2}" ]]; then
      TOOL_NAME="rocprofiler-compute"
      TOOL_COMMAND="rocprof-compute"
   fi
   echo " ------------------------------- "
   echo " "
   echo "loaded ${TOOL_NAME} from ROCm"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "module load ${TOOL_NAME}${VERSION}"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "tool command is ${TOOL_NAME}"
   echo " "
   echo " ------------------------------- "
   module show ${TOOL_NAME}${VERSION}
   module load ${TOOL_NAME}${VERSION}
   echo " "
fi

export HSA_XNACK=1
${TOOL_COMMAND} profile -n v1 --no-roof -- ./saxpy
${TOOL_COMMAND} analyze -p workloads/v1/* --block 7.1.0 7.1.1 7.1.2 7.1.0: Grid size 7.1.1: Workgroup size 7.1.2: Total Wavefronts

cd ..
rm -rf build_for_test
popd
