#!/bin/bash

# This test checks that
# rocprofiler-compute (formerly omniperf) analyze with mpi runs

VERSION=""
TOOL_NAME="omniperf"
TOOL_COMMAND="omniperf"
TOOL_ORIGIN="AMD Research"

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


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
pushd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1
rm -rf build_for_test
mkdir build_for_test
cd build_for_test
cmake -DCMAKE_CXX_COMPILER=/opt/rocm-${ROCM_VERSION}/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm-${ROCM_VERSION}/bin/amdclang ..
make -j

result=`echo ${ROCM_VERSION} | awk '$1>6.1.2'` && echo $result
if [[ "${result}" ]]; then
   TOOL_ORIGIN="ROCm"
fi
result=`echo ${ROCM_VERSION} | awk '$1>6.2.9'` && echo $result
if [[ "${result}" ]]; then
   TOOL_NAME="rocprofiler-compute"
   TOOL_COMMAND="rocprof-compute"
fi

if [[ "${VERSION}" != "" ]]; then
   VERSION="/${VERSION}"
   if [[ "${VERSION}" == "/develop" ]]; then
      TOOL_ORIGIN="source"
      TOOL_NAME="rocprofiler-compute"
      TOOL_COMMAND="rocprof-compute"
   fi
fi

echo " ------------------------------- "
echo " "
echo "loaded ${TOOL_NAME} from ${TOOL_ORIGIN}"
echo " "
echo " ------------------------------- "
echo " "
echo "module load ${TOOL_NAME}${VERSION}"
echo " "
echo " ------------------------------- "
echo " "
echo "tool command is ${TOOL_COMMAND}"
echo " "
echo " ------------------------------- "
module show ${TOOL_NAME}${VERSION}
if [ ! -f "`which rocprof-compute`.bin" ]; then
   module load ${TOOL_NAME}${VERSION}
fi

export HSA_XNACK=1
${TOOL_COMMAND} profile -n v1 --no-roof -- ./GhostExchange -x 1 -y 1 -i 200 -j 200 -h 2 -t -c -I 100

cd ..
rm -rf build_for_test
git checkout ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1/GhostExchange.hip
popd

