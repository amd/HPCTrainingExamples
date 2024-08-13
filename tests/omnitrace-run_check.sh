#!/bin/bash

# This test checks that the omnitrace-run
# binary exists and it is able to run and complete

OMNITRACE_VERSION=""

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--omnitrace-version : specifies the omnitrace version"
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
      "--omnitrace-version")
          shift
          OMNITRACE_VERSION=${1}
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
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
rm -rf build_for_test
mkdir build_for_test; cd build_for_test
cmake ../
make -j

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1<=6.1.2'` && echo $result
module unload rocm

if [[ "${OMNITRACE_VERSION}" != "" ]]; then
   OMNITRACE_VERSION="/${OMNITRACE_VERSION}"
fi

if [[ "${result}" ]]; then
   echo " ------------------------------- "
   echo " "
   echo "loaded omnitrace from AMD Research"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "module load omnitrace${OMNITRACE_VERSION}"
   echo " "
   echo " ------------------------------- "
   module show omnitrace${OMNITRACE_VERSION}
   module load omnitrace${OMNITRACE_VERSION}
else
   echo " ------------------------------- "
   echo " "
   echo "loaded omnitrace from ROCm"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "module load omnitrace${OMNITRACE_VERSION}"
   echo " "
   echo " ------------------------------- "
   module show omnitrace${OMNITRACE_VERSION}
   module load rocm
   module load omnitrace${OMNITRACE_VERSION}
   echo " "
fi

omnitrace-run -- ./compute_comm_overlap 2

cd ..
rm -rf build_for_test

popd

