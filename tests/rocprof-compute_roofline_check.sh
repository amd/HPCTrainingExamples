#!/bin/bash

# This test checks that 
# omniperf roofline generation works

OMNIPERF_VERSION=""

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--omniperf-version : specifies the omniperf version"
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
      "--omniperf-version")
          shift
          OMNIPERF_VERSION=${1}
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
pushd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1
rm -rf build_for_test
mkdir build_for_test
cd build_for_test
cmake -DCMAKE_CXX_COMPILER=/opt/rocm-${ROCM_VERSION}/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm-${ROCM_VERSION}/bin/amdclang ..
make -j

result=`echo ${ROCM_VERSION} | awk '$1<=6.1.2'` && echo $result

if [[ "${OMNIPERF_VERSION}" != "" ]]; then
   OMNIPERF_VERSION="/${OMNIPERF_VERSION}"
fi	

if [[ "${result}" ]]; then
   echo " ------------------------------- "
   echo " "
   echo "loaded omniperf from AMD Research"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "module load omniperf${OMNIPERF_VERSION}"
   echo " "
   echo " ------------------------------- "
   module show omniperf${OMNIPERF_VERSION}
   module load omniperf${OMNIPERF_VERSION}
else
   echo " ------------------------------- "
   echo " "
   echo "loaded omniperf from ROCm"
   echo " "
   echo " ------------------------------- "
   echo " "
   echo "module load omniperf${OMNIPERF_VERSION}"
   echo " "
   echo " ------------------------------- "
   module show omniperf${OMNIPERF_VERSION}
   module load omniperf${OMNIPERF_VERSION}
   echo " "
fi   

export HSA_XNACK=1
unset ROOFLINE_BIN
omniperf profile -n rooflines_PDF --roof-only --kernel-names -- ./GhostExchange -x 1 -y 1 -i 200 -j 200 -h 2 -t -c -I 100

cd ..
rm -rf build_for_test
popd
