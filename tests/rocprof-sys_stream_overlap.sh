#!/bin/bash

# This test checks that rocprof-sys
# is able to produce .proto files
# NOTE: the test does not check whether
# what is in those files is correct


VERSION=""
TOOL_NAME="omnitrace"
TOOL_COMMAND="omnitrace"
TOOL_ORIGIN="AMD Research"
TOOL_CONFIG="OMNITRACE"
TOOL_OUTPUT="omnitrace"

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

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
rm -rf build_for_test
mkdir build_for_test; cd build_for_test
cmake ../
make -j

result=`echo ${ROCM_VERSION} | awk '$1>6.1.2'` && echo $result
if [[ "${result}" ]]; then
   TOOL_ORIGIN="ROCm"
fi
result=`echo ${ROCM_VERSION} | awk '$1>6.2.9'` && echo $result
if [[ "${result}" ]]; then
   TOOL_NAME="rocprofiler-systems"
   TOOL_COMMAND="rocprof-sys"
   TOOL_CONFIG="ROCPROFSYS"
   TOOL_OUTPUT="rocprofsys"
fi

if [[ "${VERSION}" != "" ]]; then
   VERSION="/${VERSION}"
else
   VERSION=${ROCM_VERSION}
   VERSION="/${VERSION}"
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
echo "tool commands are:"
echo "${TOOL_COMMAND}-avail"
echo "${TOOL_COMMAND}-instrument"
echo "${TOOL_COMMAND}-run"
echo " "
echo " ------------------------------- "
module show ${TOOL_NAME}${VERSION}
module load ${TOOL_NAME}${VERSION}

${TOOL_COMMAND}-avail -G $PWD/.configure.cfg
export ${TOOL_CONFIG}_CONFIG_FILE=$PWD/.configure.cfg
${TOOL_COMMAND}-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
${TOOL_COMMAND}-run -- ./compute_comm_overlap.inst 2
cd ${TOOL_OUTPUT}-compute_comm_overlap.inst-output/
ls *

cd ..
rm -rf build_for_test

popd

