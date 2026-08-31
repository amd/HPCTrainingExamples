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


module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `

# Version-sorted comparison so two-digit majors (10.x, 11.x, ...) order
# correctly. The old `awk '$1>6.2.9'` compared version strings lexically, so
# "10.1.0" tested as LESS THAN "6.2.9" and the ROCm-era tool names
# (rocprof-sys) were never selected -- the script then called the retired
# omnitrace-* commands, which do not exist in ROCm >= 10.
ver_gt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]; }

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
BUILD_DIR=$(mktemp -d $PWD/build_XXXXXX)
cd ${BUILD_DIR}
cmake ../
make -j

result=""; ver_gt "${ROCM_VERSION}" 6.1.2 && result="${ROCM_VERSION}"; echo $result
if [[ "${result}" ]]; then
   TOOL_ORIGIN="ROCm"
fi
result=""; ver_gt "${ROCM_VERSION}" 6.2.9 && result="${ROCM_VERSION}"; echo $result
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

module avail 2>&1 | grep -q -w "${TOOL_NAME}"
   if [ $? -eq 0 ]; then
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
fi

${TOOL_COMMAND}-avail -G $PWD/.configure.cfg
export ${TOOL_CONFIG}_CONFIG_FILE=$PWD/.configure.cfg
${TOOL_COMMAND}-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
${TOOL_COMMAND}-run -- ./compute_comm_overlap.inst 2
# Check for a real .proto artifact instead of matching tool output. The output
# directory name differs between omnitrace and rocprofiler-systems, so glob on
# the shared suffix and require a .proto inside it.
assert_proto_output() {
  local outdir n
  outdir="$(ls -d ./*-compute_comm_overlap.inst-output 2>/dev/null | head -1)"
  if [[ -z "${outdir}" || ! -d "${outdir}" ]]; then
    echo "ROCPROFSYS_STREAM_OVERLAP_RESULT: FAIL no *-compute_comm_overlap.inst-output directory produced"
    return 1
  fi
  # The tool nests a timestamped subdirectory, so recurse.
  n="$(find "${outdir}" -type f -name '*.proto' | wc -l)"
  find "${outdir}" -type f | sed 's/^/    /' | head -20
  if [[ "${n}" -gt 0 ]]; then
    echo "ROCPROFSYS_STREAM_OVERLAP_RESULT: PASS ${n} proto file(s) under ${outdir}"
  else
    echo "ROCPROFSYS_STREAM_OVERLAP_RESULT: FAIL 0 proto files under ${outdir}"
    return 1
  fi
}
assert_proto_output
cd ..
rm -rf ${BUILD_DIR}

popd

