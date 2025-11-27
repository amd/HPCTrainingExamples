#!/bin/bash

# This test checks that the rocprof-sys-avail
# (formerly) omnitrace-avail
# binary exists and it is able to write
# a config file

VERSION=""
TOOL_NAME="omnitrace"
TOOL_COMMAND="omnitrace"
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
echo "tool command is ${TOOL_COMMAND}-avail"
echo " "
echo " ------------------------------- "
module show ${TOOL_NAME}${VERSION}
module load ${TOOL_NAME}${VERSION}

${TOOL_COMMAND}-avail -G $PWD/.configure.cfg

rm .configure.cfg
