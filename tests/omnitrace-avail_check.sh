#!/bin/bash

# This test checks that the omnitrace-avail
# binary exists and it is able to write
# an Omnitrace cfg file

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

omnitrace-avail -G $PWD/.omnitrace.cfg

rm .omnitrace.cfg
