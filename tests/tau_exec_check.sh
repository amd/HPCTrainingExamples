#!/bin/bash

TAU_TRACE=0
TAU_PROFILE=0

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--tau-trace : sets TAU_TRACE=1 - default is TAU_TRACE=0"
    echo "--tau-profile : sets TAU_PROFILE=1 - default is TAU_PROFILE=0"
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
      "--tau-trace")
          shift
          TAU_TRACE=1
          reset-last
          ;;
      "--tau-profile")
          shift
          TAU_PROFILE=1
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

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi
module load tau

export TAU_PROFILE=${TAU_PROFILE}
export TAU_TRACE=${TAU_TRACE}

rm -rf profile.0*
rm -rf tautrace.0*
make clean
make

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1>6.1.9'` && echo $result
if [[ "${result}" ]]; then
   mpirun -n 2 tau_exec -rocm -T rocm,rocprofsdk ./Jacobi_hip -g 2 1
else
   mpirun -n 2 tau_exec -T rocm,roctracer,rocprofiler ./Jacobi_hip -g 2 1
fi

ls
pprof

make clean
rm -rf profile.0*
rm -rf tautrace.0*

