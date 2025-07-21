#!/bin/bash

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

# Can be one to four GPUS. Four GPUs is the default.
NUM_GPUS=4
MATRIX_SIZE=200000

n=0
while [[ $# -gt 0 ]]
do
    case "${1}" in
       "--num-gpus")
          shift
          NUM_GPUS=${1}
	  reset-last
          ;;
       "--matrix-size")
          shift
          MATRIX_SIZE=${1}
	  reset-last
          ;;
        *)
          last ${1}
          ;;
    esac
    n=$((${n} + 1))
    shift
done

#if [[ "$NUM_GPUS" == 1 ]];then
#   P=1; Q=1;
if [[ "$NUM_GPUS" == 2 ]];then
   P=2; Q=1;
elif [[ "$NUM_GPUS" == 4 ]];then
   P=2; Q=2;
elif [[ "$NUM_GPUS" == 8 ]];then
   P=4; Q=2;
else
   echo "Error with requested number of GPUs requested $NUM_GPUS"
   exit
fi

#echo "Removing old version if it it exists"
rm -rf rocHPL
#echo "Getting apu branch version from https://github.com/ROCm/rocHPL"
git clone -b apu https://github.com/ROCm/rocHPL
#echo "Building HPL code"
module load amdclang openmpi rocm
cd rocHPL/
./install.sh --with-rocm=${ROCM_PATH} --with-mpi=/opt/rocmplus-6.1.1/openmpi
# Running HPL
echo "build/rocHPL/mpirun_rochpl -P ${P} -Q ${Q} -N ${MATRIX_SIZE} --NB 512 build/rocHPL/bin/rochpl |& tee rochpl.out"
build/rocHPL/mpirun_rochpl -P ${P} -Q ${Q} -N ${MATRIX_SIZE} --NB 512 build/rocHPL/bin/rochpl |& tee rochpl.out
echo ""
grep "Residual Check: PASSED" rochpl.out
if [ "${NUM_GPUS}" -gt "1" ]; then
   FOM=`grep "Final Score" rochpl.out |awk -v NUM_GPUS=$NUM_GPUS '{print "For all NUM_GPUS = " $3/1000}'`
   echo "FOM: $FOM TFlops"
fi

FOM=`grep "Final Score" rochpl.out |awk -v NUM_GPUS=$NUM_GPUS '{print "For single GPU = " $3/1000/NUM_GPUS}'`
echo "FOM: $FOM TFlops"

TEMP_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $5}' | echo $(cat $1)`
echo "TEMP: ${TEMP_LIST}"
SCLK_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $10}' | echo $(cat $1)`
echo "SCLK: ${SCLK_LIST}"
MCLK_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $11}' | echo $(cat $1)`
echo "MCLK: ${MCLK_LIST}"
PWR_LIST=`rocm-smi | grep -v "Junction" | grep '[0-9]' | awk '{print $14}' | echo $(cat $1)`
echo "PWR: ${PWR_LIST}"
