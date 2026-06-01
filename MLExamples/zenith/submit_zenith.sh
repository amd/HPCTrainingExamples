#!/bin/bash
#flux: --nodes=1
#flux: --exclusive
#flux: --time-limit=6h
#flux: --job-name=zenith.train
#
#SBATCH --jobname=zenith.train
#SBATCH --time=06:00:00
#SBATCH --account=tcp
#SBATCH --nodes=1
#SBATCH --gres=gpu:4


# path containing all of the LLMs here on the host, container location is always /app/models
SCRATCH="/p/lustre5"
MODELPATH="${SCRATCH}/`whoami`/hfmodels"
# path containing datasets for training
DATASETSPATH="${SCRATCH}/`whoami`/datasets"

# load the rocm module and python environment
ROCMVERSION=7.2.2
module load rocm/$ROCMVERSION

# number of nodes assigned to this batch job
#RESOURCEINFO=`flux resource info`
#NNODES=`echo ${RESOURCEINFO} | awk '{print $1}'`
#NCORES=`echo ${RESOURCEINFO} | awk '{print $3}'`
#NGPUS=`echo ${RESOURCEINFO} | awk '{print $5}'`
#CORESPERNODE=$((NCORES / NNODES))
#GPUSPERNODE=$((NGPUS / NNODES))
NNODES=1
NCORES=96
NGPUS=4
CORESPERNODE=96
GPUSPERNODE=4

# when the workload gets heavy, ray spews this annoying message - filter it out
WARNING_STRING="PlacementGroupCleaner"
echo "Booting up Ray containers on ${NNODES} nodes with ${CORESPERNODE} cores per node and ${GPUSPERNODE} gpus per node"
# Usage: zenith_container_launcher.sh <directory holding the LLMs> <directory holding datasets>
#stdbuf -oL flux run -N ${NNODES} --ntasks=${NNODES} --gpus-per-task=${GPUSPERNODE} --cores-per-task=${CORESPERNODE} zenith_container_launcher.sh ${MODELPATH} ${DATASETSPATH} 2>&1 | grep --line-buffered -v ${WARNING_STRING}
stdbuf -oL srun -N ${NNODES} --ntasks=${NNODES} --gpus-per-task=${GPUSPERNODE} --cores-per-task=${CORESPERNODE} zenith_container_launcher.sh ${MODELPATH} ${DATASETSPATH} 2>&1 | grep --line-buffered -v ${WARNING_STRING}

