#!/bin/bash

export GPU_NUMBER_STRIDE=`expr ${OMPI_COMM_WORLD_SIZE} / ${SLURM_GPUS}`
export GPU_NUMBER_INDEX=`expr ${OMPI_COMM_WORLD_LOCAL_RANK} / ${GPU_NUMBER_STRIDE}`
gpu_list=(0 1 3 2 4 5 7 6)
export my_gpu=${gpu_list[${GPU_NUMBER_INDEX}]}
#echo "MPI Rank  ${OMPI_COMM_WORLD_LOCAL_RANK} will run on index ${GPU_NUMBER_INDEX} GPU ${my_gpu}"

export ROCR_VISIBLE_DEVICES=$my_gpu

"$@"
