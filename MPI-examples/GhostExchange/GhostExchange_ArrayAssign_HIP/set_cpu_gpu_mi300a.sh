#!/bin/bash

export global_rank=${OMPI_COMM_WORLD_RANK}
export local_rank=${OMPI_COMM_WORLD_LOCAL_RANK}
export ranks_per_node=${OMPI_COMM_WORLD_LOCAL_SIZE}

if [ -z "${NUM_CPUS}" ]; then
    let NUM_CPUS=96
fi

if [ -z "${RANK_STRIDE}" ]; then
    let RANK_STRIDE=$(( ${NUM_CPUS}/${ranks_per_node} ))
fi

if [ -z "${OMP_STRIDE}" ]; then
    let OMP_STRIDE=1
fi

if [ -z "${NUM_GPUS}" ]; then
    let NUM_GPUS=4
fi

if [ -z "${GPU_START}" ]; then
    let GPU_START=0
fi

if [ -z "${GPU_STRIDE}" ]; then
    let GPU_STRIDE=1
fi

cpu_list=($(seq 0 95))
let cpus_per_gpu=${NUM_CPUS}/${NUM_GPUS}
let cpu_start_index=$(( ($RANK_STRIDE*${local_rank})+${GPU_START}*$cpus_per_gpu ))
let cpu_start=${cpu_list[$cpu_start_index]}
let cpu_stop=$(($cpu_start+$OMP_NUM_THREADS*$OMP_STRIDE-1))

gpu_list=(0 1 2 3)
let ranks_per_gpu=$(((${ranks_per_node}+${NUM_GPUS}-1)/${NUM_GPUS}))
let my_gpu_index=$(($local_rank*$GPU_STRIDE/$ranks_per_gpu))+${GPU_START}
let my_gpu=${gpu_list[${my_gpu_index}]}

export GOMP_CPU_AFFINITY=$cpu_start-$cpu_stop:$OMP_STRIDE
export ROCR_VISIBLE_DEVICES=$my_gpu

"$@"

