#!/bin/bash
let mygpu=${OMPI_COMM_WORLD_SIZE}-${OMPI_COMM_WORLD_LOCAL_RANK}-1
export ROCR_VISIBLE_DEVICES=$mygpu
exec $*
