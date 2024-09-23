#!/bin/bash
let mygpu=${OMPI_COMM_WORLD_LOCAL_RANK}
export ROCR_VISIBLE_DEVICES=$mygpu
exec $*
