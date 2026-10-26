#!/bin/bash
let mygpu=${OMPI_COMM_WORLD_LOCAL_RANK}
export HIP_VISIBLE_DEVICES=$mygpu
exec $*
