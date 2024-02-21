#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/HIPFort/hipgemm
make gemm_local_sd
./gemm_local_sd

make clean
