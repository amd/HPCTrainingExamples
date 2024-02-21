#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/HIPFort/hipgemm
make gemm_global_sd
./gemm_global_sd

make clean
