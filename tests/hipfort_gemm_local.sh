#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/HIPFort/hipgemm
make gemm_local
./gemm_local

make clean
