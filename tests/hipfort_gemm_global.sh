#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/HIPFort/hipgemm
make gemm_global
./gemm_global

make clean
