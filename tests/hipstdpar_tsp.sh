#!/bin/bash

mkdir tsp
git clone https://github.com/pkestene/tsp
cd tsp
git checkout 51587
git clone https://raw.githubusercontent.com/ROCm/roc-stdpar/main/data/patches/tsp/TSP.patch

patch -p1 < TSP.patch

cd stdpar


cd ~/HPCTrainingExamples/HIPStdPar/CXX/saxpy

export HSA_XNACK=1
module load llvm-latest
export STDPAR_PATH=${CPLUS_INCLUDE_PATH}
export STDPAR_CXX=${CXX}
export STDPAR_TARGET=gfx90a

export AMD_LOG_LEVEL=3

make tsp_clang_stdpar_gpu
./tsp_clang_stdpar_gpu

make clean
