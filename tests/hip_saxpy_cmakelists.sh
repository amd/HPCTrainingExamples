#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIP/saxpy

mkdir build && cd build
cmake ..
make
./saxpy
cd ..
rm -rf build
