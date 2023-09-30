#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIP/hip-stream

mkdir build && cd build
cmake ..
make
./stream
cd ..
rm -rf build
