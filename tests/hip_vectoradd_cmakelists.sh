#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIP/vectorAdd

mkdir build && cd build
cmake ..
make
./vectoradd
cd ..
rm -rf build
