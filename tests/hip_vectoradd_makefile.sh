#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIP/vectorAdd

make vectoradd
./vectoradd
make clean
