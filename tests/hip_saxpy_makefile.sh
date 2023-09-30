#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIP/saxpy

make saxpy
./saxpy
make clean
