#!/bin/bash

module load rocm

cd ~/HPCTrainingExamples/HIP/hip-stream

make stream
./stream
make clean
