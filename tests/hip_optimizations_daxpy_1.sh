#!/bin/bash

cd ~/HPCTrainingExamples/HIP-Optimizations/daxpy
make daxpy_1
./daxpy_1 1000000

make clean
