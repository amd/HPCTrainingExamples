#!/bin/bash

cd ~/HPCTrainingExamples/HIP-Optimizations/daxpy
make daxpy_3
./daxpy_3 1000000

make clean
