#!/bin/bash

cd ~/HPCTrainingExamples/HIP-Optimizations/daxpy
make daxpy_2
./daxpy_2 1000000

make clean
