#!/bin/bash

cd ~/HPCTrainingExamples/HIP-Optimizations/daxpy
make daxpy_5
./daxpy_5 1000000

make clean
